#!/usr/bin/env python
# SBM Phase 0 — Baseline Backtest (FULL FILE, UPDATED)

import sys, os
# ✅ Make Python see the repo root and the ./src package (works locally + in GitHub Actions)
sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), "src")])

import argparse, yaml
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import skellam
from sklearn.isotonic import IsotonicRegression


def skellam_probs(mu_h, mu_a, k_range=10):
    """
    Convert xG-home/xG-away to 1X2 probabilities by summing Skellam mass across goal-difference range.
    Returns (pH, pD, pA).
    """
    if not np.isfinite(mu_h) or not np.isfinite(mu_a) or mu_h < 0 or mu_a < 0:
        return (np.nan, np.nan, np.nan)
    diffs = np.arange(-k_range, k_range+1)
    pmf = skellam.pmf(diffs, mu_h, mu_a)
    pH = pmf[diffs > 0].sum()
    pD = pmf[diffs == 0].sum()
    pA = pmf[diffs < 0].sum()
    s = pH + pD + pA
    return (pH/s, pD/s, pA/s) if s > 0 else (np.nan, np.nan, np.nan)


def fit_isotonic_binary(y, p):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(p, y)
    return ir


def calibrate_triplet(y_true, P):
    """
    Multiclass isotonic via one-vs-all then renormalize.
    y_true: array of {0,1,2} (A, D, H)
    P: Nx3 raw probabilities
    """
    y = np.array(y_true)
    P = np.array(P)
    out = np.zeros_like(P)
    for k in range(3):
        yk = (y == k).astype(float)
        pk = P[:, k]
        ir = fit_isotonic_binary(yk, pk)
        out[:, k] = ir.transform(pk)
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def expected_calibration_error(y_true, proba, n_bins=15):
    """ECE for multiclass: mean bin-wise absolute gap between confidence and accuracy."""
    y = np.array(y_true)
    P = np.array(proba)
    bins = np.linspace(0, 1, n_bins+1)
    eces = []
    for k in range(3):
        pk = P[:, k]
        yk = (y == k).astype(float)
        for i in range(n_bins):
            m = (pk >= bins[i]) & (pk < bins[i+1])
            if m.sum() < 30:  # ignore tiny bins
                continue
            conf = pk[m].mean()
            acc = yk[m].mean()
            eces.append(abs(acc - conf) * m.mean())
    return float(np.nansum(eces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    k_range = int(cfg.get('skellam', {}).get('goal_diff_range', 10))
    frac = float(cfg.get('kelly', {}).get('fraction', 0.25))
    reports = Path(cfg.get('reports_dir', 'reports')); reports.mkdir(parents=True, exist_ok=True)

    # Load merged dataset
    df = pd.read_parquet(cfg.get('training_parquet', 'data/training/historical_training_dataset.parquet')).copy()

    # Raw Skellam probs from xG
    df['pH_raw'], df['pD_raw'], df['pA_raw'] = zip(*df.apply(
        lambda r: skellam_probs(r.get('xg_home', np.nan), r.get('xg_away', np.nan), k_range), axis=1
    ))

    # If xG missing, back off to fair 1X2 (still produces something calibratable)
    m = df[['pH_raw','pD_raw','pA_raw']].isna().any(axis=1)
    if m.any():
        df.loc[m, ['pH_raw','pD_raw','pA_raw']] = df.loc[m, ['fair_home','fair_draw','fair_away']].values

    # Labels: A=0, D=1, H=2
    y = np.where(df['goals_home'] > df['goals_away'], 2,
        np.where(df['goals_home'] < df['goals_away'], 0, 1))

    # Calibrate per league (skip small samples)
    chunks = []
    min_samples = int(cfg.get('calibration', {}).get('min_samples', 500))
    for lg, g in df.groupby('league'):
        if len(g) < min_samples:
            g[['pA_cal','pD_cal','pH_cal']] = g[['pA_raw','pD_raw','pH_raw']]
        else:
            P_cal = calibrate_triplet(y[g.index], g[['pA_raw','pD_raw','pH_raw']].values)
            g[['pA_cal','pD_cal','pH_cal']] = P_cal
        chunks.append(g)
    df = pd.concat(chunks).sort_index()

    # Global + per-league ECE
    ece_global = expected_calibration_error(y, df[['pA_cal','pD_cal','pH_cal']].values)
    per_lg = []
    for lg, g in df.groupby('league'):
        e = expected_calibration_error(y[g.index], g[['pA_cal','pD_cal','pH_cal']].values)
        per_lg.append({'league': lg, 'ece': float(e), 'rows': int(len(g))})
    pd.DataFrame(per_lg).to_csv(reports / "CALIBRATION_REPORT.csv", index=False)

    # ROI vs neutral “fair odds” benchmark (1/p_fair) — diagnostic only
    df['oddsH_fair'] = 1.0 / df['fair_home'].clip(1e-9, 1.0)
    df['oddsD_fair'] = 1.0 / df['fair_draw'].clip(1e-9, 1.0)
    df['oddsA_fair'] = 1.0 / df['fair_away'].clip(1e-9, 1.0)

    def kelly(p, odds, f):
        b = odds - 1.0
        edge = p*b - (1-p)
        if edge <= 0:
            return 0.0, edge
        k = edge / b
        return f*k, edge

    stakes = []; edges = []
    for ph, pd_, pa, oh, od, oa in zip(df['pH_cal'], df['pD_cal'], df['pA_cal'],
                                       df['oddsH_fair'], df['oddsD_fair'], df['oddsA_fair']):
        sh, eh = kelly(ph, oh, frac)
        sd, ed = kelly(pd_, od, frac)
        sa, ea = kelly(pa, oa, frac)
        stakes.append((sh, sd, sa))
        edges.append((eh, ed, ea))
    df[['stakeH','stakeD','stakeA']] = pd.DataFrame(stakes, index=df.index)
    df[['edgeH','edgeD','edgeA']] = pd.DataFrame(edges, index=df.index)

    # Simple realized ROI against fair odds (neutral)
    ret = []
    for _, r in df.iterrows():
        v = 0.0
        if r['stakeH'] > 0: v += r['stakeH'] * (r['oddsH_fair'] - 1.0) if r['goals_home'] > r['goals_away'] else -r['stakeH']
        if r['stakeD'] > 0: v += r['stakeD'] * (r['oddsD_fair'] - 1.0) if r['goals_home'] == r['goals_away'] else -r['stakeD']
        if r['stakeA'] > 0: v += r['stakeA'] * (r['oddsA_fair'] - 1.0) if r['goals_home'] < r['goals_away'] else -r['stakeA']
        ret.append(v)
    df['kelly_return'] = ret
    roi = float(np.nansum(df['kelly_return']))

    pd.DataFrame([{
        'rows': int(len(df)),
        'ece_multiclass': float(ece_global),
        'roi_fractional_kelly_vs_fair': roi
    }]).to_csv(reports / "BACKTEST_SUMMARY.csv", index=False)

    print(f"[OK] Reports written to {reports}. Global ECE={ece_global:.4f}")
    print("[Note] CLV is skipped unless opening/closing odds exist in inputs.")


if __name__ == "__main__":
    main()