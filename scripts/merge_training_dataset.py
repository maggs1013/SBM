#!/usr/bin/env python
# SBM Phase 0 — Merge Training Dataset (FULL FILE, UPDATED)

import sys, os
# ✅ Make Python see the repo root and the ./src package (works locally + in GitHub Actions)
sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), "src")])

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from dateutil import parser as dtp

# ✅ Import from src/ without the "src." prefix
from sbm.utils.odds import shin_fair_probs
from sbm.utils.names import build_name_map, canonicalize


def parse_date(x):
    """Robust date parser → UTC-aware Timestamp (or NaT)."""
    try:
        return pd.to_datetime(dtp.parse(str(x))).tz_localize('UTC')
    except Exception:
        return pd.NaT


def discover(base: Path, patterns):
    """Recursively collect files under base using glob patterns."""
    files = []
    if patterns and isinstance(patterns, list):
        for p in patterns:
            files += [str(x) for x in base.rglob(p)]
    else:
        files = [str(x) for x in base.rglob('*.*')]
    return sorted(files)


def load_fdorg(paths, names):
    """
    Load Football-Data.org style CSVs (results + odds).
    Returns a DataFrame with: date, league, home_team, away_team, goals_home/away, fair_home/draw/away, season
    """
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        # Flexible column resolution for team/date/goals
        col_home = next((c for c in ['HomeTeam','Home','home_team'] if c in df.columns), None)
        col_away = next((c for c in ['AwayTeam','Away','away_team'] if c in df.columns), None)
        col_date = next((c for c in ['Date','date','match_date'] if c in df.columns), None)
        if not all([col_home, col_away, col_date]):
            continue

        df = df.rename(columns={col_home:'home_raw', col_away:'away_raw', col_date:'date'})
        df['date'] = df['date'].apply(parse_date)
        df = df[df['date'].notna()]

        # Canonicalize team names
        df['home_team'] = df['home_raw'].apply(lambda x: canonicalize(x, names, 'fd'))
        df['away_team'] = df['away_raw'].apply(lambda x: canonicalize(x, names, 'fd'))

        # Goals (optional in some CSVs)
        if 'FTHG' in df.columns: df = df.rename(columns={'FTHG':'goals_home'})
        if 'FTAG' in df.columns: df = df.rename(columns={'FTAG':'goals_away'})

        # Grab odds (prefer Pinnacle PS*, else B365*, else others)
        def pick(cols):
            for c in cols:
                if c in df.columns: return df[c]
            return pd.Series(np.nan, index=df.index)

        oh = pick(['PSH','B365H','WHH','IWH','VCH'])
        od = pick(['PSD','B365D','WHD','IWD','VCD'])
        oa = pick(['PSA','B365A','WHA','IWA','VCA'])

        # Fair probabilities via Shin de-vig
        fair = df.apply(lambda r: shin_fair_probs(r.get(oh.name, np.nan),
                                                 r.get(od.name, np.nan),
                                                 r.get(oa.name, np.nan)), axis=1)
        fair = pd.DataFrame(fair.tolist(), columns=['fair_home','fair_draw','fair_away'])

        league = df['Div'] if 'Div' in df.columns else 'UNKNOWN'
        out = pd.DataFrame({
            'date': df['date'],
            'league': league.astype(str),
            'home_team': df['home_team'].astype(str),
            'away_team': df['away_team'].astype(str),
            'goals_home': df.get('goals_home', pd.Series(np.nan, index=df.index)),
            'goals_away': df.get('goals_away', pd.Series(np.nan, index=df.index)),
            'fair_home': fair['fair_home'],
            'fair_draw': fair['fair_draw'],
            'fair_away': fair['fair_away'],
        })
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Season label: Aug–May → season = year of August
    season = df['date'].dt.year
    season = np.where(df['date'].dt.month >= 8, season, season - 1)
    df['season'] = season.astype('Int64')
    return df


def load_understat(paths, names):
    """
    Load Understat (Parquet) match-level aggregates: xg_home/xg_away/goals_home/goals_away/date/home_team/away_team
    """
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue

        # Standardize column names
        rename = {'home_team':'home_raw','away_team':'away_raw','date':'date'}
        for k, v in list(rename.items()):
            if k not in df.columns and f"{k}_x" in df.columns:
                rename[f"{k}_x"] = v
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        for c in ['xg_home','xg_away','goals_home','goals_away']:
            if c not in df.columns:
                df[c] = np.nan

        df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
        df['home_team'] = df['home_raw'].apply(lambda x: canonicalize(x, names, 'understat'))
        df['away_team'] = df['away_raw'].apply(lambda x: canonicalize(x, names, 'understat'))

        frames.append(df[['date','home_team','away_team','xg_home','xg_away','goals_home','goals_away']])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    raw_dir = Path(cfg.get('raw_dir', 'data/raw'))
    fd_paths = discover(raw_dir, cfg.get('football_data_hints', ['**/*.csv']))
    us_paths = discover(raw_dir, cfg.get('understat_hints', ['**/*.parquet']))

    name_map = build_name_map('config/team_dictionary.csv')

    fd = load_fdorg(fd_paths, name_map)
    if fd.empty:
        print("[ERROR] No Football-Data.org CSVs parsed. Put them under data/raw/ .")
        raise SystemExit(2)

    us = load_understat(us_paths, name_map)
    if not us.empty:
        fd = fd.merge(us, on=['date','home_team','away_team'], how='left', suffixes=('',''))

    # match_key
    fd['match_key'] = (
        fd['date'].dt.strftime('%Y%m%d') + '__' + fd['home_team'] + '__vs__' + fd['away_team']
    )

    # Reorder & write
    keep = [
        'match_key','date','league','season','home_team','away_team',
        'goals_home','goals_away','xg_home','xg_away','fair_home','fair_draw','fair_away'
    ]
    extra = [c for c in fd.columns if c not in keep]
    out = fd[keep + extra]

    out_path = cfg.get('training_parquet', 'data/training/historical_training_dataset.parquet')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[OK] Wrote {out_path} with {len(out):,} rows and {len(out.columns)} columns.")


if __name__ == "__main__":
    main()