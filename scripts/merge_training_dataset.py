#!/usr/bin/env python
# SBM Phase 0 - Merge Training Dataset (Fixed Imports)

import sys, os
# ✅ Add repo root and src folder to path so GitHub Actions & local both work
sys.path.extend([os.getcwd(), os.path.join(os.getcwd(), "src")])

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from dateutil import parser as dtp

# ✅ Use updated import path (no "src." prefix)
from sbm.utils.odds import shin_fair_probs
from sbm.utils.names import build_name_map, canonicalize

def parse_date(x):
    try:
        return pd.to_datetime(dtp.parse(str(x))).tz_localize('UTC')
    except Exception:
        return pd.NaT

def discover(base: Path, patterns):
    files = []
    if patterns and isinstance(patterns, list):
        for p in patterns:
            files += [str(x) for x in base.rglob(p)]
    else:
        files = [str(x) for x in base.rglob('*.*')]
    return sorted(files)

def load_fdorg(paths, names):
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        col_home = next((c for c in ['HomeTeam','Home','home_team'] if c in df.columns), None)
        col_away = next((c for c in ['AwayTeam','Away','away_team'] if c in df.columns), None)
        col_date = next((c for c in ['Date','date','match_date'] if c in df.columns), None)
        if not all([col_home, col_away, col_date]):
            continue
        df = df.rename(columns={col_home:'home_raw', col_away:'away_raw', col_date:'date'})
        df['date'] = df['date'].apply(parse_date)
        df = df[df['date'].notna()]
        df['home_team'] = df['home_raw'].apply(lambda x: canonicalize(x, names, 'fd'))
        df['away_team'] = df['away_raw'].apply(lambda x: canonicalize(x, names, 'fd'))
        if 'FTHG' in df.columns: df = df.rename(columns={'FTHG':'goals_home'})
        if 'FTAG' in df.columns: df = df.rename(columns={'FTAG':'goals_away'})

        def pick(cols):
            for c in cols:
                if c in df.columns: return df[c]
            return pd.Series(np.nan, index=df.index)
        oh = pick(['PSH','B365H','WHH','IWH','VCH'])
        od = pick(['PSD','B365D','WHD','IWD','VCD'])
        oa = pick(['PSA','B365A','WHA','IWA','VCA'])
        fair = df.apply(lambda r: shin_fair_probs(r.get(oh.name, np.nan),
                                                 r.get(od.name, np.nan),
                                                 r.get(oa.name, np.nan)), axis=1)
        fair = pd.DataFrame(fair.tolist(), columns=['fair_home','fair_draw','fair_away'])
        league = df['Div'] if 'Div' in df.columns else 'UNKNOWN'
        out = pd.DataFrame({
            'date': df['date'],
            'league': league,
            'home_team': df['home_team'],
            'away_team': df['away_team'],
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
    season = df['date'].dt.year
    season = np.where(df['date'].dt.month >= 8, season, season - 1)
    df['season'] = season.astype('Int64')
    return df

def load_understat(paths, names):
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        rename = {'home_team':'home_raw','away_team':'away_raw','date':'date'}
        for k,v in list(rename.items()):
            if k not in df.columns and f"{k}_x" in df.columns:
                rename[f"{k}_x"] = v
        df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
        for c in ['xg_home','xg_away','goals_home','goals_away']:
            if c not in df.columns: df[c] = np.nan
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
        df['home_team'] = df['home_raw'].apply(lambda x: canonicalize(x, names, 'understat'))
        df['away_team'] = df['away_raw'].apply(lambda x: canonicalize(x, names, 'understat'))
        frames.append(df[['date','home_team','away_team','xg_home','xg_away','goals_home','goals_away']])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    raw = Path(cfg.get('raw_dir','data/raw'))
    names = build_name_map('config/team_dictionary.csv')
    fd = load_fdorg(discover(raw, cfg.get('football_data_hints', ['**/*.csv'])), names)
    if fd.empty:
        print("[ERROR] No FD.org CSVs parsed. Put them under data/raw/.")
        raise SystemExit(2)
    us = load_understat(discover(raw, cfg.get('understat_hints', ['**/*.parquet'])), names)
    if not us.empty:
        fd = fd.merge(us, on=['date','home_team','away_team'], how='left', suffixes=('',''))
    mk = fd['date'].dt.strftime('%Y%m%d') + '__' + fd['home_team'] + '__vs__' + fd['away_team']
    fd['match_key'] = mk

    keep = ['match_key','date','league','season','home_team','away_team','goals_home','goals_away',
            'xg_home','xg_away','fair_home','fair_draw','fair_away']
    extra = [c for c in fd.columns if c not in keep]
    out = fd[keep + extra]
    Path(cfg.get('training_parquet')).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg.get('training_parquet'), index=False)
    print(f"[OK] wrote {cfg.get('training_parquet')} with {len(out)} rows and {len(out.columns)} cols.")

if __name__ == "__main__":
    main()