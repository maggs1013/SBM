# SBM — Soccer Betting Machine (Phase 0)

Welcome! This is the **beginner-friendly** seed repo for the Soccer Betting Machine (SBM).
You can run Phase 0 completely offline to build a clean, **normalized training dataset** and a **baseline backtest**.

> Phase 0 follows the blueprint and Council rules. You’ll produce:
> - `data/training/historical_training_dataset.parquet`
> - `reports/BACKTEST_SUMMARY.csv`
> - `reports/CALIBRATION_REPORT.csv`

---

## 0) Quick Start

**Prereqs:** Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Put source files under ./data/raw/ (CSV/Parquet)
python scripts/merge_training_dataset.py --config config/phase0_config.yaml
python scripts/backtest_baseline.py --config config/phase0_config.yaml
```

Artifacts:
- `data/training/historical_training_dataset.parquet`
- `reports/BACKTEST_SUMMARY.csv`
- `reports/CALIBRATION_REPORT.csv`

---

## Inputs (drop into `data/raw/`)

**Football-Data.org CSVs** (required): `Date, HomeTeam, AwayTeam, FTHG, FTAG, PSH/PSD/PSA or B365H/D/A ...`  
**Understat** (optional): Parquet with `xg_home, xg_away, goals_home, goals_away`  
**FBref** (optional): team-season aggregates (e.g., `sca90, gca90, pass_pct, psxg_prevented`)

Use `config/team_dictionary.csv` to map names.

---

## What happens

1) Normalize team names, build `match_key`  
2) Merge FD.org + Understat/FBref (if provided)  
3) Convert decimal odds → implied probs → **Shin de-vig** → `fair_home/draw/away`  
4) **Skellam baseline** (xG → 1X2), **isotonic calibration**, ROI & ECE reports

---

## Gates (Phase 0)

- ECE ≤ 0.05 on Must‑Have 9 leagues
- CLV ≥ 0 (skipped if open/close not present)

---

## CI: GitHub Actions

`.github/workflows/training-warehouse.yml` runs the end-to-end Phase 0 and uploads artifacts.
