import numpy as np
import pandas as pd

def implied_probs_from_decimal(odds: pd.Series) -> pd.Series:
    return 1.0 / odds.replace(0, np.nan)

def shin_fair_probs(odds_home, odds_draw, odds_away):
    o = np.array([odds_home, odds_draw, odds_away], dtype=float)
    if np.any(o <= 1.0) or np.any(~np.isfinite(o)):
        return (np.nan, np.nan, np.nan)
    imp = 1.0 / o
    s = imp.sum()
    z = 0.0
    for _ in range(50):
        denom = (o / (o - 1.0))
        num = (imp**2 / denom).sum()
        new_z = 1.0 - np.sqrt(1.0 - num)
        if not np.isfinite(new_z):
            break
        if abs(new_z - z) < 1e-7:
            z = new_z
            break
        z = new_z
    fair = (imp * (1 - z)) / (1 - z * imp)
    fair = np.clip(fair, 1e-9, 1.0)
    fair = fair / fair.sum()
    return tuple(fair.tolist())
