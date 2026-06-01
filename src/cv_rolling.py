"""
Rolling-origin cross-validation for the chosen model (direct_safe).

A single holdout said 0.428 but the leaderboard came back 0.485 -- that one window
was optimistic. This evaluates direct_safe over several consecutive 16-day windows
stepping back from the cutoff, for an honest, lower-variance estimate (and to see
whether the average lands near the LB). Same leakage-safe shift-16 features; each
fold trains only on data strictly before its window.
"""
import numpy as np
import pandas as pd

from validation import rmsle
from model_lgbm import HORIZON, TRAIN_START, fit, load_panel, predict, recency

N_FOLDS = 6

if __name__ == "__main__":
    panel = load_panel()
    panel = pd.concat([panel, recency(panel, HORIZON)], axis=1)
    dates = np.sort(panel["date"].unique())

    print(f"rolling-origin CV: {N_FOLDS} folds x {HORIZON}d, train from {TRAIN_START.date()}\n")
    print(f"{'fold':>4}  {'holdout window':>24}  {'RMSLE':>8}")
    scores = []
    for f in range(N_FOLDS):
        hi = len(dates) - f * HORIZON
        hold_dates = dates[hi - HORIZON:hi]
        start = hold_dates[0]
        tr = panel[(panel["date"] < start) & (panel["date"] >= TRAIN_START)]
        hold = panel[panel["date"].isin(hold_dates)]
        model = fit(tr)
        s = rmsle(hold["sales"].values, predict(model, hold))
        scores.append(s)
        w = f"{pd.Timestamp(hold_dates[0]).date()}..{pd.Timestamp(hold_dates[-1]).date()}"
        print(f"{f:>4}  {w:>24}  {s:>8.5f}", flush=True)

    scores = np.array(scores)
    print(f"\nmean {scores.mean():.5f}   std {scores.std():.5f}")
    print("(fold 0 = the original single holdout; the LB submission scored 0.48509)")
