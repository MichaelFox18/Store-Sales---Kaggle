"""
Step A: recent-window per-series mean.

The all-history mean under-predicts because sales trended up ~30%+. Fix: average
each series over only its last N days. Sweep N on the holdout to see (a) how much
recency helps and (b) whether fresher is always better or there's a sweet spot.

Caveat: picking the best N *on the holdout* is mild tuning-on-the-test; with one
holdout it's fine for one knob, but the honest score for the winner would come
from a separate window (or rolling-origin CV, a later upgrade).
"""
import numpy as np
import pandas as pd

from validation import rmsle, time_train_holdout_split, RAW

if __name__ == "__main__":
    train = pd.read_csv(
        RAW / "train.csv", usecols=["date", "store_nbr", "family", "sales"]
    )
    tr, hold, cutoff = time_train_holdout_split(train)
    y = hold["sales"].values
    keys = ["store_nbr", "family"]

    tr_dates = np.sort(tr["date"].unique())  # ascending; last entries are most recent

    print(f"holdout cutoff {cutoff} | floor to beat (all-history mean): 0.69453\n")
    print(f"{'window':>8}  {'RMSLE':>8}  {'~factor off':>11}")
    for n in [7, 14, 21, 28, 56, 84, 140, len(tr_dates)]:
        start = tr_dates[-n]                       # keep only the last n distinct dates
        window = tr[tr["date"] >= start]
        pred_tbl = window.groupby(keys)["sales"].mean().rename("pred").reset_index()
        preds = hold.merge(pred_tbl, on=keys, how="left")["pred"].fillna(0)
        score = rmsle(y, preds.values)
        label = "all" if n == len(tr_dates) else f"{n}d"
        print(f"{label:>8}  {score:>8.5f}  {np.exp(score):>10.2f}x")
