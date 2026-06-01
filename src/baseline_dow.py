"""
Step B: recent-window mean, now split by day-of-week.

Step A predicted one flat number per series. Grocery demand swings by weekday
(weekends >> midweek), so here we average each series *per weekday* over a recent
window. Sweep the window again: per-weekday cells need more samples, so the
bias-variance sweet spot may shift longer than Step A's 14 days.
"""
import numpy as np
import pandas as pd

from validation import rmsle, time_train_holdout_split, RAW

if __name__ == "__main__":
    train = pd.read_csv(
        RAW / "train.csv", usecols=["date", "store_nbr", "family", "sales"]
    )
    tr, hold, cutoff = time_train_holdout_split(train)

    # day-of-week: 0=Mon .. 6=Sun
    tr = tr.assign(dow=pd.to_datetime(tr["date"]).dt.dayofweek)
    hold = hold.assign(dow=pd.to_datetime(hold["date"]).dt.dayofweek)

    y = hold["sales"].values
    keys = ["store_nbr", "family", "dow"]
    tr_dates = np.sort(tr["date"].unique())

    print(f"holdout cutoff {cutoff} | Step A best (flat 14d): 0.51911\n")
    print(f"{'window':>8}  {'RMSLE':>8}  {'~factor off':>11}")
    for n in [7, 14, 21, 28, 56, 84, 140, len(tr_dates)]:
        start = tr_dates[-n]
        window = tr[tr["date"] >= start]
        pred_tbl = window.groupby(keys)["sales"].mean().rename("pred").reset_index()
        preds = hold.merge(pred_tbl, on=keys, how="left")["pred"].fillna(0)
        score = rmsle(y, preds.values)
        label = "all" if n == len(tr_dates) else f"{n}d"
        print(f"{label:>8}  {score:>8.5f}  {np.exp(score):>10.2f}x")
