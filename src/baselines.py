"""
Baseline A/B: arithmetic vs geometric per-series mean, on the same holdout.

RMSLE scores predictions in log space, so the error-minimizing constant for a
series is the mean of log1p(sales) converted back with expm1 -- a geometric-
style mean, not the arithmetic one. We measure both instead of trusting theory.
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

    # (A) Arithmetic mean per series.
    arith_pred = tr.groupby(keys)["sales"].mean().rename("pred").reset_index()
    arith = rmsle(y, hold.merge(arith_pred, on=keys, how="left")["pred"].fillna(0).values)

    # (B) Geometric-style mean: average in log space, convert back. RMSLE-optimal.
    log_mean = tr.assign(ls=np.log1p(tr["sales"])).groupby(keys)["ls"].mean()
    geo_pred = np.expm1(log_mean).rename("pred").reset_index()
    geo = rmsle(y, hold.merge(geo_pred, on=keys, how="left")["pred"].fillna(0).values)

    print(f"holdout: {len(hold):,} rows, {hold['date'].nunique()} days (cutoff {cutoff})")
    print(f"(A) arithmetic mean  RMSLE: {arith:.5f}")
    print(f"(B) geometric  mean  RMSLE: {geo:.5f}")
    print(f"change B vs A: {geo - arith:+.5f}  ({'BETTER' if geo < arith else 'worse'})")
