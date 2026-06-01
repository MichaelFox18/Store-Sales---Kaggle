"""
Phase 2 validation harness: a time-based holdout that mimics the real task.

The Kaggle test set is 16 future days (2017-08-16 .. 2017-08-31) with no labels.
To estimate our leaderboard score locally -- and tune for FREE without spending
submissions -- we carve the LAST 16 days off train as a 'fake test' set, train on
everything strictly before, and score with the competition metric (RMSLE).

NEVER use a random/shuffled split here: it would let the model train on future
days while predicting past ones (temporal leakage) and report a falsely good
score. Split by calendar time only.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

HORIZON = 16  # the test set is 16 days; our holdout must match it.


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error -- the competition metric.

    Predictions are clipped at 0 first: sales can't be negative, and log1p of a
    negative number is nan. log1p(x) = log(1 + x), so zeros stay well-defined.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def time_train_holdout_split(df, horizon=HORIZON, date_col="date"):
    """Split a dated frame into (train_part, holdout) by calendar time.

    The holdout is the LAST `horizon` distinct dates; train_part is everything
    strictly before the cutoff. ISO date strings sort correctly as plain text.
    Returns (train_part, holdout, cutoff_date).
    """
    dates = np.sort(df[date_col].unique())
    cutoff = dates[-horizon]                 # first date that belongs to the holdout
    train_part = df[df[date_col] < cutoff]
    holdout = df[df[date_col] >= cutoff]
    return train_part, holdout, cutoff


if __name__ == "__main__":
    # Demo: score the per-series mean baseline on the holdout (our first real number).
    train = pd.read_csv(
        RAW / "train.csv", usecols=["date", "store_nbr", "family", "sales"]
    )
    tr, hold, cutoff = time_train_holdout_split(train)
    print(f"cutoff (first holdout day): {cutoff}")
    print(
        f"train rows: {len(tr):,}  |  holdout rows: {len(hold):,}  "
        f"|  holdout days: {hold['date'].nunique()}"
    )

    # Fit the baseline on tr ONLY -- no peeking at the holdout ...
    series_mean = (
        tr.groupby(["store_nbr", "family"])["sales"].mean().rename("pred").reset_index()
    )
    # ... then predict every holdout row with its series' historical mean.
    preds = hold.merge(series_mean, on=["store_nbr", "family"], how="left")["pred"].fillna(0)

    score = rmsle(hold["sales"].values, preds.values)
    print(f"per-series mean baseline -- local RMSLE: {score:.5f}")
