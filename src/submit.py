"""
Submission builder. Currently: ITERATION 1 -- global LightGBM, direct_safe
strategy (lags >= 16d), the winner of the bake-off in model_lgbm.py.

Retrains the chosen model on ALL data from TRAIN_START through the last train day
(2017-08-15; the old holdout is no longer held out), then predicts the real test
window (2017-08-16..31, whose onpromotion is known) and writes a Kaggle-format
file. Local holdout RMSLE for this exact model: 0.42751.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from validation import RAW
from model_lgbm import CAT, FEATURES, HORIZON, ROLL, TRAIN_START, fit, predict, recency

ITERATION = 1
TAG = "lgbm_direct_safe"
LOCAL_RMSLE = 0.42751
OUT = Path(__file__).resolve().parents[1] / "submissions"
OUT.mkdir(exist_ok=True)


def add_features(df):
    """Same feature steps as model_lgbm.load_panel, for the combined train+test panel."""
    df = df.sort_values(CAT + ["date"]).reset_index(drop=True)
    df["store_nbr"] = df["store_nbr"].astype("category")
    df["family"] = df["family"].astype("category")
    df["sid"] = df.groupby(CAT, observed=True).ngroup()
    d = df["date"].dt
    df["dow"] = d.dayofweek
    df["month"] = d.month
    df["day"] = d.day
    df["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    df["payday"] = ((d.day == 15) | d.is_month_end).astype("int8")
    return df


if __name__ == "__main__":
    train = pd.read_csv(RAW / "train.csv",
                        usecols=["date", "store_nbr", "family", "sales", "onpromotion"])
    test = pd.read_csv(RAW / "test.csv",
                       usecols=["id", "date", "store_nbr", "family", "onpromotion"])
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    # One panel so test recency reads the real train tail. shift>=16 means every
    # test day's lag lands on actual train sales -> no leakage, no NaN.
    panel = add_features(pd.concat([train, test], ignore_index=True))
    panel = pd.concat([panel, recency(panel, HORIZON)], axis=1)

    last_train = train["date"].max()
    tr = panel[(panel["date"] <= last_train) & (panel["date"] >= TRAIN_START)]
    te = panel[panel["date"] > last_train].copy()
    print(f"ITERATION {ITERATION} ({TAG})")
    print(f"train {len(tr):,} rows ({TRAIN_START.date()}..{last_train.date()}) | test {len(te):,} rows")

    model = fit(tr)
    te["sales"] = predict(model, te)

    sub = te[["id", "sales"]].copy()
    sub["id"] = sub["id"].astype(int)
    sub = sub.sort_values("id").reset_index(drop=True)

    # Sanity vs the official sample_submission before we trust it.
    sample = pd.read_csv(RAW / "sample_submission.csv")
    assert len(sub) == len(sample) == 28512, f"row count {len(sub)} != 28512"
    assert set(sub["id"]) == set(sample["id"]), "id set does not match sample_submission"
    assert sub["sales"].notna().all() and (sub["sales"] >= 0).all(), "NaN or negative predictions"

    out_path = OUT / f"iter{ITERATION}_{TAG}.csv"
    sub.to_csv(out_path, index=False)
    print(f"wrote {len(sub):,} rows -> {out_path}")
    print(sub["sales"].describe())
