"""
Phase 0 baseline: per-series historical mean.

For every (store_nbr, family) series, predict that series' average historical
sales for all 16 test days. The dumbest "real" ruler: it knows GROCERY sells
more than BABY CARE, but nothing about *when* (no day-of-week, trend, or promo).
"""
from pathlib import Path
import pandas as pd

# Repo-root-relative paths so this runs from anywhere.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "submissions"
OUT.mkdir(exist_ok=True)

# 1. Load only the columns we need (train.csv is 121 MB; don't read sales' siblings).
train = pd.read_csv(RAW / "train.csv", usecols=["store_nbr", "family", "sales"])
test = pd.read_csv(RAW / "test.csv", usecols=["id", "store_nbr", "family"])

# 2. One mean per (store, family) series -> 1,782 numbers.
series_mean = (
    train.groupby(["store_nbr", "family"])["sales"].mean().rename("sales").reset_index()
)

# 3. Attach each series' mean to its test rows (left join keeps all 28,512).
pred = test.merge(series_mean, on=["store_nbr", "family"], how="left")

# 4. Safety net: any series unseen in train would be NaN -> 0. Check, don't assume.
n_missing = int(pred["sales"].isna().sum())
print(f"test rows with no matching train series: {n_missing}")
pred["sales"] = pred["sales"].fillna(0)

# 5. Write submission in the exact required format: id, sales.
submission = pred[["id", "sales"]].sort_values("id")
out_path = OUT / "baseline_mean.csv"
submission.to_csv(out_path, index=False)

print(f"wrote {len(submission):,} rows -> {out_path}")
print(submission["sales"].describe())
print(f"distinct predicted values: {submission['sales'].nunique()}")
