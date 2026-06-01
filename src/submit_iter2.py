"""
Iteration 2 submission: tuned global LightGBM.

Features = iter1 base + holidays (transferred-aware) + store metadata (oil and the
rmean56/growth "recency2" group were dropped — they hurt in ablation). Model = lr 0.03,
~600 trees, light regularization (the count the CV folds early-stopped near), L2-on-log1p,
2016+ history. Local fold-0 RMSLE 0.41765 (iter1 was 0.42751; iter1 LB 0.48509).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import RAW
from features import HORIZON, add_recency, build_panel

ITERATION = 2
TAG = "lgbm_holidays_storemeta_tuned"
TRAIN_START = pd.Timestamp("2016-01-01")
CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
FEATS = (["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
          "payday", "lag", "rmean7", "rmean28"]
         + ["is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event", "is_holiday"]
         + ["city", "state", "store_type", "cluster"])
PARAMS = dict(objective="regression", n_estimators=600, learning_rate=0.03, num_leaves=63,
              min_child_samples=100, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42)
OUT = Path(__file__).resolve().parents[1] / "submissions"

if __name__ == "__main__":
    panel, last_train = build_panel()
    panel = add_recency(panel, HORIZON)

    tr = panel[(panel["date"] <= last_train) & (panel["date"] >= TRAIN_START)]
    te = panel[panel["date"] > last_train].copy()
    catf = [c for c in FEATS if c in CAT_ALL]
    print(f"ITERATION {ITERATION} ({TAG})")
    print(f"train {len(tr):,} rows ({TRAIN_START.date()}..{last_train.date()}) | test {len(te):,} | {len(FEATS)} features")

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(tr[FEATS], np.log1p(tr["sales"]), categorical_feature=catf)
    te["sales"] = np.clip(np.expm1(model.predict(te[FEATS])), 0, None)

    sub = te[["id", "sales"]].copy()
    sub["id"] = sub["id"].astype(int)
    sub = sub.sort_values("id").reset_index(drop=True)

    sample = pd.read_csv(RAW / "sample_submission.csv")
    assert len(sub) == len(sample) == 28512, f"row count {len(sub)}"
    assert set(sub["id"]) == set(sample["id"]), "id mismatch vs sample_submission"
    assert sub["sales"].notna().all() and (sub["sales"] >= 0).all(), "bad predictions"

    out_path = OUT / f"iter{ITERATION}_{TAG}.csv"
    sub.to_csv(out_path, index=False)
    print(f"wrote {len(sub):,} rows -> {out_path}")
    print(sub["sales"].describe())
