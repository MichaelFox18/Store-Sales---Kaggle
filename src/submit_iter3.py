"""
Iteration 3 submission: the SIMPLE iter1 model + promotion features (the only change).

After iter2's complexity overfit and regressed on the LB, iter3 deliberately keeps the
simple model (150 trees / lr 0.05, no extra tuning) and adds only promotions -- the one
change with strong causal justification (promo drives sales and is known for the test
window). Promo helped across July + recent Augusts in the multi-season check. Trained on
2016+ (the rich-promo era). Holidays/store-meta deferred (not yet multi-season validated).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import RAW
from features import HORIZON, add_promo, add_recency, build_panel

ITERATION = 3
TAG = "lgbm_promo"
TRAIN_START = pd.Timestamp("2016-01-01")
CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
FEATS = (["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
          "payday", "lag", "rmean7", "rmean28"]
         + ["promo_trail7", "promo_trail28", "store_promo_day"])
PARAMS = dict(objective="regression", n_estimators=150, learning_rate=0.05, num_leaves=63,
              min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=-1, verbose=-1, random_state=42)
OUT = Path(__file__).resolve().parents[1] / "submissions"

if __name__ == "__main__":
    panel, last_train = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = add_promo(panel)

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
