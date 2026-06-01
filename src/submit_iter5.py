"""
Iteration 5 (FINAL) submission: log-space ensemble.

Average of 4 decorrelated models (predictions averaged in log1p space, then expm1):
  - iter3 config (simple + promo)      x 3 seeds  (42, 7, 123)
  - iter4 config (+ promo_lead)        x 1 seed   (42)  -- adds feature diversity
This "ens_div" beat a single iter3 on ALL multi-season holdouts (Jul2017 + Aug2014/15/16),
unlike pure seed-averaging. iter3 single LB was 0.43893; ensembles reduce variance.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import RAW
from features import HORIZON, add_promo, add_promo_lead, add_recency, build_panel

ITERATION = 5
TAG = "ensemble_iter3x3_iter4"
TRAIN_START = pd.Timestamp("2016-01-01")
CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
FEATS3 = (["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
           "payday", "lag", "rmean7", "rmean28"]
          + ["promo_trail7", "promo_trail28", "store_promo_day"])
FEATS4 = FEATS3 + ["promo_lead7"]
MEMBERS = [(FEATS3, 42), (FEATS3, 7), (FEATS3, 123), (FEATS4, 42)]
PB = dict(objective="regression", n_estimators=150, learning_rate=0.05, num_leaves=63,
          min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
          n_jobs=-1, verbose=-1)
OUT = Path(__file__).resolve().parents[1] / "submissions"

if __name__ == "__main__":
    panel, last_train = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = add_promo(panel)
    panel = add_promo_lead(panel)

    tr = panel[(panel["date"] <= last_train) & (panel["date"] >= TRAIN_START)]
    te = panel[panel["date"] > last_train].copy()
    print(f"ITERATION {ITERATION} ({TAG})")
    print(f"train {len(tr):,} rows ({TRAIN_START.date()}..{last_train.date()}) | test {len(te):,} | {len(MEMBERS)} members")

    log_preds = []
    for feats, seed in MEMBERS:
        m = lgb.LGBMRegressor(**dict(PB, random_state=seed))
        m.fit(tr[feats], np.log1p(tr["sales"]), categorical_feature=[c for c in feats if c in CAT_ALL])
        log_preds.append(m.predict(te[feats]))
        print(f"  trained member: {len(feats)} feat, seed {seed}", flush=True)

    te["sales"] = np.clip(np.expm1(np.mean(log_preds, axis=0)), 0, None)  # average in log space

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
