"""
Is our validation the wrong SEASON? The test is Aug 16-31; our rolling folds were
June/July. iter2 beat iter1 on those folds but LOST on the LB. This evaluates the simple
iter1 config vs the complex iter2 config on late-August holdouts of prior years (the
test's actual season), plus the July-2017 window for contrast.

If the August holdouts rank iter2 WORSE than iter1 (matching the LB: 0.485 vs 0.518),
then season-aligned validation tracks Kaggle and iter2 was an overfit. Each holdout
trains on the prior ~18 months; same leakage-safe shift-16 features.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle
from features import HORIZON, add_recency, build_panel

CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
BASE = ["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
        "payday", "lag", "rmean7", "rmean28"]
COMBO = BASE + ["is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event", "is_holiday",
                "city", "state", "store_type", "cluster"]
P1 = dict(objective="regression", n_estimators=150, learning_rate=0.05, num_leaves=63,
          min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
          n_jobs=-1, verbose=-1, random_state=42)
P2 = dict(objective="regression", n_estimators=600, learning_rate=0.03, num_leaves=63,
          min_child_samples=100, reg_lambda=1.0, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42)


def eval_holdout(panel, start, feats, params, lookback_months=18):
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=HORIZON - 1)
    tr = panel[(panel["date"] < start) & (panel["date"] >= start - pd.DateOffset(months=lookback_months))]
    ho = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    catf = [c for c in feats if c in CAT_ALL]
    m = lgb.LGBMRegressor(**params)
    m.fit(tr[feats], np.log1p(tr["sales"]), categorical_feature=catf)
    return rmsle(ho["sales"].values, np.clip(np.expm1(m.predict(ho[feats])), 0, None))


if __name__ == "__main__":
    panel, last = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = panel[panel["date"] <= last]

    holdouts = [("Jul2017 (old fold)", "2017-07-31"),
                ("Aug2016 (season)", "2016-08-16"),
                ("Aug2015 (season)", "2015-08-16"),
                ("Aug2014 (season)", "2014-08-16")]
    print(f"{'holdout':>20}  {'iter1':>8}  {'iter2':>8}  {'iter2-iter1':>11}")
    for name, start in holdouts:
        s1 = eval_holdout(panel, start, BASE, P1)
        s2 = eval_holdout(panel, start, COMBO, P2)
        flag = "iter2 worse" if s2 > s1 else "iter2 better"
        print(f"{name:>20}  {s1:>8.5f}  {s2:>8.5f}  {s2 - s1:>+11.5f}  {flag}", flush=True)
    print("\nLB truth: iter1 0.48509, iter2 0.51756 (iter2 WORSE by +0.0325).")
    print("If the Aug holdouts say 'iter2 worse', season-aligned validation tracks the LB.")
