"""
Iteration 2 tuning: ablation-winning features (BASE + holidays + store-meta; oil and
recency2 dropped) with two bigger levers tested before submitting --
  (a) training history start: 2013 (full) vs 2016
  (b) objective: tweedie (fits non-negative, zero-inflated sales directly) vs L2-on-log1p
Evaluated on the 2 most recent folds (fold 0 primary), seeded.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle
from features import HORIZON, add_recency, build_panel

CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
FEATS = (["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
          "payday", "lag", "rmean7", "rmean28"]
         + ["is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event", "is_holiday"]
         + ["city", "state", "store_type", "cluster"])
COMMON = dict(n_estimators=150, learning_rate=0.05, num_leaves=63, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=-1,
              verbose=-1, random_state=42)


def eval_cfg(panel, dates, train_start, objective, folds=(0, 1)):
    catf = [c for c in FEATS if c in CAT_ALL]
    log_t = objective == "regression"
    params = dict(COMMON, objective=objective)
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.5
    out = []
    for f in folds:
        hi = len(dates) - f * HORIZON
        hd = dates[hi - HORIZON:hi]
        tr = panel[(panel["date"] < hd[0]) & (panel["date"] >= train_start)]
        ho = panel[panel["date"].isin(hd)]
        y = np.log1p(tr["sales"]) if log_t else tr["sales"]
        m = lgb.LGBMRegressor(**params)
        m.fit(tr[FEATS], y, categorical_feature=catf)
        p = m.predict(ho[FEATS])
        p = np.expm1(p) if log_t else p
        out.append(rmsle(ho["sales"].values, np.clip(p, 0, None)))
    return out


if __name__ == "__main__":
    panel, last = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = panel[panel["date"] <= last]
    dates = np.sort(panel["date"].unique())

    configs = [
        ("combo @2016 L2", pd.Timestamp("2016-01-01"), "regression"),
        ("combo @2013 L2", pd.Timestamp("2013-01-01"), "regression"),
        ("combo @2016 TW", pd.Timestamp("2016-01-01"), "tweedie"),
        ("combo @2013 TW", pd.Timestamp("2013-01-01"), "tweedie"),
    ]
    print(f"{'config':>16}  {'fold0':>8}  {'fold1':>8}  {'avg':>8}")
    for name, ts, obj in configs:
        s = eval_cfg(panel, dates, ts, obj)
        print(f"{name:>16}  {s[0]:>8.5f}  {s[1]:>8.5f}  {np.mean(s):>8.5f}", flush=True)
    print("\nref: iter1 fold0 0.42751 | seeded BASE fold0 0.42944 | iter1 LB 0.48509")
