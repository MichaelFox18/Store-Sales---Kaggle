"""
Iteration 2 hyperparameter tuning: the chosen features (BASE + holidays + store-meta,
2016+ history, L2-on-log1p) with more trees + lower learning rate + EARLY STOPPING.

Early stopping uses a time-based internal validation = the last 16 days of each fold's
training data (no peek at the holdout), so the tree count is tuned honestly. Compared
on the two most recent folds. Reference: combo @150 trees/lr0.05 -> fold0 0.42378.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle
from features import HORIZON, add_recency, build_panel

TRAIN_START = pd.Timestamp("2016-01-01")
CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
FEATS = (["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
          "payday", "lag", "rmean7", "rmean28"]
         + ["is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event", "is_holiday"]
         + ["city", "state", "store_type", "cluster"])
COMMON = dict(objective="regression", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=-1, verbose=-1, random_state=42, n_estimators=4000)
CONFIGS = {
    "lr0.05 leaves63":        dict(learning_rate=0.05, num_leaves=63, min_child_samples=100),
    "lr0.03 leaves63 reg":    dict(learning_rate=0.03, num_leaves=63, min_child_samples=100, reg_lambda=1.0),
    "lr0.02 leaves127 reg":   dict(learning_rate=0.02, num_leaves=127, min_child_samples=50, reg_lambda=1.0),
}


def eval_cfg(panel, dates, extra, folds=(0, 1)):
    catf = [c for c in FEATS if c in CAT_ALL]
    rmsles, iters = [], []
    for f in folds:
        hi = len(dates) - f * HORIZON
        hd = dates[hi - HORIZON:hi]
        trall = panel[(panel["date"] < hd[0]) & (panel["date"] >= TRAIN_START)]
        tdates = np.sort(trall["date"].unique())
        vstart = tdates[-HORIZON]                       # internal val = last 16 train days
        ft, val = trall[trall["date"] < vstart], trall[trall["date"] >= vstart]
        m = lgb.LGBMRegressor(**dict(COMMON, **extra))
        m.fit(ft[FEATS], np.log1p(ft["sales"]),
              eval_set=[(val[FEATS], np.log1p(val["sales"]))], eval_metric="rmse",
              categorical_feature=catf, callbacks=[lgb.early_stopping(50, verbose=False)])
        ho = panel[panel["date"].isin(hd)]
        p = np.clip(np.expm1(m.predict(ho[FEATS])), 0, None)
        rmsles.append(rmsle(ho["sales"].values, p))
        iters.append(int(m.best_iteration_))
    return rmsles, iters


if __name__ == "__main__":
    panel, last = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = panel[panel["date"] <= last]
    dates = np.sort(panel["date"].unique())

    print(f"{'config':>22}  {'fold0':>8}  {'fold1':>8}  {'avg':>8}  {'trees':>11}")
    for name, extra in CONFIGS.items():
        s, it = eval_cfg(panel, dates, extra)
        print(f"{name:>22}  {s[0]:>8.5f}  {s[1]:>8.5f}  {np.mean(s):>8.5f}  {str(it):>11}", flush=True)
    print("\nref: combo @150 trees/lr0.05 -> fold0 0.42378 | iter1 LB 0.48509")
