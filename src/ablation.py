"""
Feature ablation: add each iter2 group ALONE on top of the iter1 base and measure
the marginal effect on the two most recent folds (fold 0 is primary). Seeded so the
differences are real, not bagging noise. Keep only groups that help.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle
from features import HORIZON, add_recency, build_panel

TRAIN_START = pd.Timestamp("2016-01-01")
PARAMS = dict(objective="regression", n_estimators=150, learning_rate=0.05, num_leaves=63,
              min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              n_jobs=-1, verbose=-1, random_state=42)
CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
BASE = ["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
        "payday", "lag", "rmean7", "rmean28"]
GROUPS = {
    "+recency2":  ["rmean56", "growth"],
    "+oil":       ["dcoilwtico"],
    "+holidays":  ["is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event", "is_holiday"],
    "+storemeta": ["city", "state", "store_type", "cluster"],
}


def eval_folds(panel, dates, feats, folds=(0, 1)):
    catf = [c for c in feats if c in CAT_ALL]
    out = []
    for f in folds:
        hi = len(dates) - f * HORIZON
        hd = dates[hi - HORIZON:hi]
        tr = panel[(panel["date"] < hd[0]) & (panel["date"] >= TRAIN_START)]
        ho = panel[panel["date"].isin(hd)]
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tr[feats], np.log1p(tr["sales"]), categorical_feature=catf)
        out.append(rmsle(ho["sales"].values, np.clip(np.expm1(m.predict(ho[feats])), 0, None)))
    return out


if __name__ == "__main__":
    panel, last = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = panel[panel["date"] <= last]
    dates = np.sort(panel["date"].unique())

    configs = [("BASE", BASE)] + [(name, BASE + extra) for name, extra in GROUPS.items()]
    print(f"{'config':>12}  {'feat':>4}  {'fold0':>8}  {'fold1':>8}  {'avg':>8}")
    results = {}
    for name, feats in configs:
        s = eval_folds(panel, dates, feats)
        results[name] = s
        print(f"{name:>12}  {len(feats):>4}  {s[0]:>8.5f}  {s[1]:>8.5f}  {np.mean(s):>8.5f}", flush=True)

    b = results["BASE"]
    print("\nmarginal vs BASE (negative = better):")
    for name in GROUPS:
        s = results[name]
        print(f"  {name:>12}  fold0 {s[0] - b[0]:+.5f}   avg {np.mean(s) - np.mean(b):+.5f}")
