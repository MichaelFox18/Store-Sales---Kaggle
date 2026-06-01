"""
Iteration 3 probe: robust promotion features on the SIMPLE (iter1) model.

After iter2 overfit, we bias to simplicity and judge changes by MULTI-SEASON agreement:
a real improvement should help (or at least not hurt) on BOTH the recent July window AND
the test-season August holdouts. Helping July but hurting August = overfitting (iter2's sin).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle
from features import HORIZON, add_promo, add_recency, build_panel

CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
BASE = ["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
        "payday", "lag", "rmean7", "rmean28"]
PROMO = ["promo_trail7", "promo_trail28", "store_promo_day"]
P1 = dict(objective="regression", n_estimators=150, learning_rate=0.05, num_leaves=63,
          min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
          n_jobs=-1, verbose=-1, random_state=42)


def eval_h(panel, start, feats, lookback_months=18):
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=HORIZON - 1)
    tr = panel[(panel["date"] < start) & (panel["date"] >= start - pd.DateOffset(months=lookback_months))]
    ho = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    catf = [c for c in feats if c in CAT_ALL]
    m = lgb.LGBMRegressor(**P1)
    m.fit(tr[feats], np.log1p(tr["sales"]), categorical_feature=catf)
    return rmsle(ho["sales"].values, np.clip(np.expm1(m.predict(ho[feats])), 0, None))


if __name__ == "__main__":
    panel, last = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = add_promo(panel)
    panel = panel[panel["date"] <= last]

    holdouts = [("Jul2017", "2017-07-31"), ("Aug2016", "2016-08-16"),
                ("Aug2015", "2015-08-16"), ("Aug2014", "2014-08-16")]
    print(f"{'holdout':>10}  {'base':>8}  {'+promo':>8}  {'delta':>9}")
    helps_all = True
    for name, start in holdouts:
        b = eval_h(panel, start, BASE)
        p = eval_h(panel, start, BASE + PROMO)
        helps_all = helps_all and (p <= b + 1e-4)
        print(f"{name:>10}  {b:>8.5f}  {p:>8.5f}  {p - b:>+9.5f}  {'helps' if p < b else 'hurts'}", flush=True)
    print(f"\npromo passes multi-season guard (helps/holds everywhere)? {helps_all}")
