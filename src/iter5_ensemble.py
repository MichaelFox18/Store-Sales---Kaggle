"""
Iteration 5 (final) candidate: ENSEMBLE for variance reduction.

iter3 (simple + promo) is our best LB (0.43893); new features keep not transferring.
A low-risk last-mile gain is to average several decorrelated models in log space:
  - ens_seeds: the iter3 config trained with 4 different seeds (bagging diversity)
  - ens_div:   3 iter3 seeds + iter4 (adds promo_lead) for feature diversity
Compared to a single iter3 on July2017 + Aug2014/15/16. We ship the variant that is <=
single across the recent seasons (ensembles should reduce error, never inflate it).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle
from features import HORIZON, add_promo, add_promo_lead, add_recency, build_panel

CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
FEATS3 = (["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
           "payday", "lag", "rmean7", "rmean28"]
          + ["promo_trail7", "promo_trail28", "store_promo_day"])
FEATS4 = FEATS3 + ["promo_lead7"]
PB = dict(objective="regression", n_estimators=150, learning_rate=0.05, num_leaves=63,
          min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
          n_jobs=-1, verbose=-1)
HOLDOUTS = [("Jul2017", "2017-07-31"), ("Aug2016", "2016-08-16"),
            ("Aug2015", "2015-08-16"), ("Aug2014", "2014-08-16")]


def logpred(tr, X, feats, seed):
    m = lgb.LGBMRegressor(**dict(PB, random_state=seed))
    m.fit(tr[feats], np.log1p(tr["sales"]), categorical_feature=[c for c in feats if c in CAT_ALL])
    return m.predict(X[feats])  # log1p space


def eval_holdout(panel, start, lookback_months=18):
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=HORIZON - 1)
    tr = panel[(panel["date"] < start) & (panel["date"] >= start - pd.DateOffset(months=lookback_months))]
    ho = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    y = ho["sales"].values
    p3 = {s: logpred(tr, ho, FEATS3, s) for s in (42, 7, 123, 2024)}
    p4 = logpred(tr, ho, FEATS4, 42)
    score = lambda logp: rmsle(y, np.clip(np.expm1(logp), 0, None))
    single = score(p3[42])
    ens_seeds = score(np.mean([p3[42], p3[7], p3[123], p3[2024]], axis=0))
    ens_div = score(np.mean([p3[42], p3[7], p3[123], p4], axis=0))
    return single, ens_seeds, ens_div


if __name__ == "__main__":
    panel, last = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = add_promo(panel)
    panel = add_promo_lead(panel)
    panel = panel[panel["date"] <= last]

    print(f"{'holdout':>10}  {'single':>8}  {'ens_seeds':>9}  {'ens_div':>8}")
    for name, start in HOLDOUTS:
        s, es, ed = eval_holdout(panel, start)
        print(f"{name:>10}  {s:>8.5f}  {es:>9.5f}  {ed:>8.5f}", flush=True)
    print("\nens should be <= single. recent seasons (Jul2017/Aug2016/Aug2015) decide.")
