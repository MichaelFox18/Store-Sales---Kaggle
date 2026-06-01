"""
Iteration 4 multi-season ablation: add each candidate ALONE on top of iter3 (base+promo),
simple model, across July2017 + Aug2014/15/16. Keep candidates that help on the RECENT,
test-relevant seasons (July2017 + Aug2016 + Aug2015); Aug2014 reported for context only
(early regime with sparse promo that misled us before).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle
from features import HORIZON, add_promo, add_promo_lead, add_recency, build_panel

CAT_ALL = {"store_nbr", "family", "city", "state", "store_type", "cluster"}
BASE3 = (["store_nbr", "family", "onpromotion", "dow", "month", "day", "is_weekend",
          "payday", "lag", "rmean7", "rmean28"]
         + ["promo_trail7", "promo_trail28", "store_promo_day"])
CANDS = {
    "+promo_lead": ["promo_lead7"],
    "+holidays":   ["is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event", "is_holiday"],
    "+storemeta":  ["city", "state", "store_type", "cluster"],
}
P1 = dict(objective="regression", n_estimators=150, learning_rate=0.05, num_leaves=63,
          min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
          n_jobs=-1, verbose=-1, random_state=42)
HOLDOUTS = [("Jul2017", "2017-07-31"), ("Aug2016", "2016-08-16"),
            ("Aug2015", "2015-08-16"), ("Aug2014", "2014-08-16")]
RECENT = {"Jul2017", "Aug2016", "Aug2015"}


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
    panel = add_promo_lead(panel)
    panel = panel[panel["date"] <= last]

    base = {name: eval_h(panel, start, BASE3) for name, start in HOLDOUTS}
    print("iter3 base (base+promo):")
    print("  " + "  ".join(f"{n} {base[n]:.4f}" for n, _ in HOLDOUTS) + "\n")

    print(f"{'candidate':>12}  " + "  ".join(f"{n:>9}" for n, _ in HOLDOUTS) + f"  {'recentAvgD':>10}  verdict")
    for cname, extra in CANDS.items():
        deltas = {}
        for name, start in HOLDOUTS:
            deltas[name] = eval_h(panel, start, BASE3 + extra) - base[name]
        recent = np.mean([deltas[n] for n, _ in HOLDOUTS if n in RECENT])
        cells = "  ".join(f"{deltas[n]:>+9.4f}" for n, _ in HOLDOUTS)
        print(f"{cname:>12}  {cells}  {recent:>+10.4f}  {'KEEP' if recent < 0 else 'drop'}", flush=True)
