"""
Iteration 2 model: direct_safe LightGBM on the enriched feature set (features.py) --
adds holidays (transferred-aware), store metadata, oil, and a trend/growth feature.
Same params/strategy as iter1, so the only change is the features.

Rolling-origin CV; fold 0 (most recent 16d) is the primary metric.
iter1 reference: fold 0 = 0.42751, rolling mean = 0.41346, LB = 0.48509.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle
from features import CAT, FEATURES, HORIZON, add_recency, build_panel

TRAIN_START = pd.Timestamp("2016-01-01")
N_FOLDS = 6
ITER1 = [0.42751, 0.40488, 0.38967, 0.40057, 0.41452, 0.44359]
PARAMS = dict(objective="regression", n_estimators=150, learning_rate=0.05,
              num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, n_jobs=-1, verbose=-1)


def fit(tr):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(tr[FEATURES], np.log1p(tr["sales"]), categorical_feature=CAT)
    return m


def predict(model, df):
    return np.clip(np.expm1(model.predict(df[FEATURES])), 0, None)


if __name__ == "__main__":
    panel, last_train = build_panel()
    panel = add_recency(panel, HORIZON)
    panel = panel[panel["date"] <= last_train]            # CV on train only
    dates = np.sort(panel["date"].unique())

    print(f"iter2: {len(FEATURES)} features (iter1 had 11)\n")
    print(f"{'fold':>4}  {'holdout window':>24}  {'iter2':>8}  {'iter1':>8}  {'delta':>8}")
    fold0_model, scores = None, []
    for f in range(N_FOLDS):
        hi = len(dates) - f * HORIZON
        hold_dates = dates[hi - HORIZON:hi]
        tr = panel[(panel["date"] < hold_dates[0]) & (panel["date"] >= TRAIN_START)]
        hold = panel[panel["date"].isin(hold_dates)]
        model = fit(tr)
        s = rmsle(hold["sales"].values, predict(model, hold))
        scores.append(s)
        if f == 0:
            fold0_model = model
        w = f"{pd.Timestamp(hold_dates[0]).date()}..{pd.Timestamp(hold_dates[-1]).date()}"
        print(f"{f:>4}  {w:>24}  {s:>8.5f}  {ITER1[f]:>8.5f}  {s - ITER1[f]:>+8.5f}", flush=True)

    scores = np.array(scores)
    print(f"\nfold 0 (primary): {scores[0]:.5f}  vs iter1 0.42751  ({scores[0] - 0.42751:+.5f})")
    print(f"mean:             {scores.mean():.5f}  vs iter1 0.41346  ({scores.mean() - 0.41346:+.5f})")

    imp = pd.Series(fold0_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\ntop features (fold 0):")
    print(imp.head(10).to_string())
