"""
Compare three multi-step forecasting strategies for the 16-day horizon. All use
ONE global LightGBM with identical params, the same features, log1p target, and
the same time-based holdout -- so the only thing that varies is the strategy.

  1. direct_safe : single model, recency lags shifted >=16d (legal for all 16
                   days); predict the whole window at once.
  2. recursive   : single model with lag-1 recency; predict day-by-day, feeding
                   each day's prediction back in as the next day's lag.
  3. per_horizon : 16 models; model_h uses lags shifted by h (freshest legal lag
                   for the h-th day ahead); each predicts its own day.

Leakage rule everywhere: a recency feature for a row at date t uses only sales
on/before t - shift, and TRAIN rows wear the same handicap as HOLDOUT rows.
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from validation import rmsle, time_train_holdout_split, RAW, HORIZON

CAT = ["store_nbr", "family"]
ROLL = [7, 28]
TRAIN_START = pd.Timestamp("2016-01-01")  # cap history for runtime; lags still see earlier data
FEATURES = CAT + ["onpromotion", "dow", "month", "day", "is_weekend", "payday",
                  "lag", "rmean7", "rmean28"]
PARAMS = dict(objective="regression", n_estimators=150, learning_rate=0.05,
              num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, n_jobs=-1, verbose=-1)


def log(msg):
    print(msg, flush=True)


def load_panel():
    df = pd.read_csv(RAW / "train.csv",
                     usecols=["date", "store_nbr", "family", "sales", "onpromotion"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(CAT + ["date"]).reset_index(drop=True)
    df["store_nbr"] = df["store_nbr"].astype("category")
    df["family"] = df["family"].astype("category")
    df["sid"] = df.groupby(CAT, observed=True).ngroup()
    d = df["date"].dt
    df["dow"] = d.dayofweek
    df["month"] = d.month
    df["day"] = d.day
    df["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    df["payday"] = ((d.day == 15) | d.is_month_end).astype("int8")
    return df


def recency(df, shift, rolls=ROLL):
    """lag/rolling features for a given shift; leakage-safe by construction."""
    sh = df.groupby(CAT, observed=True)["sales"].shift(shift)
    out = pd.DataFrame({"lag": sh})
    base = df[CAT].copy()
    base["_s"] = sh.values
    for w in rolls:
        r = base.groupby(CAT, observed=True)["_s"].rolling(w).mean()
        out[f"rmean{w}"] = r.reset_index(level=[0, 1], drop=True)
    return out


def fit(train_df):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(train_df[FEATURES], np.log1p(train_df["sales"]), categorical_feature=CAT)
    return m


def predict(model, df):
    return np.clip(np.expm1(model.predict(df[FEATURES])), 0, None)


def leakage_check(panel, cutoff):
    """The shift-16 lag for the LAST holdout day must equal sales 16 days earlier
    (which is < cutoff). If it doesn't, our features are peeking. Abort if so."""
    df = pd.concat([panel, recency(panel, HORIZON)], axis=1)
    last = cutoff + pd.Timedelta(days=HORIZON - 1)        # max holdout date
    src = last - pd.Timedelta(days=HORIZON)               # = cutoff - 1 (< cutoff)
    row = df[(df["store_nbr"] == 1) & (df["family"] == "GROCERY I") & (df["date"] == last)]
    exp = panel[(panel["store_nbr"] == 1) & (panel["family"] == "GROCERY I")
                & (panel["date"] == src)]["sales"]
    got, want = float(row["lag"].iloc[0]), float(exp.iloc[0])
    log(f"leakage check: lag for {last.date()} = {got:.1f}, sales on {src.date()} = {want:.1f}  "
        f"-> {'PASS' if abs(got - want) < 1e-6 and src < cutoff else 'FAIL'}")
    if not (abs(got - want) < 1e-6 and src < cutoff):
        sys.exit("LEAKAGE CHECK FAILED -- aborting before trusting any score.")


def run_direct_safe(panel, cutoff):
    df = pd.concat([panel, recency(panel, HORIZON)], axis=1)
    tr = df[(df["date"] < cutoff) & (df["date"] >= TRAIN_START)]
    hold = df[df["date"] >= cutoff]
    model = fit(tr)
    return rmsle(hold["sales"].values, predict(model, hold))


def run_per_horizon(panel, cutoff, hold_dates):
    truth, preds = [], []
    for h, d in enumerate(hold_dates, start=1):
        df = pd.concat([panel, recency(panel, h)], axis=1)
        tr = df[(df["date"] < cutoff) & (df["date"] >= TRAIN_START)]
        model = fit(tr)
        day = df[df["date"] == d]
        preds.append(predict(model, day))
        truth.append(day["sales"].values)
        log(f"  per_horizon h={h:>2}/{HORIZON} ({d.date()}) done")
    return rmsle(np.concatenate(truth), np.concatenate(preds))


def run_recursive(panel, cutoff, hold_dates, shift=1):
    df = pd.concat([panel, recency(panel, shift)], axis=1)
    tr = df[(df["date"] < cutoff) & (df["date"] >= TRAIN_START)]
    model = fit(tr)

    start = hold_dates[0] - pd.Timedelta(days=max(ROLL) + shift + 2)
    recent = panel[panel["date"] >= start]
    mat = recent.pivot_table(index="sid", columns="date", values="sales")
    for d in hold_dates:
        mat[d] = np.nan  # never use actual holdout sales as a feature

    truth, preds = [], []
    for d in hold_dates:
        lag_col = mat[d - pd.Timedelta(days=shift)]
        day = panel[panel["date"] == d].copy()
        day["lag"] = day["sid"].map(lag_col)
        for w in ROLL:
            end = d - pd.Timedelta(days=shift)
            cols = pd.date_range(end - pd.Timedelta(days=w - 1), end)
            day[f"rmean{w}"] = day["sid"].map(mat.reindex(columns=cols).mean(axis=1))
        p = predict(model, day)
        mat.loc[day["sid"].values, d] = p  # feed prediction back in
        preds.append(p)
        truth.append(day["sales"].values)
    return rmsle(np.concatenate(truth), np.concatenate(preds))


if __name__ == "__main__":
    panel = load_panel()
    tr, hold, cutoff = time_train_holdout_split(panel)
    cutoff = pd.Timestamp(cutoff)
    hold_dates = [pd.Timestamp(d) for d in sorted(hold["date"].unique())]
    log(f"cutoff {cutoff.date()} | train from {TRAIN_START.date()} | "
        f"holdout {len(hold):,} rows, {len(hold_dates)} days")
    log("best naive so far: 0.49871\n")

    leakage_check(panel, cutoff)

    s1 = run_direct_safe(panel, cutoff)
    log(f"\n[1] direct_safe   RMSLE = {s1:.5f}")
    s2 = run_recursive(panel, cutoff, hold_dates)
    log(f"[2] recursive     RMSLE = {s2:.5f}")
    log("[3] per_horizon training 16 models...")
    s3 = run_per_horizon(panel, cutoff, hold_dates)
    log(f"[3] per_horizon   RMSLE = {s3:.5f}")

    log("\n=== SUMMARY (lower is better) ===")
    for name, s in sorted([("direct_safe", s1), ("recursive", s2), ("per_horizon", s3)],
                          key=lambda x: x[1]):
        log(f"  {name:<12} {s:.5f}   ~{np.exp(s):.2f}x off")
