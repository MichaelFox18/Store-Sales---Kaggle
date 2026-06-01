"""
Step C: multiplicative decomposition -- fresh level x stable weekday shape.

Step B showed weekday is real signal but cell-averaging trades away freshness to
get it. Here we decouple by hand:
    prediction = level_s * factor_{s,dow}
where level_s is the recent 14d mean (fresh, won Step A) and factor is a weekday
multiplier from a LONG window. The factor is a RATIO, so a uniform trend cancels
top and bottom -- the weekday *shape* stays stable even over a long window, while
the *level* stays fresh. Sweep the factor window to confirm the shape is stable.
"""
import numpy as np
import pandas as pd

from validation import rmsle, time_train_holdout_split, RAW

LEVEL_WINDOW = 14  # fresh level: Step A's winner

if __name__ == "__main__":
    train = pd.read_csv(
        RAW / "train.csv", usecols=["date", "store_nbr", "family", "sales"]
    )
    tr, hold, cutoff = time_train_holdout_split(train)
    tr = tr.assign(dow=pd.to_datetime(tr["date"]).dt.dayofweek)
    hold = hold.assign(dow=pd.to_datetime(hold["date"]).dt.dayofweek)
    y = hold["sales"].values
    tr_dates = np.sort(tr["date"].unique())

    # Fresh level: recent 14d mean per series.
    level = (
        tr[tr["date"] >= tr_dates[-LEVEL_WINDOW]]
        .groupby(["store_nbr", "family"])["sales"].mean().rename("level").reset_index()
    )

    print(f"level window = {LEVEL_WINDOW}d (fresh) | Step A flat: 0.51911, Step B dow: 0.52054\n")
    print(f"{'factor_win':>10}  {'RMSLE':>8}")
    for fw in [28, 56, 84, 140, len(tr_dates)]:
        win = tr[tr["date"] >= tr_dates[-fw]]
        smean = win.groupby(["store_nbr", "family"])["sales"].mean().rename("smean")
        dmean = win.groupby(["store_nbr", "family", "dow"])["sales"].mean().rename("dmean")
        factor = dmean.reset_index().merge(smean.reset_index(), on=["store_nbr", "family"])
        factor["factor"] = np.where(factor["smean"] > 0, factor["dmean"] / factor["smean"], 1.0)

        pred = (
            hold.merge(level, on=["store_nbr", "family"], how="left")
            .merge(factor[["store_nbr", "family", "dow", "factor"]],
                   on=["store_nbr", "family", "dow"], how="left")
        )
        preds = (pred["level"].fillna(0) * pred["factor"].fillna(1.0)).values
        score = rmsle(y, preds)
        label = "all" if fw == len(tr_dates) else f"{fw}d"
        print(f"{label:>10}  {score:>8.5f}")
