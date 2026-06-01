"""
Iteration 2 feature builder: external joins (stores, oil, holidays) + trend.

Holiday logic handles the transferred trap:
  day off  = type in {Holiday, Transfer, Additional, Bridge}, EXCEPT a Holiday with
             transferred=True (that one was moved away -> a normal working day).
  not off  = Work Day (a normal day repaying a bridge); Event is flagged separately.
Day-off flags are matched by scope: National -> all stores, Regional -> store's
state, Local -> store's city.
"""
import numpy as np
import pandas as pd

from validation import RAW

CAT = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
ROLL = [7, 28, 56]
HORIZON = 16
FEATURES = CAT + [
    "onpromotion", "dcoilwtico", "dow", "month", "day", "is_weekend", "payday",
    "is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event", "is_holiday",
    "lag", "rmean7", "rmean28", "rmean56", "growth",
]


def _oil():
    o = pd.read_csv(RAW / "oil.csv")
    o["date"] = pd.to_datetime(o["date"])
    full = pd.date_range(o["date"].min(), pd.Timestamp("2017-08-31"))  # daily, covers test
    o = o.set_index("date").reindex(full).rename_axis("date").reset_index()
    o["dcoilwtico"] = o["dcoilwtico"].ffill().bfill()
    return o


def _holidays():
    h = pd.read_csv(RAW / "holidays_events.csv")
    h["date"] = pd.to_datetime(h["date"])
    transferred = h["transferred"].astype(str).str.lower().eq("true")  # robust to bool/str
    dayoff = h["type"].isin(["Holiday", "Transfer", "Additional", "Bridge"]) & ~(
        (h["type"] == "Holiday") & transferred)
    nat = h.loc[dayoff & (h["locale"] == "National"), ["date"]].drop_duplicates()
    nat["is_nat_hol"] = np.int8(1)
    reg = h.loc[dayoff & (h["locale"] == "Regional"), ["locale_name", "date"]] \
        .rename(columns={"locale_name": "state"}).drop_duplicates()
    reg["is_reg_hol"] = np.int8(1)
    loc = h.loc[dayoff & (h["locale"] == "Local"), ["locale_name", "date"]] \
        .rename(columns={"locale_name": "city"}).drop_duplicates()
    loc["is_loc_hol"] = np.int8(1)
    evt = h.loc[h["type"] == "Event", ["date"]].drop_duplicates()
    evt["is_event"] = np.int8(1)
    return nat, reg, loc, evt


def build_panel():
    """Combined train+test panel with all non-recency features. Returns (panel, last_train_date)."""
    train = pd.read_csv(RAW / "train.csv",
                        usecols=["date", "store_nbr", "family", "sales", "onpromotion"])
    test = pd.read_csv(RAW / "test.csv",
                       usecols=["id", "date", "store_nbr", "family", "onpromotion"])
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    last_train = train["date"].max()

    panel = pd.concat([train, test], ignore_index=True)
    stores = pd.read_csv(RAW / "stores.csv").rename(columns={"type": "store_type"})
    panel = panel.merge(stores, on="store_nbr", how="left")
    panel = panel.merge(_oil(), on="date", how="left")
    panel["dcoilwtico"] = panel["dcoilwtico"].ffill().bfill()

    d = panel["date"].dt
    panel["dow"] = d.dayofweek
    panel["month"] = d.month
    panel["day"] = d.day
    panel["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    panel["payday"] = ((d.day == 15) | d.is_month_end).astype("int8")

    nat, reg, loc, evt = _holidays()
    panel = panel.merge(nat, on="date", how="left").merge(reg, on=["state", "date"], how="left")
    panel = panel.merge(loc, on=["city", "date"], how="left").merge(evt, on="date", how="left")
    for c in ["is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event"]:
        panel[c] = panel[c].fillna(0).astype("int8")
    panel["is_holiday"] = ((panel[["is_nat_hol", "is_reg_hol", "is_loc_hol"]].sum(axis=1)) > 0).astype("int8")

    panel = panel.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    for c in CAT:
        panel[c] = panel[c].astype("category")
    return panel, last_train


def add_recency(panel, shift, rolls=ROLL):
    """Leakage-safe lag/rolling/growth for a given shift (applied to train and test alike)."""
    sh = panel.groupby(["store_nbr", "family"], observed=True)["sales"].shift(shift)
    panel["lag"] = sh.values
    base = panel[["store_nbr", "family"]].copy()
    base["_s"] = sh.values
    for w in rolls:
        r = base.groupby(["store_nbr", "family"], observed=True)["_s"].rolling(w).mean()
        panel[f"rmean{w}"] = r.reset_index(level=[0, 1], drop=True)
    panel["growth"] = (panel["rmean7"] + 1.0) / (panel["rmean56"] + 1.0)  # recent vs older; trees can apply it
    return panel


def add_promo(panel):
    """Promotion features. onpromotion is known for ALL dates (incl. test), so trailing
    sums and store-day intensity are leakage-safe -- promo is a known input, not the target.
    Assumes panel is sorted by [store_nbr, family, date]."""
    g = panel.groupby(["store_nbr", "family"], observed=True)["onpromotion"]
    panel["promo_trail7"] = g.rolling(7).sum().reset_index(level=[0, 1], drop=True)
    panel["promo_trail28"] = g.rolling(28).sum().reset_index(level=[0, 1], drop=True)
    panel["store_promo_day"] = panel.groupby(["store_nbr", "date"], observed=True)["onpromotion"].transform("sum")
    return panel


def add_promo_lead(panel, days=7):
    """Future promo intensity: onpromotion summed over the next `days` (incl. today).
    onpromotion is known for all dates (incl. test), so this is leakage-safe. Equivalent to
    a trailing `days`-sum shifted back (days-1). Assumes panel sorted by [store_nbr, family, date]."""
    g = panel.groupby(["store_nbr", "family"], observed=True)["onpromotion"]
    trail = g.rolling(days).sum().reset_index(level=[0, 1], drop=True)
    panel["_ptmp"] = trail
    panel["promo_lead7"] = panel.groupby(["store_nbr", "family"], observed=True)["_ptmp"].shift(-(days - 1))
    return panel.drop(columns="_ptmp")


if __name__ == "__main__":
    panel, last_train = build_panel()
    print(f"panel {len(panel):,} rows, last train {last_train.date()}")

    # No NaN in the joined/flag features (recency NaNs are fine — LightGBM handles them).
    joined = ["dcoilwtico", "city", "state", "store_type", "cluster",
              "is_nat_hol", "is_reg_hol", "is_loc_hol", "is_event", "is_holiday"]
    nulls = panel[joined].isna().sum()
    print("nulls in joined features:\n", nulls.to_string())
    assert nulls.sum() == 0, "unexpected NaN from a join"

    # Holiday spot-checks (the transferred trap + locale), on dates that exist in the panel.
    def chk(date, store, col, want):
        v = int(panel[(panel["date"] == pd.Timestamp(date)) & (panel["store_nbr"] == store)][col].iloc[0])
        print(f"  {pd.Timestamp(date).date()} store {store} {col} = {v}  (want {want})  {'OK' if v == want else 'FAIL'}")

    h = pd.read_csv(RAW / "holidays_events.csv")
    h["date"] = pd.to_datetime(h["date"])
    transf = h["transferred"].astype(str).str.lower().eq("true")
    pdates = set(panel["date"].unique())
    nat_off = h[(h["type"] == "Holiday") & ~transf & (h["locale"] == "National") & h["date"].isin(pdates)]["date"]
    nat_moved = h[(h["type"] == "Holiday") & transf & (h["locale"] == "National") & h["date"].isin(pdates)]["date"]
    amb_store = int(pd.read_csv(RAW / "stores.csv").query("city == 'Ambato'")["store_nbr"].iloc[0])

    print("holiday checks:")
    chk(nat_off.iloc[0], 1, "is_nat_hol", 1)         # a real national day off -> flagged
    chk(nat_moved.iloc[0], 1, "is_nat_hol", 0)       # a TRANSFERRED holiday -> NOT flagged (the trap)
    chk("2017-08-24", amb_store, "is_loc_hol", 1)    # Fundacion de Ambato, local -> Ambato store
    chk("2017-08-24", 1, "is_loc_hol", 0)            # ...but NOT a Quito store

    print(f"\nholiday coverage: is_holiday on {panel['is_holiday'].mean():.1%} of rows, "
          f"is_event on {panel['is_event'].mean():.1%}")
