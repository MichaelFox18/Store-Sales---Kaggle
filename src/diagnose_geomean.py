"""Why did the geometric mean lose? Check prediction levels, trend, and bias."""
import numpy as np
import pandas as pd

from validation import time_train_holdout_split, RAW

train = pd.read_csv(RAW / "train.csv", usecols=["date", "store_nbr", "family", "sales"])
tr, hold, cutoff = time_train_holdout_split(train)
keys = ["store_nbr", "family"]

arith = tr.groupby(keys)["sales"].mean()
geo = np.expm1(tr.assign(ls=np.log1p(tr["sales"])).groupby(keys)["ls"].mean())

print("=== average level: predictions vs reality ===")
print(f"mean ACTUAL  (holdout, Aug'17) : {hold['sales'].mean():8.2f}")
print(f"mean ARITH   prediction        : {arith.mean():8.2f}")
print(f"mean GEO     prediction        : {geo.mean():8.2f}")

print("\n=== is there a trend? (daily mean sales) ===")
print(f"full history (2013 -> Jul'17)  : {tr['sales'].mean():8.2f}")
print(f"recent (Jul 2017)              : {tr[tr['date'] >= '2017-07-01']['sales'].mean():8.2f}")
print(f"holdout (Aug 2017)             : {hold['sales'].mean():8.2f}")

print("\n=== how often does each prediction fall BELOW the actual? ===")
hm = (
    hold.merge(arith.rename("a").reset_index(), on=keys)
    .merge(geo.rename("g").reset_index(), on=keys)
)
nz = hm[hm["sales"] > 0]
print(f"arith under-predicts (nonzero actuals): {(nz['a'] < nz['sales']).mean():.1%}")
print(f"geo   under-predicts (nonzero actuals): {(nz['g'] < nz['sales']).mean():.1%}")
