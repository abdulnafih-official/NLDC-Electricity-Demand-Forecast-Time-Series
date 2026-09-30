"""
This script generates features for time series modeling, including:
- calendar features (day of week, month, year, day of year, weekend flag, sin/cos of day of year)
- holiday and festival flags (using the `holidays` package), and manually confirmed some festival dates to ensure accuracy.
- lag features (1, 7, 14 days)
- rolling statistics (7-day mean and std, 28-day mean)
It saves the features to parquet files for use in Prophet, SARIMA and XGBoost models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd, numpy as np, holidays
from config import CLEANED_DATA_DIR, FEATURES_DATA_DIR

FEAT_DIR = FEATURES_DATA_DIR

FESTIVALS = ["Diwali", "Holi", "Eid", "Dussehra", "Christmas", "Republic Day", "Independence Day"]

df = pd.read_parquet(CLEANED_DATA_DIR / "daily_peak.parquet")  # cols: ds, y (+ n_hours, missing, partial, interpolated)
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)
assert df["ds"].is_unique and df["ds"].diff().dropna().eq(pd.Timedelta("1D")).all(), "gaps in calendar"

# ---- Holidays (dates exactly as given by the package) ----
yrs = range(df.ds.dt.year.min() - 1, df.ds.dt.year.max() + 2)
hol = holidays.India(years=yrs)
hol_df = pd.DataFrame({"date": pd.to_datetime(list(hol.keys())), "name": list(hol.values())})
hol_df["is_festival"] = hol_df["name"].str.contains("|".join(FESTIVALS), case=False)

# verify festival dates against known years
print(hol_df[hol_df.is_festival].sort_values("date")[["date", "name"]].to_string())

d = df.copy()
d["is_holiday"] = d.ds.isin(set(hol_df["date"])).astype(int)
d["is_festival"] = d.ds.isin(set(hol_df.loc[hol_df.is_festival, "date"])).astype(int)

# ---- Calendar ----
d["dow"] = d.ds.dt.dayofweek
d["is_weekend"] = (d.dow >= 5).astype(int)
d["month"] = d.ds.dt.month
d["doy"] = d.ds.dt.dayofyear
d["doy_sin"] = np.sin(2 * np.pi * d.doy / 365.25)
d["doy_cos"] = np.cos(2 * np.pi * d.doy / 365.25)
d["year"] = d.ds.dt.year
d["t"] = (d.ds - d.ds.min()).dt.days

# ---- Lags / rolling (all shifted: nothing from day t or later) ----
for k in (1, 7, 14):
    d[f"lag_{k}"] = d.y.shift(k)
s = d.y.shift(1)
d["roll7_mean"] = s.rolling(7).mean()
d["roll7_std"] = s.rolling(7).std()
d["roll28_mean"] = s.rolling(28).mean()

# ---- Leakage check: perturbing y[t:] must not change features at t ----
t0 = len(d) // 2
y2 = d.y.copy(); y2.loc[t0:] += 1e6
assert y2.shift(1)[t0] == d.loc[t0, "lag_1"]
assert y2.shift(1).rolling(7).mean()[t0] == d.loc[t0, "roll7_mean"]

# ---- Save ----
FEAT_DIR.mkdir(parents=True, exist_ok=True)
d[["ds", "y"]].to_parquet(FEAT_DIR / "series.parquet", index=False)  # Prophet / SARIMA
d.dropna(subset=["lag_14", "roll28_mean"]).reset_index(drop=True) \
 .to_parquet(FEAT_DIR / "xgb_features.parquet", index=False)  # XGBoost
hol_df.rename(columns={"date": "ds", "name": "holiday"})[["ds", "holiday"]] \
      .to_parquet(FEAT_DIR / "prophet_holidays.parquet", index=False)  # Prophet holidays arg

print(d.shape, d.ds.min(), d.ds.max())
print(d.isna().sum()[lambda s: s > 0])