"""
this script resamples the combined energy data to hourly resolution (due to varying resolutions),
computes daily peak demand, and flags missing or partial days.
It also interpolates short gaps in the data (up to 2 days) and
saves the cleaned daily peak data to a parquet file.
"""
 
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COMBINED_DATA_DIR, CLEANED_DATA_DIR

TS_COL, DEMAND_COL = "datetime", "demand_mw"
MAX_FILL_DAYS = 2

CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(COMBINED_DATA_DIR / "energy_combined.parquet")
df[TS_COL] = pd.to_datetime(df[TS_COL])
df = df.drop_duplicates(subset=TS_COL).sort_values(TS_COL)
s = df.set_index(TS_COL)[DEMAND_COL]

# Check: dominant sampling interval per month (confirms the era changes)
print(
    s.index.to_series()
    .diff()
    .groupby(s.index.to_period("M"))
    .agg(lambda x: x.mode()[0])
    .to_string()
)

# 1. Common resolution: hourly mean
hourly = s.resample("h").mean()

# 2. Daily peak of hourly values
g = hourly.resample("D")
daily = pd.DataFrame({"y": g.max(), "n_hours": g.count()})

# 3. Flags
daily["missing"] = daily["n_hours"] == 0
daily["partial"] = (daily["n_hours"] > 0) & (daily["n_hours"] < 24)

# 4. Interpolate only gaps <= MAX_FILL_DAYS; longer gaps stay NaN
is_na = daily["y"].isna()
run_id = (is_na != is_na.shift()).cumsum()
run_len = is_na.groupby(run_id).transform("sum")
short_gap = is_na & (run_len <= MAX_FILL_DAYS)

interp = daily["y"].interpolate(method="time", limit_area="inside")
daily["interpolated"] = short_gap & interp.notna()
daily.loc[daily["interpolated"], "y"] = interp[daily["interpolated"]]

# 5. Report
print(daily[["missing", "partial", "interpolated"]].sum())
print("still NaN:", daily["y"].isna().sum())
print(daily["y"].describe())
print(daily.groupby(daily.index.year)["y"].agg(["count", "min", "max"]))

daily.reset_index().rename(columns={TS_COL: "ds"}).to_parquet(
    CLEANED_DATA_DIR / "daily_peak.parquet", index=False
)