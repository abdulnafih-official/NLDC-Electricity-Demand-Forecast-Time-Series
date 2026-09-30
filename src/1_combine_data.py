"""
this script combines all the raw NLDC electricity demand data files into a single DataFrame,
removes duplicates, and
saves the combined data to a parquet file.
"""


import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RAW_DATA_DIR, COMBINED_DATA_DIR

v2_data = "January 2024- June 2025.xlsx"

all_dfs = []

for file in sorted(p.name for p in RAW_DATA_DIR.glob("*.xlsx")):
    if file.startswith("~$"):
        continue

    if file == v2_data:
        df_month = pd.read_excel(
            RAW_DATA_DIR / file,
            sheet_name="Report",
            header=0,
            engine="calamine"
        )
        df_month = df_month[["Timestamp", "Demand (MW)"]].rename(columns={
            "Timestamp": "datetime",
            "Demand (MW)": "demand_mw"
        })
        dt_format = "%d-%m-%Y %H:%M:%S"
    else:
        df_month = pd.read_excel(
            RAW_DATA_DIR / file,
            sheet_name="Sheet1",
            header=1,
            engine="calamine"
        )
        df_month = df_month[["Unnamed: 0", "NLDC_DEMAND|P"]].rename(columns={
            "Unnamed: 0": "datetime",
            "NLDC_DEMAND|P": "demand_mw"
        })
        dt_format = "%Y-%m-%d %H:%M:%S.%f"

    n_raw = len(df_month)
    df_month["datetime"] = pd.to_datetime(df_month["datetime"], format=dt_format, errors="coerce")
    df_month["demand_mw"] = pd.to_numeric(df_month["demand_mw"], errors="coerce")

    n_bad_dt = df_month["datetime"].isna().sum()
    n_bad_demand = df_month.loc[df_month["datetime"].notna(), "demand_mw"].isna().sum()
    df_month = df_month.dropna(subset=["datetime"])

    all_dfs.append(df_month)
    print(f"✅ {file} | rows: {n_raw} | dropped (bad datetime): {n_bad_dt} | NaN demand kept: {n_bad_demand}")

df = pd.concat(all_dfs, ignore_index=True)

n_before = len(df)
df = df.drop_duplicates(subset="datetime", keep="last")
print(f"\nDuplicate timestamps removed: {n_before - len(df)}")

df.sort_values("datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nShape: {df.shape}")
print(df.head())
print(f"\nDate range: {df['datetime'].min()} → {df['datetime'].max()}")
print(f"Total NaN demand: {df['demand_mw'].isna().sum()}")

COMBINED_DATA_DIR.mkdir(parents=True, exist_ok=True)
output_path = COMBINED_DATA_DIR / "energy_combined.parquet"
df.to_parquet(output_path, index=False)
print(f"\nSaved to {output_path}")