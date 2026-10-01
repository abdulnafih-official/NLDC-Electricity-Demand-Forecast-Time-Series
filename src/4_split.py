"""
This script splits the time series and features data into train, calibration,
and test sets based on specified date thresholds.
It also computes baseline metrics (MAE and MAPE) for simple lag-based predictions.
The train set includes data before the calibration start date,
the calibration set includes data between the calibration start date and the test start date,
and the test set includes data after the test start date.
The script saves the split datasets to parquet files and prints summary statistics for each split.
The baseline metrics are computed using two simple lag-based predictions: "yesterday" (the value from the previous day) and
"last_week" (the value from the same day of the previous week). The results are saved to a CSV file for further analysis.

"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import FEATURES_DATA_DIR

FEAT_DIR = FEATURES_DATA_DIR
SERIES = FEAT_DIR / "series.parquet"      # ds, y  (adjust to your actual filename)
FEATURES = FEAT_DIR / "xgb_features.parquet"    # adjust to your actual filename

CALIB_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")


def split(df):
    df = df.sort_values("ds").reset_index(drop=True)
    train = df[df.ds < CALIB_START]
    calib = df[(df.ds >= CALIB_START) & (df.ds < TEST_START)]
    test = df[df.ds >= TEST_START]
    return train, calib, test


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def main():
    series = pd.read_parquet(SERIES)
    series["ds"] = pd.to_datetime(series["ds"])
    feats = pd.read_parquet(FEATURES)
    feats["ds"] = pd.to_datetime(feats["ds"])

    for name, df in [("series", series), ("features", feats)]:
        tr, ca, te = split(df)
        assert len(tr) + len(ca) + len(te) == len(df)
        assert tr.ds.max() < ca.ds.min() < ca.ds.max() < te.ds.min()
        print(f"{name}: train={len(tr)} calib={len(ca)} test={len(te)} | "
              f"{tr.ds.min().date()}..{tr.ds.max().date()} | "
              f"{ca.ds.min().date()}..{ca.ds.max().date()} | "
              f"{te.ds.min().date()}..{te.ds.max().date()}")
        tr.to_parquet(FEAT_DIR / f"{name}_train.parquet", index=False)
        ca.to_parquet(FEAT_DIR / f"{name}_calib.parquet", index=False)
        te.to_parquet(FEAT_DIR / f"{name}_test.parquet", index=False)

    # Baselines: lags computed on the full series, then sliced (past values only)
    s = series.sort_values("ds").set_index("ds")["y"]
    base = pd.DataFrame({"y": s, "yesterday": s.shift(1), "last_week": s.shift(7)})
    rows = []
    for seg, lo, hi in [("calib", CALIB_START, TEST_START),
                        ("test", TEST_START, pd.Timestamp.max)]:
        d = base[(base.index >= lo) & (base.index < hi)].dropna()
        for m in ["yesterday", "last_week"]:
            rows.append({"segment": seg, "baseline": m,
                         "MAE": float(np.mean(np.abs(d.y - d[m]))),
                         "MAPE": mape(d.y, d[m])})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(FEAT_DIR / "baseline_metrics.csv", index=False)


if __name__ == "__main__":
    main()