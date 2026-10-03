# src/XGB_Calibrate.py
"""
Computes the split-conformal quantile for the saved XGBoost model (models/xgb.json,
trained on train only by 5_XGBoost.py) from absolute residuals on the 2024 calibration split.
Writes models/calibration.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from config import CALIBRATION_PATH, FEATURES_DATA_DIR, MODELS_DIR, XGB_PARAMS

ALPHA = 0.10  # 90% interval

train = pd.read_parquet(FEATURES_DATA_DIR / "features_train.parquet").dropna().reset_index(drop=True)
calib = pd.read_parquet(FEATURES_DATA_DIR / "features_calib.parquet").dropna().reset_index(drop=True)
test = pd.read_parquet(FEATURES_DATA_DIR / "features_test.parquet").dropna().reset_index(drop=True)
cols = [c for c in train.columns if c not in ("ds", "y")]  

model = XGBRegressor()
model.load_model(MODELS_DIR / "xgb.json")

# Split conformal with finite-sample correction
resid = np.abs(calib["y"].to_numpy() - model.predict(calib[cols]))
n = len(resid)
level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
q = float(np.quantile(resid, level, method="higher"))

# Sanity check: 1-step coverage on test with real lags (target about 0.90)
cover = float(np.mean(np.abs(test["y"].to_numpy() - model.predict(test[cols])) <= q))

CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
CALIBRATION_PATH.write_text(json.dumps({
    "alpha": ALPHA,
    "q": q,
    "n_calib": n,
    "quantile_level": float(level),
    "fit_on": f"{train['ds'].min():%Y-%m-%d} to {train['ds'].max():%Y-%m-%d}",
    "calib_on": f"{calib['ds'].min():%Y-%m-%d} to {calib['ds'].max():%Y-%m-%d}",
    "xgb_params": XGB_PARAMS,
}, indent=2))

print(f"q = {q:,.0f}  (n={n}, level={level:.4f})")
print(f"test coverage = {cover:.3f}  (target {1 - ALPHA:.2f})")
print(f"wrote {CALIBRATION_PATH}")