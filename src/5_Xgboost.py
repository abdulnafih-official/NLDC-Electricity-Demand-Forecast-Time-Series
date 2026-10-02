# src/5_XGBoost.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from xgboost import XGBRegressor
from config import FEATURES_DATA_DIR, MODELS_DIR, XGB_PARAMS  # params chosen in notebooks/01_xgb_tuning.ipynb

train = pd.read_parquet(FEATURES_DATA_DIR / "features_train.parquet").dropna().reset_index(drop=True)
cols = [c for c in train.columns if c not in ("ds", "y")]

model = XGBRegressor(**XGB_PARAMS, n_jobs=-1).fit(train[cols], train["y"])

# Save the fitted model and its feature order
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model.save_model(MODELS_DIR / "xgb.json") 
