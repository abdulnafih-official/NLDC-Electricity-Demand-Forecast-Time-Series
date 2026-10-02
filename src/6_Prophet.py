# src/6_Prophet.py
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
from config import FEATURES_DATA_DIR, MODELS_DIR, PROPHET_PARAMS  # params chosen in notebooks/02_prophet_tuning.ipynb

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

train = pd.read_parquet(FEATURES_DATA_DIR / "series_train.parquet")[["ds", "y"]]
holidays_df = pd.read_parquet(FEATURES_DATA_DIR / "prophet_holidays.parquet")

model = Prophet(**PROPHET_PARAMS, holidays=holidays_df).fit(train)

# Native Prophet JSON: safer across versions than a pickle
MODELS_DIR.mkdir(parents=True, exist_ok=True)
with open(MODELS_DIR / "prophet.json", "w") as f:
    f.write(model_to_json(model))
print("saved:", MODELS_DIR / "prophet.json", "| rows:", len(train))