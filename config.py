"""Project configuration and constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
COMBINED_DATA_DIR = PROCESSED_DATA_DIR / "combined"
CLEANED_DATA_DIR = PROCESSED_DATA_DIR / "cleaned"


XGB_PARAMS = {
  "max_depth": 1,
  "learning_rate": 0.05,
  "n_estimators": 1000,
  "min_child_weight": 5,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
}

PROPHET_PARAMS = {
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.90,
    "changepoint_prior_scale": 0.0005,
    "seasonality_prior_scale": 1,
    "holidays_prior_scale": 10,
    "seasonality_mode": "additive",
}

SERIES_TRAIN_PATH = FEATURES_DATA_DIR / "series_train.parquet"
SERIES_CALIB_PATH = FEATURES_DATA_DIR / "series_calib.parquet"
SERIES_TEST_PATH = FEATURES_DATA_DIR / "series_test.parquet"

SARIMA_PARAMS = {"order": (2, 0, 1), "seasonal_order": (0, 1, 1, 7)}
SARIMA_FOURIER_K = 6                                                
SARIMA_INTERVAL_ALPHA = 0.10