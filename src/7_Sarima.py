"""
This script trains a SARIMA model on the features dataset and saves the fitted model to disk.
The model parameters are defined in the config.py file,
which is generated from the hyperparameter tuning notebook (notebooks/sarima_tuning.ipynb).
model is saved in joblib format at models/sarima.joblib, which can be loaded later for inference or further evaluation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import config` / `src.` work

import joblib
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (SARIMA_PARAMS, SARIMA_FOURIER_K, MODELS_DIR , SARIMA_INTERVAL_ALPHA)
from src.sarima_utils import fourier_terms, load_splits

SARIMA_MODEL_PATH = MODELS_DIR / "sarima.joblib"

def fit_sarima(y):
    return SARIMAX(y, exog=fourier_terms(y.index, SARIMA_FOURIER_K), **SARIMA_PARAMS,
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False, maxiter=200)


train, calib, test = load_splits()
fit_data = pd.concat([train, calib])   # final model: train + calib

# 1) conformal quantile: model fit on train only, absolute errors over the calibration period
cal_pred = fit_sarima(train).forecast(
    len(calib), exog=fourier_terms(calib.index, SARIMA_FOURIER_K)).values
abs_err = np.abs(calib.values - cal_pred)
n = len(abs_err)
level = min(1.0, np.ceil((n + 1) * (1 - SARIMA_INTERVAL_ALPHA)) / n)
q = float(np.quantile(abs_err, level, method="higher"))

# 2) final model on train+calib, forecast the test period (test used once)
res = fit_sarima(fit_data)
fc = res.get_forecast(len(test), exog=fourier_terms(test.index, SARIMA_FOURIER_K))
point = fc.predicted_mean.values
lo, hi = point - q, point + q
ci = fc.conf_int(alpha=SARIMA_INTERVAL_ALPHA).values      # native interval, kept for the README only
y = test.values

metrics = {
    "mae": float(np.mean(np.abs(y - point))),
    "mape": float(np.mean(np.abs((y - point) / y)) * 100),
    "coverage": float(np.mean((y >= lo) & (y <= hi))),
    "avg_width": float(2 * q),
    "native_coverage": float(np.mean((y >= ci[:, 0]) & (y <= ci[:, 1]))),
    "native_avg_width": float(np.mean(ci[:, 1] - ci[:, 0])),
    "conformal_q": q,
    "n_calib": int(n),
    "n_test": int(len(test)),
    "fit_rows": int(len(fit_data)),
    "last_train_date": str(fit_data.index[-1].date()),
}
print(json.dumps(metrics, indent=2))

#res.remove_data()                      # strip training data, smaller file
SARIMA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "model": res,
    "params": SARIMA_PARAMS,
    "fourier_k": SARIMA_FOURIER_K,
    "interval_alpha": SARIMA_INTERVAL_ALPHA,
    "conformal_q": q,
    "metrics": metrics,
    "versions": {"statsmodels": statsmodels.__version__, "pandas": pd.__version__},
}, SARIMA_MODEL_PATH)
print("saved", SARIMA_MODEL_PATH)
