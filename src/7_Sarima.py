"""Train final SARIMA on train+calib using SARIMA_PARAMS from config.py,
evaluate once on test, save to models/sarima.joblib."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import config` works

import joblib
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (DAILY_SERIES_PATH, CALIB_END, SARIMA_PARAMS,
                    SARIMA_MODEL_PATH, SARIMA_INTERVAL_ALPHA)


s = pd.read_parquet(DAILY_SERIES_PATH).set_index("ds")["y"].asfreq("D")
assert s.isna().sum() == 0, "gaps in daily series"

fit_data = s[:CALIB_END]               # train + calib
test = s[CALIB_END:].iloc[1:]          # test only, used once

res = SARIMAX(fit_data, **SARIMA_PARAMS,
    enforce_stationarity=False,
    enforce_invertibility=False).fit(disp=False, maxiter=200)

fc = res.get_forecast(len(test))
point = fc.predicted_mean.values
ci = fc.conf_int(alpha=SARIMA_INTERVAL_ALPHA).values
lo, hi = ci[:, 0], ci[:, 1]
y = test.values

metrics = {
    "mae": float(np.mean(np.abs(y - point))),
    "mape": float(np.mean(np.abs((y - point) / y)) * 100),
    "coverage": float(np.mean((y >= lo) & (y <= hi))),
    "avg_width": float(np.mean(hi - lo)),
    "n_test": int(len(test)),
    "fit_rows": int(len(fit_data)),
    "last_train_date": str(fit_data.index[-1].date()),
    }
print(json.dumps(metrics, indent=2))

res.remove_data()                      # strip training data, smaller file
SARIMA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "model": res,
    "params": SARIMA_PARAMS,
    "interval_alpha": SARIMA_INTERVAL_ALPHA,
    "metrics": metrics,
    "versions": {"statsmodels": statsmodels.__version__,
                "pandas": pd.__version__},
}, SARIMA_MODEL_PATH)
