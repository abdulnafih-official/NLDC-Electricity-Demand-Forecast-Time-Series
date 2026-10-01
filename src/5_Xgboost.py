# src/5_XGBoost.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from config import FEATURES_DATA_DIR

FEAT_DIR = FEATURES_DATA_DIR
ALPHA = 0.10  # 90% interval

train = pd.read_parquet(FEAT_DIR / "features_train.parquet").dropna().reset_index(drop=True)
calib = pd.read_parquet(FEAT_DIR / "features_calib.parquet").dropna().reset_index(drop=True)
test = pd.read_parquet(FEAT_DIR / "features_test.parquet").dropna().reset_index(drop=True)

cols = [c for c in train.columns if c not in ("ds", "y")]
Xtr, ytr = train[cols], train["y"]
Xca, yca = calib[cols], calib["y"]
Xte, yte = test[cols], test["y"]

# Tune with walk-forward CV inside train only (calib stays untouched for conformal)
grid = {
    "max_depth": [1, 2, 3],
    "learning_rate": [0.02, 0.05],
    "n_estimators": [100, 200, 300],
    "min_child_weight": [5, 10],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}
tscv = TimeSeriesSplit(n_splits=4)
best, best_mae = None, np.inf
for vals in itertools.product(*grid.values()):
    params = dict(zip(grid.keys(), vals))
    maes = []
    for tr_i, va_i in tscv.split(Xtr):
        m = XGBRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        maes.append(np.mean(np.abs(ytr.iloc[va_i] - m.predict(Xtr.iloc[va_i]))))
    if np.mean(maes) < best_mae:
        best, best_mae = params, float(np.mean(maes))
print("best params:", best, "| CV MAE:", round(best_mae, 1))

model = XGBRegressor(**best, random_state=42, n_jobs=-1).fit(Xtr, ytr)

# Split-conformal interval from calibration residuals
res = np.abs(yca - model.predict(Xca))
n = len(res)
q = float(np.quantile(res, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher"))

pred = model.predict(Xte)
lo, hi = pred - q, pred + q
print({
    "MAE": float(np.mean(np.abs(yte - pred))),
    "MAPE": float(np.mean(np.abs((yte - pred) / yte)) * 100),
    "coverage": float(np.mean((yte >= lo) & (yte <= hi))),
    "avg_width": float(np.mean(hi - lo)),
    "conformal_q": q,
})