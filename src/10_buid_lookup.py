# src/9_build_lookup.py
"""Offline builder (needs the data). Precomputes every model's prediction for every date
into models/lookup.parquet. Upload it to the private HF repo; the app never touches the data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holidays
import numpy as np
import pandas as pd

from config import FEATURES_DATA_DIR, MODELS_DIR
from src.sarima_utils import fourier_terms, load_splits
from src.storage import load_json, load_model

END = pd.Timestamp("2030-12-31")
RECOMMENDED_HORIZON = {"prophet": 90, "sarima": 30, "xgboost": 7}  # days
FESTIVALS = ["Diwali", "Holi", "Eid", "Dussehra", "Christmas", "Republic Day", "Independence Day"]  # as in 3_Features.py
XGB_BURN, SARIMA_BURN = 28, 14  # first rows with no valid lags / unreliable one-step fit (ASSUMED)

train, calib, test = load_splits()
full = pd.concat([train, calib, test])
y, N, T0 = full.values, len(full), full.index[0]
train_end, fit_end, test_end = train.index[-1], calib.index[-1], test.index[-1]

hol = holidays.India(years=range(T0.year - 1, END.year + 2))
HOL = set(pd.to_datetime(list(hol.keys())))
FEST = set(pd.to_datetime([k for k, v in hol.items() if any(f.lower() in v.lower() for f in FESTIVALS)]))


def segment(d):
    return "train" if d <= train_end else "calibration" if d <= fit_end else "test" if d <= test_end else "future"


def warnings_for(model, seg, h):
    w, rec = [], RECOMMENDED_HORIZON[model]
    if seg in ("train", "calibration") and model != "xgboost":
        w.append("In-sample: the model was fit on this date, so this is not a real forecast.")
    if model == "xgboost" and seg == "train":
        w.append("In-sample: the model was trained on this date, so this is not a real forecast.")
    if model == "xgboost" and seg == "calibration":
        w.append("2024 residuals set the interval width, so coverage here is optimistic.")
    if model == "xgboost" and h > 1:
        w.append("Interval is calibrated for 1-day-ahead; recursive forecasts cover far less than 90%.")
    if h > 2 * rec:
        w.append(f"Horizon {h}d is more than twice the recommended {rec}d: indicative only.")
    elif h > rec:
        w.append(f"Horizon {h}d exceeds the recommended {rec}d: accuracy drops.")
    return w


def frame(model, dates, point, lower, upper, horizon):
    df = pd.DataFrame({"model": model, "ds": pd.to_datetime(dates), "lower": lower, "point": point,
                       "upper": upper, "horizon_days": np.asarray(horizon, dtype=int)})
    df["segment"] = [segment(d) for d in df.ds]
    df["actual"] = [full.get(d, np.nan) if s == "test" else np.nan for d, s in zip(df.ds, df.segment)]
    df["warnings"] = [" | ".join(warnings_for(model, s, h)) for s, h in zip(df.segment, df.horizon_days)]
    return df


parts = []

# ---------------- Prophet ----------------
d = pd.date_range(T0, END)
pr = load_model("prophet", "local").predict(pd.DataFrame({"ds": d}))
parts.append(frame("prophet", d, pr["yhat"].values, pr["yhat_lower"].values, pr["yhat_upper"].values,
                   np.maximum(0, (d - fit_end).days)))

# ---------------- SARIMA (saved model already holds train+calib, ends at fit_end) ----------------
sd = load_model("sarima", "local")
res, K, q_s = sd["model"], sd["fourier_k"], float(sd["conformal_q"])
assert len(res.model.endog) == len(train) + len(calib), "saved SARIMA does not cover train+calib"

d_in = full.loc[:fit_end].index[SARIMA_BURN:]
p_in = res.get_prediction().predicted_mean.values[SARIMA_BURN:]
d_fut = pd.date_range(fit_end + pd.Timedelta(days=1), END)
p_fut = res.get_forecast(len(d_fut), exog=fourier_terms(d_fut, K)).predicted_mean.values
assert np.isclose(np.mean(np.abs(test.values - p_fut[:len(test)])), sd["metrics"]["mae"]), \
    "SARIMA test MAE differs from saved metrics"

d = d_in.append(d_fut)
p = np.concatenate([p_in, p_fut])
parts.append(frame("sarima", d, p, p - q_s, p + q_s, np.maximum(0, (d - fit_end).days)))

# ---------------- XGBoost (1-day-ahead inside the data, recursive after it) ----------------
xgb = load_model("xgboost", "local")
cols = [c for c in pd.read_parquet(FEATURES_DATA_DIR / "features_train.parquet").columns if c not in ("ds", "y")]
q_x = float(load_json("calibration.json", "local")["q"])


def row(date, hist):
    """Feature vector for `date` given earlier y values. Same as 8_evaluate.py (mirrors 3_Features.py)."""
    w = np.asarray(hist[-28:], dtype=float)
    f = {"is_holiday": int(date in HOL), "is_festival": int(date in FEST),
         "dow": date.dayofweek, "is_weekend": int(date.dayofweek >= 5), "month": date.month,
         "doy": date.dayofyear, "doy_sin": np.sin(2 * np.pi * date.dayofyear / 365.25),
         "doy_cos": np.cos(2 * np.pi * date.dayofyear / 365.25), "year": date.year, "t": (date - T0).days,
         "lag_1": w[-1], "lag_7": w[-7], "lag_14": w[-14],
         "roll7_mean": w[-7:].mean(), "roll7_std": w[-7:].std(ddof=1), "roll28_mean": w.mean()}
    return [f[c] for c in cols]


# checkpoint: rebuilt features must equal features_test
o = len(train) + len(calib)
ft = pd.read_parquet(FEATURES_DATA_DIR / "features_test.parquet")
assert np.allclose([row(dt, y[o + i - 28:o + i]) for i, dt in enumerate(test.index)],
                   ft[cols].values.astype(float)), "rebuilt features != features_test"

d_in = full.index[XGB_BURN:]
p_in = xgb.predict(pd.DataFrame([row(full.index[i], y[i - 28:i]) for i in range(XGB_BURN, N)], columns=cols))
d_fut = pd.date_range(full.index[-1] + pd.Timedelta(days=1), END)
hist, p_fut = list(y[-28:]), []
for dt in d_fut:
    pt = float(xgb.predict(pd.DataFrame([row(dt, hist)], columns=cols))[0])
    hist.append(pt)
    p_fut.append(pt)
d = d_in.append(d_fut)
p = np.concatenate([p_in, p_fut])
h = np.concatenate([np.ones(len(d_in), dtype=int), np.arange(1, len(d_fut) + 1)])
parts.append(frame("xgboost", d, p, p - q_x, p + q_x, h))

# ---------------- save + checks ----------------
out = pd.concat(parts, ignore_index=True)
out.to_parquet(MODELS_DIR / "lookup.parquet", index=False)

t = out[out.segment == "test"]
print(t.assign(mae=(t.actual - t.point).abs()).groupby("model").mae.mean().round(0))
print(out.groupby("model").ds.agg(["min", "max", "count"]))
print(f"wrote {MODELS_DIR / 'lookup.parquet'}")