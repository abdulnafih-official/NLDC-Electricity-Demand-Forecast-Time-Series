"""One evaluation script for all models, in two tracks:
  short: rolling origin, 1-7 days ahead  (XGBoost recursive, SARIMA via extend(); Prophet not included, it would need a refit per origin)
  long : fixed origin 2024-12-31, whole test period (XGBoost recursive, Prophet, SARIMA)
Writes artifacts/metrics.csv and artifacts/test_predictions.parquet (long-term, fixed-origin predictions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import holidays
import joblib
import numpy as np
import pandas as pd
from prophet.serialize import model_from_json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from config import FEATURES_DATA_DIR, MODELS_DIR, ROOT_DIR, SARIMA_INTERVAL_ALPHA
from src.sarima_utils import fourier_terms, load_splits

H = 7
ALPHA = SARIMA_INTERVAL_ALPHA
ART = ROOT_DIR / "artifacts"   # not models/artifacts: models/ is gitignored, the app needs these files
FESTIVALS = ["Diwali", "Holi", "Eid", "Dussehra", "Christmas", "Republic Day", "Independence Day"]  # as in 3_Features.py

train, calib, test = load_splits()
full = pd.concat([train, calib, test])
y_all = full.values
N, n_tr, n_ca, n_te = len(full), len(train), len(calib), len(test)
o_cal, o_test = n_tr, n_tr + n_ca       # position of the first calibration / test target
y_te = test.values
T0 = full.index[0]


def conformal_q(abs_err, alpha=ALPHA):
    n = len(abs_err)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_err, level, method="higher"))


# ---------------- XGBoost: feature rebuild + recursive forecast ----------------
xgb = XGBRegressor()
xgb.load_model(MODELS_DIR / "xgb.json")
cols = [c for c in pd.read_parquet(FEATURES_DATA_DIR / "features_train.parquet").columns if c not in ("ds", "y")]

hol = holidays.India(years=range(T0.year - 1, full.index[-1].year + 2))
HOL = set(pd.to_datetime(list(hol.keys())))
FEST = set(pd.to_datetime([k for k, v in hol.items() if any(f.lower() in v.lower() for f in FESTIVALS)]))


def row(date, hist):
    """Feature vector for `date` given all earlier y values (known or predicted). Mirrors 3_Features.py."""
    w = np.asarray(hist[-28:], dtype=float)
    f = {"is_holiday": int(date in HOL), "is_festival": int(date in FEST),
         "dow": date.dayofweek, "is_weekend": int(date.dayofweek >= 5), "month": date.month,
         "doy": date.dayofyear, "doy_sin": np.sin(2 * np.pi * date.dayofyear / 365.25),
         "doy_cos": np.cos(2 * np.pi * date.dayofyear / 365.25), "year": date.year, "t": (date - T0).days,
         "lag_1": w[-1], "lag_7": w[-7], "lag_14": w[-14],
         "roll7_mean": w[-7:].mean(), "roll7_std": w[-7:].std(ddof=1), "roll28_mean": w.mean()}
    missing = [c for c in cols if c not in f]
    if missing:
        raise ValueError(f"XGBoost uses features this script cannot rebuild for future dates: {missing}")
    return [f[c] for c in cols]


# checkpoint: with true history the rebuilt features must equal features_test
ft = pd.read_parquet(FEATURES_DATA_DIR / "features_test.parquet")
oracle = np.array([row(d, y_all[:o_test + i]) for i, d in enumerate(test.index)])
assert np.allclose(oracle, ft[cols].values.astype(float)), "rebuilt features != features_test"


def xgb_path(o, h):
    """Recursive forecast of y_all[o:o+h] using only y_all[:o]."""
    hist, preds = list(y_all[o - 28:o]), []
    for j in range(h):
        p = float(xgb.predict(pd.DataFrame([row(full.index[o + j], hist)], columns=cols))[0])
        hist.append(p)
        preds.append(p)
    return np.array(preds)


def rolling_xgb(o_start, o_end):
    out = {h: ([], []) for h in range(1, H + 1)}
    for o in range(o_start, o_end):
        hh = min(H, o_end - o)
        p = xgb_path(o, hh)
        for h in range(1, hh + 1):
            out[h][0].append(y_all[o + h - 1])
            out[h][1].append(p[h - 1])
    return {h: (np.array(a), np.array(b)) for h, (a, b) in out.items()}


# ---------------- SARIMA: rolling origin via extend() ----------------
sd = joblib.load(MODELS_DIR / "sarima.joblib")
K = sd["fourier_k"]
kw = dict(**sd["params"], enforce_stationarity=False, enforce_invertibility=False)
res_tr = SARIMAX(train, exog=fourier_terms(train.index, K), **kw).fit(disp=False, maxiter=200)  # as in 7_Sarima.py
fit_all = pd.concat([train, calib])
# remove_data() stripped endog from the saved model, so rebuild on train+calib and apply the saved params (no refit)
res_all = SARIMAX(fit_all, exog=fourier_terms(fit_all.index, K), **kw).smooth(sd["model"].params)


def rolling_sarima(res, seg):
    """`res` is fitted through the day before `seg` starts; each origin extends it with the observed days so far."""
    out = {h: ([], []) for h in range(1, H + 1)}
    n = len(seg)
    for i in range(n):
        r = res if i == 0 else res.extend(seg.iloc[:i], exog=fourier_terms(seg.index[:i], K))
        hh = min(H, n - i)
        fc = r.get_forecast(hh, exog=fourier_terms(seg.index[i:i + hh], K)).predicted_mean.values
        for h in range(1, hh + 1):
            out[h][0].append(seg.iloc[i + h - 1])
            out[h][1].append(fc[h - 1])
    return {h: (np.array(a), np.array(b)) for h, (a, b) in out.items()}


def baseline(kind):
    out = {}
    for h in range(1, H + 1):
        os_ = np.arange(o_test, N - h + 1)
        tgt = os_ + h - 1
        out[h] = (y_all[tgt], y_all[os_ - 1] if kind == "persistence" else y_all[tgt - 7])
    return out


def metric_row(track, model, horizon, y, p, cov=None, width=None):
    return {"track": track, "model": model, "horizon": horizon, "mae": np.mean(np.abs(y - p)),
            "mape": np.mean(np.abs((y - p) / y)) * 100,
            "coverage": np.nan if cov is None else np.mean(cov),
            "avg_width": np.nan if width is None else np.mean(width)}


def short_rows(model, res, q=None):
    rows = [metric_row("short", model, str(h), y, p,
                       None if q is None else np.abs(y - p) <= q[h],
                       None if q is None else np.full(len(y), 2 * q[h])) for h, (y, p) in res.items()]
    ys = np.concatenate([y for y, _ in res.values()])
    ps = np.concatenate([p for _, p in res.values()])
    cov = None if q is None else np.concatenate([np.abs(y - p) <= q[h] for h, (y, p) in res.items()])
    wid = None if q is None else np.concatenate([np.full(len(y), 2 * q[h]) for h, (y, _) in res.items()])
    return rows + [metric_row("short", model, f"1-{H}", ys, ps, cov, wid)]


def long_rows(model, pt, lo=None, hi=None):
    rows = []
    for a, b in [(1, 30), (31, 90), (91, n_te), (1, n_te)]:
        s = slice(a - 1, b)
        has = lo is not None
        rows.append(metric_row("long", model, f"{a}-{b}", y_te[s], pt[s],
                               (y_te[s] >= lo[s]) & (y_te[s] <= hi[s]) if has else None,
                               hi[s] - lo[s] if has else None))
    return rows


rows = []

# ---- short-term ----
for name, cal_res, test_res in [
    ("xgboost", rolling_xgb(o_cal, o_test), rolling_xgb(o_test, N)),
    ("sarima", rolling_sarima(res_tr, calib), rolling_sarima(res_all, test)),
]:
    q = {h: conformal_q(np.abs(y - p)) for h, (y, p) in cal_res.items()}   # per-horizon conformal quantile (calibration year)
    rows += short_rows(name, test_res, q)
rows += short_rows("persistence", baseline("persistence"))
rows += short_rows("last_week", baseline("last_week"))

# ---- long-term (fixed origin = end of calibration) ----
long_preds = {}
p_x = xgb_path(o_test, n_te)
q_x = conformal_q(np.abs(y_all[o_cal:o_test] - xgb_path(o_cal, n_ca)))   # same recipe, origin = end of train
long_preds["xgboost"] = (p_x - q_x, p_x, p_x + q_x)

with open(MODELS_DIR / "prophet.json") as f:
    pr = model_from_json(f.read()).predict(pd.DataFrame({"ds": test.index}))
long_preds["prophet"] = (pr["yhat_lower"].values, pr["yhat"].values, pr["yhat_upper"].values)

p_s = res_all.get_forecast(n_te, exog=fourier_terms(test.index, K)).predicted_mean.values
assert np.isclose(np.mean(np.abs(y_te - p_s)), sd["metrics"]["mae"]), "SARIMA MAE differs from 7_Sarima.py"
long_preds["sarima"] = (p_s - sd["conformal_q"], p_s, p_s + sd["conformal_q"])
long_preds["seasonal_naive_364"] = (None, full.shift(364).loc[test.index].values, None)

for name, (lo, pt, hi) in long_preds.items():
    rows += long_rows(name, pt, lo, hi)

out = pd.DataFrame(rows)
num = ["mae", "mape", "coverage", "avg_width"]
out[num] = out[num].round(2)
ART.mkdir(exist_ok=True)
out.to_csv(ART / "metrics.csv", index=False)
pd.concat([pd.DataFrame({"ds": test.index, "model": n, "y": y_te, "lower": lo, "point": pt, "upper": hi})
           for n, (lo, pt, hi) in long_preds.items()]).to_parquet(ART / "test_predictions.parquet", index=False)

print(out.to_string(index=False))
for track, hz in (("short", f"1-{H}"), ("long", f"1-{n_te}")):
    t = out[(out.track == track) & (out.horizon == hz)].sort_values("mae")
    print(f"\n{track}-term ranking by MAE (all rows, incl. baselines):")
    print(t[["model", "mae", "mape", "coverage", "avg_width"]].to_string(index=False))