# All-India Daily Peak Electricity Demand Forecasting (NLDC)

Forecasting India's daily peak electricity demand (MW) from NLDC data using **XGBoost, Prophet and SARIMA**, with **split-conformal 90% prediction intervals**, honest baselines, and a Streamlit app.

**[Live demo](<https://nldc-electricity-demand-forecast-time-series.streamlit.app/>)**

![App screenshot](<PATH_OR_URL_TO_SCREENSHOT>)

---

## Key results

Held-out test set: **175 days (2025-01-01 onward)**. Models were tuned on train/calibration only; the test set was used once.

### Short term: 1–7 days ahead (rolling origin)

| Model | MAE (MW) | MAPE (%) | 90% coverage | Avg interval width (MW) |
|---|---:|---:|---:|---:|
| **SARIMA** | **7,161** | **3.27** | 0.86 | 27,151 |
| Persistence (yesterday) | 7,837 | 3.57 | n/a | n/a |
| XGBoost (recursive) | 8,417 | 3.76 | 0.85 | 27,652 |
| Last week (same weekday) | 8,799 | 3.99 | n/a | n/a |

At 1 day ahead: SARIMA 4,087 MW (1.87%), XGBoost 4,201 MW (1.90%), persistence 5,110 MW (2.34%).

### Long term: up to 175 days ahead (fixed origin, 2024-12-31)

| Model | MAE (MW) | MAPE (%) | 90% coverage | Avg interval width (MW) |
|---|---:|---:|---:|---:|
| **Prophet** | **8,573** | **3.99** | 0.79 | 29,556 |
| SARIMA | 9,169 | 4.15 | 0.98 | 44,022 |
| Seasonal naive (364 d) | 11,811 | 5.34 | n/a | n/a |
| XGBoost (recursive) | 17,255 | 7.65 | 0.70 | 44,988 |

Full per-horizon results: [`artifacts/metrics.csv`](artifacts/metrics.csv). Test-set predictions: [`artifacts/test_predictions.parquet`](artifacts/test_predictions.parquet).

### Takeaways

- **SARIMA** is best for 1–7 days; **Prophet** is best at long horizons, but its coverage falls to 0.58 for days 91–175.
- **XGBoost does not beat simple baselines here.** It trails persistence at 1–7 days and seasonal-naive at long horizons. It is trained on the train split only (no 2024 data), uses shallow trees, and recursive forecasting compounds error.
- **Interval coverage degrades with horizon.** XGBoost intervals are calibrated for 1-day-ahead only, so long-horizon coverage is well below the 90% target (0.47–0.70). The app surfaces warnings for this.

---

## Data

| Item | Detail |
|---|---|
| Source | NLDC / Grid-India demand reports: `<https://data.mendeley.com/datasets/y58jknpgs8/2>` |
| Coverage | `<Sept 2021>` to `<June 2025>`   |
| Raw format | Excel files in two layouts: older monthly files (`Sheet1`, column `NLDC_DEMAND\|P`) and a newer file (`Report` sheet, column `Demand (MW)`) |
| Sampling | Mixed resolutions across files; everything is resampled to hourly means |
| Target | **Daily peak = max of the 24 hourly means** (this slightly understates the instantaneous peak) |
| Cleaning | Duplicate timestamps dropped; gaps of ≤ 2 days linearly interpolated; longer gaps left missing; missing/partial/interpolated days flagged |
| Not committed | `data/` and `models/` are git-ignored. Download raw files to `data/raw/` |
|Credits: Mukherjee, Debanjan; Kalita, Karuna; Kumar, Subhash (2025), “Electricity Demand, Solar and Wind Generation Data (September 2021- June 2025) of India at 1-hour interval”, Mendeley Data, V2, doi: 10.17632/y58jknpgs8.2|


**Splits (chronological, no shuffling)**

| Split | Dates | Days | Purpose |
|---|---|---:|---|
| Train | 2021-09-01 → 2023-12-31 | 852 | Fit models, tune hyperparameters |
| Calibration | 2024-01-01 → 2024-12-31 | 366 | Conformal quantiles, Prophet/SARIMA validation |
| Test | 2025-01-01 → | 175 | Final evaluation (used once) |

---

## Method

```
data/raw/*.xlsx
   │  1_combine_data.py         merge two Excel formats, dedupe
   │  2_resample_hourly_peak.py hourly mean → daily peak, flag/fill short gaps
   │  3_Features.py             calendar, holiday, lag & rolling features (+ leakage check)
   │  4_split.py                train / calib / test + naive baselines
   ├─ 5_Xgboost.py              XGBoost (train only)
   ├─ 6_Prophet.py              Prophet (train + calib)
   ├─ 7_Sarima.py               SARIMAX + Fourier terms, conformal interval (train + calib)
   │  9_XGB_Calibrate.py        split-conformal quantile for XGBoost
   │  8_Evaluate.py             short & long evaluation → artifacts/
   └─ 10_buid_lookup.py         precompute predictions for every date → models/lookup.parquet
                                         │
                                         ▼
                         app.py (Streamlit) ← src/predict.py
```

**Features** (XGBoost): day of week, weekend flag, month, day of year (+ sin/cos), year, time index, Indian holiday and festival flags (via the `holidays` package), lags (1, 7, 14 days), rolling mean/std (7 days) and mean (28 days).

**Leakage prevention**
- All lag and rolling features are shifted so nothing from day *t* or later is used for day *t*; an assertion perturbs `y[t:]` and checks features at *t* do not change.
- `8_Evaluate.py` and `10_build_lookup.py` rebuild features for future dates and assert they equal the saved test features when true history is supplied.

**Models and tuning**
- **XGBoost:** grid search with 4-fold walk-forward `TimeSeriesSplit` on train only (best CV MAE ≈ 4,715 ± 542 MW). Shallow trees (`max_depth=1`).
- **Prophet:** grid over changepoint/seasonality/holiday priors, scored on the calibration year (MAE 6,950 MW vs 15,441 MW for seasonal-naive).
- **SARIMA:** weekly seasonal SARIMAX with a linear trend and yearly Fourier terms; grid over (p,q,P,Q) and Fourier order, scored on the calibration year.
- <!-- TODO: `config.py` has SARIMA_PARAMS (1,0,0)(1,1,0,7), K=4, but notebooks/sarima_tuning.ipynb selected (2,0,1)(0,1,1,7), K=6. Reconcile, re-run 7_Sarima.py and 8_Evaluate.py, then update the tables above. -->

**Prediction intervals**
- Split conformal with finite-sample correction at α = 0.10. Absolute residuals on the calibration year give the interval half-width.
- Short-term track uses a separate quantile per horizon (1–7 days). Prophet uses its native intervals.

**Evaluation tracks**
- *Short:* rolling origin, 1–7 days ahead (XGBoost recursive; SARIMA via `extend()`); Prophet is excluded because it would need a refit per origin.
- *Long:* fixed origin at the end of calibration, forecasting the whole test period.
- Baselines: persistence, last week, and 364-day seasonal naive.

---

## Project structure

```
├── app.py                  # Streamlit app (model + date picker)
├── config.py               # paths, hyperparameters, split constants
├── requirements.txt
├── artifacts/              # metrics.csv, test_predictions.parquet
├── notebooks/              # xgb / prophet / sarima tuning
├── src/
│   ├── 1_combine_data.py … 10_buid_lookup.py   # pipeline, run in order
│   ├── sarima_utils.py     # Fourier terms, split loading
│   ├── predict.py          # lookup-based prediction used by the app
│   └── storage.py          # local or Hugging Face model/table loading
└── .devcontainer/
```

---

## Run it

**Requirements:** Python 3.11 (per `.devcontainer`). Prophet needs a working CmdStan toolchain.

```bash
git clone https://github.com/abdulnafih-official/NLDC-Electricity-Demand-Forecast-Time-Series.git
cd NLDC-Electricity-Demand-Forecast-Time-Series
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Rebuild the pipeline** (needs raw data in `data/raw/`; run from the repo root):

```bash
for f in 1_combine_data 2_resample_hourly_peak 3_Features 4_split \
         5_Xgboost 6_Prophet 7_Sarima 9_XGB_Calibrate 8_Evaluate 10_build_lookup; do
  python src/$f.py
done
```

**Run the app:**

```bash
streamlit run app.py
```

The app reads a precomputed `lookup.parquet` and never touches raw data. Storage is controlled by environment variables (`.env` locally, Streamlit Secrets when deployed):

| Variable | Meaning |
|---|---|
| `STORAGE_BACKEND` | `local` (default, reads `models/`) or `hf` (Hugging Face Hub) |
| `REPO_NAME`, `HF_TOKEN` | Hugging Face repo and token, required only for `hf` |

---

## Limitations and next steps

- No weather or macro features. 
- The test set is short (175 days, one half-year), so rankings carry uncertainty.
- XGBoost uses recursive forecasting and is trained without the 2024 data; a direct multi-horizon model, refit on train + calibration, is the obvious next step.
- XGBoost long-horizon intervals are under-covered; horizon-aware or adaptive conformal intervals would fix this.
- Dates beyond each model's recommended horizon (Prophet 90 d, SARIMA 30 d, XGBoost 7 d) are indicative only.
- Daily peak is computed from hourly means, not the true instantaneous peak.

---

## License

`<LICENSE>`

## Author

**Abdul Nafih**
GitHub: [abdulnafih-official](https://github.com/abdulnafih-official) · LinkedIn: `<www.linkedin.com/in/abdulnafih0001>` · Email: `<abdulnafih.official@gmail.com>`