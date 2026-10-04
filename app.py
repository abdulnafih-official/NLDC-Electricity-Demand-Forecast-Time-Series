# app.py
from datetime import date

import streamlit as st

from src.predict import predict

MODELS = {"XGBoost": "xgboost", "Prophet": "prophet", "SARIMA": "sarima"}

st.title("All-India daily peak demand")

label = st.selectbox("Model", list(MODELS))
d = st.date_input("Date", value=date(2025, 3, 15), min_value=date(2021, 9, 1), max_value=date(2030, 12, 31))

if st.button("Predict"):
    try:
        r = predict(MODELS[label], d)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.metric("Peak demand (MW)", f"{r['point']:,.0f}")
    st.caption(f"90% interval: {r['lower']:,.0f} to {r['upper']:,.0f} MW")
    if r["actual"] is not None:
        st.caption(f"Actual: {r['actual']:,.0f} MW")
    for w in r["warnings"]:
        st.warning(w)

st.divider()
with st.expander("About this project", expanded=False):
    st.subheader("What it is")
    st.markdown(
        "A forecasting app built by **Abdul Nafih**."
        "Pick a date and a model and get the all-India daily peak demand with a 90% prediction interval."
    )

    st.subheader("Why I built it")
    st.markdown(
        "A few months ago I used this NLDC dataset to practice data cleaning. "
        "Recently there has been load shedding in my area, which made me wonder: "
        "if demand can be predicted in advance, couldn't it be planned for and supplied accordingly? "
        "So I tried predicting it myself, then tested the predictions on data the models had never "
        "seen to check how accurate they really are."
    )

    st.subheader("What I found")
    st.markdown(
        """
I tested on 2025 data, two ways: forecasting **1 to 7 days ahead** and forecasting
**up to 175 days ahead**. No single model won both.

- **Next 7 days:** SARIMA was best, with a MAPE of 3.27% against 3.57% for
  "same as yesterday". For the next day alone it reached 1.87% against 2.34%.
  XGBoost (3.76%) did not beat "same as yesterday" overall, and its error grew with
  the horizon.
- **Up to 175 days(~6 months):** Prophet was best with a MAPE of 3.99%, against 5.34% for
  "same day last year". SARIMA was close at 4.15%. XGBoost (7.65%) was worse than
  the naive baseline, because it depends on recent values that are not available
  that far ahead.
- **Intervals:** the target is 90% coverage. Short-term, SARIMA hit 86% and XGBoost 85%.
  Long-term, Prophet reached only 79% overall and just 58% on days 91-175 (about
  April to June 2025), when demand ran above what the model expected. The models
  have no weather input, so they miss extreme-heat periods.
"""
    )

    st.subheader("Architecture")
    st.markdown(
        """
- **Data:** NLDC all-India demand, aggregated to the daily peak
- **Features:** national holidays and festivals (±2 day window), calendar and trend features, lags for XGBoost. No weather.
- **Models:** XGBoost, Prophet, SARIMA, compared against naive baselines
- **Intervals:** 90% for every model. Built-in for Prophet and SARIMA, conformal (calibration residuals) for XGBoost.
- **Split:** chronological. Train to 2023, calibration 2024, test from 2025.
- **Storage:** models in a private Hugging Face repo, loaded on demand
- **Serving:** Streamlit Community Cloud
"""
    )

    st.subheader("Connect")
    st.markdown(
        """
- [GitHub](https://github.com/abdulnafih-official)
- [LinkedIn]( www.linkedin.com/in/abdulnafih0001/)
- [Email](mailto:abdulnafih.official@gmail.com)
"""
    )