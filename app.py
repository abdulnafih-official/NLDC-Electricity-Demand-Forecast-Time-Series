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