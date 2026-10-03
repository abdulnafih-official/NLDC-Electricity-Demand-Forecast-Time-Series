# app.py
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import predict
from src.storage import load_table

st.set_page_config(page_title="All-India Daily Peak Demand Forecast")
MODELS = {"XGBoost": "xgboost", "Prophet": "prophet", "SARIMA": "sarima"}


@st.cache_data
def get_lookup():
    return load_table("lookup.parquet")


@st.cache_data
def get_metrics():
    return pd.read_csv(Path(__file__).parent / "artifacts" / "metrics.csv")


st.title("All-India daily peak demand forecast")
st.caption("Daily peak demand (MW) with a 90% interval. Predictions are precomputed from the saved models.")

c1, c2 = st.columns(2)
label = c1.selectbox("Model", list(MODELS))
d = c2.date_input("Date", value=date(2025, 3, 15), min_value=date(2021, 9, 1), max_value=date(2030, 12, 31))
key = MODELS[label]

try:
    r = predict(key, d)
except ValueError as e:
    st.error(str(e))
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Prediction (MW)", f"{r['point']:,.0f}")
m2.metric("90% interval (MW)", f"{r['lower']:,.0f} to {r['upper']:,.0f}")
if r["actual"] is not None:
    err = r["actual"] - r["point"]
    m3.metric("Actual (MW)", f"{r['actual']:,.0f}", f"error {err:+,.0f} ({err / r['actual']:+.1%})", delta_color="off")

note = " (XGBoost uses real recent demand, 1-day-ahead)" if key == "xgboost" and r["horizon_days"] == 1 else ""
st.caption(f"Segment: {r['segment']} | horizon: {r['horizon_days']} day(s){note}")
for w in r["warnings"]:
    st.warning(w)

# chart: +-90 days around the selected date
t = get_lookup()
t = t[t.model == key]
ds = pd.Timestamp(d)
w = t[t.ds.between(ds - pd.Timedelta(days=90), ds + pd.Timedelta(days=90))]
fig = go.Figure()
fig.add_trace(go.Scatter(x=w.ds, y=w.upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=w.ds, y=w.lower, fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
                         line=dict(width=0), name="90% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=w.ds, y=w.point, name="Prediction", line=dict(color="#1f77b4")))
a = w.dropna(subset=["actual"])
if not a.empty:
    fig.add_trace(go.Scatter(x=a.ds, y=a.actual, name="Actual", line=dict(color="black")))
fig.add_trace(go.Scatter(x=[ds], y=[r["point"]], mode="markers", name="Selected date",