# src/predict.py  (unchanged from before; needs load_table in storage.py)
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.storage import load_table


@lru_cache(maxsize=1)
def _lookup():
    return load_table("lookup.parquet").set_index(["model", "ds"]).sort_index()


def predict(model_name: str, date) -> dict:
    t = _lookup()
    models = t.index.get_level_values("model").unique().tolist()
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}; choose from {models}")
    date = pd.Timestamp(date).normalize()
    try:
        r = t.loc[(model_name, date)]
    except KeyError:
        d = t.loc[model_name].index
        raise ValueError(f"{model_name} covers {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}") from None
    return {"lower": float(r["lower"]), "point": float(r["point"]), "upper": float(r["upper"]),
            "actual": None if pd.isna(r["actual"]) else float(r["actual"]),
            "segment": r["segment"], "horizon_days": int(r["horizon_days"]),
            "warnings": [w for w in r["warnings"].split(" | ") if w]}