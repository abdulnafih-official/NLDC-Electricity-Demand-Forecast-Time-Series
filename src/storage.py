# src/storage.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os

import joblib
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

import config

load_dotenv()

MODEL_FILES = {"xgboost": "xgb.json", "prophet": "prophet.json", "sarima": "sarima.joblib"}


def _secret(name: str, default: str | None = None) -> str:
    """st.secrets first (deployed), then .env / environment (local)."""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # streamlit missing or no secrets.toml
        pass
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"{name} not set (.env locally, Streamlit Secrets on deploy)")
    return value


def _local_path(filename: str) -> str:
    return str(config.MODELS_DIR / filename)


def _hf_path(filename: str) -> str:
    return hf_hub_download(
        repo_id=_secret("REPO_NAME"),
        filename=filename,
        token=_secret("HF_TOKEN"),
    )


def _path(filename: str, backend: str | None) -> str:
    """backend: 'local' | 'hf'. Default from STORAGE_BACKEND, else 'local'."""
    backend = backend or _secret("STORAGE_BACKEND", "local")
    if backend == "local":
        return _local_path(filename)
    if backend == "hf":
        return _hf_path(filename)
    raise ValueError(f"backend must be 'local' or 'hf', got {backend!r}")


def load_model(name: str, backend: str | None = None):
    if name not in MODEL_FILES:
        raise ValueError(f"Unknown model {name!r}; choose from {list(MODEL_FILES)}")
    path = _path(MODEL_FILES[name], backend)

    if name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(path)
        return model
    if name == "prophet":
        from prophet.serialize import model_from_json
        with open(path) as f:
            return model_from_json(f.read())
    return joblib.load(path)


def load_json(filename: str, backend: str | None = None) -> dict:
    """e.g. load_json("calibration.json")."""
    with open(_path(filename, backend)) as f:
        return json.load(f)