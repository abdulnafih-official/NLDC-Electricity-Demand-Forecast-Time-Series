"""
Shared by the tuning notebook, 7_Sarima.py and the later prediction function.
"""
import numpy as np
import pandas as pd

from config import SERIES_TRAIN_PATH, SERIES_CALIB_PATH, SERIES_TEST_PATH

_ORIGIN = pd.Timestamp("2020-01-01")   # fixed origin: train/calib/test/future share one phase
PERIOD = 365.25


def fourier_terms(index, k):
    """SARIMAX exog: const + linear trend (years) + yearly Fourier terms (harmonics 1..k).
    Depends only on the date, never on y."""
    t = np.asarray((index - _ORIGIN).days, dtype=float)
    cols = {"const": np.ones(len(t)), "trend": t / PERIOD}
    for j in range(1, k + 1):
        cols[f"sin{j}"] = np.sin(2 * np.pi * j * t / PERIOD)
        cols[f"cos{j}"] = np.cos(2 * np.pi * j * t / PERIOD)
    return pd.DataFrame(cols, index=index)


def load_splits():
    """Load the project's saved train/calib/test series; check they are contiguous and gap-free."""
    train, calib, test = [pd.read_parquet(p).set_index("ds")["y"].asfreq("D")
                          for p in (SERIES_TRAIN_PATH, SERIES_CALIB_PATH, SERIES_TEST_PATH)]
    for a, b in ((train, calib), (calib, test)):
        assert b.index[0] - a.index[-1] == pd.Timedelta(days=1), "splits are not contiguous"
    assert not any(x.isna().any() for x in (train, calib, test)), "gaps in a split"
    return train, calib, test
