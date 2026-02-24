import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils

def make_trending_df(n=120):
    rng = np.random.default_rng(42)
    base = np.linspace(100.0, 110.0, n)
    noise = rng.normal(0, 0.2, size=n)
    close = base + noise
    high = close + np.abs(rng.normal(0, 0.15, size=n))
    low = close - np.abs(rng.normal(0, 0.15, size=n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    vol = np.maximum(100, (rng.normal(200, 30, size=n))).astype(int)
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": vol,
        "real_volume": vol
    })
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    df.index = idx
    return df

def test_get_adx_nonzero_on_trend():
    df = make_trending_df(120)
    val = utils.get_adx(df, period=14)
    assert val is not None
    assert val > 12.0

def test_get_adx_series_length():
    df = make_trending_df(120)
    s = utils.get_adx_series(df, period=14)
    assert isinstance(s, pd.Series)
    assert len(s) == len(df)
