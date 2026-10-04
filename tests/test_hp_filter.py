"""Focused regression tests for local HP correction invariants."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

# The project module normally supplies this constant via src.data.
data_mod = types.ModuleType("src.data")
data_mod.TICKER = "XU030"
sys.modules.setdefault("src.data", data_mod)
from src import hp_filter as hp


def series(n=80):
    idx = pd.bdate_range("2024-01-01", periods=n)
    x = np.arange(n)
    return pd.Series(100 * np.exp(.001*x + .012*np.sin(x/4)), index=idx)


def test_date_order_independent():
    p = series()
    dates = [p.index[20], p.index[22], p.index[50]]
    a = hp.find_minimal_hp_correction(p, jump_dates=dates)[0]
    b = hp.find_minimal_hp_correction(p, jump_dates=dates[::-1])[0]
    pd.testing.assert_series_equal(a, b)


def test_total_log_return_preserved_and_window_edges_fixed():
    p = series()
    date = p.index[30]
    corrected = hp.find_minimal_hp_correction(p, jump_dates=[date], window=3)[0]
    assert corrected.iloc[0] == pytest.approx(np.log(p.iloc[0]))
    assert corrected.iloc[-1] == pytest.approx(np.log(p.iloc[-1]))
    assert corrected.diff().dropna().sum() == pytest.approx(np.log(p.iloc[-1]/p.iloc[0]))


def test_window_clips_at_series_edges():
    p = series(20)
    corrected, replaced, *_ = hp.find_minimal_hp_correction(
        p, jump_dates=[p.index[0]], window=3, est_window=30)
    assert len(corrected) == len(p)
    # Cosine blending leaves the clipped block's endpoints unchanged.
    assert replaced == list(p.index[1:3])


def test_missing_date_warns():
    p = series()
    with pytest.warns(UserWarning, match="indekste bulunamadı"):
        hp.find_minimal_hp_correction(p, jump_dates=["2030-01-01"])
