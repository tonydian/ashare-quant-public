# -*- coding: utf-8 -*-
"""ops_fast 滚动 min/max 回归测试：大窗口（w>40）曾因 chunk 行数 < w 输出全 NaN。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qf import ops_fast as of


def _make(n=320, cols=6, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, cols)).astype(np.float32)
    mask = rng.random((n, cols)) < 0.06
    a[mask] = np.nan
    return a


@pytest.mark.parametrize("w", [5, 14, 55, 250])
def test_roll_max_min_large_windows(w):
    a = _make()
    n, c = a.shape
    w = min(w, n)
    minp = of._minp(w)  # ops_fast 的默认 min_periods（比 w 宽松，用于提前 warmup）
    df = pd.DataFrame(a)
    ref_max = df.rolling(w, min_periods=minp).max().to_numpy()
    ref_min = df.rolling(w, min_periods=minp).min().to_numpy()
    got_max = of.roll_max(a, w)
    got_min = of.roll_min(a, w)
    assert got_max.shape == a.shape and got_min.shape == a.shape
    # 大窗口不应整列 NaN（回归点）
    if w > 40:
        assert np.isfinite(got_max).sum() > 0, f"w={w} roll_max 全 NaN"
        assert np.isfinite(got_min).sum() > 0, f"w={w} roll_min 全 NaN"
    np.testing.assert_allclose(got_max, ref_max, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(got_min, ref_min, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_roll_max_w250_shape_and_nan_semantics():
    a = _make(n=400, cols=4, seed=3)
    got = of.roll_max(a, 250)
    assert got.shape == a.shape
    # 大窗口回归点：不能整列 NaN（旧实现 w>40 全 NaN）
    assert np.isfinite(got).sum() > 0
    minp = of._minp(250)
    assert np.isnan(got[:minp - 1]).all(), f"前 {minp-1} 行应无足够有效值"
    ref = pd.DataFrame(a).rolling(250, min_periods=minp).max().to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6, equal_nan=True)
