# -*- coding: utf-8 -*-
"""``qf/pvdiv.py`` 的不变量测试。

这个模块目前是研究产物（未进生产池），但它一旦被复用，三条底线必须成立：

1. **无未来函数**：第 t 行的值只能用到 t 及之前的数据（回测最贵的 bug）
2. **变体关系自洽**：B = −Δcorr、A = B×(1+w)、w∈[−0.5,0.5]
3. **停牌不伪造**：停牌日（alive=False）不得用 ffill 价格与 0 成交量配对
"""
from __future__ import annotations

import numpy as np
import pytest

from qf import pvdiv


class FakeMD:
    """只提供 pvdiv 需要的字段，避免测试依赖 4GB 的真面板。"""

    def __init__(self, T: int = 300, N: int = 200, seed: int = 11):
        rng = np.random.default_rng(seed)
        self.T, self.N = T, N
        # 几何随机游走 + 成交量对数正态，保证 corr 不为常数
        ret = rng.normal(0.0, 0.02, size=(T, N))
        self.close = (10.0 * np.exp(np.cumsum(ret, axis=0))).astype(np.float32)
        self.high = (self.close * (1.0 + np.abs(rng.normal(0, 0.01, (T, N))))).astype(np.float32)
        self.volume_orig = np.exp(rng.normal(12.0, 0.8, size=(T, N))).astype(np.float32)
        self.alive = np.ones((T, N), dtype=bool)
        self.eligible = np.ones((T, N), dtype=bool)


@pytest.fixture(scope="module")
def md():
    return FakeMD()


def test_shapes_and_dtype(md):
    p = pvdiv.panels(md)
    assert set(p) == set(pvdiv.VARIANTS)
    for k, v in p.items():
        assert v.shape == (md.T, md.N), k
        assert v.dtype == np.float32, k
        assert np.isfinite(v).sum() > 0, k


def test_no_lookahead(md):
    """改掉最后 50 行，前 T-50 行必须一字不变。"""
    base = pvdiv.panels(md)["A"]
    cut = md.T - 50

    alt = FakeMD(md.T, md.N, seed=11)
    rng = np.random.default_rng(999)
    alt.close[cut:] *= np.float32(1.3)
    alt.high[cut:] *= np.float32(1.3)
    alt.volume_orig[cut:] *= np.float32(5.0)
    del rng

    after = pvdiv.panels(alt)["A"]
    np.testing.assert_array_equal(
        np.nan_to_num(base[:cut], nan=-1e9), np.nan_to_num(after[:cut], nan=-1e9),
        err_msg="前 cut 行受未来数据影响 → 存在未来函数",
    )
    assert not np.allclose(np.nan_to_num(base[cut:], nan=0),
                           np.nan_to_num(after[cut:], nan=0)), "改未来数据后尾部也没变，测试本身失效"


def test_variant_algebra(md):
    p = pvdiv.panels(md)
    parts = pvdiv.raw_parts(md)
    d = parts["d_corr"]
    w = parts["rank_std20"] - 0.5

    ok = np.isfinite(d) & np.isfinite(w) & (np.abs(d) > 1e-6)
    np.testing.assert_allclose(p["B"][ok], -d[ok], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p["A"][ok], p["B"][ok] * (1.0 + w[ok]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p["D"][ok], p["B"][ok] * (w[ok] + 0.5), rtol=1e-5, atol=1e-6)

    # 权重有界 → A 的绝对值不超过 1.5×|B|
    assert np.all(np.abs(p["A"][ok]) <= 1.5 * np.abs(p["B"][ok]) + 1e-6)


def test_rank_in_unit_interval(md):
    r = pvdiv.raw_parts(md)["rank_std20"]
    v = r[np.isfinite(r)]
    assert v.min() > 0.0 and v.max() <= 1.0
    assert abs(float(v.mean()) - 0.5) < 0.05


def test_slow_panels_algebra(md):
    """降频版 E/F/G 必须是「窗口/平滑」的纯改组，不能偷偷换成别的信号。"""
    from qf import ops

    p = pvdiv.slow_panels(md)
    assert set(p) == set(pvdiv.SLOW_VARIANTS)
    d5 = pvdiv._dcorr(md, 5, 5)
    d20 = pvdiv._dcorr(md, 20, 20)

    def same(a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        assert ok.sum() > 1000, "有效格太少，测试没意义"
        np.testing.assert_allclose(a[ok], b[ok], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))

    same(p["E"], -d20)
    same(p["F"], ops.roll_mean(-d5, pvdiv.W_SMOOTH, minp=15))
    same(p["G"], ops.roll_mean(-d20, pvdiv.W_SMOOTH, minp=15))


def test_slow_no_lookahead(md):
    """平滑 + 长窗口的记忆更长，未来函数更容易藏在这里。"""
    base = pvdiv.slow_panels(md)["G"]
    cut = md.T - 60

    alt = FakeMD(md.T, md.N, seed=11)
    alt.close[cut:] *= np.float32(1.4)
    alt.volume_orig[cut:] *= np.float32(3.0)
    after = pvdiv.slow_panels(alt)["G"]

    np.testing.assert_array_equal(
        np.nan_to_num(base[:cut], nan=-1e9), np.nan_to_num(after[:cut], nan=-1e9),
        err_msg="降频版存在未来函数",
    )


def test_suspended_not_faked():
    """停牌段（alive=False）不得用 ffill 价格与 NaN 成交量配对。

    停牌第 1 天仍会有值：5 日窗口里还剩 4 个真实观测（min_periods=4），
    这属于「用陈旧窗口」，不是伪造。从第 2 天起窗口内有效点不足，必须全 NaN。
    """
    m = FakeMD(120, 80, seed=5)
    m.alive[40:60, :] = False
    m.volume_orig[40:60, :] = np.nan
    p = pvdiv.panels(m)["B"]
    assert not np.isfinite(p[41:60]).any(), "停牌第 2 天起仍在出值 → 停牌被当成真实观测"

    # 恢复交易后应重新有值（说明是暂时剔除，不是永久置 NaN）
    assert np.isfinite(p[70:]).sum() > 0
