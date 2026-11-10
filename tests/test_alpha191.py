# -*- coding: utf-8 -*-
"""Alpha191 算子与实现口径的单元测试。

校核方式：所有算子都与 pandas 的独立实现逐格比对（随机面板 + 手工构造的边界），
不依赖本模块自身的假设。运行：``pytest tests/test_alpha191.py -q``
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qf import alpha_ops as op


def _panel(T=40, N=7, seed=0, nan_head=3):
    rng = np.random.default_rng(seed)
    a = rng.normal(10.0, 1.0, size=(T, N)).cumsum(axis=0)
    a[:nan_head] = np.nan                      # 模拟上市前缺失
    a[10:12, 0] = np.nan                       # 模拟中途停牌
    return a


def _pd(a):
    return pd.DataFrame(a)


# ------------------------------------------------------------------ 基础算子
def test_shift_and_delta():
    a = _panel()
    s = op.shift(a, 2)
    assert np.isnan(s[:2]).all()
    np.testing.assert_allclose(s[2:], a[:-2])
    d = op.delta(a, 3)
    np.testing.assert_allclose(d[3:], a[3:] - a[:-3], equal_nan=True)


@pytest.mark.parametrize("w", [1, 5, 20])
def test_roll_sum_mean_vs_pandas(w):
    a = _panel()
    r = _pd(a).rolling(w, min_periods=w)
    np.testing.assert_allclose(op.roll_sum(a, w), r.sum().to_numpy(), equal_nan=True, rtol=1e-9)
    np.testing.assert_allclose(op.roll_mean(a, w), r.mean().to_numpy(), equal_nan=True, rtol=1e-9)


@pytest.mark.parametrize("w", [2, 5, 20])
def test_roll_std_vs_pandas(w):
    a = _panel()
    r = _pd(a).rolling(w, min_periods=w)
    np.testing.assert_allclose(op.roll_std(a, w), r.std(ddof=0).to_numpy(), equal_nan=True, rtol=1e-7)


@pytest.mark.parametrize("w", [1, 2, 5, 20])
def test_ts_max_min_vs_pandas(w):
    a = _panel()
    r = _pd(a).rolling(w, min_periods=w)
    np.testing.assert_allclose(op.ts_max(a, w), r.max().to_numpy(), equal_nan=True)
    np.testing.assert_allclose(op.ts_min(a, w), r.min().to_numpy(), equal_nan=True)


@pytest.mark.parametrize("w", [3, 5, 10])
def test_ts_rank_vs_pandas(w):
    """TSRANK 的分位口径：等于"窗口内小于当日值的个数 / w"（无并列时=percentile rank）。"""
    a = _panel(seed=1)
    got = op.ts_rank(a, w)
    exp = np.full(a.shape, np.nan)
    for t in range(w - 1, a.shape[0]):
        win = a[t - w + 1:t + 1]
        last = a[t]
        ok = np.isfinite(win).all(axis=0) & np.isfinite(last)
        cnt = (last[None, :] > win).sum(axis=0)
        exp[t, ok] = cnt[ok] / w
    np.testing.assert_allclose(got, exp, equal_nan=True)


@pytest.mark.parametrize("w", [2, 5, 12])
def test_decay_weights(w):
    a = _panel()
    got = op.decay(a, w)
    acc = np.zeros(a.shape)
    for k in range(w):
        acc += (w - k) * op.shift(a, k)
    exp = acc / (w * (w + 1) / 2.0)
    valid = op.roll_count(a, w) >= w
    np.testing.assert_allclose(got[valid], exp[valid], rtol=1e-10)


def test_wma_weights_are_0p9_powers():
    w = 5
    a = _panel()
    got = op.wma(a, w)
    wt = 0.9 ** np.arange(w)
    acc = sum(wt[k] * op.shift(a, k) for k in range(w))
    exp = acc / wt.sum()
    valid = op.roll_count(a, w) >= w
    np.testing.assert_allclose(got[valid], exp[valid], rtol=1e-10)


@pytest.mark.parametrize("n,m", [(6, 1), (13, 2), (20, 1)])
def test_sma_matches_pandas_ewm(n, m):
    """SMA(n,m) 等价 pandas ewm(alpha=m/n, adjust=False)，差异只允许来自预热期。"""
    a = _panel(T=200, nan_head=60)
    got = op.sma(a, n, m)
    exp = _pd(a).ewm(alpha=m / n, adjust=False).mean().to_numpy()
    tail = slice(60 + 3 * n, None)
    np.testing.assert_allclose(got[tail], exp[tail], rtol=1e-6)


@pytest.mark.parametrize("w", [4, 10])
def test_roll_corr_vs_pandas(w):
    a = _panel(seed=2)
    b = _panel(seed=3)
    got = op.roll_corr(a, b, w)
    exp = _pd(a).rolling(w, min_periods=w).corr(_pd(b)).to_numpy()
    m = np.isfinite(exp)
    assert m.sum() > 50
    np.testing.assert_allclose(got[m], exp[m], rtol=1e-6, atol=1e-9)


def test_roll_count_and_rank_pct():
    a = _panel(T=10, N=5, nan_head=0)
    a[3, 1] = np.nan
    c = op.roll_count(a, 4)
    np.testing.assert_allclose(c[3, 1], 3.0)
    r = op.rank_pct(a)
    assert np.isfinite(r).sum() == np.isfinite(a).sum()
    # 秩必须与数值同序：按 a 排序后 r 应严格递增
    o = np.argsort(a[0])
    assert np.all(np.diff(r[0][o]) > 0)


def test_ts_argmin_argmax():
    a = np.array([[5.0], [3.0], [7.0], [2.0], [9.0]])
    assert op.ts_argmin(a, 3)[4, 0] == 1      # 窗口 [7,2,9] 最小是 2（1 天前）
    assert op.ts_argmax(a, 3)[4, 0] == 0      # 最大是 9（当天）
    assert np.isnan(op.ts_argmin(a, 3)[1, 0])


def test_sequence_beta_recovers_slope():
    n = 10
    x = np.arange(60)[:, None].astype(float)
    got = op.sequence_beta(x, n)[-1, 0]
    assert got == pytest.approx(1.0)


def test_sum_where():
    a = np.arange(12, dtype=float).reshape(6, 2)
    cond = a % 2 == 0
    got = op.sum_where(a, 3, cond)[-1]
    exp = np.where(cond[-3:], a[-3:], 0.0).sum(axis=0)
    np.testing.assert_allclose(got, exp)


# ------------------------------------------------------- 端到端：整条公式链路
class _FakeMD:
    """最小行情对象：只提供 Alpha191 公式用到的字段。"""

    def __init__(self, T=60, N=9, seed=7):
        rng = np.random.default_rng(seed)
        base = 30 + rng.normal(0, 1, (T, N)).cumsum(axis=0)
        self.close = base
        self.open = base * (1 + rng.normal(0, 0.005, (T, N)))
        self.high = np.maximum(self.open, self.close) * (1 + np.abs(rng.normal(0, 0.01, (T, N))))
        self.low = np.minimum(self.open, self.close) * (1 - np.abs(rng.normal(0, 0.01, (T, N))))
        self.volume = np.abs(rng.normal(1e7, 2e6, (T, N)))
        self.amount = self.volume * base
        self.T, self.N = T, N


def test_alpha001_matches_independent_pandas():
    """用 pandas 独立实现 alpha_001，与公式表实现逐格比对（验证算子+解析全链路）。"""
    from qf import alpha191 as A
    md = _FakeMD()
    ns = A.build_inputs(md, vwap=(md.high + md.low + md.close) / 3, verbose=False)
    got = A.compute_one(1, ns)

    df_vol = pd.DataFrame(np.log(ns["VOL"]))
    df_ret = pd.DataFrame((md.close - md.open) / md.open)

    def centered_rank(df):
        r = df.rank(axis=1, method="average") - 1.0
        n = df.notna().sum(axis=1)
        return r.sub((n - 1) / 2.0, axis=0).div((n - 1).replace(0, np.nan), axis=0)

    a = centered_rank(df_vol.diff())
    b = centered_rank(df_ret)
    ref = -a.rolling(6, min_periods=6).corr(b).to_numpy()

    m = np.isfinite(ref) & np.isfinite(got)
    assert m.sum() > 200
    np.testing.assert_allclose(got[m], ref[m], rtol=1e-6, atol=1e-9)


def test_alpha014_simple_price_diff():
    from qf import alpha191 as A
    md = _FakeMD()
    ns = A.build_inputs(md, vwap=(md.high + md.low + md.close) / 3, verbose=False)
    got = A.compute_one(14, ns)          # CLOSE - DELAY(CLOSE,5)
    exp = md.close - np.vstack([np.full((5, md.N), np.nan), md.close[:-5]])
    # 基础面板以 float32 存储（省内存），故允许 1e-5 的量化误差
    np.testing.assert_allclose(got, exp, equal_nan=True, rtol=1e-5, atol=1e-5)
