# -*- coding: utf-8 -*-
"""独立审计：算子层（qf/ops.py + qf/ops_fast.py）数值正确性。

不依赖业务因子，直接用 pandas 参考实现对比随机面板（形状刻意触发 ops_fast 的
分块 sliding-window 路径），并检查窗口覆盖是否连续（无空洞）。

审计历史：F-01（roll_min/roll_max 分块边界空洞）已由 Lead 于 2026-10-21 修复
（step=rows-(w-1) 块间重叠），原 xfail(strict) 已转为上述回归测试。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import ops                                  # noqa: E402

W = 10
T, N = 620, 2600     # N 大 → ops_fast 分块 size = 8e6/(w*N) ≈ 307 < T，触发分块路径


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(7)
    return rng.normal(size=(T, N)).astype(np.float32)


def _pd(a, w, how):
    s = pd.DataFrame(a)
    r = s.rolling(w, min_periods=w)
    if how == "std":
        # 参考实现（vendor factor.py）与 ops 统一使用总体标准差 ddof=0
        return r.std(ddof=0).to_numpy(dtype=np.float64)
    return getattr(r, how)().to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------
def test_shift_and_pct_change(panel):
    k = 3
    exp = np.full_like(panel, np.nan)
    exp[k:] = panel[:-k]
    assert np.array_equal(ops.shift(panel, k), exp, equal_nan=True)
    r = ops.pct_change(panel, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        exp_r = panel.astype(np.float64) / exp.astype(np.float64) - 1.0
    ok = np.isfinite(r) & np.isfinite(exp_r)
    assert np.allclose(r[ok], exp_r[ok], rtol=1e-5, atol=1e-6)


def test_roll_mean_sum_std(panel):
    for how, fn in (("mean", ops.roll_mean), ("sum", ops.roll_sum), ("std", ops.roll_std)):
        got = fn(panel, W)
        exp = _pd(panel, W, how)
        ok = np.isfinite(got) & np.isfinite(exp)
        assert ok.sum() > 0.9 * got.size
        np.testing.assert_allclose(got[ok], exp[ok], rtol=1e-5, atol=1e-5,
                                   err_msg=f"roll_{how} 与 pandas 不一致")


def test_roll_skew_kurt(panel):
    for how, fn in (("skew", ops.roll_skew), ("kurt", ops.roll_kurt)):
        got = fn(panel, W)
        exp = _pd(panel, W, how)
        ok = np.isfinite(got) & np.isfinite(exp)
        assert ok.sum() > 0.5 * got.size
        # pandas 为无偏估计，ops_fast 已实现相同修正；容差放宽到 1e-3
        np.testing.assert_allclose(got[ok], exp[ok], rtol=1e-3, atol=1e-3,
                                   err_msg=f"roll_{how} 与 pandas 不一致")


def test_roll_corr(panel):
    a = panel
    b = np.roll(panel, 1, axis=0).astype(np.float32)
    b[0] = np.nan
    got = ops.roll_corr(a, b, W)
    exp = pd.DataFrame(a).rolling(W, min_periods=W).corr(pd.DataFrame(b)).to_numpy()
    ok = np.isfinite(got) & np.isfinite(exp)
    assert ok.sum() > 0.5 * got.size
    np.testing.assert_allclose(got[ok], exp[ok], rtol=1e-3, atol=1e-3)


def test_slope_matches_polyfit(panel):
    got = ops.roll_slope(panel, W)
    x = np.arange(W, dtype=np.float64)
    x = x - x.mean()
    den = float((x ** 2).sum())
    for t in (W - 1, 500, T - 1):
        y = panel[t - W + 1: t + 1].astype(np.float64)     # (W, N)
        exp = (x[:, None] * y).sum(axis=0) / den
        assert np.allclose(got[t], exp, rtol=1e-4, atol=1e-5)


def test_rank_last_matches_pandas(panel):
    got = ops.roll_rank_last(panel, W)
    exp = pd.DataFrame(panel).rolling(W, min_periods=W).rank(pct=True).to_numpy()
    ok = np.isfinite(got) & np.isfinite(exp)
    assert ok.sum() > 0.9 * got.size
    np.testing.assert_allclose(got[ok], exp[ok], rtol=1e-5, atol=1e-5)


def test_ema_matches_pandas(panel):
    span = 12
    got = ops.ema(panel, span)
    exp = pd.DataFrame(panel).ewm(span=span, adjust=False,
                                  min_periods=ops._minp(span, 3)).mean().to_numpy()
    ok = np.isfinite(got) & np.isfinite(exp)
    assert ok.sum() > 0.9 * got.size
    np.testing.assert_allclose(got[ok], exp[ok], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 回归测试：F-01（roll_min / roll_max 分块边界空洞，已于 2026-10-21 修复）
# 修复前 3 个测试为 xfail(strict)；修复后转为正常断言，防止回归。
# ---------------------------------------------------------------------------
def test_roll_max_min_no_chunk_gaps(panel):
    """窗口覆盖必须连续：t >= w-1 的每一行都应有值（全 finite 输入）。"""
    for fn, name in ((ops.roll_max, "roll_max"), (ops.roll_min, "roll_min")):
        got = fn(panel, W)
        nan_rows = np.flatnonzero(~np.isfinite(got[W - 1:]).all(axis=1)) + (W - 1)
        assert len(nan_rows) == 0, (
            f"{name} 在分块边界产生空洞：{len(nan_rows)} 行全 NaN，"
            f"前几行={nan_rows[:10].tolist()}（应为 0）")


def test_roll_max_values_match_pandas(panel):
    got = ops.roll_max(panel, W)
    exp = _pd(panel, W, "max")
    ok = np.isfinite(got) & np.isfinite(exp)
    nan_gap = np.flatnonzero(~np.isfinite(got[W - 1:]).all(axis=1))
    assert len(nan_gap) == 0, f"roll_max 分块边界有 {len(nan_gap)} 行空洞"
    np.testing.assert_allclose(got[ok], exp[ok], rtol=1e-5, atol=1e-5)


def test_roll_min_values_match_pandas(panel):
    got = ops.roll_min(panel, W)
    exp = _pd(panel, W, "min")
    nan_gap = np.flatnonzero(~np.isfinite(got[W - 1:]).all(axis=1))
    assert len(nan_gap) == 0, f"roll_min 分块边界有 {len(nan_gap)} 行空洞"
    ok = np.isfinite(got) & np.isfinite(exp)
    np.testing.assert_allclose(got[ok], exp[ok], rtol=1e-5, atol=1e-5)
