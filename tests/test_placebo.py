# -*- coding: utf-8 -*-
"""``build/placebo.py`` 的不变量测试。

这个工具存在的唯一理由，是防止「加过滤 → 少持仓 → 白拿降杠杆红利」被当成 alpha
（2026-11-06 HVRP 事件）。所以测试也围绕它的三条底线：

1. **同比例**：随机组的实际剔除比例必须接近请求值，否则对照组比实验组"轻"或"重"，
   比较就失去意义。
2. **作用域一致**：只能从「可交易且分数有限」的候选里抽，且同种子可复现。
3. **判据能挡住假信号**：F2 那种「8 相位 7/8 胜、但只优于 3/5 个随机种子」的结果，
   必须被判为不通过。
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("placebo_mod", ROOT / "build" / "placebo.py")
placebo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(placebo)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(7)
    T, N = 30, 200
    S = rng.normal(size=(T, N))
    S[rng.random((T, N)) < 0.1] = np.nan
    E = rng.random((T, N)) > 0.25
    return S, E


# ---------------------------------------------------------------------------
# 一、随机剔除本身
# ---------------------------------------------------------------------------
def test_drop_ratio_matches_request(panel):
    S, E = panel
    m = placebo.random_mask(S, E, drop_frac=0.20, seed=1)
    frac = placebo.actual_drop_frac(S, m, E)
    assert abs(frac - 0.20) < 0.01


def test_scope_is_tradable_and_scored(panel):
    """被剔除的格子必须原本既有分数、又可交易 —— 否则对照组与实验组作用域不同。"""
    S, E = panel
    m = placebo.random_mask(S, E, drop_frac=0.20, seed=3)
    legal = E & np.isfinite(S)
    assert (m & ~legal).sum() == 0


def test_same_seed_reproducible_different_seed_differs(panel):
    S, E = panel
    a = placebo.random_mask(S, E, drop_frac=0.20, seed=1)
    b = placebo.random_mask(S, E, drop_frac=0.20, seed=1)
    c = placebo.random_mask(S, E, drop_frac=0.20, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_apply_mask_only_nans_dropped(panel):
    S, E = panel
    m = placebo.random_mask(S, E, drop_frac=0.20, seed=1)
    out = placebo.apply_mask(S, m)
    assert not np.isfinite(out[m]).any()
    keep = ~m
    assert np.array_equal(out[keep], S[keep], equal_nan=True)
    # 原面板不被就地修改
    assert np.isfinite(S[m]).any()


def test_rows_restriction(panel):
    """只传部分行时，其余行不得被剔除。"""
    S, E = panel
    rows = np.array([2, 5, 9])
    m = placebo.random_mask(S, E, drop_frac=0.20, seed=1, rows=rows)
    assert m.sum(axis=1)[~np.isin(np.arange(S.shape[0]), rows)].sum() == 0
    assert m[rows].sum() > 0


def test_min_universe_skips_tiny_rows(panel):
    S, E = panel
    m = placebo.random_mask(S, E, drop_frac=0.20, seed=1, min_universe=10_000)
    assert m.sum() == 0


def test_invalid_drop_frac(panel):
    S, E = panel
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            placebo.random_mask(S, E, drop_frac=bad, seed=1)


def test_shape_mismatch(panel):
    S, E = panel
    with pytest.raises(ValueError):
        placebo.random_mask(S, E[:-1], drop_frac=0.20, seed=1)


def test_build_set_shapes(panel):
    S, E = panel
    d = placebo.build_set(S, E, drop_frac=0.20, seeds=(1, 2))
    assert set(d) == {"R1", "R2"}
    for v in d.values():
        assert v.shape == S.shape
        assert np.isnan(v).sum() > np.isnan(S).sum()


# ---------------------------------------------------------------------------
# 二、对照表与判据
# ---------------------------------------------------------------------------
def _make_df(treat_sharpe: float, seed_sharpes=(0.666, 0.700, 0.720, 0.745, 0.765),
             n_phases: int = 8) -> pd.DataFrame:
    """构造长表：每相位 treat 略高于随机均值，差值恒定（t 由相位间方差定）。"""
    rows = [{"config": "BASE", "phase": p, "sharpe": 0.590, "cagr": 0.0713,
             "maxdd": -0.1484, "avg_hold": 34.5, "avg_expo": 0.564} for p in range(n_phases)]
    for p in range(n_phases):
        rows.append({"config": "F2", "phase": p, "sharpe": treat_sharpe,
                     "cagr": 0.0649, "maxdd": -0.088, "avg_hold": 20.4, "avg_expo": 0.349})
    for i, s in enumerate(seed_sharpes, start=1):
        for p in range(n_phases):
            # 相位内抖动让配对差有非零方差
            jitter = 0.02 * np.sin(p + i)
            rows.append({"config": f"R{i}", "phase": p, "sharpe": s + jitter,
                         "cagr": 0.0621, "maxdd": -0.0912,
                         "avg_hold": 20.0, "avg_expo": 0.350})
    return pd.DataFrame(rows)


def test_compare_identifies_placebo_band():
    df = _make_df(0.796)
    cmp = placebo.compare(df, treat="F2")
    r = cmp.loc["sharpe"]
    assert r["n_seeds"] == 5
    assert r["win"] == 5            # 0.796 高于所有随机种子
    # 相位抖动让每个种子的八相位均值略有偏移，故用 0.01 容差
    assert r["placebo_mean"] == pytest.approx(
        np.mean([0.666, 0.700, 0.720, 0.745, 0.765]), abs=0.01)
    assert r["placebo_lo"] <= r["placebo_mean"] <= r["placebo_hi"]
    assert r["diff"] > 0


def test_judge_rejects_inside_band_result():
    """F2 实际结果：只优于 3/5 个种子 → 必须判不通过（旧判据会放过它）。"""
    df = _make_df(0.720)
    cmp = placebo.compare(df, treat="F2")
    ok, reasons = placebo.judge(cmp)
    assert ok is False
    assert any("随机种子" in x for x in reasons)


def test_judge_passes_clear_winner():
    df = _make_df(0.900)
    cmp = placebo.compare(df, treat="F2")
    ok, reasons = placebo.judge(cmp)
    assert ok is True, reasons


def test_judge_fails_when_other_metric_worse():
    """Sharpe 赢但 CAGR 输 → 仍不通过（不允许用回撤换收益当 alpha）。"""
    df = _make_df(0.900)
    df.loc[df.config == "F2", "cagr"] = 0.0300     # 远低于随机组 0.0621
    cmp = placebo.compare(df, treat="F2")
    ok, reasons = placebo.judge(cmp)
    assert ok is False
    assert any("cagr" in x for x in reasons)


def test_missing_placebo_group_raises():
    df = _make_df(0.796)
    df = df[~df.config.str.startswith("R")]
    with pytest.raises(KeyError):
        placebo.compare(df, treat="F2")


def test_leverage_check_flags_position_drop():
    df = _make_df(0.796)
    lv = placebo.leverage_check(df, treat="F2")
    assert lv.loc["avg_hold", "diff"] < 0          # 持仓下降
    assert lv.loc["avg_expo", "rel"] < -0.3        # 仓位掉超 30%
