# -*- coding: utf-8 -*-
"""波动率目标仓位开关（vol_target_scale）的行为测试。"""
import numpy as np
import pytest

from qf import config as C
from qf.backtest import vol_target_scale


def _nav_series(n=60, daily_vol=0.01, seed=7):
    """恒定日波动 random walk 净值（年化波动 ≈ daily_vol*√252 ≈ 15.9%）。"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, daily_vol, n)
    return 100.0 * np.cumprod(1 + r)


def test_disabled_returns_identity():
    nav = _nav_series()
    s, v = vol_target_scale(nav, 59, C)
    assert s == 1.0 and np.isnan(v)


def test_target_above_realized_vol_scales_up_to_cap():
    nav = _nav_series(daily_vol=0.01)          # 年化 ≈ 15.9%
    cfg = type("Cfg", (), {**{k: getattr(C, k) for k in dir(C) if k.isupper()},
                           "VOL_TARGET": 0.10, "VOL_LOOKBACK": 20,
                           "VOL_MIN_EXP": 0.5, "VOL_MAX_EXP": 1.0})
    s, v = vol_target_scale(nav, 59, cfg)
    assert 0.5 <= s <= 1.0                     # 目标低于实际波动 → 缩仓
    assert v == pytest.approx(0.159, rel=0.35)
    assert s == pytest.approx(0.10 / v, abs=0.05)


def test_target_below_realized_vol_scales_down_to_floor():
    nav = _nav_series(daily_vol=0.03)          # 年化 ≈ 47.6%
    cfg = type("Cfg", (), {**{k: getattr(C, k) for k in dir(C) if k.isupper()},
                           "VOL_TARGET": 0.10, "VOL_LOOKBACK": 20,
                           "VOL_MIN_EXP": 0.5, "VOL_MAX_EXP": 1.0})
    s, v = vol_target_scale(nav, 59, cfg)
    assert s == 0.5                            # 0.10/0.476 ≈ 0.21 → 截到下限
    assert v == pytest.approx(0.476, rel=0.35)


def test_no_lookahead_only_uses_data_up_to_i():
    nav = _nav_series()
    cfg = type("Cfg", (), {**{k: getattr(C, k) for k in dir(C) if k.isupper()},
                           "VOL_TARGET": 0.10, "VOL_LOOKBACK": 20,
                           "VOL_MIN_EXP": 0.5, "VOL_MAX_EXP": 1.0})
    s1, v1 = vol_target_scale(nav, 40, cfg)
    nav2 = nav.copy()
    nav2[41:] = 999.0                          # 未来数据炸掉也不影响 i=40
    s2, v2 = vol_target_scale(nav2, 40, cfg)
    assert s1 == s2 and v1 == v2


def test_warmup_too_short_returns_identity():
    nav = _nav_series()
    cfg = type("Cfg", (), {**{k: getattr(C, k) for k in dir(C) if k.isupper()},
                           "VOL_TARGET": 0.10, "VOL_LOOKBACK": 20,
                           "VOL_MIN_EXP": 0.5, "VOL_MAX_EXP": 1.0})
    s, v = vol_target_scale(nav, 4, cfg)       # 只有 5 个点、4 个收益 → 不干预
    assert s == 1.0 and np.isnan(v)


def test_flat_nav_zero_vol_returns_identity():
    nav = np.full(60, 100.0)
    cfg = type("Cfg", (), {**{k: getattr(C, k) for k in dir(C) if k.isupper()},
                           "VOL_TARGET": 0.10, "VOL_LOOKBACK": 20,
                           "VOL_MIN_EXP": 0.5, "VOL_MAX_EXP": 1.0})
    s, v = vol_target_scale(nav, 59, cfg)
    assert s == 1.0                            # 零波动（全现金段）不放大仓位


def test_nan_segments_are_ignored():
    nav = _nav_series()
    nav[:30] = np.nan
    cfg = type("Cfg", (), {**{k: getattr(C, k) for k in dir(C) if k.isupper()},
                           "VOL_TARGET": 0.10, "VOL_LOOKBACK": 20,
                           "VOL_MIN_EXP": 0.5, "VOL_MAX_EXP": 1.0})
    s, v = vol_target_scale(nav, 59, cfg)      # 窗口内 30 个有效点，正常计算
    assert np.isfinite(v) and 0.5 <= s <= 1.0
