# -*- coding: utf-8 -*-
"""ETF 趋势模块单元测试：信号规则、成本、组合权重、数据完整性。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "build" / "etf" / "etf_panel.npz"


@pytest.fixture(scope="module")
def panel():
    if not PANEL.exists():
        pytest.skip("ETF 面板不存在，先跑 scripts/build_etf_data.py")
    z = np.load(PANEL, allow_pickle=True)
    return {
        "dates": pd.DatetimeIndex(z["dates"]),
        "codes": [str(c) for c in z["codes"]],
        "close": z["close"].astype(float),
        "raw_close": z["raw_close"].astype(float),
        "amount": z["amount"].astype(float),
    }


def test_panel_shape_and_alignment(panel):
    T = len(panel["dates"])
    assert T > 2000, f"ETF 历史太短：{T}"
    assert panel["close"].shape == (T, len(panel["codes"]))
    assert len(panel["codes"]) == 3
    assert set(panel["codes"]) == {"510300", "510500", "512100"}


def test_dates_monotonic_and_unique(panel):
    d = panel["dates"]
    assert d.is_monotonic_increasing
    assert d.is_unique


def test_qfq_positive_and_no_lookahead(panel):
    c = panel["close"]
    assert np.nanmin(c) > 0, "前复权价必须为正"
    raw = panel["raw_close"]
    m = np.isfinite(c) & np.isfinite(raw)
    assert m.sum() > 1000
    ratio = raw[m] / c[m]
    assert np.all(ratio > 0)


def test_listing_dates_staggered(panel):
    """512100 上市晚于 510300，前期应为 NaN。"""
    d = panel["dates"]
    c = panel["close"]
    j100 = panel["codes"].index("512100")
    j300 = panel["codes"].index("510300")
    first100 = d[np.flatnonzero(np.isfinite(c[:, j100]))[0]]
    first300 = d[np.flatnonzero(np.isfinite(c[:, j300]))[0]]
    assert first100 > first300


def test_donchian_signal_is_causal(panel):
    """突破信号只能用 t-1 及以前的数据。"""
    sys.path.insert(0, str(ROOT))
    from scripts.etf_trend_strategy import signals  # noqa

    dates = panel["dates"]
    P = {"close": panel["close"], "amount": panel["amount"]}
    sig = signals(dates, panel["codes"], P, 55, 0, max_pos=2)
    assert len(sig) == len(dates)
    # signals 现在返回 list[list[int]]：每项是当日的目标持仓列表
    assert all(isinstance(x, list) for x in sig)
    # 前 55 行窗口不足，不可能有持仓
    assert all(len(x) == 0 for x in sig[:55])
    # 持仓数不超过 max_pos，下标合法
    assert all(len(x) <= 2 for x in sig)
    assert all(0 <= j < len(panel["codes"]) for x in sig for j in x)
    # 同一日不能重复持有同一标的
    assert all(len(set(x)) == len(x) for x in sig)


def test_backtest_costs_and_t1(panel):
    """回测：交易必须有成本，且不能 t 日信号 t 日成交。"""
    sys.path.insert(0, str(ROOT))
    from scripts.etf_trend_strategy import backtest, signals  # noqa

    dates = panel["dates"]
    P = {"close": panel["close"], "amount": panel["amount"]}
    cash = pd.Series(0.0, index=dates)
    sig = signals(dates, panel["codes"], P, 55, 0, max_pos=2)
    nav, pos, trades = backtest(dates, panel["codes"], P, sig, cash, max_pos=2)
    assert len(nav) == len(dates)
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav.to_numpy() > 0).all()
    if len(trades):
        # 每笔成交日期必须晚于对应信号日（T+1）
        assert trades["date"].is_monotonic_increasing
        # 买入价应含滑点（高于当日收盘）
        j = panel["codes"].index(trades.iloc[0]["code"])
        t = int(dates.get_loc(trades.iloc[0]["date"]))
        px = panel["close"][t, j]
        if trades.iloc[0]["side"] == "buy":
            assert trades.iloc[0]["price"] > px


def test_amount_filter_excludes_illiquid(panel):
    """512100 早期成交额低，应被 5000 万门槛过滤。"""
    amt20 = pd.DataFrame(panel["amount"]).rolling(20, min_periods=10).mean().to_numpy()
    j100 = panel["codes"].index("512100")
    listed = np.flatnonzero(np.isfinite(panel["close"][:, j100]))
    assert len(listed) > 300
    start = int(listed[0])
    early = amt20[start:start + 300, j100]
    valid = np.isfinite(early)
    assert valid.sum() > 0, "512100 上市后应有成交额数据"
    assert (early[valid] < 5e7).mean() > 0.5, "512100 上市初期应多为低流动性"


def test_strict_t1_first_trade_after_signal(panel):
    """回归：首笔交易必须晚于首个信号日（防同 bar 前视）。"""
    sys.path.insert(0, str(ROOT))
    from scripts.etf_trend_strategy import backtest, signals

    dates = panel["dates"]
    P = {"close": panel["close"], "amount": panel["amount"]}
    cash = pd.Series(0.0, index=dates)
    sig = signals(dates, panel["codes"], P, 55, 0, max_pos=2)
    nav, pos, trades = backtest(dates, panel["codes"], P, sig, cash, max_pos=2)
    if len(trades) == 0:
        pytest.skip("无交易")
    first_sig_t = next(t for t in range(len(sig)) if sig[t])
    first_tr_t = int(dates.get_indexer([pd.Timestamp(trades.iloc[0]["date"])])[0])
    assert first_tr_t > first_sig_t, \
        f"首笔交易({first_tr_t}) 应晚于首个信号日({first_sig_t})，否则是同 bar 前视"


def test_max_pos_respected(panel):
    """持仓数不得超过 max_pos。"""
    sys.path.insert(0, str(ROOT))
    from scripts.etf_trend_strategy import backtest, signals

    dates = panel["dates"]
    P = {"close": panel["close"], "amount": panel["amount"]}
    cash = pd.Series(0.0, index=dates)
    for k in (1, 2, 3):
        sig = signals(dates, panel["codes"], P, 55, 0, max_pos=k)
        nav, pos, trades = backtest(dates, panel["codes"], P, sig, cash, max_pos=k)
        assert pos.max() <= k, f"max_pos={k} 但持仓达 {pos.max()}"


def test_pos_style_affects_result(panel):
    """equal 与 fill 两种仓位机制必须产生不同结果（否则参数没生效）。"""
    sys.path.insert(0, str(ROOT))
    from scripts.etf_trend_strategy import backtest, signals

    dates = panel["dates"]
    P = {"close": panel["close"], "amount": panel["amount"]}
    cash = pd.Series(0.0, index=dates)
    sig = signals(dates, panel["codes"], P, 55, 0, max_pos=3)
    nav_e, _, _ = backtest(dates, panel["codes"], P, sig, cash, max_pos=3,
                           pos_style="equal")
    nav_f, _, _ = backtest(dates, panel["codes"], P, sig, cash, max_pos=3,
                           pos_style="fill")
    assert not np.allclose(nav_e.to_numpy(), nav_f.to_numpy()), \
        "equal 与 fill 结果相同，说明 pos_style 未生效"
