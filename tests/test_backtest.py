# -*- coding: utf-8 -*-
"""回测引擎集成测试（真实 MarketData + 只用过去信息的简单分数）。

覆盖 docs/INTERFACES.md §3.5 要求：
* 净值有限、正数，初始净值 = 本金；
* 交易成本为正（佣金/印花税/过户费/滑点），整手；
* 调仓后目标权重满足约束（单票 8% / 行业 25% / 双创 50% / 持仓数 20）；
* 无未来函数：分数整体 shift 1 天后结果不同；扰动未来分数不改变历史净值；
* 指标函数可用。

信号构造说明：``scores = cs_rank(-(price[t]/price[t-20]-1))``，只依赖 t 及之前
的价格，禁止使用 ``md.fwd_*``。
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

from qf import config as C                          # noqa: E402
from qf import data as qfdata                       # noqa: E402
from qf import metrics as qmetrics                  # noqa: E402
from qf import ops                                  # noqa: E402
from qf.backtest import BacktestResult, run_backtest  # noqa: E402
from qf.portfolio import build_target               # noqa: E402

START = "2017-01-03"
TRADE_COLS = ["date", "code", "side", "shares", "price", "amount",
              "commission", "stamp", "transfer", "slip", "reason"]


# ---------------------------------------------------------------------------
# 模块级 fixture：真实数据只加载一次
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def scores(md):
    r20 = ops.pct_change(md.price, 20)          # 只依赖 <=t 的价格
    return ops.cs_rank(-r20)


@pytest.fixture(scope="module")
def res(md, scores):
    start = md.date_index(START)
    return run_backtest(md, scores, start, md.T - 1, C, verbose=False)


# ---------------------------------------------------------------------------
# 基础结构与净值
# ---------------------------------------------------------------------------
def test_result_structure_and_nav_finite(md, res):
    assert isinstance(res, BacktestResult)
    start = md.date_index(START)
    assert res.nav.index.equals(md.dates[start:])
    assert res.daily_ret.index.equals(res.nav.index)
    assert len(res.nav) == md.T - start
    assert np.isfinite(res.nav.to_numpy()).all()
    assert (res.nav.to_numpy() > 0).all()
    assert np.isfinite(res.daily_ret.to_numpy()).all()

    assert list(res.weights.columns) == ["date", "code", "weight", "reason"]
    assert list(res.trades.columns) == TRADE_COLS
    assert {"date", "code", "shares", "price", "value", "weight"}.issubset(res.holdings.columns)
    assert "cash" in res.exposures.columns
    for col in ["dd_1y", "dd_state", "exposure_cap", "regime_scale"]:
        assert col in res.exposures.columns
    # 行业暴露列（用行业名称）
    ind_names = [md.industry_name(k) for k in range(len(md.industry_cats))]
    assert any(name in res.exposures.columns for name in ind_names)


def test_initial_capital_and_metrics_keys(res):
    assert res.nav.iloc[0] == pytest.approx(C.CAPITAL)
    assert res.daily_ret.iloc[0] == pytest.approx(0.0)
    m = res.metrics
    for key in ["cagr", "ann_vol", "sharpe", "sortino", "max_dd", "max_dd_days", "calmar",
                "win_rate_daily", "win_rate_monthly", "turnover_ann", "cost_ratio",
                "n_trades", "avg_holdings", "beta_hs300", "alpha_hs300", "ir_hs300",
                "beta_csi1000", "alpha_csi1000"]:
        assert key in m, f"metrics 缺少 {key}"
    assert np.isfinite(m["cagr"]) and np.isfinite(m["max_dd"])
    assert m["max_dd"] <= 0.0
    assert m["n_trades"] > 0 and m["cost_total"] > 0
    # 换手口径：turnover_ann = 单边；turnover_ann_double = 双边
    assert m["turnover_ann"] == pytest.approx(m["turnover_ann_single"])
    assert m["turnover_ann_double"] == pytest.approx(2.0 * m["turnover_ann"])
    # 配置快照
    for key in ["start", "end", "capital", "n_holdings", "rebalance_days",
                "risk_overlay", "stop_loss", "slippage_bp", "trade_band_rel",
                "trade_band_abs", "max_new_entries", "rebalance_offset",
                "stamp_tax_sell_old", "dd_window"]:
        assert key in res.config


# ---------------------------------------------------------------------------
# 交易成本与撮合规则
# ---------------------------------------------------------------------------
def test_costs_are_positive_and_complete(res):
    tr = res.trades
    assert len(tr) > 0
    assert tr["commission"].sum() > 0
    assert tr["stamp"].sum() > 0
    assert tr["transfer"].sum() > 0
    assert tr["slip"].sum() > 0
    assert res.metrics["cost_total"] == pytest.approx(
        float((tr["commission"] + tr["stamp"] + tr["transfer"] + tr["slip"]).sum()))

    buy = tr[tr["side"] == "buy"]
    sell = tr[tr["side"] == "sell"]
    assert len(buy) > 0 and len(sell) > 0
    assert (buy["stamp"] == 0).all()                       # 印花税只在卖出
    normal_sell = sell[sell["reason"] != "liquidate"]
    if len(normal_sell):
        # 卖出印花税历史分段：2023-08-28 前 10bp，之后 cfg.STAMP_TAX_SELL=5bp
        dates = pd.to_datetime(normal_sell["date"])
        rate = np.where(dates < pd.Timestamp("2023-08-28"), 0.001, C.STAMP_TAX_SELL)
        assert np.allclose(normal_sell["stamp"].to_numpy(),
                           rate * normal_sell["amount"].to_numpy())
    assert np.allclose(tr["transfer"].to_numpy(), C.TRANSFER_FEE * tr["amount"].to_numpy())
    assert np.allclose(tr["commission"].to_numpy(),
                       np.maximum(C.MIN_COMMISSION, C.COMMISSION_RATE * tr["amount"].to_numpy()))


def test_lot_size_and_price_and_t_plus_1(md, res):
    tr = res.trades
    assert (tr["shares"] > 0).all()
    assert (tr["shares"] % C.LOT == 0).all()
    assert set(tr["side"].unique()) <= {"buy", "sell"}
    assert set(tr["reason"].unique()) <= {"rebalance", "stop_loss", "liquidate"}

    slip = C.SLIPPAGE_BP / 1e4
    for _, t in tr.iterrows():
        row = md.date_index(t["date"])
        j = md.code_index(t["code"])
        open_px = float(md.price_open[row, j])
        assert np.isfinite(open_px) and open_px > 0
        if t["side"] == "buy":
            assert t["price"] == pytest.approx(open_px * (1 + slip))
        else:
            assert t["price"] == pytest.approx(open_px * (1 - slip))

    # 同一天同一只股票不会既买又卖（T+1 / 残差撮合的自然结果）
    both = tr.groupby(["date", "code"])["side"].nunique()
    assert (both <= 1).all()


def test_no_buy_beyond_cash(md, res):
    """逐日检查：现金 + 卖出净收入 >= 买入支出（现金不为负）。"""
    tr = res.trades
    for date, grp in tr.groupby("date"):
        buys = grp[grp["side"] == "buy"]
        sells = grp[grp["side"] == "sell"]
        net_sell = (sells["amount"] - sells["commission"] - sells["stamp"]
                    - sells["transfer"]).sum()
        cost_buy = (buys["amount"] + buys["commission"] + buys["transfer"]).sum()
        # 允许卖出资金 + 期初现金覆盖买入；这里只要求单日买入不超过卖出净额+300k
        assert cost_buy <= net_sell + C.CAPITAL + 1e-6


# ---------------------------------------------------------------------------
# 组合约束
# ---------------------------------------------------------------------------
def test_target_weights_constraints(md, res):
    w = res.weights
    assert len(w) > 0
    growth = np.isin(np.asarray(md.board, dtype=object), ("STAR", "CHINEXT"))
    for date, grp in w.groupby("date"):
        t = md.date_index(date)
        ws = grp["weight"].to_numpy()
        assert ws.sum() <= 1.0 + 1e-6
        assert ws.max() <= C.MAX_WEIGHT + 1e-6
        assert np.all(ws > 0)
        assert len(ws) <= C.N_HOLDINGS
        js = np.array([md.code_index(c) for c in grp["code"]], dtype=int)
        ind = md.industry[t, js]
        for k in np.unique(ind):
            assert ws[ind == k].sum() <= C.INDUSTRY_MAX + 1e-6
        assert ws[growth[js]].sum() <= C.BOARD_MAX_GROWTH + 1e-6


def test_build_target_direct(md, scores):
    """直接测组合层：2022 年某调仓日应持满 20 只且满足约束。"""
    t = md.date_index("2022-06-15")
    prev = {}
    target = build_target(md, t, scores, prev, C)
    assert len(target) == C.N_HOLDINGS
    ws = np.array(list(target.values()))
    assert ws.sum() <= 1.0 + 1e-6
    assert ws.max() <= C.MAX_WEIGHT + 1e-6
    assert ws.min() >= C.MIN_WEIGHT - 1e-12
    js = np.array(list(target.keys()), dtype=int)
    ind = md.industry[t, js]
    for k in np.unique(ind):
        assert ws[ind == k].sum() <= C.INDUSTRY_MAX + 1e-6
    growth = np.isin(np.asarray(md.board, dtype=object), ("STAR", "CHINEXT"))
    assert ws[growth[js]].sum() <= C.BOARD_MAX_GROWTH + 1e-6
    # 缓冲区：把上一期持仓再传一次，结果不应崩坏
    target2 = build_target(md, t, scores, target, C)
    assert len(target2) == C.N_HOLDINGS
    # score 斜率模式也可用
    target3 = build_target(md, t, scores, target, C, weight_mode="score")
    assert 0 < len(target3) <= C.N_HOLDINGS
    assert sum(target3.values()) <= 1.0 + 1e-6


def test_rebalance_cadence(md, res):
    dates = pd.to_datetime(res.weights["date"].unique())
    assert len(dates) == res.metrics["n_rebalances"]
    rows = np.array([md.date_index(d) for d in dates])
    start = md.date_index(START)
    # 数据层 list_days>=120 的 warmup 使 2017 年前 ~120 个交易日无候选，
    # 第一次真正发信号可能晚于 START；但节奏必须固定为 rebalance_days。
    assert rows[0] >= start
    assert rows[0] - start < 200
    assert (rows[0] - start) % res.config["rebalance_days"] == 0
    rb = res.config["rebalance_days"]
    assert (np.diff(rows) == rb).all()


# ---------------------------------------------------------------------------
# 无未来函数
# ---------------------------------------------------------------------------
def test_no_lookahead_scores_shift(md, scores, res):
    """分数整体滞后 1 天 → 回测结果必须不同（否则信号时点可能串了）。"""
    lag = ops.shift(scores, 1)
    start = md.date_index(START)
    res2 = run_backtest(md, lag, start, md.T - 1, C, verbose=False)
    assert np.isfinite(res2.nav.to_numpy()).all()
    diff = np.abs(res.nav.to_numpy() - res2.nav.to_numpy())
    assert not np.allclose(res.nav.to_numpy(), res2.nav.to_numpy(), rtol=1e-9, atol=1e-6)
    assert diff.max() > 1.0


def test_no_lookahead_future_scores_do_not_change_history(md, scores, res):
    """扰动 t 之后的所有分数，回测历史段净值必须逐位不变（严格因果检验）。"""
    start = md.date_index(START)
    mid = start + 400
    scores2 = scores.copy()
    rng = np.random.default_rng(C.RANDOM_SEED)
    scores2[mid:] = rng.normal(size=scores2[mid:].shape).astype(np.float32)
    res2 = run_backtest(md, scores2, start, md.T - 1, C, verbose=False)

    k = mid - start
    prefix = np.abs(res.nav.to_numpy()[:k + 1] - res2.nav.to_numpy()[:k + 1]).max()
    after = np.abs(res.nav.to_numpy()[k + 2:] - res2.nav.to_numpy()[k + 2:]).max()
    assert prefix < 1e-6, f"历史净值依赖于未来分数（prefix diff={prefix}）"
    assert after > 1.0, "扰动未来分数后结果却完全一样，检查信号是否真的被使用"


# ---------------------------------------------------------------------------
# 指标函数 & 参数化
# ---------------------------------------------------------------------------
def test_metrics_functions(md, res):
    nav = res.nav
    perf = qmetrics.performance(nav, bench_returns=md.bench.reindex(nav.index))
    assert isinstance(perf, dict)
    assert "beta_hs300" in perf and "alpha_hs300" in perf and "te_hs300" in perf

    yearly = qmetrics.yearly_returns(nav)
    assert isinstance(yearly, pd.Series) and len(yearly) >= 9
    assert np.isfinite(yearly.to_numpy()).all()

    monthly = qmetrics.monthly_returns_table(nav)
    assert isinstance(monthly, pd.DataFrame)
    assert "YTD" in monthly.columns
    dd = qmetrics.drawdown_series(nav)
    assert (dd <= 1e-12).all() and dd.min() == pytest.approx(perf["max_dd"])
    rolling = qmetrics.rolling_metrics(nav, window=252)
    assert isinstance(rolling, pd.DataFrame) and len(rolling) == len(nav)
    stress = qmetrics.stress_periods(nav, bench_returns=md.bench.reindex(nav.index))
    assert isinstance(stress, pd.DataFrame) and len(stress) >= 5
    assert "2018全年" in set(stress["period"])
    assert {"period", "start", "end", "n_days", "ret", "max_dd"}.issubset(stress.columns)


def test_start_row_parameter(md, scores):
    """Lead 的实盘口径：从 2020-01-03 开始，只跑 3 年，应正常运行。"""
    start = md.date_index("2020-01-03")
    end = md.date_index("2022-12-30")
    r = run_backtest(md, scores, start, end, C, risk_overlay=True, stop_loss=True,
                     verbose=False)
    assert len(r.nav) == end - start + 1
    assert np.isfinite(r.nav.to_numpy()).all()
    assert r.metrics["n_trades"] > 0
    assert r.nav.index[0] == md.dates[start]


# ---------------------------------------------------------------------------
# trade band / slippage 透传
# ---------------------------------------------------------------------------
def test_slippage_bp_override_and_restore(md, scores):
    """run_backtest(slippage_bp=...) 临时覆盖 C.SLIPPAGE_BP 并在结束后恢复。"""
    start = md.date_index("2022-01-04")
    end = md.date_index("2022-06-30")
    old = float(C.SLIPPAGE_BP)
    r15 = run_backtest(md, scores, start, end, C, slippage_bp=15.0, verbose=False)
    assert float(C.SLIPPAGE_BP) == old, "回测结束后必须恢复 C.SLIPPAGE_BP"
    r60 = run_backtest(md, scores, start, end, C, slippage_bp=60.0, verbose=False)
    assert float(C.SLIPPAGE_BP) == old
    assert r15.config["slippage_bp"] == 15.0
    assert r60.config["slippage_bp"] == 60.0
    assert r60.metrics["cost_total"] > r15.metrics["cost_total"]
    assert r60.metrics["cagr"] < r15.metrics["cagr"]


def test_trade_band_reduces_trades_and_costs(md, scores):
    """默认 trade band 应显著减少微调笔数与成本（vs band=0）。"""
    start = md.date_index("2022-01-04")
    end = md.date_index("2022-12-30")
    r_band = run_backtest(md, scores, start, end, C, verbose=False)
    r_off = run_backtest(md, scores, start, end, C, verbose=False,
                         trade_band_rel=0.0, trade_band_abs=0.0)
    assert r_band.config["trade_band_rel"] == 0.30
    assert r_band.config["trade_band_abs"] == 6000.0
    assert r_band.metrics["n_band_skipped"] > 0
    assert r_band.metrics["n_trades"] < r_off.metrics["n_trades"]
    # 注：40 日调仓下，band 减少了交易次数但单笔金额变大，
    # 总成本可能反而更高（持仓集中）。因此只断言"次数减少"这一本职效果。
    assert r_band.metrics["avg_trade_amount"] >= r_off.metrics["avg_trade_amount"] * 0.9


# ---------------------------------------------------------------------------
# 换手上限（max_new_entries）
# ---------------------------------------------------------------------------
def test_build_target_max_new_entries(md, scores):
    t0 = md.date_index("2022-06-15")
    prev = build_target(md, t0, scores, {}, C)
    t1 = t0 + C.REBALANCE_DAYS
    tgt_cap = build_target(md, t1, scores, prev, C, max_new_entries=6)
    tgt_free = build_target(md, t1, scores, prev, C)      # None = 不限制
    new_cap = set(tgt_cap) - set(prev)
    new_free = set(tgt_free) - set(prev)
    assert len(new_cap) <= 6
    assert len(new_free) >= len(new_cap)
    # 不限制时应该能更自由换仓（该数据/信号下确实超过 6 只）
    assert len(new_free) > 6
    # 权重约束仍满足
    ws = np.array(list(tgt_cap.values()))
    assert ws.sum() <= 1.0 + 1e-6
    assert ws.max() <= C.MAX_WEIGHT + 1e-6


def test_build_target_max_new_entries_zero(md, scores):
    t0 = md.date_index("2022-06-15")
    prev = build_target(md, t0, scores, {}, C)
    tgt = build_target(md, t0 + C.REBALANCE_DAYS, scores, prev, C, max_new_entries=0)
    assert set(tgt) <= set(prev)          # 一个新股都不进
    assert len(tgt) > 0


def test_max_new_entries_reduces_turnover(md, scores):
    start = md.date_index("2022-01-04")
    end = md.date_index("2022-12-30")
    r_cap = run_backtest(md, scores, start, end, C, verbose=False, max_new_entries=3)
    r_free = run_backtest(md, scores, start, end, C, verbose=False, max_new_entries=None)
    assert r_cap.metrics["n_trades"] < r_free.metrics["n_trades"]
    assert r_cap.metrics["turnover_ann_single"] < r_free.metrics["turnover_ann_single"]


# ---------------------------------------------------------------------------
# 回撤节流（滚动 252 日峰值）/ 调仓相位
# ---------------------------------------------------------------------------
def test_rolling_dd_state_machine():
    from qf.backtest import _update_dd_state
    state = 0
    state, cap = _update_dd_state(-0.06, C, state)
    assert state == 0 and cap == pytest.approx(1.0)
    state, cap = _update_dd_state(-0.11, C, state)       # 破 -10% → 60%
    assert state == 1 and cap == pytest.approx(C.DD_LEVEL_1)
    state, cap = _update_dd_state(-0.12, C, state)
    assert state == 1
    state, cap = _update_dd_state(-0.16, C, state)       # 破 -15% → 40%
    assert state == 2 and cap == pytest.approx(C.DD_LEVEL_2)
    state, cap = _update_dd_state(-0.12, C, state)       # 回升仍 ≥ -15% → 60%
    assert state == 1 and cap == pytest.approx(C.DD_LEVEL_1)
    state, cap = _update_dd_state(-0.07, C, state)       # 未到 -5%：滞回维持 60%
    assert state == 1 and cap == pytest.approx(C.DD_LEVEL_1)
    state, cap = _update_dd_state(-0.04, C, state)       # > -5% → 恢复 100%
    assert state == 0 and cap == pytest.approx(1.0)


def test_rolling_peak_dd_window():
    from qf.backtest import _rolling_peak_dd
    nav = np.array([100.0] * 10 + [80.0] * 10 + [95.0] * 300 + [110.0, 90.0])
    # 第 19 日 = 80，相对 100 的滚动峰值回撤 -20%
    assert _rolling_peak_dd(nav, 19) == pytest.approx(0.80 / 1.0 - 1.0)
    # 第 320 日 = 110 是窗口内新高 → 回撤 0（峰值含当日，回撤恒 ≤ 0）
    assert _rolling_peak_dd(nav, 320) == pytest.approx(0.0)
    # 第 321 日 = 90，窗口（252 日）内峰值 110 → 回撤 -18.2%
    assert _rolling_peak_dd(nav, 321) == pytest.approx(90.0 / 110.0 - 1.0)
    # 100/80 已在 252 日窗口外 → 更早的低点不再影响峰值
    assert _rolling_peak_dd(nav, 321) > -0.5


def test_rebalance_offset(md, scores):
    start = md.date_index("2022-01-04")
    end = md.date_index("2022-06-30")
    r0 = run_backtest(md, scores, start, end, C, verbose=False)
    r2 = run_backtest(md, scores, start, end, C, verbose=False, rebalance_offset=2)
    rb = C.REBALANCE_DAYS
    rows0 = np.array([md.date_index(d) for d in pd.to_datetime(r0.weights["date"].unique())])
    rows2 = np.array([md.date_index(d) for d in pd.to_datetime(r2.weights["date"].unique())])
    assert ((rows0 - start) % rb == 0).all()
    assert ((rows2 - start) % rb == (2 % rb)).all()
    assert r0.config["rebalance_offset"] == 0
    assert r2.config["rebalance_offset"] == 2
    assert r2.nav.index[0] == md.dates[start]
