# -*- coding: utf-8 -*-
"""执行引擎单元测试（构造小型面板，不依赖真实数据）。

覆盖 docs/INTERFACES.md §3.5 要求：整手取整、最低佣金、卖出印花税、
涨停不能买、跌停不能卖、停牌不可成交、现金不为负（含 1% 现金缓冲）、T+1。
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C                      # noqa: E402
from qf.execution import ExecutionEngine, Position, Trade  # noqa: E402


# ---------------------------------------------------------------------------
# 构造小型面板
# ---------------------------------------------------------------------------
def make_md(price=10.0, price_open=None, T: int = 4, N: int = 1,
            buyable=None, sellable=None) -> SimpleNamespace:
    price = np.broadcast_to(np.asarray(price, dtype=float), (T, N)).astype(np.float32).copy()
    if price_open is None:
        opens = price.copy()
    else:
        opens = np.broadcast_to(np.asarray(price_open, dtype=float),
                                (T, N)).astype(np.float32).copy()
    md = SimpleNamespace()
    md.T, md.N = T, N
    md.price = price
    md.price_open = opens
    md.alive = np.ones((T, N), dtype=bool)
    md.buyable = np.ones((T, N), dtype=bool) if buyable is None else np.asarray(buyable, bool)
    md.sellable = np.ones((T, N), dtype=bool) if sellable is None else np.asarray(sellable, bool)
    md.codes = np.array([f"{i:06d}" for i in range(N)], dtype=object)
    md.dates = pd.date_range("2020-01-02", periods=T, freq="B")
    return md


def fill_buy(open_px: float) -> float:
    return open_px * (1.0 + C.SLIPPAGE_BP / 1e4)


def fill_sell(open_px: float) -> float:
    return open_px * (1.0 - C.SLIPPAGE_BP / 1e4)


# ---------------------------------------------------------------------------
# 整手 / 佣金 / 费用
# ---------------------------------------------------------------------------
def test_lot_rounding_and_position_field():
    px = 7.77
    md = make_md(price=px, price_open=px)
    eng = ExecutionEngine(md, C)
    tr = eng.execute(0, {0: 0.05}, target_nav=300000.0)

    assert len(tr) == 1
    t = tr[0]
    # floor(300000*0.05 / 7.77 / 100) * 100 = 1900
    assert t.shares == 1900
    assert t.shares % C.LOT == 0
    assert t.side == "buy"
    assert t.price == pytest.approx(fill_buy(px))
    pos = eng.positions[0]
    assert isinstance(pos, Position)
    assert pos.code == 0 and pos.shares == 1900
    assert pos.entry_price == pytest.approx(t.price)
    assert pos.entry_date == 0 and pos.buy_date == 0


def test_lot_rounding_exact_multiple():
    md = make_md(price=5.0, price_open=5.0)
    eng = ExecutionEngine(md, C)
    tr = eng.execute(0, {0: 0.02}, target_nav=300000.0)   # 6000 / 5 = 1200 股
    assert tr[0].shares == 1200


def test_min_commission():
    md = make_md(price=10.0)
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    tr = eng.execute(0, {0: 0.01}, target_nav=100000.0)   # 目标 1000 元 → 100 股
    assert len(tr) == 1 and tr[0].shares == 100
    amount = tr[0].amount
    assert amount == pytest.approx(100 * fill_buy(10.0))
    # 名义佣金 2.5bp×1000=0.25 元 < 最低 5 元 → 收 5 元
    assert C.COMMISSION_RATE * amount < C.MIN_COMMISSION
    assert tr[0].commission == pytest.approx(C.MIN_COMMISSION)


def test_stamp_tax_only_on_sell():
    md = make_md(price=10.0)                  # 2020 年（2023-08-28 前 → 10bp）
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    buy = eng.execute(0, {0: 0.5}, target_nav=100000.0)    # 5000 股
    assert buy[0].side == "buy" and buy[0].stamp == 0.0
    sell = eng.execute(1, {})
    assert len(sell) == 1 and sell[0].side == "sell"
    assert sell[0].shares == 5000
    old_rate = 0.001                       # 2023-08-28 之前卖出印花税 10bp
    assert eng.stamp_rate(1) == pytest.approx(old_rate)
    assert sell[0].stamp == pytest.approx(old_rate * sell[0].amount)
    assert sell[0].transfer == pytest.approx(C.TRANSFER_FEE * sell[0].amount)
    assert sell[0].commission == pytest.approx(
        max(C.MIN_COMMISSION, C.COMMISSION_RATE * sell[0].amount))


def test_stamp_tax_historical_cut():
    """卖出印花税历史分段：< 2023-08-28 → 10bp；>= 2023-08-28 → 5bp。"""
    md_pre = make_md(price=10.0, T=3)
    eng_pre = ExecutionEngine(md_pre, C)
    assert eng_pre.stamp_rate(0) == pytest.approx(0.001)
    eng_pre.cash = 100000.0
    eng_pre.execute(0, {0: 0.5}, target_nav=100000.0)
    sell_pre = eng_pre.execute(1, {})[0]
    assert sell_pre.stamp == pytest.approx(0.001 * sell_pre.amount)

    md_post = make_md(price=10.0, T=3)
    md_post.dates = pd.date_range("2024-01-02", periods=3, freq="B")
    eng_post = ExecutionEngine(md_post, C)
    assert eng_post.stamp_rate(1) == pytest.approx(C.STAMP_TAX_SELL)
    eng_post.cash = 100000.0
    eng_post.execute(0, {0: 0.5}, target_nav=100000.0)
    sell_post = eng_post.execute(1, {})[0]
    assert sell_post.stamp == pytest.approx(C.STAMP_TAX_SELL * sell_post.amount)


def test_slippage_direction_and_cost():
    md = make_md(price=10.0)
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    buy = eng.execute(0, {0: 0.2}, target_nav=100000.0)[0]
    assert buy.price == pytest.approx(10.0 * (1 + C.SLIPPAGE_BP / 1e4))
    assert buy.slip == pytest.approx(
        buy.shares * (buy.price - 10.0))
    sell = eng.execute(1, {})[0]
    assert sell.price == pytest.approx(10.0 * (1 - C.SLIPPAGE_BP / 1e4))
    assert sell.slip == pytest.approx(sell.shares * (10.0 - sell.price))
    assert isinstance(buy, Trade) and buy.cost > 0


# ---------------------------------------------------------------------------
# 可交易性：涨停 / 跌停 / 停牌
# ---------------------------------------------------------------------------
def test_limit_up_cannot_buy_then_retry():
    md = make_md(price=10.0, T=3)
    md.buyable[0, 0] = False          # 开盘涨停
    eng = ExecutionEngine(md, C)
    cash0 = eng.cash
    tr = eng.execute(0, {0: 0.5}, target_nav=100000.0)
    assert tr == [] and eng.cash == cash0 and 0 not in eng.positions

    # 次日打开 → 残差继续尝试并成交
    tr2 = eng.execute(1, {0: 0.5}, target_nav=100000.0)
    assert len(tr2) == 1 and 0 in eng.positions
    assert eng.positions[0].shares == 5000


def test_limit_down_cannot_sell_then_retry():
    md = make_md(price=10.0, T=4)
    md.sellable[1, 0] = False         # 次日一字跌停
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    eng.execute(0, {0: 0.5}, target_nav=100000.0)
    assert 0 in eng.positions

    tr = eng.execute(1, {})           # 跌停卖不出
    assert tr == [] and 0 in eng.positions
    tr2 = eng.execute(2, {})          # 次日可卖
    assert len(tr2) == 1 and tr2[0].side == "sell" and 0 not in eng.positions


def test_suspended_cannot_trade():
    md = make_md(price=10.0, T=4)
    md.price_open[0, 0] = np.nan
    md.buyable[0, 0] = False          # 停牌
    eng = ExecutionEngine(md, C)
    cash0 = eng.cash
    assert eng.execute(0, {0: 0.5}, target_nav=100000.0) == []
    assert eng.cash == cash0

    # 持仓遇到停牌：卖不出
    tr = eng.execute(1, {0: 0.5}, target_nav=100000.0)
    assert len(tr) == 1
    md.price_open[2, 0] = np.nan
    md.sellable[2, 0] = False
    assert eng.execute(2, {}) == []
    assert 0 in eng.positions


# ---------------------------------------------------------------------------
# 现金约束 / T+1
# ---------------------------------------------------------------------------
def test_cash_never_negative_and_reserve():
    N = 10
    md = make_md(price=10.0, T=3, N=N)
    eng = ExecutionEngine(md, C)
    eng.cash = 10000.0
    target = {j: 0.1 for j in range(N)}          # 目标 100% 满仓
    tr = eng.execute(0, target, target_nav=10000.0)
    buys = [t for t in tr if t.side == "buy"]
    assert len(buys) == 9                        # 留 1% 缓冲 + 整手后少买 1 只
    assert eng.cash >= 0.0
    assert eng.cash >= 0.01 * 10000.0 - 1e-6     # 1% 现金缓冲
    assert eng.cash < 10.0 * C.LOT * (1 + C.SLIPPAGE_BP / 1e4) + C.MIN_COMMISSION
    spent = sum(t.amount + t.commission + t.transfer for t in buys)
    assert spent <= 10000.0 + 1e-6


def test_t_plus_1_same_day_no_sell():
    md = make_md(price=10.0, T=4)
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    eng.execute(0, {0: 0.5}, target_nav=100000.0)
    assert 0 in eng.positions

    tr = eng.execute(0, {})           # 当天买当天不能卖
    assert tr == [] and 0 in eng.positions
    tr2 = eng.execute(1, {})          # 次日可以卖
    assert len(tr2) == 1 and tr2[0].side == "sell" and 0 not in eng.positions


def test_cash_reconciliation_exact():
    md = make_md(price=10.0, T=3)
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    buy = eng.execute(0, {0: 0.2}, target_nav=100000.0)[0]
    expected = (100000.0
                - buy.amount - buy.commission - buy.transfer)
    assert eng.cash == pytest.approx(expected, rel=1e-12)

    sell = eng.execute(1, {})[0]
    expected += (sell.amount - sell.commission - sell.stamp - sell.transfer)
    assert eng.cash == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 目标股数冻结 / 止损
# ---------------------------------------------------------------------------
def test_frozen_target_shares_no_churn_on_price_move():
    md = make_md(price=10.0, T=3)
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    weights = {0: 0.5}
    plan = eng.plan_target_shares(0, weights, target_nav=100000.0)
    assert plan[0] == 5000
    tr = eng.execute(0, weights, target_nav=100000.0, target_shares=plan)
    assert [t.shares for t in tr] == [5000]

    # 次日价格涨到 11：目标股数冻结，不应产生任何"再平衡"交易
    md.price[1:, 0] = 11.0
    md.price_open[1:, 0] = 11.0
    tr2 = eng.execute(1, weights, target_nav=100000.0, target_shares=plan)
    assert tr2 == []
    assert eng.positions[0].shares == 5000


def test_stop_loss_detection():
    md = make_md(price=10.0, T=4)
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    eng.execute(0, {0: 0.5}, target_nav=100000.0)
    entry = eng.positions[0].entry_price
    # 阈值取自 config（曾为 -0.15、现为 -0.30），测试跟随参数而非硬编码
    md.price[1, 0] = entry * (1.0 + C.STOP_LOSS - 0.10)   # 比阈值再跌 10pp → 必触发
    assert eng.stop_loss_sells(1) == [0]
    md.price[2, 0] = entry * 0.995      # -0.5% → 不触发
    assert eng.stop_loss_sells(2) == []


def test_stop_loss_forced_sell_and_limit_down_retry():
    md = make_md(price=10.0, T=4)
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    eng.execute(0, {0: 0.5}, target_nav=100000.0)
    entry = eng.positions[0].entry_price
    md.price[1, 0] = entry * (1.0 + C.STOP_LOSS - 0.10)   # 跟随 config 的止损阈值
    stops = eng.stop_loss_sells(1)
    assert stops == [0]
    md.sellable[2, 0] = False           # 触发次日一字跌停：止损单也卖不出
    assert eng.execute(2, {0: 0.0}, reasons={0: "stop_loss"}) == []
    md.sellable[3, 0] = True
    tr = eng.execute(3, {0: 0.0}, reasons={0: "stop_loss"})
    assert len(tr) == 1 and tr[0].side == "sell"
    assert tr[0].reason == "stop_loss"
    assert 0 not in eng.positions


def test_nav_and_current_weights():
    md = make_md(price=10.0, T=3)
    eng = ExecutionEngine(md, C)
    eng.cash = 100000.0
    eng.execute(0, {0: 0.5}, target_nav=100000.0)
    nav = eng.nav(0)
    assert nav == pytest.approx(eng.cash + eng.positions[0].shares * 10.0)
    w = eng.current_weights(0)
    assert set(w) == {0}
    assert w[0] == pytest.approx(eng.positions[0].shares * 10.0 / nav)


# ---------------------------------------------------------------------------
# 交易带宽（trade band，防微调）
# ---------------------------------------------------------------------------
def test_trade_band_suppresses_small_rebalance():
    md = make_md(price=10.0, T=4)
    eng = ExecutionEngine(md, C, trade_band_rel=0.20, trade_band_abs=3000.0)
    eng.cash = 100000.0
    plan0 = eng.plan_target_shares(0, {0: 0.2}, 100000.0)
    eng.execute(0, {0: 0.2}, target_nav=100000.0, target_shares=plan0)
    assert eng.positions[0].shares == 2000

    # 价格涨到 10.05 → 新目标股数 1900，市值差 ~1005 元 < band(3000) → 不交易
    md.price[1:, 0] = 10.05
    md.price_open[1:, 0] = 10.05
    plan1 = eng.plan_target_shares(1, {0: 0.2}, 100000.0)
    assert plan1[0] == 1900
    tr = eng.execute(1, {0: 0.2}, target_nav=100000.0, target_shares=plan1)
    assert tr == [] and eng.positions[0].shares == 2000
    assert eng.n_band_skipped == 1

    # 目标权重降到 10% → 市值差远超 band → 正常调仓
    plan2 = eng.plan_target_shares(2, {0: 0.1}, 100000.0)
    tr2 = eng.execute(2, {0: 0.1}, target_nav=100000.0, target_shares=plan2)
    assert len(tr2) == 1 and tr2[0].side == "sell"
    assert tr2[0].shares == 2000 - plan2[0]


def test_trade_band_zero_still_trades():
    """band 参数置 0 时退化为无带宽（小偏差也调仓）。"""
    md = make_md(price=10.0, T=4)
    eng = ExecutionEngine(md, C, trade_band_rel=0.0, trade_band_abs=0.0)
    eng.cash = 100000.0
    plan0 = eng.plan_target_shares(0, {0: 0.2}, 100000.0)
    eng.execute(0, {0: 0.2}, target_nav=100000.0, target_shares=plan0)
    md.price[1:, 0] = 10.05
    md.price_open[1:, 0] = 10.05
    plan1 = eng.plan_target_shares(1, {0: 0.2}, 100000.0)
    tr = eng.execute(1, {0: 0.2}, target_nav=100000.0, target_shares=plan1)
    assert len(tr) == 1 and tr[0].shares == plan0[0] - plan1[0]


def test_trade_band_does_not_block_new_position_or_exit():
    md = make_md(price=10.0, T=4)
    eng = ExecutionEngine(md, C, trade_band_rel=0.20, trade_band_abs=3000.0)
    eng.cash = 100000.0
    # 新建仓：目标市值 2000 元 < band_abs，也必须买入
    tr = eng.execute(0, {0: 0.02}, target_nav=100000.0)
    assert len(tr) == 1 and tr[0].side == "buy" and 0 in eng.positions
    # 清仓：目标权重 0（不在目标里）不受 band 限制
    tr2 = eng.execute(1, {})
    assert len(tr2) == 1 and tr2[0].side == "sell" and 0 not in eng.positions
