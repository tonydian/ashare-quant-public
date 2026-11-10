# -*- coding: utf-8 -*-
"""A 股执行模拟（T+1 开盘成交、整手、费用、涨跌停/停牌约束）。

设计口径（与 docs/INTERFACES.md §3.1 一致）：

* **时间**：t 日收盘生成目标权重 → t+1 日**开盘**成交（本模块只负责"在 t 日开盘
  按目标执行"，不自己决定信号时点）。同一个 execute() 一天最多调用一次；
  T+1 通过 ``Position.buy_date`` 约束：当天买入的持仓当天不能卖。
* **价格**：全部用 ``md.price / md.price_open``（常数缩放后复权价，≈实际价），
  这正是整手取整、最小佣金、止损判断需要的口径；因子计算才用后复权 close/open。
* **可交易性**：直接使用数据层算好的 ``md.buyable[t] / md.sellable[t]``
  （已含选股域、涨跌停、停牌、有无开盘价）。这里只额外兜底检查
  ``price_open`` 有限且 > 0。
* **费用**：佣金 ``max(MIN_COMMISSION, COMMISSION_RATE×成交额)``（双边）、
  卖出印花税**历史分段**（成交日 < 2023-08-28 为 10bp，之后为
  ``STAMP_TAX_SELL``=5bp）、过户费 ``TRANSFER_FEE×成交额``（双边）、
  滑点 ``SLIPPAGE_BP``（买价上浮、卖价下调，成交价即含滑点）。
* **整手**：买入股数向下取整到 100 股；卖出股数 = 现有股数 - 目标股数（两者都是
  整手，所以差值也是整手）。
* **目标权重 → 目标股数**：``目标市值 = weight × ref_nav``，``目标股数 =
  floor(目标市值 / 开盘价 / 100) × 100``。ref_nav 由调用方给（通常＝信号日 t 收盘
  NAV）；不给则用执行日开盘市值。**目标股数只算一次并冻结**（``plan_target_shares``），
  后续每天只是"重试残差"，不会因为股价波动每天重新算目标而制造无谓换手。
* **现金**：先卖后买（卖出所得当天可用于买入），买入后现金不得为负，并保留
  1% NAV 的现金缓冲；买不起时缩量到最大可负担整手数，剩余残差留到下一日重试。
* **交易带宽（trade band，防微调）**：对"已持有且目标仍持有"的标的，若
  ``|目标市值 - 当前市值| < max(trade_band_rel×当前市值, trade_band_abs)``
  则不交易（默认 rel=20%、abs=3000 元），避免为维持等权产生大量 1 手级别小单
  （最低佣金 5 元会让小单实际费率升到 15bp+）。**新建仓、清仓、止损单不受
  带宽限制**。带宽判断用执行日开盘价计算的市值。
* **退市**：若持仓股票在 t 日已不在存续期（alive=False），按
  "最后一次有效价 × (1 - delist_haircut)" 强制清算（reason='liquidate'，
  缺省 haircut=0.60，见踩坑清单 §2.4），避免净值因价格 NaN 而失真。

对外主接口::

    eng = ExecutionEngine(md)
    plan = eng.plan_target_shares(t, target_weights, target_nav)   # 冻结目标股数
    trades = eng.execute(t, target_weights, target_shares=plan)    # t 日开盘执行
    nav = eng.nav(t)                                                # t 日收盘净值
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

__all__ = ["Position", "Trade", "ExecutionEngine"]

# 卖出印花税历史分段：2023-08-28 起由 10bp 降为 5bp（cfg.STAMP_TAX_SELL）。
# 2023-08-28 之前的历史成交按 10bp 计提，回测更诚实（踩坑清单 §2.2）。
STAMP_TAX_SELL_BEFORE_CUT = 0.001
STAMP_TAX_CUT_DATE = "2023-08-28"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Position:
    """持仓（股数按缩放后复权价空间，整手）。"""

    code: int          # md.codes 的下标 j
    shares: int        # 股数（整手）
    entry_price: float  # 买入成交价（md.price_open 口径，含滑点）
    entry_date: int    # 首次建仓的交易日行号
    buy_date: int      # 最近一次买入的交易日行号（T+1 判断）


@dataclass
class Trade:
    """一笔成交（买入 amount>0 记为流出，卖出为流入，见 ExecutionEngine.execute）。"""

    date: int
    code: int
    side: str          # 'buy' / 'sell'
    shares: int
    price: float       # 成交价（含滑点）
    amount: float      # 成交额 = shares × price
    commission: float  # 佣金（元）
    stamp: float       # 印花税（元，仅卖出）
    transfer: float    # 过户费（元）
    slip: float        # 滑点成本（元）= |price - price_open| × shares
    reason: str = "rebalance"  # 'rebalance' / 'stop_loss' / 'liquidate'

    @property
    def cost(self) -> float:
        return self.commission + self.stamp + self.transfer + self.slip


# ---------------------------------------------------------------------------
# 执行引擎
# ---------------------------------------------------------------------------
class ExecutionEngine:
    """单账户、单标的池的 A 股执行模拟器（现金 + 持仓字典）。"""

    def __init__(self, md, cfg=C, delist_haircut: float = 0.60,
                 trade_band_rel: float = 0.20, trade_band_abs: float = 3000.0,
                 stamp_tax_old: float = STAMP_TAX_SELL_BEFORE_CUT,
                 stamp_cut_date: str = STAMP_TAX_CUT_DATE):
        self.md = md
        self.cfg = cfg
        self.cash: float = float(cfg.CAPITAL)
        self.positions: dict[int, Position] = {}
        # 退市计提比例：持有到退市时按"最后有效价 × (1 - haircut)"清算。
        # 来源：踩坑清单 §2.4（持有到退市日的股票不能按停牌价计价，本系统按 -60% 计提）。
        self.delist_haircut = float(delist_haircut)
        # 交易带宽（trade band，防微调）：对**已持有且目标仍持有**的标的，
        # 若 |目标市值 - 当前市值| < max(rel×当前市值, abs) 则不交易。
        # 作用：避免为维持等权产生大量 1 手级别小单（最低佣金 5 元 → 实际费率 15bp+）。
        self.trade_band_rel = float(trade_band_rel)
        self.trade_band_abs = float(trade_band_abs)
        # 卖出印花税历史分段（2023-08-28 前 10bp、之后 cfg.STAMP_TAX_SELL=5bp）
        self.stamp_tax_old = float(stamp_tax_old)
        self.stamp_cut_date = pd.Timestamp(stamp_cut_date)
        # 最近一次有效收盘价（退市清算/估值兜底用）
        self._mark: dict[int, float] = {}
        self._mark_date: dict[int, int] = {}
        self.n_settled_delisted: int = 0
        self.n_band_skipped: int = 0

    # ------------------------------------------------------------------ 费用参数
    def stamp_rate(self, t: int) -> float:
        """t 日卖出印花税率：< 2023-08-28 为 10bp，之后为 5bp（cfg.STAMP_TAX_SELL）。"""
        dates = getattr(self.md, "dates", None)
        if dates is not None and 0 <= t < len(dates):
            if pd.Timestamp(dates[t]) < self.stamp_cut_date:
                return self.stamp_tax_old
        return float(self.cfg.STAMP_TAX_SELL)

    # ------------------------------------------------------------------ 内部工具
    def _open_px(self, t: int, j: int) -> float:
        """t 日开盘价（**后复权口径**，与 md.price 同一价格空间，用于持仓估值/股数换算）。"""
        if t < 0 or t >= self.md.T:
            return np.nan
        p = float(self.md.price_open[t, j])
        if np.isfinite(p) and p > 0:
            return p
        return np.nan


    def _close_px(self, t: int, j: int) -> float:
        """t 日收盘价；停牌/退市用最近有效价兜底（用于估值，不做成交）。"""
        if t < 0 or t >= self.md.T:
            return self._mark.get(j, np.nan)
        p = float(self.md.price[t, j])
        if np.isfinite(p) and p > 0:
            self._mark[j] = p
            self._mark_date[j] = t
            return p
        return self._mark.get(j, np.nan)

    def _mask(self, name: str) -> np.ndarray | None:
        m = getattr(self.md, name, None)
        return m if m is not None else None

    def _can_buy(self, t: int, j: int) -> bool:
        m = self._mask("buyable")
        if m is not None and not bool(m[t, j]):
            return False
        return np.isfinite(self._open_px(t, j))

    def _can_sell(self, t: int, j: int) -> bool:
        m = self._mask("sellable")
        if m is not None and not bool(m[t, j]):
            return False
        return np.isfinite(self._open_px(t, j))

    def _alive(self, t: int, j: int) -> bool:
        m = self._mask("alive")
        if m is None:
            return True
        return bool(m[t, j])

    # ------------------------------------------------------------------ 估值
    def nav(self, t: int) -> float:
        """t 日收盘净值 = 现金 + Σ 股数 × md.price[t]（缺失价用最近有效价兜底）。"""
        v = self.cash
        for j, pos in self.positions.items():
            p = self._close_px(t, j)
            if not np.isfinite(p):
                p = pos.entry_price
            v += pos.shares * p
        return float(v)

    def nav_open(self, t: int) -> float:
        """t 日开盘净值（按开盘价估值；开盘价缺失用收盘价/最近价）。"""
        v = self.cash
        for j, pos in self.positions.items():
            p = self._open_px(t, j)
            if not np.isfinite(p):
                p = self._close_px(t, j)
            if not np.isfinite(p):
                p = pos.entry_price
            v += pos.shares * p
        return float(v)

    def holdings_value(self, t: int) -> float:
        return float(self.nav(t) - self.cash)

    def current_weights(self, t: int) -> dict[int, float]:
        """t 日收盘的持仓权重 {j: weight}（含 0 权重不返回）。"""
        nav = self.nav(t)
        out: dict[int, float] = {}
        if nav <= 0:
            return out
        for j, pos in self.positions.items():
            p = self._close_px(t, j)
            if not np.isfinite(p):
                p = pos.entry_price
            w = pos.shares * p / nav
            if w > 0:
                out[j] = float(w)
        return out

    # ------------------------------------------------------------------ 目标股数
    def plan_target_shares(self, t: int, target_weights: dict[int, float],
                           target_nav: float | None = None) -> dict[int, int | None]:
        """把目标权重换算成目标股数（整手向下取整），供后续执行冻结使用。

        * ``target_nav``：目标权重对应的参考净值（通常＝信号日 t-1 收盘 NAV）。
          缺省用执行日开盘净值。
        * 返回值 ``None`` 表示当日开盘价不可用、暂无法定价（如停牌），
          调用方应在后续交易日重试补算。
        """
        ref_nav = float(target_nav) if target_nav is not None else self.nav_open(t)
        lot = int(self.cfg.LOT)
        out: dict[int, int | None] = {}
        for j, w in target_weights.items():
            w = float(w)
            if not np.isfinite(w) or w <= 0:
                out[int(j)] = 0
                continue
            px = self._open_px(t, int(j))
            if not np.isfinite(px):
                out[int(j)] = None
                continue
            shares = int(np.floor(w * ref_nav / px / lot) * lot)
            out[int(j)] = max(0, shares)
        return out

    # ------------------------------------------------------------------ 费用工具
    def _buy_cost(self, shares: int, px: float) -> tuple[float, float, float]:
        """返回 (成交额, 佣金, 过户费)。"""
        amount = float(shares) * float(px)
        comm = max(float(self.cfg.MIN_COMMISSION), float(self.cfg.COMMISSION_RATE) * amount)
        transfer = float(self.cfg.TRANSFER_FEE) * amount
        return amount, comm, transfer

    def _sell_net(self, t: int, shares: int, px: float) -> tuple[float, float, float, float]:
        """返回 (成交额, 佣金, 印花税, 过户费)；印花税按成交日历史费率分段。"""
        amount = float(shares) * float(px)
        comm = max(float(self.cfg.MIN_COMMISSION), float(self.cfg.COMMISSION_RATE) * amount)
        stamp = self.stamp_rate(t) * amount
        transfer = float(self.cfg.TRANSFER_FEE) * amount
        return amount, comm, stamp, transfer

    def max_affordable_lots(self, cash_avail: float, px: float) -> int:
        """给定可用现金/买入成交价，返回最大可买整手数（含佣金/过户费）。"""
        lot = int(self.cfg.LOT)
        if not (np.isfinite(cash_avail) and np.isfinite(px)) or cash_avail <= 0 or px <= 0:
            return 0

        def total_cost(lots: int) -> float:
            shares = lots * lot
            amount, comm, transfer = self._buy_cost(shares, px)
            return amount + comm + transfer

        hi = int(cash_avail / (px * lot * (1.0 + self.cfg.COMMISSION_RATE + self.cfg.TRANSFER_FEE))) + 2
        lo = 0
        if total_cost(hi) <= cash_avail + 1e-9:
            return hi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total_cost(mid) <= cash_avail + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        return max(0, lo)

    # ------------------------------------------------------------------ 退市清算
    def _settle_delisted(self, t: int) -> list[Trade]:
        out: list[Trade] = []
        sold = [j for j in self.positions if not self._alive(t, j)]
        for j in sold:
            pos = self.positions.pop(j)
            # 退市计提：最后有效价 × (1 - haircut)，避免按停牌价高估回收金额
            px = self._mark.get(j, pos.entry_price) * (1.0 - self.delist_haircut)
            amount = pos.shares * float(px)
            comm = max(float(self.cfg.MIN_COMMISSION), float(self.cfg.COMMISSION_RATE) * amount)
            transfer = float(self.cfg.TRANSFER_FEE) * amount
            self.cash += amount - comm - transfer
            out.append(Trade(date=t, code=j, side="sell", shares=pos.shares, price=float(px),
                             amount=amount, commission=comm, stamp=0.0, transfer=transfer,
                             slip=0.0, reason="liquidate"))
            self.n_settled_delisted += 1
        return out

    # ------------------------------------------------------------------ 主接口
    def execute(self, t: int, target_weights: dict[int, float] | None = None,
                reasons: dict[int, str] | None = None,
                target_nav: float | None = None,
                target_shares: dict[int, int | None] | None = None) -> list[Trade]:
        """在 **t 日开盘** 按目标权重执行：先卖出（超配/清仓），再买入（低配）。

        参数
        ----
        target_weights : {j: weight}，目标权重（和 ≤ 1，其余为现金）。
            **其插入顺序即买入优先级**（组合层按排名生成，见 portfolio.build_target）。
        reasons        : {j: 'stop_loss'/'liquidate'/...}，覆盖该股票的成交原因。
        target_nav     : 目标权重对应的参考净值；缺省用 t 日开盘净值。
        target_shares  : 预先冻结的目标股数（``plan_target_shares`` 的返回值）。
            缺省时在本次调用内按 t 日开盘价现算（等价于首日执行）。冻结可避免
            "每天重新按现价算目标 → 价格波动导致无谓买卖" 的换手。

        返回本次成交列表（已更新 self.cash 与 self.positions）。
        """
        target_weights = dict(target_weights or {})
        if target_shares is None:
            target_shares = self.plan_target_shares(t, target_weights, target_nav)
        nav_open = self.nav_open(t)
        lot = int(self.cfg.LOT)
        slip = float(self.cfg.SLIPPAGE_BP) / 1e4
        trades: list[Trade] = []

        # 0) 退市股强制清算（避免净值 NaN）
        trades.extend(self._settle_delisted(t))

        # 1) 卖出：现有持仓 > 目标股数（含清仓）
        for j in list(self.positions.keys()):
            pos = self.positions.get(j)
            if pos is None:
                continue
            tgt = target_shares.get(j, 0)
            if tgt is None:
                continue  # 目标价不可用（停牌等）→ 不动
            tgt = int(max(0, tgt))
            delta = int(pos.shares - tgt)
            if delta <= 0:
                continue
            if pos.buy_date >= t:
                continue  # T+1：当天买入不能当天卖
            if not self._can_sell(t, j):
                continue  # 跌停/停牌 → 残差留到下一交易日
            px_open = self._open_px(t, j)
            reason = (reasons or {}).get(j, "rebalance")
            # trade band：已持有且目标仍持有（目标股数 > 0）的等权微调不交易
            if tgt > 0 and float(target_weights.get(j, 0.0)) > 0 and reason == "rebalance":
                band = max(self.trade_band_rel * pos.shares * px_open, self.trade_band_abs)
                if abs(delta * px_open) < band:
                    self.n_band_skipped += 1
                    continue
            fill = px_open * (1.0 - slip)
            amount, comm, stamp, transfer = self._sell_net(t, delta, fill)
            self.cash += amount - comm - stamp - transfer
            pos.shares -= delta
            if pos.shares <= 0:
                self.positions.pop(j, None)
            trades.append(Trade(date=t, code=j, side="sell", shares=delta, price=fill,
                                amount=amount, commission=comm, stamp=stamp,
                                transfer=transfer, slip=abs(fill - px_open) * delta,
                                reason=reason))

        # 2) 买入：现有持仓 < 目标股数（按 target_weights 插入顺序 = 排名优先级）
        cash_floor = max(0.0, 0.01 * nav_open)  # 保底留 1% 现金缓冲
        for j, w in target_weights.items():
            w = float(w)
            if not np.isfinite(w) or w <= 0:
                continue
            tgt = target_shares.get(j)
            if tgt is None:
                continue
            tgt = int(max(0, tgt))
            cur = self.positions[j].shares if j in self.positions else 0
            delta = tgt - cur
            if delta < lot:
                continue
            if not self._can_buy(t, j):
                continue  # 涨停/停牌/现金不足 → 残差留到下一交易日
            px_open = self._open_px(t, j)
            reason = (reasons or {}).get(j, "rebalance")
            # trade band：已持有标的的加仓微调不交易（新建仓不受 band 限制）
            if cur > 0 and reason == "rebalance":
                cur_val = cur * px_open
                band = max(self.trade_band_rel * cur_val, self.trade_band_abs)
                if abs(delta * px_open) < band:
                    self.n_band_skipped += 1
                    continue
            fill = px_open * (1.0 + slip)
            avail = self.cash - cash_floor
            max_lots = self.max_affordable_lots(avail, fill)
            lots = min(delta // lot, max_lots)
            if lots <= 0:
                continue
            shares = lots * lot
            amount, comm, transfer = self._buy_cost(shares, fill)
            self.cash -= amount + comm + transfer
            if j in self.positions:
                old = self.positions[j]
                new_shares = old.shares + shares
                old.entry_price = float((old.entry_price * old.shares + fill * shares) / new_shares)
                old.shares = new_shares
                old.buy_date = t
            else:
                self.positions[j] = Position(code=j, shares=shares, entry_price=fill,
                                             entry_date=t, buy_date=t)
            trades.append(Trade(date=t, code=j, side="buy", shares=shares, price=fill,
                                amount=amount, commission=comm, stamp=0.0,
                                transfer=transfer, slip=abs(fill - px_open) * shares,
                                reason=reason))
        return trades

    # ------------------------------------------------------------------ 止损
    def stop_loss_sells(self, t: int, cfg=None) -> list[int]:
        """收盘后检查个股止损：md.price[t]/entry_price-1 <= STOP_LOSS 的持仓下标。

        返回需要"下一交易日开盘强制卖出"的 j 列表；由 backtest 在 t 日收盘调用
        并把相应目标权重置 0（reason='stop_loss'）。
        """
        cfg = cfg or self.cfg
        out: list[int] = []
        thr = float(cfg.STOP_LOSS)
        for j, pos in self.positions.items():
            if pos.entry_price <= 0:
                continue
            p = float(self.md.price[t, j])
            if not np.isfinite(p):
                p = self._mark.get(j, np.nan)
            if not np.isfinite(p):
                continue
            if p / pos.entry_price - 1.0 <= thr:
                out.append(int(j))
        return out
