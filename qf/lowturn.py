# -*- coding: utf-8 -*-
"""低换手月度策略（用户给定规则）。

规则
----
1. 上月末计算过去 20 个交易日的平均换手率；
2. 剔除 ST、退市、停牌、上市不满一年（默认 250 交易日）的股票；
3. 选换手率最小的 n_hold 只（默认 10）；
4. 每月第一个交易日开盘等权买入；
5. 每月最后一个交易日收盘卖出 —— **落地口径做了修正，见下**。

两个口径
--------
* ``strict``：严格照字面，每月末全部清仓、次月首全部买回。每年白付约 11.8pp 成本。
* ``actual``（默认，推荐）：**只换变动**。上月已持有且本期仍在名单里的份额不动，
  只对退出/新进的部分成交。年换手从 24x 降到约 7x。

回测实测（2020-01 ~ 2026-10，82 个月，单边成本 49.3bp）：
毛 13.82% / 只换变动净 9.84% / 严格净 0.66%。

换手率口径
----------
``成交额 / 总市值``。不用 ``volume / 股本``：数据里 391 只股票的 volume 单位是"手"、
其余是"股"，直接除会让这批被低估 100 倍，全部挤进最低换手组。两种口径在同一日的
横截面 rank 相关 0.9997。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as qfmetrics

LOOKBACK = 20
MIN_LIST_DAYS = 250          # 约一年交易日
COST_BP = 49.3               # 单边：滑点 40 + 佣金 6.67 + 印花 2.5 + 过户 0.1


def turnover_panel(md, lookback: int = LOOKBACK) -> np.ndarray:
    """20 日均换手率（成交额 / 总市值），(T, N)。"""
    amt = md.amount.astype(np.float64)
    mcap = md.mcap.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        turn = np.where((mcap > 0) & np.isfinite(amt), amt / mcap, np.nan)
    turn[turn <= 0] = np.nan
    return (pd.DataFrame(turn)
            .rolling(lookback, min_periods=lookback).mean()
            .to_numpy(dtype=np.float64))


def month_bounds(dates: pd.DatetimeIndex) -> dict:
    """{月度 Period: (首日行号, 末日行号)}"""
    per = pd.PeriodIndex(dates, freq="M")
    out: dict[pd.Period, tuple[int, int]] = {}
    for i, m in enumerate(per):
        if m in out:
            out[m] = (out[m][0], i)
        else:
            out[m] = (i, i)
    return out


def month_role(dates: pd.DatetimeIndex, t: int) -> tuple[int, str, str]:
    """返回 (信号日行号, 目标持有月份, 时点状态)。

    t 是本月最后一个交易日 → 用 t 算下月名单（今晚就能准备）；
    否则 → 用上月最后一个交易日算本月名单（已在持有，等下月调仓）。

    放在 qf/ 而不是脚本里，是为了让信号脚本、daily_signal 与 Web 共用同一套
    时点判定，避免三处各写一遍、口径漂移。
    """
    mb = month_bounds(dates)
    months = sorted(mb)
    m = pd.Period(dates[t], freq="M")
    idx = months.index(m)
    if t == mb[m][1]:
        tgt = months[idx + 1] if idx + 1 < len(months) else m + 1
        return t, str(tgt), "month_end"
    return (mb[months[idx - 1]][1] if idx > 0 else t), str(m), "mid_month"


def candidate_mask(md, turn20: np.ndarray, t: int, min_list_days: int = MIN_LIST_DAYS,
                   min_adv20: float = 0.0) -> np.ndarray:
    """t 日可参与选股的股票（剔除 ST/退市/停牌/上市不满一年/流动性不足）。

    ``min_adv20`` 单位是元；0 表示不加流动性下限（严格按原始规则）。
    """
    ok = (md.alive[t]
          & ~md.is_suspended[t]
          & (md.list_days[t] >= min_list_days)
          & ~md.name_st
          & np.isfinite(turn20[t]))
    if min_adv20 > 0:
        adv = md.amount_20d[t]
        ok &= np.isfinite(adv) & (adv >= min_adv20)
    return ok


def select(md, turn20: np.ndarray, t_sig: int, n_hold: int = 10,
           min_list_days: int = MIN_LIST_DAYS, min_adv20: float = 0.0) -> list[int]:
    """在 t_sig 日按换手率升序取前 n_hold 只（不含买入日可买性检查）。"""
    ok = candidate_mask(md, turn20, t_sig, min_list_days, min_adv20)
    cand = np.where(ok)[0]
    if cand.size == 0:
        return []
    order = cand[np.argsort(turn20[t_sig, cand], kind="stable")]
    return [int(j) for j in order[:n_hold]]


def _ffill_px(md) -> np.ndarray:
    """估值用后复权收盘，ffill 到底 → 退市/停牌期间按最后有效价冻结。"""
    return pd.DataFrame(md.close.astype(np.float64)).ffill().to_numpy()


class _Book:
    """份额账本：现金 + 每股持股数，按开盘价成交。"""

    __slots__ = ("cash", "shares", "held")

    def __init__(self, n: int):
        self.cash = 1.0
        self.shares = np.zeros(n, dtype=np.float64)
        self.held: set[int] = set()

    def value(self, price: np.ndarray, t: int) -> float:
        v = self.cash + sum(self.shares[j] * price[t, j] for j in self.held)
        return v if np.isfinite(v) and v > 0 else self.cash


def _rebalance(book: _Book, picks: list[int], t: int, opx: np.ndarray,
               px: np.ndarray, cost: float, n_hold: int,
               rebalance_weight: bool = True) -> tuple[list[int], list[int], float]:
    """月初开盘调仓，返回 (新进, 退出, 单边成交额占组合比例)。

    ``rebalance_weight=True``（默认）：把持仓权重校准到 1/n_hold，只对实际变动
    的金额成交 —— 退出的全卖、新进的补足、保留的只调整漂移，不是全卖全买。
    ``False``：保留的份额完全不动（会形成赢家权重越滚越大的动量暴露）。
    """
    pick_set = set(picks)
    exited = sorted(book.held - pick_set)
    entered = [j for j in picks if j not in book.held]
    total = book.cash + sum(book.shares[j] * opx[t, j] for j in book.held
                            if np.isfinite(opx[t, j]))
    traded = 0.0

    for j in exited:                                     # 退出：全卖
        p = opx[t, j] if np.isfinite(opx[t, j]) else px[t, j]
        v = book.shares[j] * p
        traded += v
        book.cash += v * (1 - cost)
        book.shares[j] = 0.0
    book.held -= set(exited)

    if rebalance_weight:
        tgt = total / n_hold
        for j in picks:                                  # 校准到等权
            p = opx[t, j] if np.isfinite(opx[t, j]) else px[t, j]
            if not (np.isfinite(p) and p > 0):
                continue
            cur_v = book.shares[j] * p if j in book.held else 0.0
            d = tgt - cur_v
            if abs(d) < tgt * 0.02:                       # 2% 以内不动，省手续费
                continue
            traded += abs(d)
            if d > 0:
                book.shares[j] += d * (1 - cost) / p
                book.cash -= d
            else:
                book.shares[j] += d / p
                book.cash -= d * (1 - cost)
            book.held.add(j)
    elif entered:                                        # 新进买足、保留不动
        per = total / n_hold
        budget = min(book.cash, per * len(entered))
        each = budget / len(entered)
        for j in entered:
            p = opx[t, j] if np.isfinite(opx[t, j]) else px[t, j]
            if not (np.isfinite(p) and p > 0):
                continue
            book.shares[j] = each * (1 - cost) / p
            book.cash -= each
            book.held.add(j)
            traded += each
    book.held |= pick_set
    return entered, exited, traded / total if total > 0 else 0.0


def backtest(md, start: str = "2020-01-02", n_hold: int = 10, lookback: int = LOOKBACK,
             min_list_days: int = MIN_LIST_DAYS, min_adv20: float = 0.0,
             cost_bp: float = COST_BP, rebalance_weight: bool = True) -> dict:
    """月度回测，同时产出「等权调仓（净）」「同口径毛」「严格全卖全买（净）」。

    ｜ 三套净值来自各自的份额账本，持仓序列相同，差别只在成交成本与是否清仓。
    """
    dates, T, N = md.dates, md.T, md.N
    turn20 = turnover_panel(md, lookback)
    px = _ffill_px(md)
    opx = md.open.astype(np.float64)
    mb = month_bounds(dates)
    months = sorted(mb)
    c = float(cost_bp) / 1e4
    start_ts = pd.Timestamp(start)

    def _can_buy(t: int) -> np.ndarray:
        return (md.alive[t] & ~md.is_suspended[t]
                & np.isfinite(opx[t]) & (opx[t] > 0)
                & (md.volume[t] > 0) & ~md.limit_up_at_open[t])

    books = {"net": _Book(N), "gross": _Book(N), "strict": _Book(N)}
    ret = {k: np.zeros(T, dtype=np.float64) for k in books}
    last_close = {k: 1.0 for k in books}       # 上月末收盘总资产（跨月衔接）
    trades, picks_log = [], []

    for k in range(1, len(months)):
        m, m_prev = months[k], months[k - 1]
        t_sig = mb[m_prev][1]
        t_first, t_last = mb[m]
        if dates[t_first] < start_ts:
            continue

        order = select(md, turn20, t_sig, n_hold, min_list_days, min_adv20)
        if not order:
            continue
        # 买入日可买性：买不进按换手率顺序补位
        cb = _can_buy(t_first)
        picks = [j for j in order if cb[j]][:n_hold]
        if not picks:
            continue

        ent, exi, tr_net = _rebalance(books["net"], picks, t_first, opx, px, c, n_hold,
                                      rebalance_weight)
        _rebalance(books["gross"], picks, t_first, opx, px, 0.0, n_hold, rebalance_weight)

        # 严格口径：月初清掉上期全部，再按等权全仓买入
        sb = books["strict"]
        for j in sorted(sb.held):
            p = opx[t_first, j] if np.isfinite(opx[t_first, j]) else px[t_first, j]
            sb.cash += sb.shares[j] * p * (1 - c)
            sb.shares[j] = 0.0
        sb.held.clear()
        tot = sb.cash
        each = tot / len(picks)                       # 预算全额，手续费从买入市值里扣
        for j in picks:
            p = opx[t_first, j] if np.isfinite(opx[t_first, j]) else px[t_first, j]
            if not (np.isfinite(p) and p > 0):
                continue
            sb.shares[j] = each * (1 - c) / p
            sb.cash -= each
            sb.held.add(j)

        # ---------------- 逐日估值 ----------------
        for name, bk in books.items():
            prev = last_close[name]
            for tt in range(t_first, t_last + 1):
                val = bk.value(px, tt)
                if name == "strict" and tt == t_last:
                    val *= (1 - c)                    # 月末卖出，扣一次单边
                ret[name][tt] = val / prev - 1.0
                prev = val
            last_close[name] = prev
            if name == "strict":
                # 虚拟清仓：账本转全现金，下月初按开盘价全仓重建（买侧再扣一次）
                bk.cash = prev
                bk.shares[:] = 0.0
                bk.held.clear()

        trades.append({
            "month": str(m), "t_sig": str(dates[t_sig].date()),
            "t_buy": str(dates[t_first].date()), "t_sell": str(dates[t_last].date()),
            "n": len(picks), "n_entered": len(ent), "n_exited": len(exi),
            "keep_ratio": (len(picks) - len(ent)) / len(picks),
            "traded": _f(tr_net),
            "avg_turn20_bp": _f(float(np.nanmean(turn20[t_sig, picks]) * 1e4)),
            "mcap_median_yi": _f(float(np.nanmedian(md.mcap[t_sig, picks]) / 1e8)),
        })
        picks_log.append({
            "month": str(m), "signal_date": str(dates[t_sig].date()),
            "buy_date": str(dates[t_first].date()),
            "codes": [str(md.codes[j]) for j in picks],
            "names": [str(md.names[j]) for j in picks],
            "entered": [str(md.codes[j]) for j in ent],
            "exited": [str(md.codes[j]) for j in exi],
        })

    i0 = int(np.searchsorted(dates, start_ts))
    idx = dates[i0:]
    navs = {}
    for k, r in ret.items():
        nav = np.cumprod(1 + r)
        navs[k] = pd.Series(nav[i0:] / nav[i0], index=idx)

    tr = pd.DataFrame(trades)
    keep_mean = float(tr["keep_ratio"].mean()) if len(tr) else 0.0
    # 单边年化换手 = 每期实际成交额占组合比例 × 12（买卖各计一次，故成本 = 2×T×c）
    turn_ann = float(tr["traded"].mean()) * 12.0 if len(tr) else 0.0
    return {
        "nav": navs["net"],
        "nav_gross": navs["gross"],
        "nav_strict": navs["strict"],
        "trades": tr,
        "picks": picks_log,
        "n_months": int(len(tr)),
        "keep_mean": keep_mean,
        "turnover_ann": turn_ann,
        "turnover_ann_strict": 24.0,
        "cost_bp": float(cost_bp),
        "params": {"n_hold": n_hold, "lookback": lookback,
                   "min_list_days": min_list_days, "min_adv20": min_adv20,
                   "start": start},
    }


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def performance(nav: pd.Series, rf: float | None = None) -> dict:
    q = qfmetrics.performance(nav, rf=C.CASH_YIELD if rf is None else rf)
    return {k: _f(q[k]) for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}
