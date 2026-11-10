# -*- coding: utf-8 -*-
"""彻底修正 A2 回测的两个时序 bug 后重测。

Bug 1（已修）：t 日买入，当日就计收益 → 白拿 t-1→t 涨幅
Bug 2（待修）：t 日卖出，当日收益仍计入 → 但卖在 t 日收盘，t 日收益应算持有到收盘

正确语义（保守口径）：
  * t 日收盘决策，**t+1 日开盘或收盘**执行 → 本次用"t+1 日收盘价"成交（保守）
  * 因此 t 日买入 → t+1 日才持有 → 收益从 t+2 开始算（t+1 的收益是买入当天，不含）
  * 简化并保守：买入成交日不计收益，卖出成交日计收益（因为持有到卖出时点）

更简单可靠的做法：**用 pending 机制**，把"决策日"和"生效日"分开。

本脚本用一个干净的实现：
  decision at t (收盘) → execute at t+1 (按 t+1 收盘价成交)
  → 收益区间：从 t+1 收盘持到 t+2 收盘
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402

PANEL = ROOT / "build" / "etf" / "sector_panel_fixed.npz"
UNIV = ROOT / "build" / "etf" / "sector_universe.json"
EXCLUDE = {"纳指", "标普", "日经", "港股", "恒生科技",
           "沪深300", "中证500", "中证1000", "创业板", "科创50", "红利", "金融"}

z = np.load(PANEL, allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
close = z["close"].astype(float)
amount = z["amount"].astype(float)
univ = json.load(open(UNIV, encoding="utf-8"))
s2 = {v["code"]: k for k, v in univ["sectors"].items()}
keep = [j for j, c in enumerate(codes) if s2.get(c) not in EXCLUDE]
codes = [codes[j] for j in keep]
close = close[:, keep]
amount = amount[:, keep]
sector_of = {c: s2[c] for c in codes}

bond = pd.read_parquet(ROOT.parent / "data_extract" / "was_stock_data" / "bond_000012.parquet")
gold = pd.read_parquet(ROOT.parent / "data_extract" / "was_stock_data" / "gold_518880.parquet")
b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
cash = (_blend_cash(b, g, dates))


def _blend_cash(b, g, idx):
    """按 C.CASH_BOND_WEIGHT / C.CASH_GOLD_WEIGHT 混合现金收益（黄金权重可为 0）。"""
    bw, gw = float(C.CASH_BOND_WEIGHT), float(C.CASH_GOLD_WEIGHT)
    tot = bw + gw
    if tot <= 0:
        return pd.Series(0.0, index=idx)
    r = (bw / tot) * b.reindex(idx).ffill().pct_change().fillna(0)
    if gw > 0:
        r = r + (gw / tot) * g.reindex(idx).ffill().pct_change().fillna(0)
    return r


def backtest_v2(*, entry_n=55, mom_n=60, top_n=3, min_amount=2.5e7,
                min_hold=0, exit_ma=0, check_every=1, start_row=500):
    """严格 T+1 语义：t 日收盘决策 → t+1 日收盘成交 → t+2 起计收益。"""
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    mom = (cs / cs.shift(mom_n) - 1.0).to_numpy()
    ma = cs.rolling(exit_ma, min_periods=exit_ma).mean().to_numpy() if exit_ma > 0 else None

    nav = np.ones(T)
    holdings: dict[int, int] = {}      # j -> 成交日 t
    pending: dict[int, int] | None = None
    trades = []
    expo = np.zeros(T)

    for t in range(1, T):
        tcost = 0.0

        # ---------- ① 执行上一日收盘的决策（t 日收盘成交）
        if pending is not None:
            old = set(holdings)
            new = set(pending)
            for j in old - new:
                trades.append({"date": str(dates[t].date()), "code": codes[j],
                               "sector": sector_of.get(codes[j], ""), "side": "sell",
                               "price": float(close[t, j])})
            for j in new - old:
                trades.append({"date": str(dates[t].date()), "code": codes[j],
                               "sector": sector_of.get(codes[j], ""), "side": "buy",
                               "price": float(close[t, j])})
                holdings[j] = t
            for j in old - new:
                del holdings[j]
            changed = len(old ^ new)
            if changed:
                tcost = (changed / max(top_n, 1)) * 5.0 / 1e4
            pending = None

        # ---------- ② 当日收益：只算 t-1 就已持有的（当日成交的不算）
        rs = []
        for j, bought_t in holdings.items():
            if bought_t >= t:
                continue
            if np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0:
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        invest = len(rs) / max(top_n, 1)
        port_r = float(np.mean(rs)) if rs else 0.0
        cr = float(cash.iloc[t]) if np.isfinite(cash.iloc[t]) else 0.0
        nav[t] = nav[t - 1] * (1 + invest * port_r + (1 - invest) * cr - tcost)
        expo[t] = invest

        # ---------- ③ t 日收盘决策
        if t >= start_row and (t - start_row) % check_every == 0:
            to_sell = []
            for j, bought in holdings.items():
                if t - bought < min_hold:
                    continue
                c = close[t, j]
                if not np.isfinite(c):
                    to_sell.append(j); continue
                if exit_ma > 0:
                    out = np.isfinite(ma[t, j]) and c < ma[t, j]
                else:
                    out = not (np.isfinite(hi[t, j]) and c > hi[t, j])
                if out:
                    to_sell.append(j)
            keep_j = [j for j in holdings if j not in to_sell]
            slots = top_n - len(keep_j)
            add = []
            if slots > 0:
                best = {}
                for j in range(N):
                    if j in holdings and j not in to_sell:
                        continue
                    c = close[t, j]
                    if not np.isfinite(c):
                        continue
                    if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= min_amount):
                        continue
                    if not (np.isfinite(hi[t, j]) and c > hi[t, j]):
                        continue
                    sec = sector_of.get(codes[j], codes[j])
                    m = float(mom[t, j]) if np.isfinite(mom[t, j]) else -9.0
                    if sec not in best or m > best[sec][1]:
                        best[sec] = (j, m)
                cand = sorted(best.values(), key=lambda x: -x[1])[:slots]
                add = [j for j, _ in cand]
            pending = set(keep_j) | set(add)

    return pd.Series(nav, index=dates), pd.DataFrame(trades), pd.Series(expo, index=dates)


def perf(nav, s, e):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


S, E = "2020-01-02", "2026-10-23"
cfg = [
    ("A1 每10日调仓（基准）", dict(check_every=10)),
    ("A2 每日+紧出场", dict(check_every=1)),
    ("A3 每日+最短持有5日", dict(check_every=1, min_hold=5)),
    ("A4 每日+最短持有10日", dict(check_every=1, min_hold=10)),
    ("A5 每日+MA20出场", dict(check_every=1, exit_ma=20)),
    ("A6 每日+MA20+最短5日", dict(check_every=1, exit_ma=20, min_hold=5)),
]

print("严格 T+1（t日决策 → t+1日成交 → t+2起计收益）")
print("=" * 96)
print(f"{'方案':<24}{'全期CAGR':>10}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
      f"{'2026年':>9}{'交易':>7}{'仓位':>7}")
print("-" * 96)
res = {}
for tag, kw in cfg:
    nav, tr, ex = backtest_v2(**kw)
    p = perf(nav, S, E)
    p26 = perf(nav, "2026-01-01", E)
    res[tag] = {**p, "y2026": p26.get("cagr"), "n_trades": len(tr),
                "avg_expo": float(ex[ex.index >= S].mean())}
    print(f"{tag:<24}{p['cagr']:>9.2%}{p['sharpe']:>8.3f}{p['max_dd']:>8.2%}"
          f"{p['calmar']:>8.2f}{p26.get('cagr', 0):>8.1%}{len(tr):>7}"
          f"{res[tag]['avg_expo']:>7.1%}")

json.dump(res, open(ROOT / "build" / "eval" / "sector_daily_v2.json", "w"),
          ensure_ascii=False, indent=1, default=float)
print(f"\n→ build/eval/sector_daily_v2.json")
