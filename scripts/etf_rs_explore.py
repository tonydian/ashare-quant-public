# -*- coding: utf-8 -*-
"""ETF 右侧交易的改进空间探测（先探路，不急着定方案）。

现状：Donchian55 紧出场，CAGR 10.57%、Sharpe 0.91、**平均仓位仅 14.6%**。
瓶颈：85% 时间空仓，资金利用率极低。

探测 6 个方向的上限：
  D1 多标的并行（同时持有多个 ETF，而非最多 1 只）
  D2 缩短出场（更早止盈，提高周转）
  D3 放宽入场（MA20 上方即买，不只突破新高）
  D4 分层仓位（趋势强度决定仓位，而非全有全无）
  D5 加入行业板块 ETF（不止宽基）
  D6 用杠杆/融资（不可行，跳过）

每个都测 CAGR / Sharpe / MaxDD / 平均仓位。
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

OUT = ROOT / "build" / "eval" / "etf_rs_explore.json"

# 载入宽基 ETF
z = np.load(ROOT / "build" / "etf" / "etf_panel.npz", allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
close = z["close"].astype(float)
amount = z["amount"].astype(float)

# 债券/黄金
bond = pd.read_parquet(ROOT.parent / "data_extract" / "was_stock_data" / "bond_000012.parquet")
gold = pd.read_parquet(ROOT.parent / "data_extract" / "was_stock_data" / "gold_518880.parquet")
b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
cash = _blend_cash(b, g, dates)

T, N = close.shape
print(f"宽基 ETF {N} 只：{codes}")
print(f"区间 {dates[0].date()} ~ {dates[-1].date()}")


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


def backtest(entry_n=55, exit_mode="tight", top_k=1,
             ma_filter=0, pos_mode="binary", start_row=None):
    """通用 ETF 右侧回测。

    entry_n   : 突破窗口
    exit_mode : tight（跌破前高即走）| ma（跌破 MA20）| none（只在换标的时卖）
    top_k     : 最多同时持有几只
    ma_filter : >0 时要求收盘也在该均线之上
    pos_mode  : binary（全仓/空仓）| scaled（按趋势强度缩放 0.4~1.0）
    """
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    ma20 = cs.rolling(20, min_periods=20).mean().to_numpy()
    ma60 = cs.rolling(60, min_periods=60).mean().to_numpy()
    ma_filter_arr = (cs.rolling(ma_filter, min_periods=ma_filter).mean().to_numpy()
                     if ma_filter > 0 else None)
    mom60 = (cs / cs.shift(60) - 1.0).to_numpy()

    nav = np.ones(T)
    holdings: dict[int, int] = {}   # j -> 买入日
    pending: dict[int, int] | None = None
    trades = []
    expo = np.zeros(T)
    s0 = 250 if start_row is None else start_row

    for t in range(1, T):
        tcost = 0.0
        # ① 执行上一日决策
        if pending is not None:
            old = set(holdings)
            new = set(pending)
            changed = len(old ^ new)
            if changed:
                tcost = (changed / max(top_k, 1)) * 5.0 / 1e4
            for j in old - new:
                trades.append({"date": str(dates[t].date()), "code": codes[j], "side": "sell"})
                del holdings[j]
            for j in new - old:
                trades.append({"date": str(dates[t].date()), "code": codes[j], "side": "buy"})
                holdings[j] = t
            pending = None

        # ② 当日收益
        rs = []
        for j in holdings:
            if np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0:
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        invest = len(rs) / max(top_k, 1)
        port_r = float(np.mean(rs)) if rs else 0.0
        cr = float(cash.iloc[t]) if np.isfinite(cash.iloc[t]) else 0.0
        nav[t] = nav[t - 1] * (1 + invest * port_r + (1 - invest) * cr - tcost)
        expo[t] = invest

        # ③ 收盘决策
        if t >= s0:
            # 出场判断
            keep = []
            for j in holdings:
                c = close[t, j]
                if not np.isfinite(c):
                    continue
                out = False
                if exit_mode == "tight":
                    out = not (np.isfinite(hi[t, j]) and c > hi[t, j])
                elif exit_mode == "ma":
                    out = np.isfinite(ma20[t, j]) and c < ma20[t, j]
                if not out:
                    keep.append(j)
            # 入场候选
            cand = []
            for j in range(N):
                if j in keep:
                    continue
                c = close[t, j]
                if not np.isfinite(c):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= 5e7):
                    continue
                if not (np.isfinite(hi[t, j]) and c > hi[t, j]):
                    continue
                if ma_filter_arr is not None:
                    if not (np.isfinite(ma_filter_arr[t, j]) and c > ma_filter_arr[t, j]):
                        continue
                m = float(mom60[t, j]) if np.isfinite(mom60[t, j]) else -9.0
                cand.append((j, m))
            cand.sort(key=lambda x: -x[1])
            slots = top_k - len(keep)
            new = keep + [j for j, _ in cand[:slots]]
            if set(new) != set(holdings):
                pending = {j: holdings.get(j, -1) for j in new}

    return pd.Series(nav, index=dates), trades, pd.Series(expo, index=dates)


def perf(nav, s="2020-01-02", e="2026-10-23"):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


res = {}
print("\n" + "=" * 100)
print("ETF 右侧改进方向探测")
print("=" * 100)
print(f"{'方案':<34}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'仓位':>8}{'交易':>7}")
print("-" * 100)

cases = [
    ("D0 基准：top1 + 紧出场", dict(top_k=1, exit_mode="tight")),
    ("D1a top2 并行", dict(top_k=2, exit_mode="tight")),
    ("D1b top3 并行", dict(top_k=3, exit_mode="tight")),
    ("D2 紧出场 → MA20出场", dict(top_k=1, exit_mode="ma")),
    ("D3a +MA60过滤", dict(top_k=1, exit_mode="tight", ma_filter=60)),
    ("D3b +MA200过滤", dict(top_k=1, exit_mode="tight", ma_filter=200)),
    ("D1c top2 + MA20出场", dict(top_k=2, exit_mode="ma")),
    ("D1d top3 + MA20出场", dict(top_k=3, exit_mode="ma")),
    ("D3c top2 + MA60过滤 + MA20出场", dict(top_k=2, exit_mode="ma", ma_filter=60)),
    ("D3d top3 + MA60过滤 + MA20出场", dict(top_k=3, exit_mode="ma", ma_filter=60)),
]
for tag, kw in cases:
    nav, tr, ex = backtest(**kw)
    p = perf(nav)
    res[tag] = {**p, "avg_expo": float(ex[ex.index >= "2020-01-02"].mean()),
                "n_trades": len(tr)}
    print(f"{tag:<34}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}{p['max_dd']:>8.2%}"
          f"{p['calmar']:>8.2f}{res[tag]['avg_expo']:>7.1%}{len(tr):>7}")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
