# -*- coding: utf-8 -*-
"""确认：高频换仓是亏损主因（不是信号无 alpha）。

从 trace 看到：
  策略几乎每天都在买卖（2022-01 到 03，60 天里约 40 次交易）
  → 每次往返 ~20bp → 40 次 = 8% 成本
  → 60 天亏 15%

这符合文章《ETF动量轮动深度拆解》的诊断：
  "每天都重新排名，如果没有【最小持有期】限制，
   策略会因为排名的细微起伏来回倒仓，交易成本会吃掉大半收益"

验证方法：
  同一套动量信号，只改"换仓频率"：
    A. 每日换仓（当前）
    B. 加最小持有期 8 天
    C. 加排名缓冲（新目标必须比当前持仓强一定幅度才换）
    D. 固定 20 日调仓

如果 B/C/D 明显改善，说明是换手问题，不是信号问题。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402

z = np.load(ROOT / "build" / "etf" / "etf_ohlc_panel_v2.npz", allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
meta = json.load(open(ROOT / "build" / "etf" / "etf_ohlc_meta_v2.json",
                      encoding="utf-8"))
close = z["close"].astype(np.float64)
op = z["open"].astype(np.float64)
T, N = close.shape

BOND = {"511010", "511260", "511220", "511020"}
is_bond = np.array([c in BOND for c in codes])

mom20 = pd.DataFrame(close).pct_change(20).to_numpy()


def run(start, end, top_n=1, min_hold=1, buffer=0.0, fixed_period=0,
        use_filter=None, slip_bp=10.0):
    """带最小持有期 / 排名缓冲 / 固定周期的轮动。

    buffer: 新标的的动量必须比现有持仓高出该值才换（相对值）
    fixed_period: >0 时改为固定周期调仓
    """
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    slip = slip_bp / 1e4
    cash = 1.0
    hold: dict[int, dict] = {}
    nav = np.full(T, np.nan)
    nav[t0] = 1.0
    pend = None
    n_trades = 0
    last_rebal = t0

    for t in range(t0 + 1, t1 + 1):
        # 执行
        if pend is not None:
            target = set(pend)
            for j in list(hold):
                if j not in target:
                    px = op[t, j]
                    if np.isfinite(px) and px > 0:
                        cash += hold[j]["sh"] * px * (1 - slip)
                        del hold[j]
                        n_trades += 1
            slots = top_n - len(hold)
            if slots > 0 and target:
                tot = cash + sum(hold[k]["sh"] * op[t, k]
                                 for k in hold if np.isfinite(op[t, k]))
                per = tot / top_n
                for j in list(target):
                    if j in hold:
                        continue
                    px = op[t, j]
                    if not (np.isfinite(px) and px > 0):
                        continue
                    b = min(per, cash)
                    if b <= 1e-9:
                        break
                    sh = b / (px * (1 + slip))
                    cash -= sh * px * (1 + slip)
                    hold[j] = {"sh": sh, "day": t}
                    n_trades += 1
                    if len(hold) >= top_n:
                        break
            pend = None

        v = cash
        for j, h in hold.items():
            px = close[t, j]
            if np.isfinite(px):
                v += h["sh"] * px
        nav[t] = v

        # 是否检查信号
        if fixed_period > 0 and (t - last_rebal) < fixed_period:
            continue
        last_rebal = t

        m = mom20[t].copy()
        ok = np.isfinite(m) & np.isfinite(close[t]) & (m > 0)
        if use_filter == "bond":
            ok &= is_bond
        elif use_filter == "equity":
            ok &= ~is_bond
        idx = np.flatnonzero(ok)
        if not len(idx):
            pend = set()
            continue
        idx = idx[np.argsort(-m[idx])][:top_n]
        target = set(idx.tolist())

        # 最小持有期
        if min_hold > 1:
            cur_ok = {j for j in hold if (t - hold[j]["day"]) >= min_hold}
            target = (target & set(hold)) | (target - cur_ok)

        # 排名缓冲：只有当新标的动量明显更强才换
        if buffer > 0 and hold:
            cur_min = min(m[j] for j in hold if j in m and np.isfinite(m[j])) \
                if any(j in m and np.isfinite(m[j]) for j in hold) else -9
            keep = set()
            for j in hold:
                if j in target:
                    keep.add(j)
                elif j in m and np.isfinite(m[j]) and (m[j] >= cur_min + buffer):
                    keep.add(j)
                # else 保留（不卖）
            # 保留所有现持仓除非被明确替换
            target = target | hold.keys() if len(keep) == 0 else (target | keep)

        pend = target

    s = pd.Series(nav, index=dates).dropna()
    if len(s) < 60:
        return None, n_trades
    p = qfm.performance(s, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in
            ("cagr", "sharpe", "max_dd", "calmar")}, n_trades


print("=" * 96)
print("换仓频率对收益的影响（同一 20 日动量信号）")
print("=" * 96)
print(f"{'配置':<28}{'期间':<7}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'交易':>7}")
print("-" * 96)

configs = [
    ("A 每日换仓（无限制）", dict(top_n=1, min_hold=1)),
    ("B 最小持有 8 天", dict(top_n=1, min_hold=8)),
    ("C 最小持有 20 天", dict(top_n=1, min_hold=20)),
    ("D 固定 20 日调仓", dict(top_n=1, fixed_period=20)),
    ("E 固定 10 日调仓", dict(top_n=1, fixed_period=10)),
    ("F top3 + 持有8天", dict(top_n=3, min_hold=8)),
    ("G top3 + 固定20日", dict(top_n=3, fixed_period=20)),
    ("H top2 + 固定20日", dict(top_n=2, fixed_period=20)),
]

results = {}
for label, kw in configs:
    for pname, (a, b) in (("DEV", ("2018-01-01", "2021-12-31")),
                          ("OOS", ("2022-01-01", "2026-10-31"))):
        p, nt = run(a, b, **kw)
        if p is None:
            continue
        results[(label, pname)] = (p, nt)
        print(f"{label:<28}{pname:<7}{p['cagr']:>8.2%}{p['sharpe']:>9.3f}"
              f"{p['max_dd']:>9.2%}{nt:>7}")

print("\n" + "=" * 96)
print("结论")
print("=" * 96)
print(f"{'配置':<28}{'DEV CAGR':>11}{'OOS CAGR':>11}{'OOS交易':>10}")
print("-" * 96)
for label, _ in configs:
    d = results.get((label, "DEV"))
    o = results.get((label, "OOS"))
    if d and o:
        print(f"{label:<28}{d[0]['cagr']:>10.2%}{o[0]['cagr']:>11.2%}{o[1]:>10}")

best = max([(k, v) for k, v in results.items() if k[1] == "OOS"],
           key=lambda kv: kv[1][0]["cagr"], default=None)
if best:
    print(f"\n  OOS 最优: {best[0][0]}")
    print(f"    CAGR {best[1][0]['cagr']:.2%}, Sharpe {best[1][0]['sharpe']:.3f}, "
          f"交易 {best[1][1]} 笔")

json.dump({f"{k[0]}|{k[1]}": {"perf": v[0], "trades": v[1]}
           for k, v in results.items()},
          open(ROOT / "build" / "eval" / "etf_rebal_freq_test.json", "w"),
          ensure_ascii=False, indent=1, default=float)
print(f"\n→ build/eval/etf_rebal_freq_test.json")
