# -*- coding: utf-8 -*-
"""决定性检验：ETF 轮动在 OOS 期（2022-2026）到底有没有 alpha？

我们试了多种配置，OOS 都是负的。需要判断：
  H1 是策略实现问题
  H2 还是"A股+海外ETF在这个区间本来就没有趋势可跟"

关键检验（最干净）：
  把"动量轮动"简化到不能再简：
    · 每日检查，取 20 日动量最高的 1 只
    · 无任何过滤、无止盈损、无择时
    · 每日可换仓（不加 min_hold）
  如果这个最简单的版本 OOS 也是负的，说明问题在"信号无 alpha"，
  而不是我们的实现。
  
对照组（同期基准）：
  · 沪深300 买入持有
  · 等权持有全部 41 只
  · 只持有债券
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
is_equity = ~is_bond

# 简单动量
mom20 = pd.DataFrame(close).pct_change(20).to_numpy()
mom60 = pd.DataFrame(close).pct_change(60).to_numpy()


def perf_nav(nav_series, label):
    s = nav_series.dropna()
    if len(s) < 60:
        return None
    p = qfm.performance(s, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


def run_simple(start, end, top_n=1, use_filter=None, min_hold=1):
    """最简轮动：每日取动量最高的 N 只，等权。

    use_filter: None / "bond" / "equity"
    """
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    cash = 1.0
    hold = {}
    nav = np.full(T, np.nan)
    nav[t0] = 1.0
    n_switch = 0
    pend = None

    for t in range(t0 + 1, t1 + 1):
        # 执行昨日信号
        if pend is not None:
            target = set(pend)
            # 卖
            for j in list(hold):
                if j not in target:
                    px = op[t, j]
                    if np.isfinite(px) and px > 0:
                        cash += hold[j]["sh"] * px * 0.9990   # 5bp 滑点
                        del hold[j]
            # 买
            slots = top_n - len(hold)
            if slots > 0:
                tot = cash + sum(hold[k]["sh"] * op[t, k]
                                 for k in hold if np.isfinite(op[t, k]))
                per = tot / top_n
                for j in list(target)[:slots]:
                    if j in hold:
                        continue
                    px = op[t, j]
                    if not (np.isfinite(px) and px > 0):
                        continue
                    b = min(per, cash)
                    if b <= 1e-9:
                        break
                    sh = b / (px * 1.0010)
                    cash -= sh * px * 1.0010
                    hold[j] = {"sh": sh, "day": t}
            pend = None

        # 净值
        v = cash
        for j, h in hold.items():
            px = close[t, j]
            if np.isfinite(px):
                v += h["sh"] * px
        nav[t] = v

        # 信号（t 收盘算，t+1 开盘执行）
        m = mom20[t].copy()
        ok = np.isfinite(m) & np.isfinite(close[t])
        if use_filter == "bond":
            ok &= is_bond
        elif use_filter == "equity":
            ok &= is_equity
        ok &= (m > 0)
        idx = np.flatnonzero(ok)
        if len(idx):
            idx = idx[np.argsort(-m[idx])][:top_n]
            new_target = set(idx.tolist())
        else:
            new_target = set()
        if new_target != set(hold):
            n_switch += 1
        pend = new_target

    s = pd.Series(nav, index=dates)
    return s, n_switch


print("=" * 94)
print("最简轮动检验（无过滤、无止损、无择时）")
print("=" * 94)
print(f"{'配置':<26}{'期间':<8}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'换仓':>7}")
print("-" * 94)

rows = []
for label, kw in (
        ("动量top1 全池", dict(top_n=1)),
        ("动量top2 全池", dict(top_n=2)),
        ("动量top3 全池", dict(top_n=3)),
        ("动量top1 仅股票", dict(top_n=1, use_filter="equity")),
        ("动量top1 仅债券", dict(top_n=1, use_filter="bond")),
):
    for pname, (a, b) in (("DEV", ("2018-01-01", "2021-12-31")),
                          ("OOS", ("2022-01-01", "2026-10-31"))):
        s, nsw = run_simple(a, b, **kw)
        p = perf_nav(s, label)
        if p is None:
            continue
        rows.append({"label": label, "period": pname, **p})
        print(f"{label:<26}{pname:<8}{p['cagr']:>8.2%}{p['sharpe']:>9.3f}"
              f"{p['max_dd']:>9.2%}{nsw:>7}")

print("\n" + "=" * 94)
print("对照基准")
print("=" * 94)
print(f"{'基准':<26}{'期间':<8}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}")
print("-" * 94)

def bench_hold(mask, a, b, label):
    t0 = int(dates.searchsorted(pd.Timestamp(a)))
    t1 = int(dates.searchsorted(pd.Timestamp(b), side="right")) - 1
    sub = close[t0:t1 + 1][:, mask]
    # 等权：每日再平衡（近似）
    rets = pd.DataFrame(sub).pct_change().mean(axis=1)
    nav = (1 + rets.fillna(0)).cumprod()
    nav.index = dates[t0:t1 + 1]
    p = perf_nav(nav, label)
    return p

for label, mask in (("沪深300 单只", np.array([c == "510300" for c in codes])),
                    ("全部股票等权", is_equity),
                    ("全部债券等权", is_bond),
                    ("全池等权", np.ones(N, dtype=bool))):
    if not mask.any():
        continue
    for pname, (a, b) in (("DEV", ("2018-01-01", "2021-12-31")),
                          ("OOS", ("2022-01-01", "2026-10-31"))):
        p = bench_hold(mask, a, b, label)
        if p is None:
            continue
        print(f"{label:<26}{pname:<8}{p['cagr']:>8.2%}{p['sharpe']:>9.3f}"
              f"{p['max_dd']:>9.2%}")

print("\n" + "=" * 94)
print("结论")
print("=" * 94)
oos_rows = [r for r in rows if r["period"] == "OOS"]
if oos_rows:
    best = max(oos_rows, key=lambda r: r["sharpe"])
    print(f"  OOS 最优配置: {best['label']}")
    print(f"    CAGR {best['cagr']:.2%}, Sharpe {best['sharpe']:.3f}, "
          f"MaxDD {best['max_dd']:.2%}")
    n_neg = sum(1 for r in oos_rows if r["cagr"] < 0)
    print(f"\n  {n_neg}/{len(oos_rows)} 个配置 OOS 为负")
    if n_neg == len(oos_rows):
        print("  → 所有动量配置在 OOS 都亏钱，说明信号无 alpha")
    else:
        print("  → 有配置 OOS 为正，需进一步分析")
