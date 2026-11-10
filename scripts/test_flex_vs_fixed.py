# -*- coding: utf-8 -*-
"""修正 min_hold 实现 + 最终对比：弹性周期 vs 固定周期。

文章说弹性周期（每日检查 + 最小持有期）应该优于固定周期。
但我的测试里 min_hold 没生效，需要修正后重新对比。

正确的 min_hold 逻辑：
  · 每日重新排名
  · 对于当前持有的标的：如果它已满 min_hold 天且不在目标中 → 卖
  · 如果它未满 min_hold 天 → 即使不在目标中也**不卖**（但目标里的新标的可以买）
  · 注意：目标集合要保持"数量 ≤ top_n"
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
close = z["close"].astype(np.float64)
op = z["open"].astype(np.float64)
T, N = close.shape
mom20 = pd.DataFrame(close).pct_change(20).to_numpy()


def run(start, end, top_n=2, min_hold=0, fixed_period=20, slip_bp=10.0,
        verbose=False):
    """弹性周期：每日检查，但持仓未满 min_hold 天不卖。

    min_hold=0 表示无限制（每日可换）
    fixed_period>0 表示只在固定周期检查（传统方式）
    """
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    slip = slip_bp / 1e4
    cash = 1.0
    hold: dict[int, dict] = {}
    nav = np.full(T, np.nan)
    nav[t0] = 1.0
    pend_sell, pend_buy = set(), []
    n_trades = 0
    last_check = t0

    for t in range(t0 + 1, t1 + 1):
        # ---- 执行（开盘）
        for j in list(pend_sell):
            if j in hold:
                px = op[t, j]
                if np.isfinite(px) and px > 0:
                    cash += hold[j]["sh"] * px * (1 - slip)
                    del hold[j]
                    n_trades += 1
        pend_sell = set()

        if pend_buy:
            slots = top_n - len(hold)
            if slots > 0 and cash > 1e-9:
                tot = cash + sum(hold[k]["sh"] * op[t, k]
                                 for k in hold if np.isfinite(op[t, k]))
                per = tot / top_n
                for j in pend_buy[:slots]:
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
            pend_buy = []

        # ---- 净值
        v = cash
        for j, h in hold.items():
            px = close[t, j]
            if np.isfinite(px):
                v += h["sh"] * px
        nav[t] = v

        # ---- 决定是否检查信号
        if fixed_period > 0:
            if (t - last_check) < fixed_period:
                continue
        last_check = t

        m = mom20[t]
        ok = np.isfinite(m) & np.isfinite(close[t]) & (m > 0)
        idx = np.flatnonzero(ok)
        if not len(idx):
            # 无合格标的 → 全部卖出（下一日开盘）
            pend_sell = set(hold.keys())
            continue
        idx = idx[np.argsort(-m[idx])]

        # ---- 计算目标持仓
        if fixed_period > 0:
            # 固定周期：直接取前 top_n，全部替换
            target = set(idx[:top_n].tolist())
            pend_sell = set(hold.keys()) - target
            pend_buy = [j for j in target if j not in hold]
        else:
            # 弹性周期：持仓满 min_hold 天才允许卖
            sellable = {j for j in hold if (t - hold[j]["day"]) >= min_hold}
            keepers = set(hold.keys()) - sellable      # 未满期，必须保留

            # 在剩余名额里选新标的
            need = top_n - len(keepers)
            if need > 0:
                cand = [j for j in idx if j not in keepers][:need]
                target = keepers | set(cand)
            else:
                target = keepers

            # 已满期且不在目标 → 卖
            pend_sell = sellable - target
            # 目标中不在持仓 → 买
            pend_buy = [j for j in target if j not in hold]

    s = pd.Series(nav, index=dates).dropna()
    if len(s) < 60:
        return None, n_trades
    p = qfm.performance(s, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in
            ("cagr", "sharpe", "max_dd", "calmar")}, n_trades


print("=" * 96)
print("弹性周期 vs 固定周期（修正 min_hold 实现后）")
print("=" * 96)
print(f"{'配置':<32}{'期间':<7}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'交易':>7}")
print("-" * 96)

configs = [
    ("固定20日 top2", dict(top_n=2, fixed_period=20)),
    ("固定20日 top3", dict(top_n=3, fixed_period=20)),
    ("固定10日 top2", dict(top_n=2, fixed_period=10)),
    ("弹性 min_hold=5 top2", dict(top_n=2, min_hold=5, fixed_period=0)),
    ("弹性 min_hold=10 top2", dict(top_n=2, min_hold=10, fixed_period=0)),
    ("弹性 min_hold=20 top2", dict(top_n=2, min_hold=20, fixed_period=0)),
    ("弹性 min_hold=5 top3", dict(top_n=3, min_hold=5, fixed_period=0)),
    ("弹性 min_hold=10 top3", dict(top_n=3, min_hold=10, fixed_period=0)),
    ("每日换仓 top2（无限制）", dict(top_n=2, min_hold=0, fixed_period=0)),
]

results = {}
for label, kw in configs:
    for pname, (a, b) in (("DEV", ("2018-01-01", "2021-12-31")),
                          ("OOS", ("2022-01-01", "2026-10-31"))):
        p, nt = run(a, b, **kw)
        if p is None:
            continue
        results[(label, pname)] = (p, nt)
        print(f"{label:<32}{pname:<7}{p['cagr']:>8.2%}{p['sharpe']:>9.3f}"
              f"{p['max_dd']:>9.2%}{nt:>7}")

print("\n" + "=" * 96)
print("汇总对比")
print("=" * 96)
print(f"{'配置':<32}{'DEV CAGR':>11}{'OOS CAGR':>11}{'OOS Sharpe':>12}{'OOS交易':>9}")
print("-" * 96)
for label, _ in configs:
    d = results.get((label, "DEV"))
    o = results.get((label, "OOS"))
    if d and o:
        print(f"{label:<32}{d[0]['cagr']:>10.2%}{o[0]['cagr']:>11.2%}"
              f"{o[0]['sharpe']:>12.3f}{o[1]:>9}")

# 找 OOS 最好的
oos = [(k[0], v) for k, v in results.items() if k[1] == "OOS"]
if oos:
    best = max(oos, key=lambda kv: kv[1][0]["cagr"])
    print(f"\n  OOS 最优: {best[0]}")
    print(f"    CAGR {best[1][0]['cagr']:.2%}, Sharpe {best[1][0]['sharpe']:.3f}, "
          f"MaxDD {best[1][0]['max_dd']:.2%}, 交易 {best[1][1]} 笔")

json.dump({f"{k[0]}|{k[1]}": {"perf": v[0], "trades": v[1]}
           for k, v in results.items()},
          open(ROOT / "build" / "eval" / "etf_flex_vs_fixed.json", "w"),
          ensure_ascii=False, indent=1, default=float)
print(f"\n→ build/eval/etf_flex_vs_fixed.json")
