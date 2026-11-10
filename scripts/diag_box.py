# -*- coding: utf-8 -*-
"""诊断：为什么箱体突破策略净值几乎不动且持续下跌？

现象：
  ann_vol 2.79%（极低）、CAGR -2.43%、每年稳定 -2~3%
  卖出原因：fade 588 次、surge 43 次、friday 3 次、stoploss 2 次

疑问：
  H1 仓位极低（大部分时间空仓）→ 波动才会这么小
  H2 "冲高回落"条件过于宽松 → 几乎每天都触发卖出
  H3 买入从未真正发生（或买入量极小）
  H4 净值计算有 bug（现金没赚利息，但成本照扣）
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

# 直接导入策略模块
sys.path.insert(0, str(ROOT / "scripts"))
import box_breakout as bb

print("=" * 92)
print("诊断：仓位与交易行为")
print("=" * 92)

P = bb.build_panel()
dy = bb.load_dividend(P)
sig = bb.build_signals(P, dy)

# 检查信号分布
print(f"\n信号日数: {len(sig)}")
if sig:
    first = sorted(sig.keys())[:5]
    for d in first:
        print(f"  {P['dates'][d].date()}: {len(sig[d])} 个候选")
    print(f"  总候选数: {sum(len(v) for v in sig.values())}")

# 手动追踪仓位变化
md = P["md"]
dates = P["dates"]
T, N = P["T"], P["N"]
close, openp, high = P["close"], P["open"], P["high"]
wl = set(int(x) for x in P["weekly_last"])
start = int(md.date_index("2020-01-08"))

holdings: dict[int, dict] = {}
cash = 1.0
nav = np.ones(T)
nav[:start] = np.nan
pending_buy = None
pending_sell = None
cost = bb.PARAMS["cost_bp"] / 1e4

stats = {"days_hold": 0, "days_flat": 0, "max_hold": 0,
         "buy_attempt": 0, "buy_ok": 0, "sell": 0}
holds_hist = []

for t in range(start + 1, T):
    # 执行卖
    if pending_sell:
        for j in list(pending_sell):
            if j in holdings:
                px = openp[t, j]
                if np.isfinite(px) and px > 0:
                    cash += holdings[j]["shares"] * px * (1 - cost / 2)
                    del holdings[j]
                    stats["sell"] += 1
        pending_sell = None
    # 执行买
    if pending_buy:
        stats["buy_attempt"] += 1
        slots = bb.PARAMS["max_pos"] - len(holdings)
        total_now = cash + sum(
            holdings[j2]["shares"] * openp[t, j2]
            for j2 in holdings if np.isfinite(openp[t, j2]))
        per = total_now / bb.PARAMS["max_pos"]
        for j, top in pending_buy[:max(0, slots)]:
            px = openp[t, j]
            if not np.isfinite(px) or px <= 0:
                continue
            budget = min(per, cash)
            if budget <= 1e-9:
                break
            sh = budget / (px * (1 + cost / 2))
            cash -= sh * px * (1 + cost / 2)
            holdings[j] = {"shares": sh, "cost": px, "box_top": top}
            stats["buy_ok"] += 1
        pending_buy = None

    val = cash + sum(holdings[j]["shares"] * close[t, j]
                     for j in holdings if np.isfinite(close[t, j]))
    nav[t] = val
    holds_hist.append(len(holdings))
    if holdings:
        stats["days_hold"] += 1
        stats["max_hold"] = max(stats["max_hold"], len(holdings))
    else:
        stats["days_flat"] += 1

    # 决策（收盘）
    sells = set()
    for j, h in holdings.items():
        c = close[t, j]
        pc = close[t - 1, j]
        hi = high[t, j]
        if not (np.isfinite(c) and np.isfinite(pc) and pc > 0):
            continue
        chg = c / pc - 1
        hi_chg = (hi / pc - 1) if np.isfinite(hi) else chg
        if chg > bb.PARAMS["sell_surge"]:
            sells.add(j); continue
        if hi_chg > bb.PARAMS["sell_fade_high"] and chg < bb.PARAMS["sell_fade_close"]:
            sells.add(j); continue
        top = h.get("box_top")
        if top is not None and np.isfinite(top) and c < top:
            sells.add(j); continue
        if t in wl and c > h["cost"] * (1 + bb.PARAMS["sell_fri_profit"]):
            sells.add(j)

    for j in list(sells):
        if t in wl and j in holdings:
            cash += holdings[j]["shares"] * close[t, j] * (1 - cost / 2)
            del holdings[j]
            stats["sell"] += 1

    if t in sig and len(holdings) < bb.PARAMS["max_pos"]:
        cand = [(j, top) for j, top in sig[t] if j not in holdings]
        if cand:
            pending_buy = cand

print("\n" + "=" * 92)
print("统计")
print("=" * 92)
tot = stats["days_hold"] + stats["days_flat"]
print(f"  有持仓天数: {stats['days_hold']} / {tot} = {stats['days_hold']/tot:.1%}")
print(f"  空仓天数:   {stats['days_flat']} / {tot} = {stats['days_flat']/tot:.1%}")
print(f"  最大同时持仓: {stats['max_hold']} 只")
print(f"  买入尝试: {stats['buy_attempt']} 次，成功 {stats['buy_ok']} 笔")
print(f"  卖出: {stats['sell']} 笔")
print()
h = np.array(holds_hist)
print(f"  平均持仓数: {h.mean():.2f}")
print(f"  持仓数分布: {dict(zip(*np.unique(h, return_counts=True)))}")

print("\n" + "=" * 92)
print("结论")
print("=" * 92)
if h.mean() < 2:
    print(f"""
  ❌ 平均持仓仅 {h.mean():.2f} 只 —— 仓位极低！

  这解释了为什么波动只有 2.79%：
    大部分时间空仓，钱躺在现金里（回测里现金不赚利息）
    但买入卖出都要付 25bp 成本 → 持续小幅亏损

  根因（需修）：
    1. 信号太少（280 个信号日，但每次只有几个候选）
    2. "冲高回落"条件过于敏感 → 588 次触发，持仓刚建立就卖
    3. 箱体突破后立即触发"跌破上沿止损"→ 反复买了卖

  注意：sell_reasons 里 stoploss 只有 2 次，
        但 fade 588 次 —— 说明"冲高回落"是主要杀手
""")
elif h.mean() < 10:
    print(f"  🟡 平均持仓 {h.mean():.2f} 只，仓位偏低")
else:
    print(f"  ✅ 平均持仓 {h.mean():.2f} 只，仓位正常")
