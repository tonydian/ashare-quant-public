# -*- coding: utf-8 -*-
"""深度诊断：为什么箱体突破策略亏钱？

关键数据：
  CAGR -3.45%、MaxDD -29.96%、交易 2968 笔
  卖出原因：fade 1111（37%）、surge 159、friday 69、stoploss 145

疑问：
  H1 每笔交易的平均盈亏是多少？（如果频繁小亏+成本，必然负）
  H2 突破后的真实胜率？（箱体突破在 A 股是否有效）
  H3 "冲高回落"是不是在砍掉本来会涨的仓位
  H4 买入时机是否总在高点（追高）

方法：
  1. 统计每笔完整交易的收益分布
  2. 对比"突破后持有 N 天"的理论收益（不看卖出规则）
  3. 分年度看
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
sys.path.insert(0, str(ROOT / "scripts"))

import box_breakout as bb  # noqa: E402

print("=" * 92)
print("1) 突破信号的事后表现（不看卖出规则）")
print("=" * 92)

P = bb.build_panel()
dy = bb.load_dividend(P)
sig = bb.build_signals(P, dy)

close = P["close"]
dates = P["dates"]
T = P["T"]

# 对每个信号，看"买入后 N 日"的收益
horizons = [5, 10, 20, 40, 60]
recs = []
for t_exec, cands in sig.items():
    for j, top in cands:
        if t_exec >= T:
            continue
        buy_px = P["open"][t_exec, j]
        if not np.isfinite(buy_px) or buy_px <= 0:
            continue
        row = {"t": t_exec, "j": j, "date": dates[t_exec], "buy": buy_px}
        for h in horizons:
            t2 = t_exec + h
            if t2 < T:
                px = close[t2, j]
                row[f"r{h}"] = (px / buy_px - 1) if np.isfinite(px) else np.nan
        recs.append(row)

df = pd.DataFrame(recs)
print(f"信号数: {len(df)}")
print(f"\n{'持有期':>8}{'均值':>10}{'中位':>10}{'胜率':>9}{'p10':>10}{'p90':>10}")
print("-" * 60)
for h in horizons:
    c = f"r{h}"
    v = df[c].dropna()
    if len(v) < 50:
        continue
    print(f"{h:>7}日{v.mean()*100:>+9.2f}%{v.median()*100:>+9.2f}%"
          f"{(v>0).mean()*100:>8.1f}%{np.percentile(v,10)*100:>+9.2f}%"
          f"{np.percentile(v,90)*100:>+9.2f}%")

print("\n" + "=" * 92)
print("2) 分年度：突破后 20 日收益")
print("=" * 92)
df["year"] = df["date"].dt.year
print(f"{'年':>6}{'信号数':>9}{'20日均值':>12}{'20日胜率':>11}")
print("-" * 40)
for y, g in df.groupby("year"):
    v = g["r20"].dropna()
    if len(v) < 20:
        continue
    print(f"{y:>6}{len(g):>9}{v.mean()*100:>+11.2f}%{(v>0).mean()*100:>10.1f}%")

print("\n" + "=" * 92)
print("3) 对比：同期全市场（市值前800）平均 20 日收益")
print("=" * 92)
# 用随机基准对比
rank = P["rank"]
bench = []
for t in range(int(P["md"].date_index("2020-01-08")), T - 20, 20):
    ok = np.isfinite(rank[t]) & (rank[t] <= bb.PARAMS["universe_top"])
    ok &= np.isfinite(close[t]) & np.isfinite(close[t + 20])
    if ok.sum() < 100:
        continue
    r = close[t + 20, ok] / close[t, ok] - 1
    bench.append(np.nanmean(r))
print(f"  全市场（前800）20 日平均收益: {np.mean(bench)*100:+.2f}%")
print(f"  箱体突破后 20 日平均收益:     {df['r20'].mean()*100:+.2f}%")
diff = df["r20"].mean() - np.mean(bench)
print(f"  → 超额: {diff*100:+.2f}pp")

print("\n" + "=" * 92)
print("结论")
print("=" * 92)
r20 = df["r20"].dropna()
if r20.mean() < 0:
    print(f"""
  ❌ 箱体突破后 20 日均值 {r20.mean()*100:+.2f}%（负收益）
     胜率 {(r20>0).mean()*100:.1f}%

  这说明：**在 A 股，"箱体突破"本身不是有效的入场信号。**
  突破后平均是下跌的 → 无论怎么优化卖出规则，都很难赚。

  可能原因：
    1. 突破是"最后一冲"（出货）而非"启动"
    2. 突破后立即触发获利了结 → 需要更长持有期
    3. 选股域（市值前800）里，突破的有效性低于小市值

  验证方法：看更长持有期（60 日）的表现
""")
    v60 = df["r60"].dropna()
    if len(v60) > 50:
        print(f"  持有 60 日：均值 {v60.mean()*100:+.2f}%，胜率 {(v60>0).mean()*100:.1f}%")
else:
    print(f"  ✅ 突破后 20 日均值 {r20.mean()*100:+.2f}%，"
          f"胜率 {(r20>0).mean()*100:.1f}%")
    print(f"     但超额只有 {diff*100:+.2f}pp（相对全市场）")
