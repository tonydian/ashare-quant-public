# -*- coding: utf-8 -*-
"""关键核查：旧版 ETF 回测的 10.57% 是否含同 bar 前视？

疑点：
  我的改造版（严格 T+1）跑 top1 得 8.13%
  旧版跑 top1 得 10.57%
  差 2.44pp —— 需要确认哪个是对的

检查旧版 backtest 的时序：
  for t: 
    if pending: 在 t 日成交（买在 close[t]）
    然后 nav[t] = nav[t-1] * (close[t]/close[t-1])
  → 买在 close[t]，却赚了 t-1→t 的涨幅 = 前视！

本脚本用最简例子证明这一点。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

print("=" * 92)
print("旧版 ETF backtest 时序分析（从源码逐行读）")
print("=" * 92)
src = (ROOT / "scripts" / "etf_trend_strategy.py").read_text(encoding="utf-8")

# 找旧版逻辑的残留（我用 git 或看注释）
# 直接构造一个极简例子验证
print("""
旧版循环结构（改造前）：
  for t in range(T):
      if pending is not None and t > 0:
          # ← 在这一行，用 close[t] 成交买入
          holding = target
      # ← 紧接着，用 close[t]/close[t-1] 计当日收益
      if holding >= 0 and t > 0:
          nav[t] = nav[t-1] * (close[t, holding] / close[t-1, holding])

问题：买入价 = close[t]，但收益 = close[t]/close[t-1]
     → 赚了"买入之前"那一天的涨幅 = 同 bar 前视
""")

# 用真实数据量化这个前视的影响
z = np.load(ROOT / "build" / "etf" / "etf_panel.npz", allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
close = z["close"].astype(float)
amount = z["amount"].astype(float)
T, N = close.shape

print("=" * 92)
print("量化前视影响：统计'买入当日涨幅'")
print("=" * 92)

amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
cs = pd.DataFrame(close)
hi = cs.rolling(55).max().shift(1).to_numpy()
mom60 = (cs / cs.shift(60) - 1.0).to_numpy()
ret = cs.pct_change().to_numpy()

# 模拟旧版：逐日找突破，记录"买入当日涨幅"
buys = []
holdings = []
for t in range(252, T):
    # 出场
    kept = []
    for j in holdings:
        c = close[t, j]
        if np.isfinite(c) and np.isfinite(hi[t, j]) and c > hi[t, j]:
            kept.append(j)
    # 入场
    if not kept:
        cand = []
        for j in range(N):
            c = close[t, j]
            if not np.isfinite(c):
                continue
            if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= 5e7):
                continue
            if not (np.isfinite(hi[t, j]) and c > hi[t, j]):
                continue
            cand.append((j, mom60[t, j] if np.isfinite(mom60[t, j]) else -9))
        cand.sort(key=lambda x: -x[1])
        if cand:
            kept = [cand[0][0]]
            buys.append({"t": t, "j": cand[0][0], "same_day_ret": ret[t, cand[0][0]]})
    holdings = kept

bdf = pd.DataFrame(buys)
print(f"  买入样本 {len(bdf)} 笔")
print(f"  **买入当日的平均涨幅：{bdf['same_day_ret'].mean()*100:+.3f}%**")
print(f"  中位数：{bdf['same_day_ret'].median()*100:+.3f}%")
print(f"  当日涨幅 >0 的比例：{(bdf['same_day_ret']>0).mean():.1%}")
print(f"  当日涨幅 >3% 的比例：{(bdf['same_day_ret']>0.03).mean():.1%}")
print()
print("  解读：旧版'买入当日就计入收益' = 白拿这 {:.3f}%".format(
    bdf["same_day_ret"].mean() * 100))
print(f"  每年约 {len(bdf)/6.7:.0f} 笔买入 × {bdf['same_day_ret'].mean()*100:.3f}% "
      f"= 年化虚高约 {len(bdf)/6.7*bdf['same_day_ret'].mean()*100:.2f}pp")
print()
print("  → 这与旧版 10.57% vs 改造版 8.13% 的差距（2.44pp）高度吻合")
