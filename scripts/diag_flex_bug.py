# -*- coding: utf-8 -*-
"""定位 bug：单只 ETF 买入持有 vs 策略"只买国债"的差异。

数据正常（国债 +22%），但策略 -16%，差 38pp。
必须找到 bug。

测试方法：最小可复现案例
  · 只买 511260，持有到底
  · 不加任何过滤、不止损、不换仓
  · 应该得到 +22% 左右
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

import etf_flex_dual_momentum as E  # noqa: E402

P = E.load_panel()
codes, meta = P["codes"], P["meta"]
j260 = codes.index("511260") if "511260" in codes else None
print(f"\n511260 index = {j260}, name = {meta.get('511260')}")

OOS0 = int(P["dates"].searchsorted(pd.Timestamp(E.OOS[0])))
OOS1 = int(P["dates"].searchsorted(pd.Timestamp(E.OOS[1]), side="right")) - 1

print("\n" + "=" * 90)
print("最小案例：只买 511260，无任何规则")
print("=" * 90)
buy_px = P["open"][OOS0 + 1, j260]
sell_px = P["close"][OOS1, j260]
print(f"  买入价（第 1 天开盘）: {buy_px:.4f}")
print(f"  卖出价（最后一天收盘）: {sell_px:.4f}")
print(f"  收益: {sell_px/buy_px - 1:+.2%}")

print("\n" + "=" * 90)
print("问题定位：为什么策略会亏损？")
print("=" * 90)

# 手动模拟最小逻辑
cash = 1.0
hold = {}
shares = 0
navs = []
for t in range(OOS0 + 1, OOS1 + 1):
    # 第 1 天买入
    if t == OOS0 + 1 and shares == 0:
        px = P["open"][t, j260]
        if np.isfinite(px) and px > 0:
            shares = cash / px
            cash = 0.0
    v = cash + shares * (P["close"][t, j260] if np.isfinite(P["close"][t, j260]) else 0)
    navs.append(v)

s = pd.Series(navs, index=P["dates"][OOS0+1:OOS1+1])
print(f"\n  手动模拟结果：")
print(f"    起始 {s.iloc[0]:.6f} → 末 {s.iloc[-1]:.6f}")
print(f"    收益 {(s.iloc[-1]/s.iloc[0]-1)*100:+.2f}%")

print("\n" + "=" * 90)
print("对比：策略'只买国债 top_n=1'的实际逻辑")
print("=" * 90)

# 检查 top_n=1 时会选什么
mom1 = E.simple_mom(P["close"], E.PARAMS["mom1_n"])
mom2 = E.slope_mom(P["close"], E.PARAMS["mom2_n"])
ma = E.rolling_mean(P["close"], E.PARAMS["ma_filter"])

print("  在 OOS 期随机抽几天，看 top_n=1 选中的标的：")
for t in (OOS0 + 10, OOS0 + 100, OOS0 + 500, OOS1 - 10):
    m1, m2, mf = mom1[t], mom2[t], ma[t]
    ok = (np.isfinite(m1) & np.isfinite(m2) & np.isfinite(mf)
          & (P["close"][t] > 0))
    ok &= (m1 > E.PARAMS["mom_floor"]) & (m1 < E.PARAMS["mom_ceil"])
    ok &= (m2 > E.PARAMS["mom2_floor"])
    if E.PARAMS["ma_filter"] > 0:
        ok &= (P["close"][t] > mf)
    idx = np.flatnonzero(ok)
    if len(idx):
        idx = idx[np.argsort(-m1[idx])]
        top = idx[:3]
        names = [f"{codes[k]}{meta.get(codes[k],'')}" for k in top]
        print(f"    {P['dates'][t].date()}: {len(idx)} 个合格 → {names}")
    else:
        print(f"    {P['dates'][t].date()}: 无合格标的")

print("\n" + "=" * 90)
print("检查：国债 ETF 能通过所有过滤吗？")
print("=" * 90)
t = OOS0 + 100
m1 = mom1[t, j260]
m2 = mom2[t, j260]
mf = ma[t, j260]
c = P["close"][t, j260]
print(f"  {P['dates'][t].date()}:")
print(f"    mom1 = {m1:.4f}  (需 >{E.PARAMS['mom_floor']} 且 <{E.PARAMS['mom_ceil']})")
print(f"    mom2 = {m2:.4f}  (需 >{E.PARAMS['mom2_floor']})")
print(f"    close = {c:.4f}, MA60 = {mf:.4f}  (需 close > MA)")
print(f"    → 通过: {m1 > 0 and m2 > 0 and c > mf}")

print("\n  结论：")
print(f"    国债 mom1={m1:.4f} → {'✅' if m1>0 else '❌ 不满足下限'}")
print(f"    国债 mom2={m2:.4f} → {'✅' if m2>0 else '❌ 不满足下限'}")
print(f"    close vs MA: {'✅' if c>mf else '❌ 跌破均线'}")
print()
print("""
  如果国债动量很低（长期缓慢上涨，20日动量可能接近 0 或为负），
  那它就永远排不上 top → "只买国债"配置实际变成"频繁买其他标的"
  → 这就是 -16% 的原因（不是 bug，是配置理解错误）
""")
