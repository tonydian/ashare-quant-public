# -*- coding: utf-8 -*-
"""根因分析：为什么 OOS 期间所有配置都是负收益？

关键数据（消融实验）：
  完整版        OOS -8.36%
  关掉所有过滤  OOS -11.18%
  只买国债      OOS -16.32%  ← 这个最可疑！

"只买国债 top_n=1" 怎么会是 -16%？国债 OOS 涨了 19%！

必然有 bug。可能原因：
  H1 国债 ETF 的 price 数据有问题（前复权 vs 后复权混用）
  H2 债券 ETF 的 amount 单位与其他不同，导致过滤失效
  H3 备选/买入逻辑错误
  H4 close 数据在债券上是"净值"而非"交易价"，导致收益计算错误
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

z = np.load(ROOT / "build" / "etf" / "etf_ohlc_panel_v2.npz", allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
meta = json.load(open(ROOT / "build" / "etf" / "etf_ohlc_meta_v2.json",
                      encoding="utf-8"))
close = z["close"].astype(np.float64)
OOS0 = int(dates.searchsorted(pd.Timestamp("2022-01-01")))
OOS1 = int(dates.searchsorted(pd.Timestamp("2026-10-31"), side="right")) - 1

print("=" * 90)
print("检查 1：国债 ETF 的实际表现（OOS 期间）")
print("=" * 90)
for code in ("511010", "511260", "511220", "511020"):
    if code not in codes:
        continue
    j = codes.index(code)
    s = close[OOS0:OOS1, j]
    ok = np.isfinite(s)
    if ok.sum() < 100:
        print(f"  {code}: 数据不足")
        continue
    v = s[ok]
    print(f"\n  {code} {meta.get(code,'')}")
    print(f"    首 {v[0]:.3f} → 末 {v[-1]:.3f}，涨幅 {v[-1]/v[0]-1:+.2%}")
    print(f"    范围 {v.min():.3f} ~ {v.max():.3f}")
    # 检查是否有异常跳空
    r = pd.Series(v).pct_change().dropna()
    print(f"    日收益 std {r.std()*100:.3f}%")
    big = r[r.abs() > 0.05]
    if len(big):
        print(f"    ⚠️ 单日 |涨跌|>5% 共 {len(big)} 次（国债不该有）")
        print(f"       最大 {r.abs().max()*100:.2f}%")

print("\n" + "=" * 90)
print("检查 2：债券 ETF 的 open 与 close 关系（是否有异常）")
print("=" * 90)
op = z["open"].astype(np.float64)
for code in ("511010", "511260"):
    if code not in codes:
        continue
    j = codes.index(code)
    o = op[OOS0:OOS1, j]
    c = close[OOS0:OOS1, j]
    ok = np.isfinite(o) & np.isfinite(c) & (c > 0)
    if ok.sum() < 100:
        continue
    ratio = o[ok] / c[ok]
    print(f"\n  {code}: open/close 比值 {ratio.min():.4f} ~ {ratio.max():.4f}")
    print(f"    中位 {np.median(ratio):.4f}")
    if ratio.max() > 1.5 or ratio.min() < 0.5:
        print(f"    🔴 异常！open 与 close 量级不匹配")

print("\n" + "=" * 90)
print("检查 3：成交额单位（债券 vs 行业）")
print("=" * 90)
amt = z["amount"].astype(np.float64)
for code in ("511010", "510300", "512880"):
    if code not in codes:
        continue
    j = codes.index(code)
    a = amt[OOS0:OOS1, j]
    ok = np.isfinite(a)
    if ok.sum() < 10:
        continue
    print(f"  {code} {meta.get(code,''):<10}: 中位 {np.median(a[ok]):>12,.0f}  "
          f"最近20日均 {np.nanmean(a[-20:]):>12,.0f}")

print("\n" + "=" * 90)
print("检查 4：模拟'只买国债'最简单情形")
print("=" * 90)
# 如果从 2022-01-01 买入 511260 持有到 2026-10-31，收益多少？
j = codes.index("511260") if "511260" in codes else None
if j is not None:
    s = close[OOS0:OOS1, j]
    ok = np.isfinite(s)
    v = s[ok]
    print(f"  511260 买入持有: {v[0]:.4f} → {v[-1]:.4f} = {v[-1]/v[0]-1:+.2%}")
    print(f"  年化: {(v[-1]/v[0])**(252/len(v))-1:+.2%}")

print("\n" + "=" * 90)
print("结论")
print("=" * 90)
print("""
  如果国债 ETF 本身收益正常（+19%），但"只买国债"策略亏 -16%，
  说明策略逻辑有严重 bug（不是策略无效，而是实现错误）。

  最可能：open/close 混用、或买入价与卖出价口径不一致。
""")
