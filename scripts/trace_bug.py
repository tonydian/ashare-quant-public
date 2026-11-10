# -*- coding: utf-8 -*-
"""追查 bug：为什么最简动量轮动会亏 29%、回撤 83%？

对照基准：
  全部股票等权 OOS = +0.52%
  全池等权     OOS = +0.99%
  而动量 top1  = -28.94%

差 30pp，绝不可能是"选股能力差"，必然是 bug。

可疑点：
  H1 买入时用了 open，但净值用 close，而 open/close 在某些 ETF 上量级不同
  H2 换仓时"卖"和"买"的顺序导致现金为负
  H3 某些 ETF 的价格数据有断层（上市前是 NaN，导致净值算错）
  H4 遗漏：买入后 hold[j]["sh"] 计算有误
  H5 净值 nav[t] 在某些天为 NaN，导致 cumprod 出错

最小复现：只买 510300，看看能不能复现买入持有的 7.45%
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")

z = np.load(ROOT / "build" / "etf" / "etf_ohlc_panel_v2.npz", allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
close = z["close"].astype(np.float64)
op = z["open"].astype(np.float64)
T, N = close.shape

print("=" * 88)
print("检查 1：只买 510300，能否复现买入持有")
print("=" * 88)
j = codes.index("510300")
t0 = int(dates.searchsorted(pd.Timestamp("2018-01-01")))
t1 = int(dates.searchsorted(pd.Timestamp("2021-12-31"), side="right")) - 1

buy = op[t0 + 1, j]
print(f"  买入价 {buy:.4f}（{dates[t0+1].date()} 开盘）")
sh = 1.0 / (buy * 1.0010)
nav = np.full(T, np.nan)
for t in range(t0 + 1, t1 + 1):
    px = close[t, j]
    nav[t] = sh * px if np.isfinite(px) else np.nan
s = pd.Series(nav, index=dates).dropna()
print(f"  末值 {s.iloc[-1]:.6f}，收益 {(s.iloc[-1]-1)*100:+.2f}%")

# 对照：close 直接算
sub = close[t0+1:t1+1, j]
ok = np.isfinite(sub)
print(f"  直接算: {sub[ok][-1]/sub[ok][0]-1:+.2%}")

print("\n" + "=" * 88)
print("检查 2：数据是否有断层（导致净值算错）")
print("=" * 88)
# 统计每只 ETF 在 OOS 期的 NaN 情况
oos0 = int(dates.searchsorted(pd.Timestamp("2022-01-01")))
oos1 = int(dates.searchsorted(pd.Timestamp("2026-10-31"), side="right")) - 1
print(f"  OOS 区间: {dates[oos0].date()} ~ {dates[oos1].date()}（{oos1-oos0+1} 天）")
print(f"\n{'代码':<8}{'名称':<12}{'有效天':>7}{'缺失率':>9}{'首日':>12}{'末日':>12}")
print("-" * 66)
bad = []
for j, c in enumerate(codes):
    s = close[oos0:oos1+1, j]
    ok = np.isfinite(s)
    miss = 1 - ok.mean()
    if miss > 0.05:
        first = dates[oos0:oos1+1][ok][0].date() if ok.any() else "-"
        last = dates[oos0:oos1+1][ok][-1].date() if ok.any() else "-"
        bad.append((c, miss))
        print(f"  {c:<8}{json.load(open(ROOT/'build/etf/etf_ohlc_meta_v2.json',encoding='utf-8')).get(c,''):<12}"
              f"{ok.sum():>7}{miss:>8.1%}{str(first):>12}{str(last):>12}")

print(f"\n  缺失 >5% 的标的: {len(bad)} 只")
if bad:
    print(f"  → 这些标的上市晚或退市早，若买入后变成 NaN，净值会出错")

print("\n" + "=" * 88)
print("检查 3：模拟最简轮动，逐笔打印")
print("=" * 88)
mom20 = pd.DataFrame(close).pct_change(20).to_numpy()

cash = 1.0
hold = {}
nav = np.full(T, np.nan)
nav[oos0] = 1.0
pend = None
trades = []

for t in range(oos0 + 1, min(oos0 + 60, oos1) + 1):
    if pend is not None:
        target = set(pend)
        for jj in list(hold):
            if jj not in target:
                px = op[t, jj]
                if np.isfinite(px) and px > 0:
                    cash += hold[jj]["sh"] * px * 0.9990
                    trades.append((dates[t].date(), "SELL", codes[jj],
                                   f"{px:.3f}", f"cash={cash:.4f}"))
                    del hold[jj]
        if len(hold) < 1:
            for jj in list(target):
                px = op[t, jj]
                if not (np.isfinite(px) and px > 0):
                    continue
                b = cash
                if b <= 1e-9:
                    break
                sh = b / (px * 1.0010)
                cash -= sh * px * 1.0010
                hold[jj] = {"sh": sh}
                trades.append((dates[t].date(), "BUY", codes[jj],
                               f"{px:.3f}", f"sh={sh:.4f}"))
                break
        pend = None

    v = cash
    for jj, h in hold.items():
        px = close[t, jj]
        if np.isfinite(px):
            v += h["sh"] * px
    nav[t] = v

    m = mom20[t].copy()
    ok = np.isfinite(m) & np.isfinite(close[t]) & (m > 0)
    idx = np.flatnonzero(ok)
    pend = set(np.argsort(-m[idx])[:1].tolist()) if len(idx) else set()

print("  前 40 个交易日交易记录:")
for tr in trades[:40]:
    print(f"    {tr[0]} {tr[1]:<5} {tr[2]:<8} {tr[3]:>10}  {tr[4]}")

s2 = pd.Series(nav, index=dates).dropna()
print(f"\n  60 天后净值: {s2.iloc[-1]:.4f}")
print(f"  期间收益: {(s2.iloc[-1]-1)*100:+.2f}%")
