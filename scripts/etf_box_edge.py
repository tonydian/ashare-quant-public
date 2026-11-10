# -*- coding: utf-8 -*-
"""关键检验：ETF 箱体突破本身有没有 alpha？（不看卖出规则）

这一步是决定性的。如果不做这一步而直接放宽参数，就是过拟合。

方法：
  1. 用当前（严）参数取所有确认信号，看突破后 N 日收益
  2. 对比同期"随便买一只 ETF"的收益
  3. 严格按 DEV/OOS 分开看
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import etf_box_breakout as E  # noqa: E402

P = E.load_panel()
F = E.build_features(P)
wc, wv, ma = F["wc"], F["wv"], F["ma"]
wl, W = F["wl"], F["W"]
close, openp = P["close"], P["open"]
dates = P["dates"]
T = P["T"]

HOR = [5, 10, 20, 40]


def collect(band=0.10, bw=10, vol_mult=1.5, ma_weeks=52):
    """收集确认信号 + 未来收益。"""
    maX = pd.DataFrame(wc).rolling(ma_weeks,
                                   min_periods=max(20, ma_weeks // 2)).mean().to_numpy()
    recs = []
    for wi in range(bw + 2, W - 1):
        base = maX[wi - 1]
        ok = np.isfinite(base) & (base > 0) & np.isfinite(wc[wi - 1])
        if ok.sum() < 3:
            continue
        lo, hi = base * (1 - band), base * (1 + band)
        for j in np.flatnonzero(ok):
            k = 0
            for back in range(1, min(45, wi) + 1):
                c = wc[wi - back, j]
                if not np.isfinite(c) or not (lo[j] <= c <= hi[j]):
                    break
                k += 1
            if k < bw:
                continue
            top = float(np.nanmax(wc[wi - k:wi, j]))
            if not np.isfinite(top) or not np.isfinite(wc[wi, j]):
                continue
            if not (wc[wi, j] > top):
                continue
            avg = np.nanmean(wv[wi - E.PARAMS["vol_win"]:wi, j])
            if not (np.isfinite(avg) and avg > 0):
                continue
            if not (wv[wi, j] > avg * vol_mult):
                continue
            # 确认
            if not (np.isfinite(wc[wi + 1, j]) and wc[wi + 1, j] > wc[wi, j]):
                continue
            t_exec = int(wl[wi + 1]) + 1
            if t_exec >= T:
                continue
            bp = openp[t_exec, j]
            if not (np.isfinite(bp) and bp > 0):
                continue
            row = {"t": t_exec, "j": j, "date": dates[t_exec],
                   "name": P["meta"].get(P["codes"][j], P["codes"][j])}
            for h in HOR:
                t2 = t_exec + h
                row[f"r{h}"] = ((close[t2, j] / bp - 1)
                                if t2 < T and np.isfinite(close[t2, j]) else np.nan)
            recs.append(row)
    return pd.DataFrame(recs)


def PARAMS_vol(x):
    return 0


def random_baseline(n_iter=2000, seed=7):
    """同期随机买入 ETF 的 N 日收益（对照基准）。"""
    rng = np.random.default_rng(seed)
    out = {h: [] for h in HOR}
    start = int(dates.searchsorted(pd.Timestamp("2017-01-03")))
    for _ in range(n_iter):
        t = int(rng.integers(start, T - max(HOR) - 5))
        j = int(rng.integers(0, P["N"]))
        bp = openp[t, j]
        if not (np.isfinite(bp) and bp > 0):
            continue
        for h in HOR:
            t2 = t + h
            if t2 < T and np.isfinite(close[t2, j]):
                out[h].append(close[t2, j] / bp - 1)
    return {h: (np.mean(v) if v else np.nan) for h, v in out.items()}


print("=" * 92)
print("ETF 箱体突破：信号有效性检验（不含卖出规则）")
print("=" * 92)

df = collect()
print(f"\n确认信号数: {len(df)}")
if len(df) < 10:
    print("信号太少，无法统计")
    sys.exit(0)

print(f"\n{'持有期':>8}{'样本':>6}{'均值':>10}{'中位':>10}{'胜率':>9}")
print("-" * 46)
for h in HOR:
    v = df[f"r{h}"].dropna()
    if len(v) < 10:
        continue
    print(f"{h:>7}日{len(v):>6}{v.mean()*100:>+9.2f}%"
          f"{v.median()*100:>+9.2f}%{(v>0).mean()*100:>8.1f}%")

bl = random_baseline()
print(f"\n{'持有期':>8}{'随机基准':>12}{'突破信号':>12}{'超额':>10}")
print("-" * 46)
for h in HOR:
    v = df[f"r{h}"].dropna()
    if len(v) < 10:
        continue
    b = bl[h]
    print(f"{h:>7}日{b*100:>+11.2f}%{v.mean()*100:>+11.2f}%"
          f"{(v.mean()-b)*100:>+9.2f}pp")

print("\n" + "=" * 92)
print("DEV / OOS 拆分")
print("=" * 92)
df["period"] = np.where(df["date"] <= pd.Timestamp(E.DEV_END), "DEV", "OOS")
for per, g in df.groupby("period"):
    print(f"\n  [{per}] {len(g)} 个信号")
    for h in (10, 20):
        v = g[f"r{h}"].dropna()
        if len(v) < 5:
            continue
        print(f"    {h}日: 均值 {v.mean()*100:+.2f}% | "
              f"中位 {v.median()*100:+.2f}% | 胜率 {(v>0).mean()*100:.1f}%")

print("\n" + "=" * 92)
print("按标的分类（哪些 ETF 的突破有效）")
print("=" * 92)
for name, g in df.groupby("name"):
    if len(g) < 3:
        continue
    v = g["r20"].dropna()
    if len(v) < 3:
        continue
    print(f"  {name:<10} {len(g):>3} 次 | 20日均值 {v.mean()*100:>+6.2f}% "
          f"| 胜率 {(v>0).mean()*100:>5.1f}%")
