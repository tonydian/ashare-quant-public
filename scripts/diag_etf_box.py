# -*- coding: utf-8 -*-
"""诊断：为什么 ETF 箱体突破信号这么少（76 笔 / 2381 天）？

现象：
  · 平均仓位只有 5%
  · DEV 期 5 年只有 26 笔交易
  · 结果 CAGR -0.8%，但这其实是"几乎全空仓 + 国债收益"的结果

关键问题：
  Q1 是"箱体条件太严"还是"突破+放量条件太严"？
  Q2 ETF 的 MA52 ±10% 箱体是否太窄/太宽？
  Q3 放宽哪一条最有效（但不能过拟合）？

方法：逐一放松单个条件，看信号量变化（这是诊断，不是调参）
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


def count(ma_weeks=52, band=0.10, bw=10, vol_mult=1.5, vol_win=10,
          confirm=True):
    """统计满足条件（不含确认）的候选数。"""
    maX = pd.DataFrame(wc).rolling(ma_weeks, min_periods=max(20, ma_weeks//2)).mean().to_numpy()
    n_box = n_break = n_conf = 0
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
            n_box += 1
            top = float(np.nanmax(wc[wi - k:wi, j]))
            if not np.isfinite(top) or not (wc[wi, j] > top):
                continue
            avg = np.nanmean(wv[wi - vol_win:wi, j])
            if not (np.isfinite(avg) and avg > 0):
                continue
            if wv[wi, j] > avg * vol_mult:
                n_break += 1
                if np.isfinite(wc[wi + 1, j]) and wc[wi + 1, j] > wc[wi, j]:
                    n_conf += 1
    return n_box, n_break, n_conf


print("=" * 92)
print("基准（当前参数）")
print("=" * 92)
b = count()
print(f"  箱体 {b[0]:,} | 突破 {b[1]:,} | 确认 {b[2]:,}")

print("\n" + "=" * 92)
print("A. 放宽箱体宽度（MA52 ±X%）")
print("=" * 92)
for band in (0.10, 0.15, 0.20, 0.25):
    r = count(band=band)
    print(f"  ±{band*100:.0f}%: 箱体 {r[0]:>7,} | 突破 {r[1]:>5} | 确认 {r[2]:>4}")

print("\n" + "=" * 92)
print("B. 放宽箱体周数（≥N 周）")
print("=" * 92)
for bw in (10, 8, 6, 5):
    r = count(bw=bw)
    print(f"  ≥{bw} 周: 箱体 {r[0]:>7,} | 突破 {r[1]:>5} | 确认 {r[2]:>4}")

print("\n" + "=" * 92)
print("C. 放宽放量倍数")
print("=" * 92)
for vm in (2.0, 1.5, 1.2, 1.0):
    r = count(vol_mult=vm)
    print(f"  ×{vm}: 箱体 {r[0]:>7,} | 突破 {r[1]:>5} | 确认 {r[2]:>4}")

print("\n" + "=" * 92)
print("D. 放宽均线周期（MA 用更短的，箱体更贴近趋势）")
print("=" * 92)
for mw in (52, 40, 30, 20):
    r = count(ma_weeks=mw)
    print(f"  MA{mw}: 箱体 {r[0]:>7,} | 突破 {r[1]:>5} | 确认 {r[2]:>4}")

print("\n" + "=" * 92)
print("E. 组合放宽（温和，不算过拟合）")
print("=" * 92)
combos = [
    ("±15% + ≥8周 + ×1.2", dict(band=0.15, bw=8, vol_mult=1.2)),
    ("±20% + ≥6周 + ×1.2", dict(band=0.20, bw=6, vol_mult=1.2)),
    ("±15% + ≥10周 + ×1.2 + MA40", dict(band=0.15, bw=10, vol_mult=1.2, ma_weeks=40)),
    ("±20% + ≥8周 + ×1.0 + MA30", dict(band=0.20, bw=8, vol_mult=1.0, ma_weeks=30)),
]
for name, kw in combos:
    r = count(**kw)
    print(f"  {name:<30} 确认 {r[2]:>4} 个")

print("\n" + "=" * 92)
print("关键判断")
print("=" * 92)
print(f"""
  当前参数下 10 年只有 {b[2]} 个确认信号 → 平均每年 {b[2]/10:.1f} 个
  ETF 池只有 38 只，且 10 个持仓位 → 信号密度太低，无法形成策略。

  ⚠️ 但注意：放宽参数属于"调参"，必须只在 DEV 期决定，
     且要在 OOS 期验证。否则就是过拟合。
""")
