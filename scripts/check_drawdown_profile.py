# -*- coding: utf-8 -*-
"""回答：市场整体下跌时策略会不会越抄越跌？有没有战术性空仓？
统计主回测的仓位/风控状态、熊市年份与压力期表现、与不降仓版本的对比。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import qf.config as C  # noqa: E402

run = ROOT / "build" / "runs" / "main"
nav = pd.read_parquet(run / "nav.parquet")["nav"]
nav.index = pd.to_datetime(nav.index)
ex = pd.read_parquet(run / "exposures.parquet")
ex.index = pd.to_datetime(ex.index)
stress = pd.read_csv(run / "stress.csv")

print("=" * 70)
print("1) 主回测：仓位与风控状态分布（2020-01-08 ~ %s）" % nav.index[-1].date())
print("=" * 70)
cols = ex.columns.tolist()
print("exposures 列:", cols)
stock = ex["stock"] if "stock" in ex else ex.iloc[:, 0]
tgt = ex["target_exposure"] if "target_exposure" in ex else None
cap = ex["exposure_cap"] if "exposure_cap" in ex else None
reg = ex["regime_scale"] if "regime_scale" in ex else None

print(f"实际股票仓位：平均 {stock.mean():.1%}，中位 {stock.median():.1%}，",
      f"最小 {stock.min():.1%}（{stock.idxmin().date()}），最大 {stock.max():.1%}")
print(f"实际仓位 <30% 天数: {(stock < 0.30).sum()} / {len(stock)}（{(stock < 0.30).mean():.1%}）；"
      f"<20%: {(stock < 0.20).sum()}；<10%: {(stock < 0.10).sum()}")
if tgt is not None:
    print(f"目标仓位：平均 {tgt.mean():.1%}，最小 {tgt.min():.1%}，<30% 天数 "
          f"{(tgt < 0.30).sum()}，<20% 天数 {(tgt < 0.20).sum()}")
if cap is not None:
    print(f"回撤节流上限分布: 100% {(cap >= 0.999).mean():.1%}，60% {(np.isclose(cap, 0.6)).mean():.1%}，"
          f"40% {(np.isclose(cap, 0.4)).mean():.1%}；最低 cap {cap.min():.2f}")
if reg is not None:
    for v, name in [(0.45, "双条件不满足 0.45"), (0.75, "hs300 破 MA60 → 0.75"),
                    (0.60, "csi_all 破 MA120 → 0.60"), (1.0, "满仓 1.0")]:
        print(f"regime 系数 {name}: {(np.isclose(reg, v)).mean():.1%}")
# 连续低仓期
for thr in (0.5, 0.35):
    low = (stock < thr).astype(int).to_numpy()
    best = cur = 0
    for v in low:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    print(f"实际仓位连续 <{thr:.0%} 的最长天数: {best}")

print()
print("=" * 70)
print("2) 熊市年份/压力期：策略 vs 基准")
print("=" * 70)
yearly = nav.groupby(nav.index.year).last() / nav.groupby(nav.index.year).first() - 1
bench = {}
for b in ["all_eq", "csi1000", "csi_all", "hs300"]:
    pass

print("主回测逐年:", {int(y): round(v * 100, 2) for y, v in yearly.items()})
print(stress.round(4).to_string(index=False))

print()
print("=" * 70)
print("3) 如果“不降仓、不止损”（no_overlay_no_stop）会怎样")
print("=" * 70)
for tag in ["no_overlay", "no_overlay_no_stop"]:
    d = ROOT / "build" / "runs" / "robustness" / tag
    if not (d / "nav.parquet").exists():
        continue
    n2 = pd.read_parquet(d / "nav.parquet")["nav"]
    n2.index = pd.to_datetime(n2.index)
    y2 = n2.groupby(n2.index.year).last() / n2.groupby(n2.index.year).first() - 1
    sub = n2[(n2.index >= "2020-01-01") & (n2.index <= "2026-12-31")]
    dd = (sub / sub.cummax() - 1).min()
    print(f"{tag:20s} 全期CAGR={ (sub.iloc[-1]/sub.iloc[0])**(252/(len(sub)-1))-1 :.2%} "
          f"MaxDD={dd:.2%} | 2022年={y2.get(2022, np.nan):.2%} "
          f"2024年={y2.get(2024, np.nan):.2%} 2026年={y2.get(2026, np.nan):.2%}")

print()
print("=" * 70)
print("4) 主回测前三大回撤区间")
print("=" * 70)
peak = nav.cummax()
dd = nav / peak - 1
# 找每个回撤段（从峰值到恢复）
in_dd = dd < 0
segs = []
start = None
for dt, v in in_dd.items():
    if v and start is None:
        start = dt
    elif not v and start is not None:
        segs.append((start, dt))
        start = None
if start is not None:
    segs.append((start, nav.index[-1]))
recs = []
for s, e in segs:
    sub = dd[s:e]
    trough = sub.idxmin()
    recs.append((s, trough, e, sub.min()))
recs.sort(key=lambda x: x[3])
for s, tr, e, m in recs[:3]:
    s2 = ex.loc[s:e]
    print(f"{s.date()} ~ {e.date()}：最大回撤 {m:.2%}（低点 {tr.date()}），"
          f"该段平均实际股票仓位 {s2['stock'].mean():.1%}，最低 {s2['stock'].min():.1%}")
