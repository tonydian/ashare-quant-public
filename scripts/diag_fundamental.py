# -*- coding: utf-8 -*-
"""诊断：为什么 5 个条件叠加后是 0 只？

单条件通过率：
  PB 0~10        94.6%
  市现率 <5       1.9%   ← 极严
  ROE >30%       0.6%   ← 极严
  毛利率 >30%     35.8%
  净利同比 >150%  10.8%

市现率<5 且 ROE>30% 的交集几乎为 0。

原因分析：
  市现率 = 市值 / 经营现金流
    低市现率 = 现金流相对市值很高 = "便宜"
  ROE > 30% = 盈利能力极强 = "优质"
  
  "又便宜又优质" 在 A 股几乎不存在（有效市场）
  → 这两个条件天然互斥

验证：看市现率<5 的 92 只里，ROE 分布如何
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
from qf import config as C, data as qfdata  # noqa: E402

md = qfdata.load_market(verbose=False)
code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}

fdir = C.DATA_RAW / "fund"
periods = sorted([p.stem for p in fdir.glob("*.parquet")])
d = pd.read_parquet(fdir / f"{periods[-1]}.parquet")
d["code6"] = d["code"].astype(str).str.zfill(6)
d = d[d["code6"].isin(code_idx)]

t = md.T - 1
px, mcap = md.price[t], md.mcap[t]

rows = []
for _, r in d.iterrows():
    j = code_idx[r["code6"]]
    p, m = px[j], mcap[j]
    if not (np.isfinite(p) and p > 0 and np.isfinite(m) and m > 0):
        continue
    bps, ocfps = r.get("bps"), r.get("ocfps")
    pb = p / bps if (bps and np.isfinite(bps) and bps > 0) else np.nan
    pcf = p / ocfps if (ocfps and np.isfinite(ocfps) and ocfps > 0) else np.nan
    rows.append({"code": r["code6"], "name": r.get("name"),
                 "pb": pb, "pcf": pcf, "roe": r.get("roe"),
                 "gpm": r.get("gpm"), "profit_yoy": r.get("profit_yoy"),
                 "mcap_yi": m / 1e8})
df = pd.DataFrame(rows)

print("=" * 88)
print("诊断 1：市现率 <5 的 92 只，它们的 ROE 有多少？")
print("=" * 88)
sub = df[(df["pcf"] > 0) & (df["pcf"] < 5)]
print(f"  样本 {len(sub)} 只")
print(f"  ROE 分布:")
for q in (10, 25, 50, 75, 90, 95, 99):
    print(f"    {q:>3}% 分位: {np.nanpercentile(sub['roe'].dropna(), q):>7.1f}%")
print(f"  ROE > 30% 的有: {int((sub['roe'] > 30).sum())} 只")
print(f"  ROE > 15% 的有: {int((sub['roe'] > 15).sum())} 只")

print("\n" + "=" * 88)
print("诊断 2：ROE >30% 的 29 只，它们的市现率是多少？")
print("=" * 88)
sub2 = df[df["roe"] > 30]
print(f"  样本 {len(sub2)} 只")
print(f"  市现率分布:")
pcf_ok = sub2["pcf"].replace([np.inf, -np.inf], np.nan).dropna()
if len(pcf_ok):
    for q in (10, 25, 50, 75, 90):
        print(f"    {q:>3}% 分位: {np.nanpercentile(pcf_ok, q):>7.2f}")
    print(f"  市现率 <5 的有: {int((sub2['pcf'] < 5).sum())} 只")
    print(f"  市现率 <20 的有: {int((sub2['pcf'] < 20).sum())} 只")
else:
    print("  市现率全为负/缺失（现金流为负）")

print("\n  这 29 只的样子:")
print(f"  {'代码':<8}{'名称':<12}{'PB':>7}{'市现率':>9}{'ROE':>8}{'毛利率':>8}{'市值(亿)':>10}")
print("  " + "-" * 64)
for _, r in sub2.head(15).iterrows():
    pcf_v = r["pcf"]
    pcf_s = f"{pcf_v:>9.1f}" if (np.isfinite(pcf_v) if isinstance(pcf_v, float) else True) else "      nan"
    print(f"  {r['code']:<8}{str(r['name'])[:10]:<12}{r['pb']:>7.2f}"
          f"{pcf_v:>9.1f}{r['roe']:>8.1f}{r['gpm']:>8.1f}{r['mcap_yi']:>10.1f}")

print("\n" + "=" * 88)
print("诊断 3：放宽哪个条件能救活策略？")
print("=" * 88)
tests = [
    ("原条件（全叠加）", dict(pb=(0, 10), pcf=5, roe=30, gpm=30, yoy=150)),
    ("市现率放宽到 20", dict(pb=(0, 10), pcf=20, roe=30, gpm=30, yoy=150)),
    ("市现率放宽到 50", dict(pb=(0, 10), pcf=50, roe=30, gpm=30, yoy=150)),
    ("去掉市现率", dict(pb=(0, 10), pcf=None, roe=30, gpm=30, yoy=150)),
    ("ROE 放宽到 15%", dict(pb=(0, 10), pcf=5, roe=15, gpm=30, yoy=150)),
    ("ROE 放宽到 15% + 市现率 20", dict(pb=(0, 10), pcf=20, roe=15, gpm=30, yoy=150)),
    ("去掉市现率 + ROE 15%", dict(pb=(0, 10), pcf=None, roe=15, gpm=30, yoy=150)),
    ("净利同比放宽到 50%", dict(pb=(0, 10), pcf=None, roe=15, gpm=30, yoy=50)),
]
print(f"{'方案':<26}{'候选数':>8}")
print("-" * 36)
for name, kw in tests:
    x = df.copy()
    if kw["pb"]:
        x = x[(x["pb"] > kw["pb"][0]) & (x["pb"] <= kw["pb"][1])]
    if kw["pcf"]:
        x = x[(x["pcf"] > 0) & (x["pcf"] < kw["pcf"])]
    if kw["roe"]:
        x = x[x["roe"] > kw["roe"]]
    if kw["gpm"]:
        x = x[x["gpm"] > kw["gpm"]]
    if kw["yoy"]:
        x = x[x["profit_yoy"] > kw["yoy"]]
    print(f"{name:<26}{len(x):>8}")
