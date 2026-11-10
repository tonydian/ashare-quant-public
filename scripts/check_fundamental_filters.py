# -*- coding: utf-8 -*-
"""检查基本面筛选条件：每个条件筛掉多少，最后剩多少。

条件（用户给的）：
  ① PB 0~10
  ② 市现率 < 5
  ③ ROE > 30%
  ④ 销售毛利率 > 30%
  ⑤ 净利润同比增长 > 150%
  ⑥ 小市值优先，取前 50 只

数据字段：
  bps → 每股净资产（PB = price / bps）
  ocfps → 每股经营现金流（市现率 = price / ocfps）
  roe → ROE
  gpm → 毛利率
  profit_yoy → 净利润同比
  eps → 每股收益
  revenue, net_profit → 规模
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
print(f"行情: {md.T} 日 × {md.N} 只")

# 取最新一期财报
fdir = C.DATA_RAW / "fund"
periods = sorted([p.stem for p in fdir.glob("*.parquet")])
print(f"财报期: {len(periods)} 期，{periods[0]} ~ {periods[-1]}")

latest = periods[-1]
d = pd.read_parquet(fdir / f"{latest}.parquet")
print(f"\n最新期 {latest}: {len(d)} 行")

# 对应到 md.codes
code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}
d["code6"] = d["code"].astype(str).str.zfill(6)
d = d[d["code6"].isin(code_idx)]
print(f"能匹配行情: {len(d)} 行")

# 取当日收盘价（最新）
t = md.T - 1
px = md.price[t]
mcap = md.mcap[t]

rows = []
for _, r in d.iterrows():
    j = code_idx[r["code6"]]
    p = px[j]
    m = mcap[j]
    if not (np.isfinite(p) and p > 0 and np.isfinite(m) and m > 0):
        continue
    bps = r.get("bps")
    ocfps = r.get("ocfps")
    pb = p / bps if (bps and np.isfinite(bps) and bps > 0) else np.nan
    pcf = p / ocfps if (ocfps and np.isfinite(ocfps) and ocfps > 0) else np.nan
    rows.append({
        "code": r["code6"], "name": r.get("name"),
        "pb": pb, "pcf": pcf,
        "roe": r.get("roe"), "gpm": r.get("gpm"),
        "profit_yoy": r.get("profit_yoy"),
        "mcap_yi": m / 1e8,
    })

df = pd.DataFrame(rows)
print(f"有效样本: {len(df)}")

print("\n" + "=" * 88)
print("逐条件筛选（每一步剩多少）")
print("=" * 88)

steps = []
cur = df.copy()

# ① PB 0~10
n0 = len(cur)
cur = cur[(cur["pb"] > 0) & (cur["pb"] <= 10)]
steps.append(("① PB 0~10", n0, len(cur)))

# ② 市现率 < 5
n0 = len(cur)
cur = cur[(cur["pcf"] > 0) & (cur["pcf"] < 5)]
steps.append(("② 市现率 <5", n0, len(cur)))

# ③ ROE > 30%
n0 = len(cur)
cur = cur[cur["roe"] > 30]
steps.append(("③ ROE >30%", n0, len(cur)))

# ④ 毛利率 > 30%
n0 = len(cur)
cur = cur[cur["gpm"] > 30]
steps.append(("④ 毛利率 >30%", n0, len(cur)))

# ⑤ 净利润同比 > 150%
n0 = len(cur)
cur = cur[cur["profit_yoy"] > 150]
steps.append(("⑤ 净利同比 >150%", n0, len(cur)))

# ⑥ 小市值前 50
n0 = len(cur)
final = cur.nsmallest(50, "mcap_yi")
steps.append(("⑥ 最小市值前 50", n0, len(final)))

print(f"{'条件':<22}{'筛选前':>8}{'筛选后':>8}{'通过率':>9}")
print("-" * 50)
for name, a, b in steps:
    print(f"{name:<22}{a:>8}{b:>8}{(b/a*100 if a else 0):>8.1f}%")

print(f"\n最终候选: {len(final)} 只")
if len(final):
    print(f"\n{'代码':<8}{'名称':<12}{'PB':>7}{'市现率':>8}{'ROE':>8}"
          f"{'毛利率':>8}{'净利同比':>10}{'市值(亿)':>10}")
    print("-" * 78)
    for _, r in final.head(15).iterrows():
        print(f"{r['code']:<8}{str(r['name'])[:10]:<12}{r['pb']:>7.2f}"
              f"{r['pcf']:>8.2f}{r['roe']:>8.1f}{r['gpm']:>8.1f}"
              f"{r['profit_yoy']:>10.1f}{r['mcap_yi']:>10.1f}")

print("\n" + "=" * 88)
print("单条件独立通过数（不叠加，看哪个条件最严）")
print("=" * 88)
conds = {
    "PB 0~10": (df["pb"] > 0) & (df["pb"] <= 10),
    "市现率 <5": (df["pcf"] > 0) & (df["pcf"] < 5),
    "ROE >30%": df["roe"] > 30,
    "毛利率 >30%": df["gpm"] > 30,
    "净利同比 >150%": df["profit_yoy"] > 150,
}
for k, mask in conds.items():
    print(f"  {k:<18} 通过 {int(mask.sum()):>5} / {len(df)}"
          f"（{mask.mean()*100:>5.1f}%）")
