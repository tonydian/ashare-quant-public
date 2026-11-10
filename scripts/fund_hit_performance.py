# -*- coding: utf-8 -*-
"""深入：原条件命中的股票，后来表现如何？

命中统计：
  2018: 平均 2.4 只/月
  2019: 1.8
  2020: 2.8
  2021: 5.4
  2022: 5.9
  2023: 4.8
  2024: 0.3  ← 几乎归零
  2025-2026: 0

关键问题：
  Q1 命中后 1/3/6/12 个月的收益是多少？
  Q2 命中股 vs 全市场，有没有超额？
  Q3 2024 后为什么归零？（条件变严了？还是市场变了？）
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
from qf import config as C, data as qfdata  # noqa: E402

md = qfdata.load_market(verbose=False)
dates = pd.DatetimeIndex(md.dates)
code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}

fdir = C.DATA_RAW / "fund"
periods = sorted([p.stem for p in fdir.glob("*.parquet")])


def usable_date(period):
    y, m = int(period[:4]), int(period[4:6])
    return {3: pd.Timestamp(y, 4, 30), 6: pd.Timestamp(y, 8, 31),
            9: pd.Timestamp(y, 10, 31)}.get(m, pd.Timestamp(y + 1, 4, 30))


fund = {}
for p in periods:
    d = pd.read_parquet(fdir / f"{p}.parquet")
    d["code6"] = d["code"].astype(str).str.zfill(6)
    d = d[d["code6"].isin(code_idx)]
    fund[p] = {"usable": usable_date(p),
               "data": d.set_index("code6")[
                   ["bps", "ocfps", "roe", "gpm", "profit_yoy", "name"]]}

PARAMS = dict(pb=(0, 10), pcf=5, roe=30, gpm=30, yoy=150, top=50)


def screen(t, period):
    d = fund[period]["data"]
    px, mcap = md.price[t], md.mcap[t]
    rows = []
    for code, r in d.iterrows():
        j = code_idx.get(code)
        if j is None:
            continue
        p_, m_ = px[j], mcap[j]
        if not (np.isfinite(p_) and p_ > 0 and np.isfinite(m_) and m_ > 0):
            continue
        bps, ocfps = r["bps"], r["ocfps"]
        pb = p_ / bps if (np.isfinite(bps) and bps > 0) else np.nan
        pcf = p_ / ocfps if (np.isfinite(ocfps) and ocfps > 0) else np.nan
        roe, gpm, yoy = r["roe"], r["gpm"], r["profit_yoy"]
        if not (np.isfinite(pb) and 0 < pb <= 10):
            continue
        if not (np.isfinite(pcf) and 0 < pcf < 5):
            continue
        if not (np.isfinite(roe) and roe > 30):
            continue
        if not (np.isfinite(gpm) and gpm > 30):
            continue
        if not (np.isfinite(yoy) and yoy > 150):
            continue
        rows.append({"code": code, "name": r["name"], "j": j,
                     "pb": pb, "pcf": pcf, "roe": roe,
                     "gpm": gpm, "yoy": yoy, "mcap": m_ / 1e8})
    df = pd.DataFrame(rows)
    return df.nsmallest(50, "mcap") if len(df) else df


# 月初
ds = pd.Series(dates)
fom = (ds.dt.month != ds.shift(1).dt.month).to_numpy()
starts = np.flatnonzero(fom)
starts = starts[starts > 260]

print("=" * 96)
print("命中股票的后续表现")
print("=" * 96)

recs = []
for t in starts:
    dt = dates[t]
    avail = [(p, v["usable"]) for p, v in fund.items() if v["usable"] <= dt]
    if not avail:
        continue
    pid = max(avail, key=lambda x: x[1])[0]
    df = screen(t, pid)
    if not len(df):
        continue
    for _, r in df.iterrows():
        j = int(r["j"])
        row = {"date": dt, "year": dt.year, "code": r["code"],
               "name": r["name"], "mcap": r["mcap"],
               "pb": r["pb"], "pcf": r["pcf"], "roe": r["roe"],
               "gpm": r["gpm"], "yoy": r["yoy"]}
        for h, lbl in ((21, "r1m"), (63, "r3m"), (126, "r6m"), (250, "r12m")):
            t2 = t + h
            if t2 < md.T:
                p0, p1 = md.price[t, j], md.price[t2, j]
                row[lbl] = (p1 / p0 - 1) if (np.isfinite(p0) and p0 > 0
                                             and np.isfinite(p1)) else np.nan
        recs.append(row)

R = pd.DataFrame(recs)
print(f"总样本: {len(R)} 个（股票-月）")
print(f"涉及 {R['code'].nunique()} 只不同股票")

print(f"\n{'持有期':>8}{'样本':>7}{'均值':>10}{'中位':>10}{'胜率':>9}")
print("-" * 46)
for lbl, name in (("r1m", "1 月"), ("r3m", "3 月"),
                  ("r6m", "6 月"), ("r12m", "12 月")):
    v = R[lbl].dropna()
    if len(v) < 10:
        continue
    print(f"{name:>8}{len(v):>7}{v.mean()*100:>+9.2f}%"
          f"{v.median()*100:>+9.2f}%{(v>0).mean()*100:>8.1f}%")

# 对照：全市场同期
print("\n" + "=" * 96)
print("对照基准（全市场同期，随机买入）")
print("=" * 96)


def bench(t, h):
    """全市场等效收益（市值加权近似）。"""
    t2 = t + h
    if t2 >= md.T:
        return np.nan
    p0, p1 = md.price[t], md.price[t2]
    ok = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & md.eligible[t]
    if ok.sum() < 100:
        return np.nan
    return float(np.median(p1[ok] / p0[ok] - 1))


bench_rows = []
for t in starts:
    for h, lbl in ((21, "r1m"), (63, "r3m"), (126, "r6m"), (250, "r12m")):
        b = bench(t, h)
        if np.isfinite(b):
            bench_rows.append({"date": dates[t], "h": lbl, "r": b})
B = pd.DataFrame(bench_rows)

print(f"{'持有期':>8}{'命中股均值':>13}{'全市场中位':>13}{'超额':>10}")
print("-" * 50)
for lbl, name in (("r1m", "1 月"), ("r3m", "3 月"),
                  ("r6m", "6 月"), ("r12m", "12 月")):
    v = R[lbl].dropna()
    bb = B[B["h"] == lbl]["r"]
    if len(v) < 10 or len(bb) < 5:
        continue
    print(f"{name:>8}{v.mean()*100:>+12.2f}%{bb.median()*100:>+12.2f}%"
          f"{(v.mean()-bb.median())*100:>+9.2f}pp")

# 分年
print("\n" + "=" * 96)
print("分年度：命中后 6 个月收益")
print("=" * 96)
print(f"{'年份':>6}{'样本':>7}{'6M均值':>10}{'胜率':>9}")
print("-" * 34)
for y, g in R.groupby("year"):
    v = g["r6m"].dropna()
    if len(v) < 3:
        continue
    print(f"{y:>6}{len(v):>7}{v.mean()*100:>+9.2f}%{(v>0).mean()*100:>8.1f}%")

# 个股
print("\n" + "=" * 96)
print("出现次数最多的股票")
print("=" * 96)
cnt = R.groupby(["code", "name"]).size().sort_values(ascending=False)
print(f"{'代码':<8}{'名称':<14}{'出现月数':>9}{'12M均值':>11}")
print("-" * 46)
for (code, name), n in cnt.head(15).items():
    v = R[(R["code"] == code)]["r12m"].dropna()
    avg = f"{v.mean()*100:+.1f}%" if len(v) else "—"
    print(f"{code:<8}{str(name)[:12]:<14}{n:>9}{avg:>11}")

print("\n" + "=" * 96)
print("2024 后为何归零？")
print("=" * 96)
# 看各年度各条件的通过数
for yr, pid in (("2021", "20201231"), ("2023", "20221231"),
                ("2024", "20231231"), ("2025", "20241231"),
                ("2026", "20251231")):
    if pid not in fund:
        continue
    t = int(dates.searchsorted(fund[pid]["usable"]))
    if t >= md.T:
        continue
    d = fund[pid]["data"]
    px, mcap = md.price[t], md.mcap[t]
    n_pb = n_pcf = n_roe = n_gpm = n_yoy = n_all = 0
    for code, r in d.iterrows():
        j = code_idx.get(code)
        if j is None:
            continue
        p_ = px[j]
        if not (np.isfinite(p_) and p_ > 0):
            continue
        bps, ocfps = r["bps"], r["ocfps"]
        pb = p_ / bps if (np.isfinite(bps) and bps > 0) else np.nan
        pcf = p_ / ocfps if (np.isfinite(ocfps) and ocfps > 0) else np.nan
        c1 = np.isfinite(pb) and 0 < pb <= 10
        c2 = np.isfinite(pcf) and 0 < pcf < 5
        c3 = np.isfinite(r["roe"]) and r["roe"] > 30
        c4 = np.isfinite(r["gpm"]) and r["gpm"] > 30
        c5 = np.isfinite(r["profit_yoy"]) and r["profit_yoy"] > 150
        n_pb += c1; n_pcf += c2; n_roe += c3; n_gpm += c4; n_yoy += c5
        n_all += (c1 and c2 and c3 and c4 and c5)
    print(f"  {yr}（{pid}）: PB {n_pb}, 市现率 {n_pcf}, ROE {n_roe}, "
          f"毛利 {n_gpm}, 净利 {n_yoy} → 全满足 {n_all}")
