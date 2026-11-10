# -*- coding: utf-8 -*-
"""检验：该策略在过去每个调仓时点能命中多少只？

严格处理财报时点（无前视）：
  报告期 → 法定披露截止 → 该日之后才可用
    一季报(3-31)   → 4-30 可用
    半年报(6-30)   → 8-31 可用
    三季报(9-30)   → 10-31 可用
    年报(12-31)    → 次年 4-30 可用

按用户原条件（严）：
  PB 0~10 / 市现率<5 / ROE>30% / 毛利率>30% / 净利同比>150%
  然后取小市值前 50

输出：
  · 每个月的候选数（原条件）
  · 放宽后的候选数（对照）
  · 候选股的后续表现（验证是否有效）
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

OUT = ROOT / "build" / "eval" / "fund_hit_rate.json"

md = qfdata.load_market(verbose=False)
dates = pd.DatetimeIndex(md.dates)
code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}
print(f"行情: {md.T} 日 × {md.N} 只，{dates[0].date()} ~ {dates[-1].date()}")

# ---------------------------------------------------------------- 加载所有财报
fdir = C.DATA_RAW / "fund"
periods = sorted([p.stem for p in fdir.glob("*.parquet")])
print(f"财报: {len(periods)} 期")


def usable_date(period: str) -> pd.Timestamp:
    """报告期 → 可用日期（法定披露截止后）。"""
    y, m = int(period[:4]), int(period[4:6])
    if m == 3:
        return pd.Timestamp(y, 4, 30)
    if m == 6:
        return pd.Timestamp(y, 8, 31)
    if m == 9:
        return pd.Timestamp(y, 10, 31)
    return pd.Timestamp(y + 1, 4, 30)


fund = {}
for p in periods:
    d = pd.read_parquet(fdir / f"{p}.parquet")
    d["code6"] = d["code"].astype(str).str.zfill(6)
    d = d[d["code6"].isin(code_idx)]
    fund[p] = {
        "usable": usable_date(p),
        "data": d.set_index("code6")[["bps", "ocfps", "roe", "gpm",
                                      "profit_yoy", "name"]],
    }
    print(f"  {p}: {len(d)} 行 → {usable_date(p).date()} 可用", flush=True)


# ---------------------------------------------------------------- 按月扫描
def screen(t: int, period: str, params: dict) -> pd.DataFrame:
    """在 t 日按 params 筛选，返回候选 DataFrame。"""
    d = fund[period]["data"]
    px = md.price[t]
    mcap = md.mcap[t]

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
        roe = r["roe"]
        gpm = r["gpm"]
        yoy = r["profit_yoy"]

        # 阈值过滤
        if params.get("pb") is not None:
            lo, hi = params["pb"]
            if not (np.isfinite(pb) and lo < pb <= hi):
                continue
        if params.get("pcf") is not None:
            if not (np.isfinite(pcf) and 0 < pcf < params["pcf"]):
                continue
        if params.get("roe") is not None:
            if not (np.isfinite(roe) and roe > params["roe"]):
                continue
        if params.get("gpm") is not None:
            if not (np.isfinite(gpm) and gpm > params["gpm"]):
                continue
        if params.get("yoy") is not None:
            if not (np.isfinite(yoy) and yoy > params["yoy"]):
                continue
        rows.append({"code": code, "name": r["name"], "pb": pb,
                     "pcf": pcf, "roe": roe, "gpm": gpm, "yoy": yoy,
                     "mcap_yi": m_ / 1e8, "j": j})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.nsmallest(params.get("top", 50), "mcap_yi")


# 参数组
PARAMS = {
    "原条件（严）": dict(pb=(0, 10), pcf=5, roe=30, gpm=30, yoy=150, top=50),
    "仅去市现率": dict(pb=(0, 10), pcf=None, roe=30, gpm=30, yoy=150, top=50),
    "ROE放宽15%": dict(pb=(0, 10), pcf=5, roe=15, gpm=30, yoy=150, top=50),
    "建议版": dict(pb=(0, 10), pcf=30, roe=15, gpm=30, yoy=50, top=50),
}

# 每月第一个交易日
dates_s = pd.Series(dates)
first_of_month = (dates_s.dt.month != dates_s.shift(1).dt.month).to_numpy()
month_starts = np.flatnonzero(first_of_month)
month_starts = month_starts[month_starts > 260]   # 需要至少 1 年历史

print(f"\n扫描 {len(month_starts)} 个月初时点")

results = {}
for label, prm in PARAMS.items():
    recs = []
    for t in month_starts:
        dt = dates[t]
        # 找该时点可用的最新财报
        avail = [(p, v["usable"]) for p, v in fund.items()
                 if v["usable"] <= dt]
        if not avail:
            recs.append({"date": dt, "n": 0, "period": None})
            continue
        pid = max(avail, key=lambda x: x[1])[0]
        df = screen(t, pid, prm)
        recs.append({"date": dt, "n": len(df),
                     "period": pid, "df": df})
    results[label] = recs
    ns = [r["n"] for r in recs]
    print(f"\n【{label}】")
    print(f"  月度候选数: 均值 {np.mean(ns):.1f}, 中位 {np.median(ns):.0f}, "
          f"最大 {max(ns)}, 最小 {min(ns)}")
    print(f"  有候选的月份: {sum(1 for x in ns if x > 0)} / {len(ns)}"
          f"（{sum(1 for x in ns if x > 0)/len(ns)*100:.0f}%）")

# ---------------------------------------------------------------- 明细输出
print("\n" + "=" * 96)
print("原条件逐月命中（只看有候选的）")
print("=" * 96)
recs = results["原条件（严）"]
hit = [r for r in recs if r["n"] > 0]
print(f"\n共 {len(hit)} 个月有候选（占 {len(hit)/len(recs)*100:.0f}%）")
if hit:
    print(f"\n{'日期':<12}{'财报期':<10}{'候选数':>7}  前 5 只")
    print("-" * 96)
    for r in hit[:30]:
        names = ", ".join(r["df"]["name"].head(5).astype(str)) if "df" in r else ""
        print(f"{r['date'].date()!s:<12}{r['period']:<10}{r['n']:>7}  {names[:60]}")

# ---------------------------------------------------------------- 严格条件能命中的年份分布
print("\n" + "=" * 96)
print("原条件：按年统计命中情况")
print("=" * 96)
by_year = {}
for r in recs:
    y = r["date"].year
    by_year.setdefault(y, []).append(r["n"])
print(f"{'年份':>6}{'月数':>6}{'有候选月':>9}{'平均候选':>10}{'最大':>7}")
print("-" * 46)
for y in sorted(by_year):
    v = by_year[y]
    print(f"{y:>6}{len(v):>6}{sum(1 for x in v if x > 0):>9}"
          f"{np.mean(v):>10.1f}{max(v):>7}")

# ---------------------------------------------------------------- 建议版对照
print("\n" + "=" * 96)
print("建议版逐月命中（对照）")
print("=" * 96)
recs2 = results["建议版"]
by_year2 = {}
for r in recs2:
    by_year2.setdefault(r["date"].year, []).append(r["n"])
print(f"{'年份':>6}{'月数':>6}{'有候选月':>9}{'平均候选':>10}{'最大':>7}")
print("-" * 46)
for y in sorted(by_year2):
    v = by_year2[y]
    print(f"{y:>6}{len(v):>6}{sum(1 for x in v if x > 0):>9}"
          f"{np.mean(v):>10.1f}{max(v):>7}")

# ---------------------------------------------------------------- 保存
summary = {}
for label, rs in results.items():
    ns = [r["n"] for r in rs]
    summary[label] = {
        "months": len(rs),
        "mean": float(np.mean(ns)),
        "median": float(np.median(ns)),
        "max": int(max(ns)),
        "min": int(min(ns)),
        "months_with_hit": int(sum(1 for x in ns if x > 0)),
        "hit_rate": float(sum(1 for x in ns if x > 0) / len(ns)),
    }
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"\n→ {OUT}")
for k, v in summary.items():
    print(f"  {k}: 命中率 {v['hit_rate']:.0%}, 均值 {v['mean']:.1f} 只")
