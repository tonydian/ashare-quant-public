# -*- coding: utf-8 -*-
"""数据体检：板块 ETF 的异常日收益（超过 ±11%）。

ETF 单日涨跌幅限制为 ±10%（科创板/创业板 ETF 可 ±20%），
超过该范围且非涨跌停的跳变，说明是**复权因子突变**（分红/拆分处理不当）。

本脚本：
  1) 找出所有异常点
  2) 用 raw_close（不复权）对照，判断是"复权问题"还是"真实价格"
  3) 给出修复建议
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "build" / "etf" / "sector_panel.npz"
UNIV = ROOT / "build" / "etf" / "sector_universe.json"
OUT = ROOT / "build" / "eval" / "sector_data_quality.json"

z = np.load(PANEL, allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
close = z["close"].astype(float)
raw = z["raw_close"].astype(float) if "raw_close" in z.files else None
univ = json.load(open(UNIV, encoding="utf-8"))
sector_of = {v["code"]: k for k, v in univ["sectors"].items()}

print(f"面板 {len(dates)} 日 × {len(codes)} 只")
print(f"raw_close 可用: {raw is not None}")

records = []
for j, c in enumerate(codes):
    col = close[:, j]
    r = col[1:] / col[:-1] - 1.0
    idx = np.flatnonzero(np.abs(r) > 0.11)
    for i in idx:
        t = i + 1
        rr = None
        if raw is not None and np.isfinite(raw[t, j]) and np.isfinite(raw[t - 1, j]) and raw[t - 1, j] > 0:
            rr = float(raw[t, j] / raw[t - 1, j] - 1.0)
        records.append({
            "date": str(dates[t].date()),
            "code": c, "sector": sector_of.get(c, ""),
            "ret_qfq": float(r[i]),
            "ret_raw": rr,
            "is_adj_issue": (rr is not None and abs(rr) <= 0.11 and abs(r[i]) > 0.11),
        })

print(f"\n异常点（|qfq 日收益| > 11%）：{len(records)} 条")
adj_issues = [r for r in records if r["is_adj_issue"]]
print(f"其中【复权因子问题】（不复权收益正常）：{len(adj_issues)} 条")
others = [r for r in records if not r["is_adj_issue"]]
print(f"其他（不复权也异常 / 无 raw 数据）：{len(others)} 条")

print("\n" + "=" * 96)
print("复权因子问题明细（这些会污染回测）")
print("=" * 96)
print(f"{'日期':<12}{'代码':<8}{'板块':<10}{'qfq收益':>10}{'不复权收益':>12}")
for r in sorted(adj_issues, key=lambda x: -abs(x["ret_qfq"]))[:25]:
    print(f"{r['date']:<12}{r['code']:<8}{r['sector']:<10}"
          f"{r['ret_qfq']*100:>9.1f}%{(r['ret_raw'] or 0)*100:>11.1f}%")

if others:
    print("\n" + "=" * 96)
    print("其他异常（不复权也超过 11%，可能是真实涨跌或数据错）")
    print("=" * 96)
    for r in sorted(others, key=lambda x: -abs(x["ret_qfq"]))[:15]:
        rr = f"{r['ret_raw']*100:+.1f}%" if r["ret_raw"] is not None else "n/a"
        print(f"{r['date']:<12}{r['code']:<8}{r['sector']:<10}"
              f"{r['ret_qfq']*100:>9.1f}%{rr:>12}")

# 按板块统计
by_sector = {}
for r in records:
    by_sector.setdefault(r["sector"], []).append(r)
print("\n" + "=" * 96)
print("按板块统计异常点数")
print("=" * 96)
for sec, rs in sorted(by_sector.items(), key=lambda x: -len(x[1])):
    n_adj = sum(1 for r in rs if r["is_adj_issue"])
    print(f"  {sec:<10} 共 {len(rs):2d} 条（其中复权问题 {n_adj} 条）")

OUT.write_text(json.dumps({"n_total": len(records), "adj_issues": adj_issues,
                           "others": others}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
