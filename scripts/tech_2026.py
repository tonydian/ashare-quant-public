# -*- coding: utf-8 -*-
"""查清 2026 年"科技牛"的真相：
  1) A 股科技板块到底涨没涨？（分行业、分市值、分个股）
  2) 如果涨了，策略有没有抓到？为什么？
  3) 涨幅是不是集中在极少数个股（策略买不到/买不够）？
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata  # noqa: E402

OUT = ROOT / "build" / "eval" / "tech_2026.json"
md = qfdata.load_market(verbose=False)
t0 = int(md.date_index("2026-01-01"))
t1 = md.T - 1
ret = md.price[t1] / md.price[t0] - 1.0
print(f"区间 {md.dates[t0].date()} ~ {md.dates[t1].date()}（{t1-t0} 个交易日）")

# ---------- 1) 全市场收益分布
valid = np.isfinite(ret)
r = ret[valid]
print("\n" + "=" * 96)
print("1) 2026 全市场收益分布")
print("=" * 96)
print(f"  有效股票 {valid.sum()} 只")
print(f"  中位数 {np.median(r)*100:+.2f}% | 均值 {r.mean()*100:+.2f}%")
print(f"  上涨家数 {(r>0).sum()}（{(r>0).mean():.1%}）")
print(f"  涨幅 > 50% 的股票: {(r>0.5).sum()} 只（{(r>0.5).mean():.2%}）")
print(f"  涨幅 > 100% 的股票: {(r>1.0).sum()} 只（{(r>1.0).mean():.2%}）")
print(f"  跌幅 > 30% 的股票: {(r<-0.3).sum()} 只（{(r<-0.3).mean():.1%}）")

# ---------- 2) 分行业
print("\n" + "=" * 96)
print("2) 分行业表现（中位数 + 龙头）")
print("=" * 96)
ind = md.industry[t1]
rows = []
for c in np.unique(ind):
    if c < 0:
        continue
    m = (ind == c) & valid
    if m.sum() < 5:
        continue
    x = ret[m]
    rows.append({
        "industry": md.industry_name(int(c)),
        "n": int(m.sum()),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
        "win_rate": float((x > 0).mean()),
    })
rows.sort(key=lambda x: -x["median"])
print(f"{'行业':<12}{'股票数':>7}{'中位':>9}{'均值':>9}{'p90':>9}{'最好':>9}{'胜率':>8}")
print("-" * 70)
for r_ in rows[:10]:
    print(f"{r_['industry']:<12}{r_['n']:>7}{r_['median']*100:>8.1f}%"
          f"{r_['mean']*100:>8.1f}%{r_['p90']*100:>8.1f}%"
          f"{r_['max']*100:>8.1f}%{r_['win_rate']:>7.0%}")
print("  ...（中间省略）")
for r_ in rows[-5:]:
    print(f"{r_['industry']:<12}{r_['n']:>7}{r_['median']*100:>8.1f}%"
          f"{r_['mean']*100:>8.1f}%{r_['p90']*100:>8.1f}%"
          f"{r_['max']*100:>8.1f}%{r_['win_rate']:>7.0%}")

# ---------- 3) 科技相关行业细看
TECH = ["电子", "计算机", "通信", "传媒", "国防军工"]
print("\n" + "=" * 96)
print("3) 科技类行业细看")
print("=" * 96)
print(f"{'行业':<10}{'股票数':>7}{'中位':>9}{'p90':>9}{'涨>50%家数':>12}{'占该行业':>10}")
for r_ in rows:
    if r_["industry"] not in TECH:
        continue
    m = (ind == next(c for c in np.unique(ind)
                     if c >= 0 and md.industry_name(int(c)) == r_["industry"])) & valid
    x = ret[m]
    n50 = int((x > 0.5).sum())
    print(f"{r_['industry']:<10}{r_['n']:>7}{r_['median']*100:>8.1f}%"
          f"{r_['p90']*100:>8.1f}%{n50:>12}{n50/len(x):>10.1%}")

# ---------- 4) 涨幅最大的个股
print("\n" + "=" * 96)
print("4) 2026 涨幅 TOP 20 个股")
print("=" * 96)
idx_sorted = np.argsort(-np.where(valid, ret, -9))[:20]
print(f"{'代码':<8}{'名称':<10}{'行业':<10}{'涨幅':>10}{'市值(亿)':>11}")
for j in idx_sorted:
    mc = md.mcap[t1, j] / 1e8 if np.isfinite(md.mcap[t1, j]) else float("nan")
    print(f"{str(md.codes[j]):<8}{str(md.names[j])[:8]:<10}"
          f"{md.industry_name(int(md.industry[t1, j])):<10}"
          f"{ret[j]*100:>9.0f}%{mc:>10.0f}")

# ---------- 5) 策略持仓在 2026 的表现
print("\n" + "=" * 96)
print("5) 策略 2026 年持仓 vs 全市场（它买的是什么）")
print("=" * 96)
tgt = ROOT / "signals" / "latest_target.csv"
if tgt.exists():
    df = pd.read_csv(tgt, dtype={"code": str})
    print(f"  当前持仓 {len(df)} 只（2026-10-23）")
    held_ret = []
    for r_ in df.itertuples():
        j = md.code_index(str(r_.code))
        if j >= 0 and np.isfinite(ret[j]):
            held_ret.append((str(r_.code), str(r_.name),
                             md.industry_name(int(md.industry[t1, j])), float(ret[j])))
    held_ret.sort(key=lambda x: -x[3])
    print(f"  {'代码':<8}{'名称':<10}{'行业':<10}{'2026涨幅':>10}")
    for c, n, ind_name, rr in held_ret[:12]:
        print(f"  {c:<8}{n[:8]:<10}{ind_name:<10}{rr*100:>9.1f}%")
    if held_ret:
        arr = np.array([x[3] for x in held_ret])
        print(f"\n  持仓 2026 中位涨幅 {np.median(arr)*100:+.1f}% | "
              f"全市场中位 {np.median(r)*100:+.1f}% | 差 {(np.median(arr)-np.median(r))*100:+.1f}pp")
    # 行业分布
    from collections import Counter
    cnt = Counter(x[2] for x in held_ret)
    print(f"  行业分布: {dict(cnt.most_common(8))}")

OUT.write_text(json.dumps({"rows": rows, "n_up": int((r > 0).sum()),
                           "median": float(np.median(r)),
                           "pct_gt50": float((r > 0.5).mean())},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
