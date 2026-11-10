# -*- coding: utf-8 -*-
"""为什么 2026 年的科技行情没抓到？

关键问题（逐层剥）：
  Q1 2026 到底是"全面科技牛"还是"窄幅结构性行情"？
  Q2 涨的这批股票，在 2026 年初能不能被选股域纳入？
  Q3 如果能纳入，模型的打分排名是多少？（是"看到了但没买"还是"根本没看到"）
  Q4 组合层面即使买到，能贡献多少？（40 只等权，单票上限 8%）
  Q5 策略的因子风格与 2026 行情是否冲突（反转 vs 动量）？
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

from qf import config as C, data as qfdata, model as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "why_miss_tech_2026.json"

md = qfdata.load_market(verbose=False)
T = md.T
t_start = int(md.date_index("2026-01-05"))
t_end = T - 1

print("=" * 100)
print("Q1 2026 是全面科技牛还是窄幅结构行情？")
print("=" * 100)
ret_2026 = md.price[t_end] / md.price[t_start] - 1.0
valid = np.isfinite(ret_2026) & md.eligible[t_start]
r = ret_2026[valid]
print(f"  有效股票 {valid.sum()} 只，区间 {md.dates[t_start].date()} ~ {md.dates[t_end].date()}")
print(f"  中位数 {np.median(r)*100:+.2f}% | 均值 {r.mean()*100:+.2f}%")
print(f"  上涨家数 {(r>0).sum()} / {len(r)}（{(r>0).mean():.1%}）")
print(f"  涨 >30%: {(r>0.3).sum()} 只（{(r>0.3).mean():.1%}）")
print(f"  涨 >50%: {(r>0.5).sum()} 只（{(r>0.5).mean():.1%}）")
print(f"  涨 >100%: {(r>1.0).sum()} 只（{(r>1.0).mean():.2%}）")
top_thr = np.percentile(r, 95)
print(f"\n  → 前 5% 股票的门槛是 {top_thr*100:+.1f}%")
print(f"  → 结论：这是**极窄的结构性行情**，不是全面科技牛"
      f"（中位数 {np.median(r)*100:+.1f}%，{(r>0).mean():.0%} 股票上涨）")

# ---------- 概念归类：看涨幅前 5% 集中在哪些行业
print("\n" + "=" * 100)
print("Q1b 涨幅前 5% 的股票集中在哪些行业？")
print("=" * 100)
ind_s = md.industry[t_start]
top_mask = valid & (ret_2026 >= top_thr)
from collections import Counter
cnt_top = Counter(md.industry_name(int(c)) for c in ind_s[top_mask])
cnt_all = Counter(md.industry_name(int(c)) for c in ind_s[valid])
print(f"{'行业':<12}{'前5%家数':>10}{'该行业总数':>11}{'集中度':>10}{'超配倍数':>10}")
print("-" * 56)
rows = []
for name, n in cnt_top.most_common(12):
    total = cnt_all.get(name, 1)
    share = n / max(len(cnt_top), 1)
    over = (n / total) / (len(cnt_top) / valid.sum())
    rows.append({"industry": name, "n_top": n, "n_all": total,
                 "share": share, "overweight": over})
    print(f"{name:<12}{n:>10}{total:>11}{share:>9.0%}{over:>9.1f}x")

# ---------- Q2 这些股票能不能被选股域纳入
print("\n" + "=" * 100)
print("Q2 涨幅前 5% 的股票，2026 年初能否进入选股域？")
print("=" * 100)
elig = md.eligible[t_start]
top_idx = np.flatnonzero(top_mask)
n_top = len(top_idx)
in_univ = int(elig[top_idx].sum()) if n_top else 0
print(f"  前 5% 共 {n_top} 只")
print(f"  其中在选股域内: {in_univ} 只（{in_univ/max(n_top,1):.1%}）")
print(f"  不在选股域: {n_top - in_univ} 只")
print("\n  选股域条件：上市≥120日、价格 2~200 元、20日均额≥3000万、市值 15亿~8000亿、非ST")
# 拆解未纳入原因
if n_top - in_univ > 0:
    out_idx = top_idx[~elig[top_idx]]
    reasons = Counter()
    for j in out_idx:
        px = md.price[t_start, j]
        amt = md.amount20[t_start, j] if hasattr(md, "amount20") else float("nan")
        mc = md.mcap[t_start, j]
        if not np.isfinite(px) or px <= 0:
            reasons["无价格/停牌"] += 1
        elif px < C.MIN_PRICE:
            reasons["价格过低"] += 1
        elif px > C.MAX_PRICE:
            reasons["价格过高"] += 1
        elif np.isfinite(mc) and mc < C.MIN_MCAP:
            reasons["市值过小"] += 1
        elif np.isfinite(mc) and mc > C.MAX_MCAP:
            reasons["市值过大"] += 1
        else:
            reasons["其他（成交额/上市天数/ST）"] += 1
    for k, v in reasons.most_common():
        print(f"    {k}: {v} 只")

# ---------- Q3 在域内的，模型打分排名多少
print("\n" + "=" * 100)
print("Q3 域内的涨幅前 5% 股票，在 2026-01-05 的模型打分排名是多少？")
print("=" * 100)
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
scores = np.asarray(res.scores)
if scores.shape[0] < T:
    scores = np.vstack([scores, np.full((T - scores.shape[0], md.N), np.nan, np.float32)])
while t_start % 5 != int(res.rows_weekly[0]) % 5:
    t_start += 1

row = scores[t_start]
ok = np.isfinite(row) & elig
ranks = np.full(md.N, np.nan)
order = np.argsort(-np.where(ok, row, -np.inf))
n_ok = int(ok.sum())
for pos, j in enumerate(order[:n_ok]):
    ranks[j] = pos + 1
print(f"  当日有效打分的股票 {n_ok} 只（40 只持仓 = 前 {40/n_ok:.1%}）")

top_in_univ = [j for j in top_idx if elig[j] and np.isfinite(ranks[j])]
if top_in_univ:
    rr = np.array([ranks[j] for j in top_in_univ])
    print(f"  涨幅前 5% 中，在域内且有打分的: {len(top_in_univ)} 只")
    print(f"    排名中位数 {np.median(rr):.0f} | 最好 {rr.min():.0f} | 最差 {rr.max():.0f}")
    print(f"    进入前 40 名的: {(rr <= 40).sum()} 只（{(rr<=40).mean():.0%}）")
    print(f"    进入前 200 名的: {(rr <= 200).sum()} 只（{(rr<=200).mean():.0%}）")
    print("\n  明细（涨幅前 10 且域内）：")
    detail = sorted([(float(ret_2026[j]), str(md.codes[j]), str(md.names[j]),
                      md.industry_name(int(ind_s[j])), float(ranks[j]))
                     for j in top_in_univ], reverse=True)[:10]
    print(f"    {'代码':<8}{'名称':<10}{'行业':<10}{'涨幅':>9}{'打分排名':>10}")
    for rr_, c, n, indn, rk in detail:
        print(f"    {c:<8}{n[:8]:<10}{indn:<10}{rr_*100:>8.0f}%{rk:>10.0f}")

# ---------- Q4 组合层面即使买到能贡献多少
print("\n" + "=" * 100)
print("Q4 组合层面：即使买到，40 只等权的单只贡献上限")
print("=" * 100)
print(f"  40 只等权 → 单只目标权重 2.5%（上限 8%）")
print(f"  若一只涨 300%，贡献 = 2.5% × 300% = 7.5%（对组合）")
print(f"  若买到 3 只各涨 200%，贡献 = 3 × 2.5% × 200% = 15%")
print(f"  → 但要抵消全市场 -15.4% 的 beta 拖累，需要极多超额")

# ---------- Q5 风格冲突
print("\n" + "=" * 100)
print('Q5 因子风格：策略买的是「超跌」还是「强势」？')
print("=" * 100)
# 用 2025 年涨幅 与 2026 年涨幅 的关系看反转效应
t_2025 = int(md.date_index("2025-01-02"))
ret_2025 = md.price[t_start] / md.price[t_2025] - 1.0
both = np.isfinite(ret_2025) & np.isfinite(ret_2026) & valid
print(f"  样本 {both.sum()} 只")
print(f"  2025 涨幅 与 2026 涨幅 的相关系数: "
      f"{np.corrcoef(ret_2025[both], ret_2026[both])[0,1]:+.3f}")
q = np.nanpercentile(ret_2025[both], [20, 40, 60, 80])
print(f"\n  按 2025 涨幅分 5 组，看 2026 表现：")
print(f"  {'2025涨幅组':<16}{'2026中位涨幅':>14}{'2026胜率':>10}")
labs = ["最低20%", "20-40%", "40-60%", "60-80%", "最高20%"]
for i in range(5):
    lo = -9 if i == 0 else q[i - 1]
    hi = 9 if i == 4 else q[i]
    m = both & (ret_2025 >= lo) & (ret_2025 < hi)
    if m.sum() > 10:
        print(f"  {labs[i]:<16}{np.median(ret_2026[m])*100:>13.1f}%"
              f"{(ret_2026[m]>0).mean():>10.0%}")

print("\n" + "=" * 100)
print("总结")
print("=" * 100)

out = {
    "median_2026": float(np.median(r)),
    "up_share": float((r > 0).mean()),
    "pct_gt50": float((r > 0.5).mean()),
    "top5_threshold": float(top_thr),
    "top_industries": rows[:8],
    "n_top": int(n_top),
    "n_top_in_univ": int(in_univ),
    "style_corr_2025_2026": float(np.corrcoef(ret_2025[both], ret_2026[both])[0, 1]),
}
if top_in_univ:
    out["rank_median"] = float(np.median(rr))
    out["rank_in_40"] = int((rr <= 40).sum())
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
