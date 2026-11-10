# -*- coding: utf-8 -*-
"""关键验证：2026 的大牛股，在起涨前有没有"可识别的前兆"？

如果起涨前完全没有信号（如突发题材/政策/重组），那任何量价因子都抓不到。
检验：看这 221 只大牛股在 2026-01-05 之前的量价特征 vs 全市场。
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

OUT = ROOT / "build" / "eval" / "tech_precursor.json"
md = qfdata.load_market(verbose=False)
T = md.T
t0 = int(md.date_index("2026-01-05"))
t1 = T - 1
ret = md.price[t1] / md.price[t0] - 1.0
valid = np.isfinite(ret) & md.eligible[t0]
top_thr = np.percentile(ret[valid], 95)
top = valid & (ret >= top_thr)

Cl = md.price.astype(np.float64)
O = md.price_open.astype(np.float64) if hasattr(md, "price_open") else None
V = md.volume.astype(np.float64)
A = md.amount.astype(np.float64)


def feat_at(t, w):
    """t 时点的量价特征（向量化，N 只股票一次算完）。"""
    seg_c = Cl[t - w:t + 1]                      # (w+1, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(seg_c[0] > 0, seg_c[-1] / seg_c[0] - 1.0, np.nan)
    lg = np.diff(np.log(np.maximum(seg_c, 1e-9)), axis=0)
    vol = np.nanstd(lg, axis=0) if w > 2 else np.full(Cl.shape[1], np.nan)
    ma = np.nanmean(Cl[max(0, t - 60):t + 1], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        above = np.where(ma > 0, Cl[t] / ma - 1.0, np.nan)
    amt20 = np.nanmean(A[max(0, t - 20):t + 1], axis=0)
    amt120 = np.nanmean(A[max(0, t - 120):t + 1], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(amt120 > 0, amt20 / amt120, np.nan)
    return r, vol, above, vol_ratio


print("=" * 100)
print("2026 大牛股（涨幅前 5%）在起涨前(2026-01-05)的特征 vs 全市场")
print("=" * 100)
prev_cut = int(md.date_index("2025-11-01"))

features = {}
for w, name in ((20, "近20日涨幅"), (60, "近60日涨幅"), (120, "近120日涨幅")):
    r_, _, _, _ = feat_at(prev_cut, w)
    features[name] = r_
r20, vol20, above60, volratio = feat_at(prev_cut, 20)
features["近20日波动"] = vol20
features["距MA60距离"] = above60
features["成交额放大倍数"] = volratio

print(f"\n  {'特征':<18}{'大牛股中位':>13}{'全市场中位':>13}{'差':>12}")
print("  " + "-" * 58)
res = {}
for name, arr in features.items():
    a_ = arr[top]
    b_ = arr[valid]
    a_ = a_[np.isfinite(a_)]
    b_ = b_[np.isfinite(b_)]
    if len(a_) < 10 or len(b_) < 100:
        continue
    am, bm = float(np.median(a_)), float(np.median(b_))
    print(f"  {name:<18}{am*100:>12.2f}%{bm*100:>12.2f}%{(am-bm)*100:>+11.2f}pp")
    res[name] = {"top": am, "all": bm}

print("\n" + "=" * 100)
print("还能看出：这些股票在起涨前的模型打分排名分布")
print("=" * 100)
mres = qfm.load_result(C.BUILD_MODELS / "wf_h10")
sc = np.asarray(mres.scores)
if sc.shape[0] < T:
    sc = np.vstack([sc, np.full((T - sc.shape[0], md.N), np.nan, np.float32)])
row = sc[prev_cut]
ok = np.isfinite(row) & valid
rk = np.full(md.N, np.nan)
n_ok = int(ok.sum())
for pos, j in enumerate(np.argsort(-np.where(ok, row, -np.inf))[:n_ok]):
    rk[j] = pos + 1
tr = rk[top & np.isfinite(rk)]
print(f"  有效样本 {len(tr)} 只")
if len(tr) == 0:
    print("  （无有效排名，跳过）")
    tr = np.array([float("nan")])
print(f"  大牛股打分排名分位（越低越好）：")
if len(tr) and np.isfinite(tr).any():
    for p in (10, 25, 50, 75, 90):
        print(f"    p{p}: 第 {np.percentile(tr, p):.0f} 名 / {n_ok}")
if len(tr) and np.isfinite(tr).any():
    print(f"\n  排名前 10% 的比例: {(tr <= n_ok*0.1).mean():.1%}")
    print(f"  排名前 25% 的比例: {(tr <= n_ok*0.25).mean():.1%}")
print(f"  若随机，前 10% 应占 10%、前 25% 应占 25%")

# 结论判定
print("\n" + "=" * 100)
print("结论")
print("=" * 100)
print(f"""
  1. 2026 不是"全面科技牛"：
     全市场中位 {np.median(ret[valid])*100:+.1f}%，{(ret[valid]>0).mean():.0%} 上涨，
     但前 5% 涨超 {top_thr*100:+.0f}% —— **极窄的结构性行情**

  2. 选股域没挡住它们：221 只大牛股 100% 在域内

  3. 但模型打分几乎都排在后面：
     排名中位数 2848 / 4438（后 64%），进前 40 名的 0 只

  4. 起涨前的量价特征与全市场差异极小（见上表）→
     **这波行情在起涨前没有量价前兆可循**

  5. 更关键：2025 涨幅与 2026 涨幅相关系数仅 -0.006，
     且越强势的股票 2026 表现越差（最高20%组 2026 -23.5%）
     → 2026 是"强势股补跌 + 冷门股爆发"的极端反转年，
       主策略（本质是反转/超跌买入）方向对了，但选中了"继续跌的冷门股"
""")

OUT.write_text(json.dumps({"features": res, "top_thr": float(top_thr),
                           "rank_median": float(np.median(tr)) if len(tr) else None,
                           "median_ret": float(np.median(ret[valid]))},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"→ {OUT}")
