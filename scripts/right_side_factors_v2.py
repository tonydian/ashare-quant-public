# -*- coding: utf-8 -*-
"""按用户材料验证并补充右侧因子：
  A. 方向库 296 因子按 base 语义精确拆分（确认"向上突破"到底有没有 alpha）；
  B. 实现材料提到但我们缺失的关键因子：
     1) 250 日新高距离（国信公式：1 - Close / max(Close,250)）
     2) 平稳创新高（位移路程比：|净涨| / Σ|日涨| 的 250 日版）
     3) 量能确认（突破日量 > 5日均量 × 1.8）
     4) OBV 创 30 日新高
     5) ATR(14) × 2.5 跟踪止损（出场层，不是因子）
     6) 防追高乖离（价格相对突破位乖离）
  C. 每个新因子做 DEV(h10) IC 检验，看能否救活"真右侧"。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, model as qfm, ops  # noqa: E402

OUT = C.BUILD_EVAL / "right_side_factors_v2.json"
md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dev = np.asarray(pd.to_datetime(md.dates[rows]) <= pd.Timestamp(C.DEV_END))
close = md.close.astype(np.float64)
high = md.high.astype(np.float64)
low = md.low.astype(np.float64)
vol = md.volume.astype(np.float64)
T, N = close.shape
print(f"面板 {T}×{N}，周频 {len(rows)} 行")


def f32(x):
    return np.asarray(x, dtype=np.float32)


def ic_stats(panel, name):
    """按 weekly_ic 口径算 DEV/全期 IC。"""
    key = name
    f = qfm.build_feature_block(md, rows, {key: panel[rows]}, key, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    d = ic[v & dev]
    stats = {"dev_ic": float(np.mean(d)) if len(d) else np.nan,
             "dev_icir": float(np.mean(d) / max(np.std(d), 1e-9)) if len(d) else np.nan,
             "full_ic": float(np.mean(ic[v])) if v.any() else np.nan,
             "pos_share": float((d > 0).mean()) if len(d) else np.nan}
    print(f"  {name:28s} DEV IC {stats['dev_ic']:+.4f} ICIR {stats['dev_icir']:+.3f} "
          f"| 全期 {stats['full_ic']:+.4f} | 正IC {stats['pos_share']:.0%}")
    return stats


res = {}

# ==================== A. 精确拆分：向上突破 vs 向下跌破
print("\n=== A. 方向库：向上突破 vs 向下跌破（语义拆分） ===")
m800 = pd.read_parquet(C.BUILD_EVAL / "factor_metrics_screen.parquet")
d = m800[m800.lib == "directional-alpha"]
for base, label_cn in (("breakout", "向上突破"), ("breakdown", "向下跌破"),
                       ("trend_strength", "风险调整趋势"), ("sma_gap", "均线偏离")):
    sub = d[d.base == base]
    res[f"dir_{base}"] = {
        "n": int(len(sub)), "dev_ic_mean": float(sub.dev_ic_5d.mean()),
        "full_ic_mean": float(sub.ic_5d.mean()),
        "pos_share": float((sub.dev_ic_5d > 0).mean()),
    }
    print(f"  {label_cn:8s}({base:14s}) n={len(sub):3d} DEV {sub.dev_ic_5d.mean():+.4f} "
          f"全期 {sub.ic_5d.mean():+.4f} 正占比 {(sub.dev_ic_5d>0).mean():.0%}")

# ==================== B. 新增因子
print("\n=== B. 材料提到但我们缺失的因子 ===")
cs = pd.DataFrame(close)

# 1) 250 日新高距离（国信公式）：越大=离新高越远
hi250 = cs.rolling(250, min_periods=200).max().to_numpy()
dist250 = np.where((hi250 > 0) & np.isfinite(hi250), 1.0 - close / hi250, np.nan)
res["new_dist_250high"] = ic_stats(f32(dist250), "250日新高距离(国信)")

# 2) 平稳创新高：位移路程比 = |close_t - close_{t-250}| / Σ|Δclose|（250日）
net = np.abs(close - np.roll(close, 250, axis=0))
daily = np.abs(np.diff(close, axis=0, prepend=np.nan))
path = pd.DataFrame(daily).rolling(250, min_periods=200).sum().to_numpy()
smooth250 = np.where(path > 0, net / path, np.nan)
smooth250[:250] = np.nan
res["new_smooth_newhigh"] = ic_stats(f32(smooth250), "平稳创新高(路程比250)")

# 3) 量能确认：突破日成交量相对 5 日均量倍数（>1.8 为有效突破）
vr = pd.DataFrame(vol / pd.DataFrame(vol).rolling(5, min_periods=5).mean().to_numpy()).to_numpy()
res["new_vol_ratio_break"] = ic_stats(f32(vr - 1.0), "量能倍数(对5日均量)")

# 4) OBV 创 30 日新高（0/1 → 用连续距离表示）
sign = np.sign(np.diff(close, axis=0, prepend=np.nan))
obv = np.nancumsum(np.where(np.isfinite(sign) & np.isfinite(vol), sign * vol, 0.0), axis=0)
obv20 = pd.DataFrame(obv)
obv_hi30 = obv20.rolling(30, min_periods=25).max().to_numpy()
obv_flag = np.where(np.isfinite(obv_hi30), (obv >= obv_hi30 - 1e-9).astype(np.float64), np.nan)
res["new_obv_high30"] = ic_stats(f32(obv_flag), "OBV创30日新高")

# 5) 防追高乖离：close 相对 20 日突破位的乖离
brk20 = cs.rolling(20, min_periods=15).max().shift(1).to_numpy()
dev_chase = np.where(np.isfinite(brk20) & (brk20 > 0), close / brk20 - 1.0, np.nan)
res["new_chase_dev"] = ic_stats(f32(dev_chase), "防追高乖离(距20日高)")

# 6) ATR 位置（防守层基准，非方向信号）：close / (ATR14*2.5) 用于止损距离
tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1, axis=0)),
                                       np.abs(low - np.roll(close, 1, axis=0))))
atr14 = pd.DataFrame(tr).rolling(14, min_periods=10).mean().to_numpy()
res["atr14_ratio"] = {"mean_atr_pct": float(np.nanmean(atr14 / close))}
print(f"  ATR(14)/close 均值 = {np.nanmean(atr14/close):.2%}（用于 2.5×ATR 跟踪止损）")

# 7) 行业热度：个股是否在近 7 日强势行业中（暂只算个股相对行业中位数）
ind = np.asarray(md.industry)
ret20 = cs / cs.shift(20) - 1.0
ind_med = np.full((T, N), np.nan)
ret20v = ret20.to_numpy()
for t in range(T):
    if t < 20:
        continue
    ids = ind[t]
    vals = ret20v[t]
    ok = np.isfinite(vals) & (ids >= 0)
    if ok.sum() < 50:
        continue
    med = {}
    for c in np.unique(ids[ok]):
        sel = ok & (ids == c)
        if sel.sum() >= 3:
            med[c] = np.nanmedian(vals[sel])
    for c, mv in med.items():
        sel = ids == c
        ind_med[t, sel] = mv
rel_to_ind = np.where(np.isfinite(ind_med) & np.isfinite(ret20v), ret20v - ind_med, np.nan)
res["new_relative_to_industry"] = ic_stats(f32(rel_to_ind), "个股相对行业中位数(20日)")

# ==================== C. 结论汇总
print("\n=== C. 结论：哪些新因子在 A 股 h10 上有正 IC ===")
positive = {k: v for k, v in res.items() if isinstance(v, dict) and (v.get("dev_ic") or -9) > 0}
for k, v in sorted(positive.items(), key=lambda x: -(x[1].get("dev_ic") or 0)):
    print(f"  {k:28s} DEV IC {v['dev_ic']:+.4f}")
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
