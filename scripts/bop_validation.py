# -*- coding: utf-8 -*-
"""BOP 因子深度验证：它是真 alpha 还是重复/噪声？

疑点：
  1) BOP = (close-open)/(high-low) 的 N 日均值 —— 本质是"日内收盘位置"的均值
     而现有库已有 `intraday` base（日内位置），可能高度重复
  2) IC 0.0429 是否稳定（分年度、分市场状态）
  3) 与现有 34 因子池的相关性（若高度相关 → 无增量）
  4) 加入模型后样本外是否有改善（最终检验）
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
from qf.stats_adjust import newey_west_tstat  # noqa: E402

OUT = ROOT / "build" / "eval" / "bop_validation.json"

md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(C.DEV_END))

O = md.price_open.astype(np.float64)
H = md.high.astype(np.float64)
L = md.low.astype(np.float64)
Cl = md.close.astype(np.float64)

rng = np.maximum(H - L, 1e-9)
body = Cl - O
bop1 = np.where(rng > 0, body / rng, 0.0)

# 与现有 intraday 因子对比（risk-pattern:intraday）
from qf import factors as qff  # noqa: E402

cat = qff.load_catalog().set_index("factor_key")
intraday_keys = [k for k in cat.index if str(cat.loc[k].get("base", "")) == "intraday"]
intraday_keys += [k for k in cat.index if str(cat.loc[k].get("base", "")) in ("upper_wick", "lower_wick", "range_ratio")]
print(f"现有 intraday 因子：{intraday_keys[:5]}")

# 计算 BOP 和现有 intraday
res = {}


def ic_series(panel, name):
    f = qfm.build_feature_block(md, rows, {name: panel[rows]}, name, mask)
    return qfm.weekly_ic(f, label, mask), f


bop = pd.DataFrame(bop1).rolling(10, min_periods=5).mean().to_numpy()
ic_bop, f_bop = ic_series(np.asarray(bop, np.float32), "BOP_10")
v = np.isfinite(ic_bop)
print("\n" + "=" * 92)
print("1) BOP_10 的 IC 统计")
print("=" * 92)
d = ic_bop[v & dev]
print(f"  DEV IC 均值 {np.mean(d):+.4f} | 标准差 {np.std(d):.4f} | 样本 {len(d)} 周")
nw = newey_west_tstat(ic_bop[v], lag=2)
print("  Newey-West t = %.2f（naive t=%.2f, n=%d）" % (nw["t"], nw["naive_t"], nw["n"]))
print(f"  IC>0 占比 {(d>0).mean():.1%}")

# 与现有 intraday 对比
print("\n" + "=" * 92)
print("2) 与现有 intraday 因子的关系（关键：是否重复）")
print("=" * 92)
best_corr = 0.0
best_key = None
for k in intraday_keys[:6]:
    try:
        pan = qff.compute_factor(qff.FactorInputs(md), cat.loc[k].to_dict())
        ic_i, f_i = ic_series(np.asarray(pan, np.float32), k)
        # 因子值相关性（在周频行上）
        a = f_bop.ravel()
        b = f_i.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 1000:
            c = float(np.corrcoef(a[m], b[m])[0, 1])
            if abs(c) > abs(best_corr):
                best_corr, best_key = c, k
            vi = np.isfinite(ic_i)
            di = ic_i[vi & dev]
            print(f"  {k:26s} DEV IC {np.mean(di):+.4f} | 与 BOP 相关 {c:+.3f}")
    except Exception as e:
        print(f"  {k}: {e}")

print(f"\n  → BOP 与现有 intraday 因子的最高相关：{best_corr:+.3f}（{best_key}）")
if abs(best_corr) > 0.8:
    print("  ⚠️ 高度重复，加进模型没有增量")
else:
    print("  ✅ 相关性不高，可能提供增量信息")

# 分年度稳定性
print("\n" + "=" * 92)
print("3) BOP_10 分年度 IC 稳定性")
print("=" * 92)
years = dates_w.year
for y in sorted(set(years)):
    m = (years == y) & np.isfinite(ic_bop)
    if m.sum() >= 10:
        print(f"  {y}: IC {np.mean(ic_bop[m]):+.4f} （{m.sum()} 周） "
              f"{'✅' if np.mean(ic_bop[m]) > 0 else '❌'}")
    res[int(y)] = float(np.mean(ic_bop[m])) if m.sum() else None

# 分市场状态（大盘涨跌）
print("\n" + "=" * 92)
print("4) BOP_10 在不同市场状态下的表现")
print("=" * 92)
bench = md.bench_prices
if bench is not None:
    bench_w = bench[rows]
    ret_bench = np.zeros(len(rows))
    ret_bench[4:] = bench_w[4:] / bench_w[:-4] - 1.0   # 20 日
    up = ret_bench > 0
    for name, m in (("大盘上涨", up), ("大盘下跌", ~up)):
        mm = m & np.isfinite(ic_bop) & dev
        if mm.sum() >= 10:
            print(f"  {name}: IC {np.mean(ic_bop[mm]):+.4f}（{mm.sum()} 周）")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
