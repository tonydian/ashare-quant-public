# -*- coding: utf-8 -*-
"""BOP 因子与右侧模型的适配性测试。

关键差异：
  * 主策略 wf_h10：全市场选股（universe=None）
  * 右侧模型 wf_rs10：**只买收盘 > MA60 的股票**（universe=above_ma60）

BOP 是"日内买盘强度"，直觉上与"右侧"（已经启动的股票）可能更契合。
但必须实测：**在 above_ma60 子集里，BOP 的 IC 是多少？**

测试：
  T1 BOP 在全市场的 IC（已知 +0.0429）
  T2 BOP 在 above_ma60 子集内的 IC（关键）
  T3 BOP 与右侧模型 6 个因子的相关性
  T4 BOP 加入右侧模型后的样本外效果
  T5 按"距 MA60 距离"分组，看 BOP 的表现是否随距离变化
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

from qf import config as C, data as qfdata, model as qfm, factors as qff  # noqa: E402

OUT = ROOT / "build" / "eval" / "bop_right_side.json"

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
bop1 = np.where(rng > 0, (Cl - O) / rng, np.nan)
bop10 = pd.DataFrame(bop1).rolling(10, min_periods=5).mean().to_numpy().astype(np.float32)

T, N = Cl.shape
print(f"面板 {T}×{N}")

# ---------- 构造 above_ma60 掩码（与右侧模型同口径）
ma60 = pd.DataFrame(Cl).rolling(60, min_periods=40).mean().to_numpy()
above60 = np.isfinite(Cl) & np.isfinite(ma60) & (Cl > ma60)
print(f"above_ma60 覆盖率（全样本）: {above60.mean():.1%}")
print(f"above_ma60 覆盖率（周频行）: {above60[rows].mean():.1%}")

mask_all = mask.copy()
mask_above = mask & above60[rows]
print(f"周频 eligible: {mask_all.sum():,} | above_ma60 内: {mask_above.sum():,} "
      f"({mask_above.sum()/mask_all.sum():.1%})")


def ic_stats(panel, name, msk, lbl=None, dev_mask=None):
    """在指定掩码下算 IC。"""
    lab = lbl if lbl is not None else label
    dm = dev_mask if dev_mask is not None else dev
    f = qfm.build_feature_block(md, rows, {name: panel[rows]}, name, msk)
    ic = qfm.weekly_ic(f, lab, msk)
    v = np.isfinite(ic)
    if v.sum() < 10:
        return None
    d = ic[v & dm]
    if len(d) < 10:
        return None
    st = {"dev_ic": float(np.mean(d)),
          "dev_icir": float(np.mean(d) / max(np.std(d), 1e-9)),
          "full_ic": float(np.mean(ic[v])),
          "pos_share": float((d > 0).mean()),
          "n_weeks": int(len(d))}
    return st


print("\n" + "=" * 94)
print("T1/T2 BOP 在不同选股域下的 IC（关键对比）")
print("=" * 94)
res = {}
for label_cn, msk in (("全市场（主策略口径）", mask_all),
                      ("above_MA60（右侧模型口径）", mask_above)):
    st = ic_stats(bop10, "BOP10", msk)
    if st:
        res[label_cn] = st
        print(f"\n  [{label_cn}]  样本 {msk.sum():,} 个股票-周")
        print(f"    DEV IC {st['dev_ic']:+.4f} | ICIR {st['dev_icir']:+.3f} | "
              f"正IC {st['pos_share']:.0%} | {st['n_weeks']} 周")

# ---------- T3 与右侧模型因子的相关性
print("\n" + "=" * 94)
print("T3 BOP 与右侧模型 6 个因子的相关性")
print("=" * 94)
RS_POOL = ['directional:R241', 'directional:R076', 'directional:R340',
           'directional:R465', 'directional:R729', 'directional:R498']
cat = qff.load_catalog().set_index("factor_key")
inp = qff.FactorInputs(md)

f_bop = qfm.build_feature_block(md, rows, {"BOP10": bop10[rows]}, "BOP10", mask_above)
corr_list = {}
for k in RS_POOL:
    try:
        pan = qff.compute_factor(inp, cat.loc[k].to_dict())
        f_k = qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask_above)
        a, b = f_bop.ravel(), f_k.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 1000:
            c = float(np.corrcoef(a[m], b[m])[0, 1])
            corr_list[k] = c
            st = ic_stats(pan, k, mask_above)
            ic_v = st["dev_ic"] if st else float("nan")
            print(f"  {k:22s} 与BOP相关 {c:+.3f} | 自身 IC {ic_v:+.4f}")
    except Exception as e:
        print(f"  {k}: {e}")

if corr_list:
    mx = max(corr_list.items(), key=lambda x: abs(x[1]))
    print(f"\n  → 最高相关 {mx[1]:+.3f}（{mx[0]}）")
    res["max_corr_with_rs"] = {"factor": mx[0], "corr": mx[1]}

# ---------- T5 按距 MA60 距离分组
print("\n" + "=" * 94)
print("T5 BOP 表现 vs 距 MA60 距离（右侧程度）")
print("=" * 94)
dist60 = np.where(np.isfinite(ma60) & (ma60 > 0), Cl / ma60 - 1.0, np.nan)
fwd = md.fwd_open_ret(10)[rows]
bop_w = bop10[rows]

# 分 5 组：深跌、MA60下方、刚上MA60、MA60上方中等、远离MA60
bins = [-9, -0.15, 0, 0.05, 0.15, 9]
names = ["深跌<-15%", "MA60下-15~0%", "刚上0~5%", "上方5~15%", "远离>15%"]
print(f"\n  {'分组':<16}{'股票-周数':>10}{'BOP高分组收益':>14}{'BOP低分组收益':>14}{'差值':>9}")
for i in range(len(bins) - 1):
    sel = (dist60[rows] >= bins[i]) & (dist60[rows] < bins[i + 1]) & mask
    if sel.sum() < 5000:
        continue
    f = bop_w[sel]
    r = fwd[sel]
    ok = np.isfinite(f) & np.isfinite(r)
    if ok.sum() < 1000:
        continue
    thr = np.nanquantile(f[ok], 0.8)
    thr_lo = np.nanquantile(f[ok], 0.2)
    hi = r[ok & (f >= thr)]
    lo = r[ok & (f <= thr_lo)]
    print(f"  {names[i]:<16}{ok.sum():>10,}{hi.mean()*100:>13.2f}%{lo.mean()*100:>13.2f}%"
          f"{(hi.mean()-lo.mean())*100:>8.2f}pp")
    res[f"dist_bin_{names[i]}"] = {
        "n": int(ok.sum()),
        "hi_ret": float(hi.mean()), "lo_ret": float(lo.mean()),
        "spread": float(hi.mean() - lo.mean()),
    }

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
