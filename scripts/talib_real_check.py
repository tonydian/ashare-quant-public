# -*- coding: utf-8 -*-
"""用真实 TA-Lib 复核关键结论（numpy 实现是否与 TA-Lib 一致）。

校验：
  1) TA-Lib 158 个函数的实际清单
  2) BOP 的 TA-Lib 版本 vs 我们的 numpy 版本，IC 是否一致
  3) TA-Lib 的形态识别（真实 CDL* 函数）IC 是否也是噪声
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import talib

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, model as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "talib_real_check.json"

print("=" * 94)
print(f"TA-Lib {talib.__version__} 函数总数：{len(talib.get_functions())}")
print("=" * 94)
groups = talib.get_function_groups()
print(f"分组数：{len(groups)}")
for g, fns in groups.items():
    print(f"  {g:<24} {len(fns):3d} 个：{', '.join(fns[:6])}{'…' if len(fns) > 6 else ''}")

# 载入数据
md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dev = np.asarray(pd.to_datetime(md.dates[rows]) <= pd.Timestamp(C.DEV_END))
O = md.price_open.astype(np.float64)
H = md.high.astype(np.float64)
L = md.low.astype(np.float64)
Cl = md.close.astype(np.float64)
T, N = Cl.shape

res = {}


def ic_of(panel, name):
    f = qfm.build_feature_block(md, rows, {name: panel[rows]}, name, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    if v.sum() < 10:
        return None
    d = ic[v & dev]
    st = {"dev_ic": float(np.mean(d)), "pos_share": float((d > 0).mean())}
    flag = "✅" if st["dev_ic"] > 0.005 else ("🟡" if st["dev_ic"] > 0 else "❌")
    print(f"  {name:24s} DEV IC {st['dev_ic']:+.4f} | 正IC {st['pos_share']:.0%} {flag}")
    res[name] = st
    return st


print("\n" + "=" * 94)
print("一、BOP：真实 TA-Lib vs numpy 实现对比")
print("=" * 94)

# TA-Lib BOP（逐列计算，要循环 N 列）
bop_talib = np.full((T, N), np.nan)
for j in range(N):
    if not np.isfinite(Cl[:, j]).any():
        continue
    try:
        bop_talib[:, j] = talib.BOP(O[:, j], H[:, j], L[:, j], Cl[:, j])
    except Exception:
        pass

# numpy 版本
rng = np.maximum(H - L, 1e-9)
bop_np = np.where(rng > 0, (Cl - O) / rng, np.nan)

# 一致性检验
m = np.isfinite(bop_talib) & np.isfinite(bop_np)
print(f"  两版本可比值 {m.sum():,} 个")
if m.sum() > 1000:
    corr = float(np.corrcoef(bop_talib[m], bop_np[m])[0, 1])
    mx = float(np.max(np.abs(bop_talib[m] - bop_np[m])))
    print(f"  相关系数 {corr:.6f} | 最大绝对差 {mx:.2e}")
    print(f"  → {'✅ 完全一致' if corr > 0.9999 else '⚠️ 有差异'}")

print("\n  TA-Lib BOP 单日 IC：", end="")
ic_of(np.asarray(bop_talib, np.float32), "BOP_talib")
print("  numpy BOP 单日 IC：  ", end="")
ic_of(np.asarray(bop_np, np.float32), "BOP_numpy")

# 10 日均值
bop10 = pd.DataFrame(bop_np).rolling(10, min_periods=5).mean().to_numpy()
print("  BOP 10日均值 IC：    ", end="")
ic_of(np.asarray(bop10, np.float32), "BOP10")

print("\n" + "=" * 94)
print("二、真实 TA-Lib 形态识别（CDL*）的 IC")
print("=" * 94)
cdl_fns = [f for f in talib.get_functions() if f.startswith("CDL")]
print(f"  共 {len(cdl_fns)} 个形态函数，取常见的 10 个实测\n")

test_cdl = ["CDLHAMMER", "CDLDOJI", "CDLENGULFING", "CDL3WHITESOLDIERS",
            "CDLMORNINGSTAR", "CDLEVENINGSTAR", "CDLSHOOTINGSTAR",
            "CDLHARAMI", "CDLPIERCING", "CDLDARKCLOUDCOVER"]
for fn in test_cdl:
    if fn not in cdl_fns:
        continue
    arr = np.full((T, N), np.nan)
    for j in range(N):
        if not np.isfinite(Cl[:, j]).any():
            continue
        try:
            arr[:, j] = getattr(talib, fn)(O[:, j], H[:, j], L[:, j], Cl[:, j])
        except Exception:
            pass
    nz = np.count_nonzero(np.nan_to_num(arr))
    print(f"  [{fn}] 非零值 {nz:,}", end="  ")
    ic_of(np.asarray(arr, np.float32), fn)

print("\n" + "=" * 94)
print("三、TA-Lib 动量类（CCI/MFI/AROON/BOP）")
print("=" * 94)
for fn, args in (("CCI", "high,low,close"), ("MFI", "high,low,close,volume"),
                 ("AROONOSC", "high,low")):
    arr = np.full((T, N), np.nan)
    V = md.volume.astype(np.float64)
    for j in range(N):
        if not np.isfinite(Cl[:, j]).any():
            continue
        try:
            if fn == "CCI":
                arr[:, j] = talib.CCI(H[:, j], L[:, j], Cl[:, j])
            elif fn == "MFI":
                arr[:, j] = talib.MFI(H[:, j], L[:, j], Cl[:, j], V[:, j])
            elif fn == "AROONOSC":
                arr[:, j] = talib.AROONOSC(H[:, j], L[:, j])
        except Exception:
            pass
    print(f"  [{fn}]", end="  ")
    ic_of(np.asarray(arr, np.float32), fn + "_talib")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
