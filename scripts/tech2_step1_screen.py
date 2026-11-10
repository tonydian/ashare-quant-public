# -*- coding: utf-8 -*-
"""第一步：在**只用 DEV 期（2017-2019）**的前提下，检验哪些因子能捕捉 2026 那类行情。

严格约束：
  * 因子筛选只用 DEV 期（2017-01-01 ~ 2019-12-31）
  * 2020-2021（验证期）、2022-2026（样本外）**不参与任何选择**
  * 2026 只作为最终检查，不回头调参

目标画像（来自 2026 大牛股的事前特征，但**用 DEV 期数据验证是否存在同类规律**）：
  A 动量：60/120 日涨幅高
  B 高波动
  C 小市值（在 DEV 期检验小市值是否有效）
  D 成交放大

方法：对候选因子在 DEV 期算 Rank IC / ICIR，按 ICIR 排序，看哪些进入前列。
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

from qf import config as C, data as qfdata, factors as qff, model as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "tech2_factor_screen.json"
DEV_END = "2019-12-31"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(DEV_END))
print(f"周频 {len(rows)} 行，DEV 期 {dev.sum()} 周（{dates_w[0].date()} ~ {DEV_END}）")

# ---------------- 载入全部候选因子（含 momentum-extra）
zf = np.load(C.BUILD_FACTORS / "weekly_top.npz")
panels = {k: zf[k] for k in zf.files}
cat = qff.load_catalog()
extra_pairs = [
    (C.BUILD_FACTORS / "momentum_extra.npz", C.BUILD_FACTORS / "momentum_catalog.parquet"),
    (C.BUILD_FACTORS / "fundamental_extra.npz", C.BUILD_FACTORS / "fundamental_catalog.parquet"),
]
for ep, cp in extra_pairs:
    if ep.exists() and cp.exists():
        ze = np.load(ep)
        ec = pd.read_parquet(cp)
        n = 0
        for k in ze.files:
            if k not in panels and ze[k].shape[0] == len(rows):
                panels[k] = ze[k]
                n += 1
        cat = pd.concat([cat, ec[ec["factor_key"].isin(panels)]], ignore_index=True)
        print(f"  并入 {ep.stem}: {n} 个")
panels = {k: v for k, v in panels.items() if k in set(cat["factor_key"])}
print(f"候选因子总数: {len(panels)}")

# ---------------- DEV 期 IC
print("\n[计算 DEV 期 IC] …", flush=True)
rows_dev = rows[dev]
mask_dev = mask[dev]
label_dev = label[dev]
recs = []
for i, (k, pan) in enumerate(panels.items()):
    try:
        f = qfm.build_feature_block(md, rows_dev, {k: pan[dev]}, k, mask_dev)
        ic = qfm.weekly_ic(f, label_dev, mask_dev)
        v = np.isfinite(ic)
        if v.sum() < 30:
            continue
        m = float(np.mean(ic[v]))
        sd = float(np.std(ic[v]))
        recs.append({"factor_key": k, "ic": m, "icir": m / max(sd, 1e-9),
                     "pos": float((ic[v] > 0).mean()), "n": int(v.sum())})
    except Exception:
        continue
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(panels)}", flush=True)

df = pd.DataFrame(recs)
df["abs_ic"] = df["ic"].abs()
df["abs_icir"] = df["icir"].abs()
print(f"  有效因子 {len(df)} 个")

# ---------------- 按"能否捕捉动量/波动/小市值"分类
base_col = "base"
df = df.merge(cat[["factor_key", base_col]], on="factor_key", how="left")

PANEL_A = {"return", "skip_return", "breakout", "sma_gap", "ema_gap",
           "trend_strength", "sma_slope", "dual_ema_gap"}
PANEL_B = {"realized_vol", "atr_ratio", "range_ratio", "drawdown",
           "downside_vol", "ret_skew", "ret_kurt"}
PANEL_D = {"volume_ratio", "amount_ratio", "dollar_volume", "obv_slope",
           "volume_z", "ts_rank_volume", "price_volume_corr"}

print("\n" + "=" * 96)
print("DEV 期（2017-2019）因子表现：按 类别 分组")
print("=" * 96)
for tag, bases in (("A 动量/趋势", PANEL_A), ("B 波动/极值", PANEL_B), ("D 成交量", PANEL_D)):
    sub = df[df[base_col].isin(bases)]
    if len(sub) == 0:
        continue
    print(f"\n  【{tag}】{len(sub)} 个因子")
    print(f"    |IC| 中位 {sub['abs_ic'].median():.4f} | |ICIR| 中位 {sub['abs_icir'].median():.3f}")
    top = sub.nlargest(8, "abs_icir")
    for r in top.itertuples():
        d = "正" if r.ic > 0 else "负"
        print(f"      {r.factor_key:<26} IC {r.ic:+.4f} ICIR {r.icir:+.3f} ({d})")

# ---------------- 最优 TOP30（DEV 期，|ICIR| 排序）
print("\n" + "=" * 96)
print("DEV 期 |ICIR| 排名 TOP 30")
print("=" * 96)
print(f"  {'因子':<28}{'base':<16}{'IC':>9}{'ICIR':>8}{'方向':>6}")
for r in df.nlargest(30, "abs_icir").itertuples():
    d = "正" if r.ic > 0 else "负"
    print(f"  {r.factor_key:<28}{str(r.base)[:14]:<16}{r.ic:>+8.4f}{r.icir:>+8.3f}{d:>6}")

# ---------------- 关键检验：DEV 期动量是否有效？
print("\n" + "=" * 96)
print("关键检验：DEV 期（2017-2019）动量因子是正 IC 还是负 IC？")
print("=" * 96)
mom = df[df[base_col].isin(PANEL_A)]
if len(mom):
    print(f"  动量类 {len(mom)} 个因子中：")
    print(f"    IC > 0 （追涨有效）: {(mom['ic']>0).sum()} 个（{(mom['ic']>0).mean():.0%}）")
    print(f"    IC < 0 （反转有效）: {(mom['ic']<0).sum()} 个（{(mom['ic']<0).mean():.0%}）")
    print(f"    平均 IC: {mom['ic'].mean():+.4f}")
    print(f"\n  → 若 DEV 期动量普遍为负 IC，说明 A 股 2017-2019 是反转市，")
    print(f"    用 DEV 期选因子的任何模型都会倾向反转，2026 的动量行情仍然抓不到。")

vol = df[df[base_col].isin(PANEL_B)]
if len(vol):
    print(f"\n  波动类 {len(vol)} 个因子：平均 IC {vol['ic'].mean():+.4f}")
    print(f"    IC > 0: {(vol['ic']>0).sum()} / IC < 0: {(vol['ic']<0).sum()}")

OUT.write_text(json.dumps({
    "dev_weeks": int(dev.sum()),
    "n_factors": int(len(df)),
    "top30": df.nlargest(30, "abs_icir")[["factor_key", "base", "ic", "icir"]].to_dict("records"),
    "momentum_avg_ic": float(mom["ic"].mean()) if len(mom) else None,
    "momentum_pos_share": float((mom["ic"] > 0).mean()) if len(mom) else None,
    "vol_avg_ic": float(vol["ic"].mean()) if len(vol) else None,
}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
df.to_parquet(ROOT / "build" / "eval" / "tech2_dev_ic.parquet")
print(f"\n→ {OUT}")
