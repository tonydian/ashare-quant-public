# -*- coding: utf-8 -*-
"""诊断：样本内(2020-2021) 与 样本外(2022+) 差距为何这么大？

三个竞争假说，逐个检验：
  H1 市场变了（beta 假说）：2020-2021 是牛市，2022+ 是熊市/震荡。
     若策略的超额收益（vs 基准）没有明显衰减，则不是因子的错。
  H2 因子衰减（alpha 假说）：因子的 IC 在 2022 后系统性下降。
  H3 波动率/回撤环境变了：同样的 alpha 在低波动市场里更难赚钱。

检验方法：
  T1 基准指数分区间收益（csi_all / hs300 / 中证500 / 中证1000 / 等权全市场）
  T2 主策略的超额收益（策略 - 基准）分区间
  T3 因子池 34 个因子的 IC 分年度走势（是否衰减）
  T4 滚动 52 周 IC（看衰减拐点）
  T5 策略 vs 基准的逐年对比
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

from qf import config as C, data as qfdata, factors as qff, model as qfm, metrics as qfm_metrics  # noqa: E402

OUT = ROOT / "build" / "eval" / "decay_diagnosis.json"

md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])

VAL = ("2020-01-08", "2021-12-31")
OOS = ("2022-01-01", "2026-10-14")
res = {}

# ============================================================
print("=" * 100)
print("T1 基准指数：样本内 vs 样本外（这是最重要的对照）")
print("=" * 100)
bench_names = {"csi_all": "中证全指", "hs300": "沪深300",
               "csi500": "中证500", "csi1000": "中证1000", "all_eq": "等权全市场"}
print(f"{'基准':<12}{'验证期2020-21':>16}{'样本外2022+':>16}{'差':>12}")
print("-" * 60)
bench_stats = {}
for key, name in bench_names.items():
    if key not in md.bench:
        continue
    b = md.bench[key]
    if not isinstance(b, pd.Series):
        continue
    for tag, (s, e) in (("val", VAL), ("oos", OOS)):
        sub = b[(b.index >= s) & (b.index <= e)]
        if len(sub) < 30:
            continue
        tot = sub.iloc[-1] / sub.iloc[0] - 1
        yrs = len(sub) / 252
        bench_stats.setdefault(key, {})[tag] = float((1 + tot) ** (1 / yrs) - 1)
    v = bench_stats[key].get("val")
    o = bench_stats[key].get("oos")
    if v is not None and o is not None:
        print(f"{name:<12}{v:>15.2%}{o:>16.2%}{(o-v)*100:>+11.2f}pp")
res["bench"] = bench_stats

# ============================================================
print("\n" + "=" * 100)
print("T2 主策略的超额收益（策略 - 基准）：若超额没衰减，就不是因子的错")
print("=" * 100)
nav = pd.read_parquet(ROOT / "build/runs/main/nav.parquet")["nav"].astype(float)
nav.index = pd.to_datetime(nav.index)


def ann(sub):
    if len(sub) < 30:
        return None
    tot = sub.iloc[-1] / sub.iloc[0] - 1
    return float((1 + tot) ** (252 / len(sub)) - 1)


strat = {}
for tag, (s, e) in (("val", VAL), ("oos", OOS)):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    strat[tag] = ann(sub)
print(f"{'':<16}{'验证期':>14}{'样本外':>14}{'变化':>12}")
print("-" * 58)
print(f"{'策略收益':<16}{strat['val']:>13.2%}{strat['oos']:>14.2%}"
      f"{(strat['oos']-strat['val'])*100:>+11.2f}pp")
for key, name in bench_names.items():
    if key not in bench_stats or "val" not in bench_stats[key]:
        continue
    bv, bo = bench_stats[key]["val"], bench_stats[key]["oos"]
    ev, eo = strat["val"] - bv, strat["oos"] - bo
    print(f"{'超额 vs '+name:<16}{ev:>13.2%}{eo:>14.2%}{(eo-ev)*100:>+11.2f}pp")
    res.setdefault("excess", {})[key] = {"val": float(ev), "oos": float(eo)}

# ============================================================
print("\n" + "=" * 100)
print("T3 因子池 34 个因子的 IC 分年度走势（检验因子是否衰减）")
print("=" * 100)
model = qfm.load_result(C.BUILD_MODELS / "wf_h10")
pool = list(model.pool)
cat = qff.load_catalog().set_index("factor_key")
inp = qff.FactorInputs(md)

feats = []
for k in pool:
    try:
        pan = qff.compute_factor(inp, cat.loc[k].to_dict())
        f = qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask)
        ic = qfm.weekly_ic(f, label, mask)
        feats.append((k, ic))
    except Exception:
        continue
print(f"  成功计算 {len(feats)} / {len(pool)} 个因子")

years = dates_w.year
print(f"\n{'年':>6}{'平均IC':>10}{'IC>0占比':>10}{'样本周数':>10}")
print("-" * 40)
yearly_ic = {}
for y in sorted(set(years)):
    vals = []
    for k, ic in feats:
        m = (years == y) & np.isfinite(ic)
        if m.sum() > 5:
            vals.append(float(np.mean(ic[m])))
    if len(vals) < 5:
        continue
    yearly_ic[int(y)] = {"mean": float(np.mean(vals)),
                         "pos": float((np.array(vals) > 0).mean()),
                         "n_weeks": int(((years == y)).sum())}
    print(f"{y:>6}{np.mean(vals):>+10.4f}{(np.array(vals)>0).mean():>10.0%}"
          f"{yearly_ic[int(y)]['n_weeks']:>10}")
res["yearly_ic"] = yearly_ic

# 样本内外对比
ic_val = [float(np.mean(ic[(years >= 2020) & (years <= 2021) & np.isfinite(ic)]))
          for _, ic in feats if np.isfinite(ic[(years >= 2020) & (years <= 2021)]).sum() > 5]
ic_oos = [float(np.mean(ic[(years >= 2022) & np.isfinite(ic)]))
          for _, ic in feats if np.isfinite(ic[(years >= 2022)]).sum() > 5]
print(f"\n  验证期(2020-21) 因子平均 IC: {np.mean(ic_val):+.4f}（{len(ic_val)} 个因子）")
print(f"  样本外(2022+)  因子平均 IC: {np.mean(ic_oos):+.4f}（{len(ic_oos)} 个因子）")
print(f"  → IC 变化: {(np.mean(ic_oos)-np.mean(ic_val)):+.4f}")
res["ic_val_oos"] = {"val": float(np.mean(ic_val)), "oos": float(np.mean(ic_oos))}

# ============================================================
print("\n" + "=" * 100)
print("T4 复合分的滚动 52 周 IC（看是否在某个时点断崖）")
print("=" * 100)
composite = np.asarray(model.scores)
if composite.shape[0] > len(rows):
    composite = composite[:len(rows)]
ic_comp = qfm.weekly_ic(composite[rows] if composite.shape[0] > rows[-1] else composite,
                        label, mask) if composite.shape[0] >= len(rows) else None
if ic_comp is None:
    # 直接用 scores 按 rows 取
    sc_row = np.asarray(model.scores)[rows]
    ic_comp = qfm.weekly_ic(sc_row, label, mask)
s_ic = pd.Series(ic_comp, index=dates_w)
roll = s_ic.rolling(52, min_periods=26).mean()
print(f"  {'时段':<16}{'滚动52周IC均值':>16}")
print("-" * 36)
for y in (2020, 2021, 2022, 2023, 2024, 2025, 2026):
    sub = roll[roll.index.year == y]
    if sub.notna().sum() > 3:
        print(f"  {y:<16}{sub.mean():>+15.4f}")
res["rolling_ic"] = {int(y): float(roll[roll.index.year == y].mean())
                     for y in (2020, 2021, 2022, 2023, 2024, 2025, 2026)
                     if roll[roll.index.year == y].notna().sum() > 3}

# ============================================================
print("\n" + "=" * 100)
print("T5 策略 vs 基准：逐年收益")
print("=" * 100)
strat_y = qfm_metrics.yearly_returns(nav[nav.index >= VAL[0]])
print(f"{'年':>6}{'策略':>10}{'中证全指':>12}{'等权全市场':>13}{'超额(全指)':>13}")
print("-" * 58)
for y in sorted(strat_y.index):
    sv = float(strat_y.loc[y])
    out_row = f"{int(y):>6}{sv:>9.2%}"
    for key in ("csi_all", "all_eq"):
        if key not in md.bench:
            out_row += f"{'—':>12}"
            continue
        b = md.bench[key]
        if not isinstance(b, pd.Series):
            out_row += f"{'—':>12}"
            continue
        sub = b[b.index.year == y]
        bv = float(sub.iloc[-1] / sub.iloc[0] - 1) if len(sub) > 1 else float("nan")
        out_row += f"{bv:>12.2%}" if key == "csi_all" else f"{bv:>13.2%}"
        if key == "csi_all":
            out_row += f"{sv-bv:>+13.2%}"
    print(out_row)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
