# -*- coding: utf-8 -*-
"""诊断（修正版）：样本内 vs 样本外差距的根因。

md.bench 是**日收益序列**（不是价格），需要 cumprod 还原净值。
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


def ann_from_ret(r: pd.Series) -> float | None:
    """日收益序列 → 年化。"""
    r = r.dropna()
    if len(r) < 30:
        return None
    tot = float(np.prod(1 + r.to_numpy()) - 1)
    return float((1 + tot) ** (252 / len(r)) - 1)


print("=" * 100)
print("T1 基准指数：样本内(2020-21) vs 样本外(2022+)")
print("=" * 100)
bench_names = {"csi_all": "中证全指", "hs300": "沪深300",
               "csi500": "中证500", "csi1000": "中证1000", "all_eq": "等权全市场"}
print(f"{'基准':<14}{'验证期2020-21':>16}{'样本外2022+':>16}{'变化':>13}")
print("-" * 62)
bench_stats = {}
for key, name in bench_names.items():
    b = md.bench.get(key)
    if b is None:
        continue
    d = {}
    for tag, (s, e) in (("val", VAL), ("oos", OOS)):
        sub = b[(b.index >= s) & (b.index <= e)]
        a = ann_from_ret(sub)
        if a is not None:
            d[tag] = a
    if len(d) < 2:
        continue
    bench_stats[key] = d
    print(f"{name:<14}{d['val']:>15.2%}{d['oos']:>16.2%}"
          f"{(d['oos']-d['val'])*100:>+12.2f}pp")
res["bench"] = bench_stats

print("\n" + "=" * 100)
print("T2 主策略超额收益：若超额没衰减，就不是因子的错")
print("=" * 100)
nav = pd.read_parquet(ROOT / "build/runs/main/nav.parquet")["nav"].astype(float)
nav.index = pd.to_datetime(nav.index)
strat = {}
for tag, (s, e) in (("val", VAL), ("oos", OOS)):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    strat[tag] = ann_from_ret(sub.pct_change().dropna())
print(f"{'':<20}{'验证期':>14}{'样本外':>14}{'变化':>12}")
print("-" * 62)
print(f"{'策略收益':<20}{strat['val']:>13.2%}{strat['oos']:>14.2%}"
      f"{(strat['oos']-strat['val'])*100:>+11.2f}pp")
for key, name in bench_names.items():
    if key not in bench_stats:
        continue
    bv, bo = bench_stats[key]["val"], bench_stats[key]["oos"]
    ev, eo = strat["val"] - bv, strat["oos"] - bo
    print(f"{'超额 vs '+name:<20}{ev:>13.2%}{eo:>14.2%}{(eo-ev)*100:>+11.2f}pp")
    res.setdefault("excess", {})[key] = {"val": float(ev), "oos": float(eo)}
res["strategy"] = strat

print("\n" + "=" * 100)
print("T3 因子池的 IC 分年度（检验因子是否衰减）")
print("=" * 100)
model = qfm.load_result(C.BUILD_MODELS / "wf_h10")
cat = qff.load_catalog().set_index("factor_key")
inp = qff.FactorInputs(md)
feats = []
for k in model.pool:
    try:
        pan = qff.compute_factor(inp, cat.loc[k].to_dict())
        f = qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask)
        feats.append((k, qfm.weekly_ic(f, label, mask)))
    except Exception:
        continue
print(f"  成功 {len(feats)} / {len(model.pool)} 个因子")

years = dates_w.year
print(f"\n{'年':>6}{'平均IC':>10}{'IC>0占比':>10}")
print("-" * 30)
yearly = {}
for y in sorted(set(years)):
    vals = [float(np.mean(ic[(years == y) & np.isfinite(ic)]))
            for _, ic in feats if np.isfinite(ic[(years == y)]).sum() > 5]
    if len(vals) >= 5:
        yearly[int(y)] = {"mean": float(np.mean(vals)),
                          "pos": float((np.array(vals) > 0).mean())}
        print(f"{y:>6}{np.mean(vals):>+10.4f}{(np.array(vals)>0).mean():>10.0%}")
res["yearly_ic"] = yearly

iv = [float(np.mean(ic[(years >= 2020) & (years <= 2021) & np.isfinite(ic)]))
      for _, ic in feats if np.isfinite(ic[(years >= 2020) & (years <= 2021)]).sum() > 5]
io = [float(np.mean(ic[(years >= 2022) & np.isfinite(ic)]))
      for _, ic in feats if np.isfinite(ic[(years >= 2022)]).sum() > 5]
print(f"\n  验证期因子平均 IC: {np.mean(iv):+.4f}")
print(f"  样本外因子平均 IC: {np.mean(io):+.4f}")
print(f"  → 变化: {np.mean(io)-np.mean(iv):+.4f}")
print(f"  → **注意：这些是 IC 的绝对值；主策略用的是 |ICIR| 排序，负 IC 因子是被反向使用的**")
res["ic_val_oos"] = {"val": float(np.mean(iv)), "oos": float(np.mean(io))}

# 用 |IC| 衡量"因子有效性"（因为主模型允许反向使用）
iv_abs = [abs(float(np.mean(ic[(years >= 2020) & (years <= 2021) & np.isfinite(ic)])))
          for _, ic in feats if np.isfinite(ic[(years >= 2020) & (years <= 2021)]).sum() > 5]
io_abs = [abs(float(np.mean(ic[(years >= 2022) & np.isfinite(ic)])))
          for _, ic in feats if np.isfinite(ic[(years >= 2022)]).sum() > 5]
print(f"\n  用 |IC| 衡量（主模型允许反向使用因子）:")
print(f"    验证期 |IC| 均值: {np.mean(iv_abs):.4f}")
print(f"    样本外 |IC| 均值: {np.mean(io_abs):.4f}")
print(f"    → 绝对强度变化: {np.mean(io_abs)-np.mean(iv_abs):+.4f}")
res["ic_abs"] = {"val": float(np.mean(iv_abs)), "oos": float(np.mean(io_abs))}

print("\n" + "=" * 100)
print("T4 复合分的滚动 52 周 IC（看是否断崖）")
print("=" * 100)
sc = np.asarray(model.scores)
if sc.shape[0] < md.T:
    sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
ic_comp = qfm.weekly_ic(sc[rows], label, mask)
s_ic = pd.Series(ic_comp, index=dates_w)
roll = s_ic.rolling(52, min_periods=26).mean()
print(f"  {'年':<8}{'滚动52周IC':>14}")
print("-" * 24)
for y in sorted(set(dates_w.year)):
    sub = roll[roll.index.year == y]
    if sub.notna().sum() > 3:
        print(f"  {y:<8}{sub.mean():>+13.4f}")
res["rolling_ic"] = {int(y): float(roll[roll.index.year == y].mean())
                     for y in sorted(set(dates_w.year))
                     if roll[roll.index.year == y].notna().sum() > 3}

print("\n" + "=" * 100)
print("T5 逐年：策略 vs 基准")
print("=" * 100)
strat_y = qfm_metrics.yearly_returns(nav[nav.index >= VAL[0]])
print(f"{'年':>6}{'策略':>10}{'中证全指':>12}{'等权全市场':>13}{'超额(全指)':>13}")
print("-" * 60)
yr_rows = {}
for y in sorted(strat_y.index):
    sv = float(strat_y.loc[y])
    row = {"strategy": sv}
    line = f"{int(y):>6}{sv:>9.2%}"
    for key in ("csi_all", "all_eq"):
        b = md.bench.get(key)
        sub = b[b.index.year == y] if b is not None else pd.Series(dtype=float)
        bv = float(np.prod(1 + sub.to_numpy()) - 1) if len(sub) > 5 else float("nan")
        row[key] = bv
        line += f"{bv:>12.2%}" if key == "csi_all" else f"{bv:>13.2%}"
        if key == "csi_all":
            line += f"{sv-bv:>+13.2%}"
            row["excess_csi_all"] = sv - bv
    print(line)
    yr_rows[int(y)] = row
res["yearly"] = yr_rows

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
