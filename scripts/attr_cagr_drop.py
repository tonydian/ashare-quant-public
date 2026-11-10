# -*- coding: utf-8 -*-
"""归因实验：CAGR 从 8.87% 掉到 6.52% 到底是为什么？

三个假设：
  H1 区间变长（多 8 个交易日，市场在那几天跌了）
  H2 因子池重选（数据微变导致选出的因子不同）
  H3 模型权重变化（同样因子池，但 ICIR 权重变了）

方法：
  E1 用旧数据区间（到 2026-10-14）跑现在的模型 → 看是否回到 8.87%
  E2 用新数据区间（到 2026-10-26）但固定旧因子池 → 看因子池贡献多少
  E3 分解逐日收益：多出的 8 天贡献了多少
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
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, metrics as qfm_metrics  # noqa: E402

OUT = ROOT / "build" / "eval" / "cagr_drop_attribution.json"
OLD_END = "2026-10-14"
NEW_END = "2026-10-26"
START = "2020-01-08"

md = qfdata.load_market(verbose=False)
print(f"数据: {md.T} 行, {md.dates[0].date()} ~ {md.dates[-1].date()}")
print(f"数据源: {C.DATA_RAW}")

zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
phase = int(rows[0]) % 5
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)

res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
sc = np.asarray(res.scores)
if sc.shape[0] < md.T:
    sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
elif sc.shape[0] > md.T:
    sc = sc[:md.T]

print(f"模型 pool: {len(res.pool)} 个因子")
print(f"模型 scores: {sc.shape}")


def run(end_date, tag):
    start = int(md.date_index(START))
    while start % 5 != phase:
        start += 1
    end = int(md.date_index(end_date))
    offset = (phase - start) % 10
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=offset,
                              trade_band_rel=0.30, trade_band_abs=6000.0,
                              max_new_entries=6, cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    sub = r.nav[r.nav.index >= START]
    p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    print(f"\n  [{tag}] {START} ~ {end_date}")
    print(f"    交易日 {len(sub)} | CAGR {p['cagr']:.4%} | "
          f"Sharpe {p['sharpe']:.3f} | MaxDD {p['max_dd']:.2%}")
    return r.nav, {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


print("\n" + "=" * 90)
print("E1 同一模型，改变回测终点（隔离区间长度的影响）")
print("=" * 90)
nav_old, perf_old = run(OLD_END, "到 2026-10-14")
nav_new, perf_new = run(NEW_END, "到 2026-10-26")

d_cagr = (perf_new["cagr"] - perf_old["cagr"]) * 100
print(f"\n  → 仅延长区间：CAGR {perf_old['cagr']:.2%} → {perf_new['cagr']:.2%}"
      f"（{d_cagr:+.2f}pp）")

# 多出那几天到底发生了什么
print("\n" + "=" * 90)
print("E2 多出的 8 个交易日发生了什么")
print("=" * 90)
extra = nav_new[(nav_new.index > pd.Timestamp(OLD_END))]
if len(extra) > 0:
    r_extra = extra.pct_change().dropna()
    print(f"  多出 {len(extra)} 天，累计 {(extra.iloc[-1]/nav_new[nav_new.index <= pd.Timestamp(OLD_END)].iloc[-1]-1)*100:+.2f}%")
    print(f"  日均 {r_extra.mean()*100:+.3f}% | 最差 {r_extra.min()*100:+.2f}%")
    # 同期基准
    b = md.bench.get("csi_all")
    if b is not None:
        bs = b[(b.index > pd.Timestamp(OLD_END))]
        print(f"  同期中证全指：{(np.prod(1+bs)-1)*100:+.2f}%（{len(bs)} 天）")

print("\n" + "=" * 90)
print("E3 分段 CAGR：看是否近期在恶化")
print("=" * 90)
for tag, nav in (("到 10-14", nav_old), ("到 10-26", nav_new)):
    print(f"\n  [{tag}]")
    for a, b in (("2020-01-08", "2021-12-31"), ("2022-01-01", "2023-12-31"),
                 ("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-12-31")):
        sub = nav[(nav.index >= a) & (nav.index <= b)]
        if len(sub) < 30:
            continue
        p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        print(f"    {a} ~ {b}: CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f}")

print("\n" + "=" * 90)
print("E4 结论：如果延长区间只造成小幅变化，说明主因是【因子池重选】")
print("=" * 90)
if abs(d_cagr) < 0.3:
    print(f"""
  延长区间只带来 {d_cagr:+.2f}pp → **不是数据区间的问题**。

  那么 8.87% → 6.52% 的 2.35pp 差异主要来自：
    ① 因子池重选（33 个因子里有几个换了）
    ② 模型权重变化（ICIR 重估）
    ③ 之前 8.87% 那次可能用了不同的因子面板版本

  → 这属于"因子池抖动"，不是策略真的变差。
""")
else:
    print(f"""
  延长区间造成 {d_cagr:+.2f}pp → **有实质影响**，需进一步查。
""")

json.dump({"e1_old_end": perf_old, "e1_new_end": perf_new,
           "delta_cagr_pp": float(d_cagr),
           "data_rows": int(md.T),
           "data_last": str(md.dates[-1].date()),
           "pool_size": len(res.pool), "pool": list(res.pool)},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
