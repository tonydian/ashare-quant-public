# -*- coding: utf-8 -*-
"""验证：30 日调仓是真的改进，还是 DEV 期过拟合？

疑点：
  DEV 期 30 日 CAGR 16.45%（异常高），OOS 只有 4.49%
  → 衰减 73%，需要确认不是"在 DEV 期挑到了运气好的参数"

检验方法（3 种，任一不过则否定该结论）：
  T1 参数敏感性：如果 30 日是"尖峰"，相邻参数应该大幅变差
     → 高原（25/30/35 都好）才是真信号，尖峰是噪声
  T2 逐年稳定性：30 日是否在多数年份都优于 10 日
  T3 滚动窗口：用不同起点重复，看结论是否稳健
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, metrics as qfm_metrics  # noqa: E402

OUT = ROOT / "build" / "eval" / "rebalance_robust.json"
SLIP = 40.0

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
phase = int(zm["rows"][0]) % 5
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
sc = np.asarray(res.scores)
if sc.shape[0] < md.T:
    sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])


def run(rb, start_s, end_s):
    start = int(md.date_index(start_s))
    while start % 5 != phase:
        start += 1
    end = int(md.date_index(end_s))
    offset = (phase - start) % rb if rb % 5 == 0 else 0
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(
            md, sc, start, end, risk_overlay=True, stop_loss=True,
            rebalance_days=rb, rebalance_offset=offset,
            trade_band_rel=0.30, trade_band_abs=6000.0,
            max_new_entries=6, cash_asset_returns=cash,
            slippage_bp=SLIP, verbose=False)
    finally:
        C.N_HOLDINGS = old
    s = r.nav[r.nav.index >= pd.Timestamp(start_s)]
    if len(s) < 60:
        return None, None
    p = qfm_metrics.performance(s, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}, s


# ================================================================ T1 参数敏感性
print("=" * 96)
print("T1 参数敏感性（是高原还是尖峰？）")
print("=" * 96)
print("  在 OOS 期（2022-2026）测相邻参数，看 30 日是否孤立")
print(f"\n{'周期':>5}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>8}")
print("-" * 46)

sweep = {}
for rb in (20, 25, 30, 35, 40, 50):
    perf, _ = run(rb, "2022-01-01", "2026-12-31")
    if perf is None:
        continue
    sweep[rb] = perf
    print(f"{rb:>5}{perf['cagr']:>9.2%}{perf['sharpe']:>9.3f}"
          f"{perf['max_dd']:>9.2%}{perf['calmar']:>8.2f}")

if sweep:
    cal = np.array([v["calmar"] for v in sweep.values()])
    cagr = np.array([v["cagr"] for v in sweep.values()])
    print(f"\n  Calmar 范围: {cal.min():.2f} ~ {cal.max():.2f}（跨度 {cal.max()-cal.min():.2f}）")
    print(f"  CAGR 范围: {cagr.min():.2%} ~ {cagr.max():.2%}"
          f"（跨度 {(cagr.max()-cagr.min())*100:.2f}pp）")
    # 高原判定：相邻参数差异是否平缓
    ks = sorted(sweep.keys())
    diffs = [abs(sweep[ks[i+1]]["cagr"] - sweep[ks[i]]["cagr"]) for i in range(len(ks)-1)]
    print(f"  相邻参数 CAGR 最大跳变: {max(diffs)*100:.2f}pp")
    if max(diffs) < 0.015:
        print("  → ✅ 高原形态（参数不敏感，结论稳健）")
    else:
        print("  → ⚠️ 存在尖峰，需谨慎")

# ================================================================ T2 逐年
print("\n" + "=" * 96)
print("T2 逐年对比：30 日 vs 10 日")
print("=" * 96)
print(f"{'年':>6}{'10日CAGR':>11}{'30日CAGR':>11}{'10日Sharpe':>12}{'30日Sharpe':>12}{'30日更优':>10}")
print("-" * 74)

yearly = {}
for y in range(2020, 2027):
    a, sa = run(10, f"{y}-01-01", f"{y}-12-31")
    b, sb = run(30, f"{y}-01-01", f"{y}-12-31")
    if a is None or b is None:
        continue
    better = "✅" if b["cagr"] > a["cagr"] else "❌"
    yearly[y] = {"d10": a, "d30": b, "d30_better": b["cagr"] > a["cagr"]}
    print(f"{y:>6}{a['cagr']:>10.2%}{b['cagr']:>10.2%}"
          f"{a['sharpe']:>12.3f}{b['sharpe']:>12.3f}{better:>10}")

if yearly:
    n_better = sum(1 for v in yearly.values() if v["d30_better"])
    print(f"\n  30 日在 {n_better}/{len(yearly)} 年更优")
    if n_better >= len(yearly) * 0.7:
        print("  → ✅ 多数年份一致，结论稳健")
    else:
        print("  → ⚠️ 结果不稳定，可能是运气")

# ================================================================ T3 滚动窗口
print("\n" + "=" * 96)
print("T3 滚动窗口：不同起点，30 日是否仍然更优")
print("=" * 96)
print(f"{'起点':>12}{'10日CAGR':>11}{'30日CAGR':>11}{'差异':>10}{'30日更优':>10}")
print("-" * 56)

roll = {}
for start_y in (2020, 2021, 2022, 2023):
    a, _ = run(10, f"{start_y}-01-01", "2026-12-31")
    b, _ = run(30, f"{start_y}-01-01", "2026-12-31")
    if a is None or b is None:
        continue
    d = (b["cagr"] - a["cagr"]) * 100
    roll[start_y] = {"d10": a["cagr"], "d30": b["cagr"], "delta_pp": d}
    print(f"{start_y:>6}年起{a['cagr']:>10.2%}{b['cagr']:>10.2%}"
          f"{d:>+9.2f}pp{('✅' if d > 0 else '❌'):>10}")

if roll:
    n = sum(1 for v in roll.values() if v["delta_pp"] > 0)
    print(f"\n  30 日在 {n}/{len(roll)} 个起点更优")

# ================================================================ 结论
print("\n" + "=" * 96)
print("结论")
print("=" * 96)

pass_t1 = bool(sweep) and max(
    abs(sweep[k]["cagr"] - sweep[k2]["cagr"])
    for k in sweep for k2 in sweep) < 0.05
pass_t2 = bool(yearly) and sum(1 for v in yearly.values() if v["d30_better"]) >= len(yearly) * 0.7
pass_t3 = bool(roll) and sum(1 for v in roll.values() if v["delta_pp"] > 0) >= len(roll) * 0.7

print(f"  T1 参数高原: {'✅ 通过' if pass_t1 else '⚠️ 需谨慎'}")
print(f"  T2 逐年稳定: {'✅ 通过' if pass_t2 else '⚠️ 不稳定'}")
print(f"  T3 滚动稳健: {'✅ 通过' if pass_t3 else '⚠️ 不稳定'}")

if pass_t1 and pass_t2 and pass_t3:
    print("\n  ✅ 三项全通过 → 30 日调仓是真实改进，不是过拟合")
else:
    print("\n  ⚠️ 有项目未通过 → 建议保守采纳（如 20 日）或继续观察")

json.dump({"sweep": {str(k): v for k, v in sweep.items()},
           "yearly": {str(k): v for k, v in yearly.items()},
           "rolling": {str(k): v for k, v in roll.items()},
           "pass_t1": pass_t1, "pass_t2": pass_t2, "pass_t3": pass_t3},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
