# -*- coding: utf-8 -*-
"""保守方案验证：不选"最优参数"，而选"成本显著降低 + 收益不劣化"的参数。

背景：
  T1/T2/T3 显示 30 日有过拟合嫌疑（逐年 4/7、参数尖峰、50 日反而更好）。
  → 不能因为 30 日 Calmar 最高就采纳（那是数据窥探）。

更稳健的思路：
  目标不是"找到最赚钱的参数"，而是：
    "在**收益不劣化**的前提下，把成本降下来"

  判定标准（预注册，不看结果调整）：
    ① OOS CAGR 不劣于 10 日（差 ≤ -0.3pp 视为劣化）
    ② OOS Sharpe 不劣化
    ③ 换手率显著下降（≥30%）
    ④ 全期 CAGR 不劣化

  在满足 ①~④ 的参数里，选**换手最低**的（成本最省）。

为什么这样更稳健：
  · 不选"最赚钱"（那是对历史的最优拟合）
  · 选"成本最低且不亏"（经济逻辑更扎实）
  · 不依赖单一参数点（看整个区间）
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

OUT = ROOT / "build" / "eval" / "rebalance_conservative.json"
SLIP = 40.0
DEV = ("2020-01-08", "2021-12-31")
OOS = ("2022-01-01", "2026-12-31")
FULL = ("2020-01-08", "2026-12-31")

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
phase = int(zm["rows"][0]) % 5
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
sc = np.asarray(res.scores)
if sc.shape[0] < md.T:
    sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])


def run(rb, period):
    start_s, end_s = period
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
        return None
    p = qfm_metrics.performance(s, rf=C.CASH_YIELD)
    tr = r.trades
    yrs = len(s) / 252
    to = float(tr["amount"].sum()) / float(s.mean()) / yrs if s.mean() > 0 else 0.0
    per = 300000.0 / 40
    comm_bp = max(C.MIN_COMMISSION, per * C.COMMISSION_RATE) / per * 1e4
    rt = SLIP * 2 + comm_bp * 2 + C.STAMP_TAX_SELL * 1e4 + 0.2
    return {"cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
            "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
            "turnover": to, "ann_cost": float(to * rt * 1e-4),
            "n_trades": int(len(tr))}


print("=" * 96)
print("保守方案：在收益不劣化的前提下，最大化成本节省")
print("=" * 96)
print(f"  预注册判定标准（不看结果调整）：")
print(f"    ① OOS CAGR 不比 10 日差超过 0.3pp")
print(f"    ② OOS Sharpe 不劣化（≥ 10日 - 0.03）")
print(f"    ③ 换手率下降 ≥ 30%")
print(f"    ④ 全期 CAGR 不劣化")

print("\n" + "=" * 96)
print("全参数扫描（DEV / OOS / FULL）")
print("=" * 96)
print(f"{'周期':>5}{'DEV CAGR':>11}{'OOS CAGR':>11}{'FULL CAGR':>12}"
      f"{'OOS Sharpe':>12}{'换手':>8}{'年化成本':>10}")
print("-" * 96)

rows = {}
for rb in (10, 15, 20, 25, 30, 40, 50, 60):
    d = run(rb, DEV)
    o = run(rb, OOS)
    f = run(rb, FULL)
    if not (d and o and f):
        continue
    rows[rb] = {"dev": d, "oos": o, "full": f}
    print(f"{rb:>5}{d['cagr']:>10.2%}{o['cagr']:>11.2%}{f['cagr']:>12.2%}"
          f"{o['sharpe']:>12.3f}{o['turnover']:>7.2f}x{o['ann_cost']:>9.2%}")

if 10 not in rows:
    print("基准 10 日缺失，退出")
    sys.exit(1)

base = rows[10]
print(f"\n  基准（10 日）: OOS CAGR {base['oos']['cagr']:.2%}, "
      f"Sharpe {base['oos']['sharpe']:.3f}, 换手 {base['oos']['turnover']:.2f}x")

# ---- 判定
print("\n" + "=" * 96)
print("逐项判定")
print("=" * 96)
print(f"{'周期':>5}{'①OOS收益':>11}{'②OOS Sharpe':>14}{'③换手降':>10}"
      f"{'④全期':>9}{'结论':>8}")
print("-" * 96)

qualified = []
for rb, v in sorted(rows.items()):
    if rb == 10:
        continue
    d_cagr = (v["oos"]["cagr"] - base["oos"]["cagr"]) * 100
    c1 = d_cagr >= -0.3
    c2 = v["oos"]["sharpe"] >= base["oos"]["sharpe"] - 0.03
    to_drop = 1 - v["oos"]["turnover"] / base["oos"]["turnover"]
    c3 = to_drop >= 0.30
    c4 = v["full"]["cagr"] >= base["full"]["cagr"] - 0.002
    ok = c1 and c2 and c3 and c4
    if ok:
        qualified.append((rb, v, to_drop))
    mark = lambda x: "✅" if x else "❌"
    print(f"{rb:>5}{d_cagr:>+10.2f}pp{mark(c2):>9}"
          f"{to_drop:>9.0%}{mark(c4):>8}{'通过' if ok else '—':>8}")

print("\n" + "=" * 96)
print("结论")
print("=" * 96)
if qualified:
    # 选换手最低的
    best = min(qualified, key=lambda x: x[1]["oos"]["turnover"])
    rb, v, drop = best
    print(f"  ✅ 通过全部判定: {[q[0] for q in qualified]}")
    print(f"  → 推荐 **{rb} 日**（换手最低，成本最省）")
    print(f"\n  对比 10 日：")
    print(f"    OOS CAGR   {base['oos']['cagr']:.2%} → {v['oos']['cagr']:.2%}"
          f"（{(v['oos']['cagr']-base['oos']['cagr'])*100:+.2f}pp）")
    print(f"    OOS Sharpe {base['oos']['sharpe']:.3f} → {v['oos']['sharpe']:.3f}")
    print(f"    OOS MaxDD  {base['oos']['max_dd']:.2%} → {v['oos']['max_dd']:.2%}")
    print(f"    年化成本   {base['oos']['ann_cost']:.2%} → {v['oos']['ann_cost']:.2%}"
          f"（省 {(base['oos']['ann_cost']-v['oos']['ann_cost'])*100:.2f}pp）")
    print(f"    换手率     {base['oos']['turnover']:.2f}x → {v['oos']['turnover']:.2f}x"
          f"（-{drop:.0%}）")
    print(f"\n  全期：")
    print(f"    CAGR {base['full']['cagr']:.2%} → {v['full']['cagr']:.2%}")
    print(f"    Calmar {base['full']['calmar']:.2f} → {v['full']['calmar']:.2f}")
    out_best = rb
else:
    print("  ❌ 无参数通过全部判定 → 建议保持 10 日调仓")
    out_best = 10

json.dump({"slip": SLIP, "rows": {str(k): v for k, v in rows.items()},
           "qualified": [q[0] for q in qualified], "recommend": out_best},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
