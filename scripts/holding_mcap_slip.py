# -*- coding: utf-8 -*-
"""补充：主策略持仓的市值分布 + 估算真实滑点。

为什么重要：
  当前假设滑点 15bp（单边）。但滑点与市值/流动性强相关：
    · 大盘股（>500亿）：盘口厚，5~15bp
    · 中盘股（100~500亿）：10~25bp
    · 小盘股（<100亿）：20~50bp

如果主策略持仓集中在中小盘，15bp 就是低估的。

方法：
  1. 统计主策略实际持仓的市值分布
  2. 按分位估算加权滑点
  3. 与该滑点档位的回测结果对照
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

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "holding_mcap.json"
START, END = "2020-01-08", "2026-12-31"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
phase = int(zm["rows"][0]) % 5
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
sc = np.asarray(res.scores)
if sc.shape[0] < md.T:
    sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])

start = int(md.date_index(START))
while start % 5 != phase:
    start += 1
end = int(md.date_index(END))
offset = (phase - start) % 40

old = C.N_HOLDINGS
C.N_HOLDINGS = 40
try:
    r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                          rebalance_days=40, rebalance_offset=offset,
                          trade_band_rel=0.30, trade_band_abs=6000.0,
                          max_new_entries=6, cash_asset_returns=cash,
                          slippage_bp=15.0, verbose=False)
finally:
    C.N_HOLDINGS = old

tr = r.trades
print("=" * 92)
print("主策略持仓市值分布")
print("=" * 92)

# 买入交易的市值
buys = tr[tr["side"] == "buy"] if "side" in tr.columns else tr
mcaps = []
dates_pos = {d.normalize(): i for i, d in enumerate(md.dates)}
code_pos = {str(c).zfill(6): j for j, c in enumerate(md.codes)}

for _, row in buys.iterrows():
    try:
        dt = pd.Timestamp(row["date"]).normalize()
        di = dates_pos.get(dt)
        code = str(row["code"]).zfill(6)
        j = code_pos.get(code)
        if di is None or j is None:
            continue
        m = md.mcap[di, j]
        if np.isfinite(m) and m > 0:
            mcaps.append(float(m) / 1e8)      # 亿元
    except Exception:
        continue

mc = np.array(mcaps)
if len(mc) == 0:
    print("无持仓市值数据")
    sys.exit(0)

print(f"  样本: {len(mc)} 笔买入")
print(f"\n{'分位':>8}{'市值(亿元)':>14}")
print("-" * 26)
for q in (5, 10, 25, 50, 75, 90, 95):
    print(f"{q:>7}%{np.percentile(mc, q):>13.1f}")
print(f"{'均值':>7} {mc.mean():>13.1f}")

# 分布
bins = [(0, 50), (50, 100), (100, 300), (300, 500), (500, 1000), (1000, 1e9)]
print(f"\n{'市值区间':>16}{'占比':>10}{'笔数':>8}")
print("-" * 36)
dist = []
for lo, hi in bins:
    n = int(((mc >= lo) & (mc < hi)).sum())
    pct = n / len(mc)
    lbl = f"{lo}~{hi if hi < 1e9 else '∞'}亿"
    print(f"{lbl:>16}{pct:>9.1%}{n:>8}")
    dist.append({"lo": lo, "hi": None if hi >= 1e9 else hi,
                 "n": n, "pct": float(pct)})

# ---- 估算滑点
# 经验映射（A股，基于盘口深度与冲击成本的保守估计）
def est_slip(mcap_yi: float) -> float:
    if mcap_yi >= 1000:
        return 8.0
    if mcap_yi >= 500:
        return 12.0
    if mcap_yi >= 300:
        return 18.0
    if mcap_yi >= 100:
        return 28.0
    if mcap_yi >= 50:
        return 40.0
    return 55.0


slips = np.array([est_slip(x) for x in mc])
w_slip = float(slips.mean())
med_slip = float(np.median(slips))

print("\n" + "=" * 92)
print("估算真实滑点（按市值映射）")
print("=" * 92)
print(f"  市值加权平均滑点: **{w_slip:.1f} bp**（单边）")
print(f"  中位数: {med_slip:.1f} bp")
print(f"  当前假设: {C.SLIPPAGE_BP:.1f} bp")
print(f"  → 低估 {w_slip - C.SLIPPAGE_BP:+.1f} bp")

print(f"\n  滑点映射依据（A股经验，保守）：")
for lo, hi, sp in ((1000, None, 8), (500, 1000, 12), (300, 500, 18),
                   (100, 300, 28), (50, 100, 40), (0, 50, 55)):
    lbl = f"{lo}~{hi}亿" if hi else f">{lo}亿"
    print(f"    {lbl:<14} {sp} bp")

# ---- 对照成本敏感性结果
cs_path = ROOT / "build" / "eval" / "cost_sensitivity.json"
if cs_path.exists():
    cs = json.load(open(cs_path, encoding="utf-8"))["results"]
    print("\n" + "=" * 92)
    print("若真实滑点 = 估算值，回测会是多少？")
    print("=" * 92)
    target = int(round(w_slip / 5) * 5)
    target = max(0, min(60, target))
    print(f"  估算滑点 {w_slip:.1f}bp → 取最接近档位 {target}bp")
    for k in (15, 20, 25, 30, 40):
        if str(k) in cs:
            mark = " ← 估算值" if k == target else ""
            print(f"    {k:>2}bp: CAGR {cs[str(k)]['cagr']:6.2%} "
                  f"Sharpe {cs[str(k)]['sharpe']:.3f}{mark}")

out = {"samples": len(mc), "mean_mcap_yi": float(mc.mean()),
       "median_mcap_yi": float(np.median(mc)),
       "p10": float(np.percentile(mc, 10)),
       "p90": float(np.percentile(mc, 90)),
       "est_slip_bp_weighted": w_slip,
       "est_slip_bp_median": med_slip,
       "current_assumption_bp": float(C.SLIPPAGE_BP),
       "distribution": dist}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
               encoding="utf-8")
print(f"\n→ {OUT}")
