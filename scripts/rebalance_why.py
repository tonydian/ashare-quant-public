# -*- coding: utf-8 -*-
"""最终确认：为什么"调仓越慢越好"？这是经济逻辑还是另有原因？

观察：
  15~60 日全部通过判定，60 日换手最低（0.69x）且 Sharpe 更高
  → 这不像过拟合（过拟合会是尖峰），更像存在**真实的经济原因**

必须查清：为什么降频会变好？
  H1 成本节省（成本从 4.30% 降到 0.68%，省 3.62pp）
  H2 模型信号是 10 日 horizon，但实际 alpha 衰减比想象慢
  H3 频繁交易会"卖掉正在上涨的股票"（换手带宽太紧）
  H4 降频意外地减少了"追高杀低"

关键检验：
  · 如果只是成本，那在 **0 成本**下，10 日应该和 60 日一样好
  · 如果 0 成本下 10 日仍更差，说明是**交易本身**损害了收益
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

OUT = ROOT / "build" / "eval" / "rebalance_why.json"
FULL = ("2020-01-08", "2026-12-31")

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
phase = int(zm["rows"][0]) % 5
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
sc = np.asarray(res.scores)
if sc.shape[0] < md.T:
    sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])


def run(rb, slip):
    start_s, end_s = FULL
    start = int(md.date_index(start_s))
    while start % 5 != phase:
        start += 1
    end = int(md.date_index(end_s))
    offset = (phase - start) % rb if rb % 5 == 0 else 0
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    # 同时把佣金/印花归零以模拟"零成本"
    old_comm, old_min, old_stamp = C.COMMISSION_RATE, C.MIN_COMMISSION, C.STAMP_TAX_SELL
    try:
        if slip == 0:
            C.COMMISSION_RATE = 0.0
            C.MIN_COMMISSION = 0.0
            C.STAMP_TAX_SELL = 0.0
        r = qfbt.run_backtest(
            md, sc, start, end, risk_overlay=True, stop_loss=True,
            rebalance_days=rb, rebalance_offset=offset,
            trade_band_rel=0.30, trade_band_abs=6000.0,
            max_new_entries=6, cash_asset_returns=cash,
            slippage_bp=float(slip), verbose=False)
    finally:
        C.N_HOLDINGS = old
        C.COMMISSION_RATE, C.MIN_COMMISSION, C.STAMP_TAX_SELL = old_comm, old_min, old_stamp

    s = r.nav[r.nav.index >= pd.Timestamp(start_s)]
    if len(s) < 60:
        return None
    p = qfm_metrics.performance(s, rf=C.CASH_YIELD)
    tr = r.trades
    yrs = len(s) / 252
    to = float(tr["amount"].sum()) / float(s.mean()) / yrs if s.mean() > 0 else 0
    return {"cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
            "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
            "turnover": float(to), "n_trades": int(len(tr))}


print("=" * 96)
print("关键实验：零成本下，10 日 vs 慢调仓")
print("=" * 96)
print("  逻辑：如果差异只来自成本，零成本时两者应该接近")
print("        如果零成本时 10 日仍差，说明交易行为本身损害收益")

print(f"\n{'周期':>5}{'零成本CAGR':>13}{'零成本Sharpe':>14}"
      f"{'40bp成本CAGR':>15}{'成本拖累':>11}")
print("-" * 96)

rows = {}
for rb in (10, 20, 40, 60):
    z = run(rb, 0)
    c = run(rb, 40)
    if not (z and c):
        continue
    drag = z["cagr"] - c["cagr"]
    rows[rb] = {"zero": z, "cost": c, "drag": float(drag)}
    print(f"{rb:>5}{z['cagr']:>12.2%}{z['sharpe']:>14.3f}"
          f"{c['cagr']:>14.2%}{drag*100:>10.2f}pp")

print("\n" + "=" * 96)
print("分析")
print("=" * 96)

if 10 in rows and 60 in rows:
    z10, z60 = rows[10]["zero"], rows[60]["zero"]
    c10, c60 = rows[10]["cost"], rows[60]["cost"]

    print(f"\n  【零成本对比】")
    print(f"    10 日: CAGR {z10['cagr']:.2%}, Sharpe {z10['sharpe']:.3f}")
    print(f"    60 日: CAGR {z60['cagr']:.2%}, Sharpe {z60['sharpe']:.3f}")
    d_zero = (z60["cagr"] - z10["cagr"]) * 100
    print(f"    → 差异 {d_zero:+.2f}pp")

    print(f"\n  【40bp 成本对比】")
    print(f"    10 日: CAGR {c10['cagr']:.2%}, Sharpe {c10['sharpe']:.3f}")
    print(f"    60 日: CAGR {c60['cagr']:.2%}, Sharpe {c60['sharpe']:.3f}")
    d_cost = (c60["cagr"] - c10["cagr"]) * 100
    print(f"    → 差异 {d_cost:+.2f}pp")

    print(f"\n  【成本拖累】")
    print(f"    10 日: {rows[10]['drag']*100:.2f}pp（换手 {c10['turnover']:.2f}x）")
    print(f"    60 日: {rows[60]['drag']*100:.2f}pp（换手 {c60['turnover']:.2f}x）")

    print("\n" + "=" * 96)
    print("结论")
    print("=" * 96)
    if abs(d_zero) < 0.5:
        print(f"""
  ✅ 零成本下两者接近（差 {d_zero:+.2f}pp）
     → 差异**主要来自成本**，这是可解释、可信的改进

     成本拖累：10 日 {rows[10]['drag']*100:.2f}pp vs 60 日 {rows[60]['drag']*100:.2f}pp
     差距 {(rows[10]['drag']-rows[60]['drag'])*100:.2f}pp 正是降频省下的
""")
    else:
        print(f"""
  ⚠️ 零成本下已有差异（{d_zero:+.2f}pp）
     → 说明**不只是成本**，交易行为本身也有影响
     需要进一步分析（可能是换手带宽、信号衰减等）
""")

# ---- 额外：信号衰减检验
print("\n" + "=" * 96)
print("补充：模型信号的衰减速度（决定最优调仓频率）")
print("=" * 96)
# 用 composite score 与未来收益的关系
rows_w = zm["rows"].astype(np.int64)
T, N = md.T, md.N
close = md.price

print(f"  {'持有期':>8}{'IC(打分 vs 未来收益)':>22}")
print("-" * 34)
for h in (5, 10, 20, 40, 60):
    ics = []
    for wi, t in enumerate(rows_w):
        if t + h >= T:
            break
        s = sc[t]
        fwd = close[t + h] / close[t] - 1
        m = np.isfinite(s) & np.isfinite(fwd) & md.eligible[t]
        if m.sum() < 200:
            continue
        a, b = s[m], fwd[m]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        ics.append(np.corrcoef(a, b)[0, 1])
    if ics:
        arr = np.array(ics)
        print(f"{h:>7}日{np.mean(arr):>+12.4f} "
              f"(ICIR {np.mean(arr)/max(np.std(arr),1e-9):>+6.2f})")

json.dump({str(k): v for k, v in rows.items()},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
