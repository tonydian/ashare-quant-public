# -*- coding: utf-8 -*-
"""融资融券模型深度诊断。

关键观察（初步）：
  wf_h10_margin     : 全期 7.40%（-0.65pp）| 样本外 4.95%（+0.50pp）
  wf_h10_margin_flow: 全期 8.00%（-0.05pp）| 样本外 5.17%（+0.72pp）
  → 全期持平/略降，但样本外提升，这是"减少过拟合"的典型迹象

需要验证：
  1) 逐年表现（不能靠单年拉动）
  2) 样本外各年的稳定性
  3) 剔除 2026（它是设计动机，可能污染判断）
  4) 风格暴露变化（融资融券因子是否引入了小市值/高波动暴露）
  5) 换手率与成本变化
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

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, metrics as qfm_metrics  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci  # noqa: E402

S, E = "2020-01-08", "2026-10-14"
md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
phase = int(rows[0]) % 5
start = int(md.date_index(S))
while start % 5 != phase:
    start += 1
end = int(md.date_index(E))
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)

MODELS = {
    "基线": "wf_h10",
    "两融": "wf_h10_margin",
    "两融+资金流": "wf_h10_margin_flow",
}
navs, trades_n = {}, {}
print("=" * 96)
print("1) 逐年收益")
print("=" * 96)
for tag, d in MODELS.items():
    p = C.BUILD_MODELS / d / "composite_score.npz"
    sc = np.load(p)["scores"]
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    elif sc.shape[0] > md.T:
        sc = sc[:md.T]
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=offset,
                              trade_band_rel=0.30, trade_band_abs=6000.0,
                              max_new_entries=6, cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    navs[tag] = r.nav
    trades_n[tag] = len(r.trades)

ys = {k: qfm_metrics.yearly_returns(v[v.index >= S]) for k, v in navs.items()}
tags = list(MODELS.keys())
print(f"{'年':>6}" + "".join(f"{t:>14}" for t in tags)
      + "".join(f"{'差'+t:>12}" for t in tags[1:]))
print("-" * (6 + 14 * len(tags) + 12 * (len(tags) - 1)))
for y in sorted(ys[tags[0]].index):
    vals = [float(ys[t].loc[y]) if y in ys[t].index else 0.0 for t in tags]
    line = f"{int(y):>6}" + "".join(f"{v:>13.2%}" for v in vals)
    line += "".join(f"{(vals[i]-vals[0])*100:>+11.2f}pp" for i in range(1, len(vals)))
    print(line)

# 剔除 2026 的对比
print("\n" + "=" * 96)
print("2) 剔除 2026（该年是设计动机，可能污染判断）")
print("=" * 96)
E2 = "2025-12-31"
print(f"{'模型':<16}{'2020-2025 CAGR':>16}{'Sharpe':>10}{'MaxDD':>10}{'Calmar':>9}")
print("-" * 62)
for tag in tags:
    sub = navs[tag][(navs[tag].index >= S) & (navs[tag].index <= E2)]
    p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    print(f"{tag:<16}{p['cagr']:>15.2%}{p['sharpe']:>10.3f}"
          f"{p['max_dd']:>10.2%}{p['calmar']:>9.2f}")

# 样本外（2022-2025，同样剔除 2026）
print(f"\n样本外 2022-2025（剔除 2026）：")
print(f"{'模型':<16}{'CAGR':>10}{'Sharpe':>10}{'MaxDD':>10}")
print("-" * 48)
for tag in tags:
    sub = navs[tag][(navs[tag].index >= "2022-01-01") & (navs[tag].index <= E2)]
    p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    print(f"{tag:<16}{p['cagr']:>9.2%}{p['sharpe']:>10.3f}{p['max_dd']:>10.2%}")

# ---------- 3) 逐年样本外（2022-2026）稳定性
print("\n" + "=" * 96)
print("3) 样本外逐年（2022-2026）")
print("=" * 96)
print(f"{'年':>6}" + "".join(f"{t:>14}" for t in tags))
print("-" * (6 + 14 * len(tags)))
oos_years = [y for y in sorted(ys[tags[0]].index) if y >= 2022]
win = {t: 0 for t in tags[1:]}
for y in oos_years:
    vals = [float(ys[t].loc[y]) if y in ys[t].index else 0.0 for t in tags]
    print(f"{int(y):>6}" + "".join(f"{v:>13.2%}" for v in vals))
    for i, t in enumerate(tags[1:], 1):
        if vals[i] > vals[0]:
            win[t] += 1
print(f"\n  样本外跑赢基线的年数: " +
      " | ".join(f"{t} {win[t]}/{len(oos_years)}" for t in tags[1:]))

# ---------- 4) 换手率与成本
print("\n" + "=" * 96)
print("4) 交易成本（换手是否上升）")
print("=" * 96)
print(f"{'模型':<16}{'交易笔数':>10}{'年均换手':>12}")
print("-" * 40)
yrs = len(navs[tags[0]][navs[tags[0]].index >= S]) / 252
for tag in tags:
    print(f"{tag:<16}{trades_n[tag]:>10}{trades_n[tag]/yrs:>11.0f}笔/年")

# ---------- 5) 分段 bootstrap（2022-2026 单独看）
print("\n" + "=" * 96)
print("5) 样本外(2022+)单独 bootstrap")
print("=" * 96)
base = navs[tags[0]]
for tag in tags[1:]:
    common = base.index.intersection(navs[tag].index)
    common = common[(common >= "2022-01-01") & (common <= E)]
    d = (navs[tag][common].pct_change() - base[common].pct_change()).dropna().to_numpy()
    ci = circular_block_bootstrap_ci(d, lambda x: float(np.mean(x) * 252),
                                     n_boot=2000, block_len=21)
    t = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(len(d)))
    print(f"  {tag:<14} 年化差 {np.mean(d)*252*100:+6.2f}% | "
          f"95%CI [{ci['lo']*100:+6.2f}%, {ci['hi']*100:+6.2f}%] | t={t:+5.2f}")

# ---------- 6) 风格暴露：持仓市值/波动分布
print("\n" + "=" * 96)
print("6) 因素暴露（持仓的平均市值/波动，看是否引入小市值倾向）")
print("=" * 96)
print("  （用 scores 最高的 40 只做近似，取 2026-01-05）")
t_chk = int(md.date_index("2026-01-05"))
while t_chk % 5 != phase:
    t_chk += 1
Cl = md.price.astype(np.float64)
vol20 = pd.DataFrame(Cl).pct_change().rolling(20).std().to_numpy()
for tag, d in MODELS.items():
    sc = np.load(C.BUILD_MODELS / d / "composite_score.npz")["scores"]
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    row = sc[t_chk]
    ok = np.isfinite(row) & md.eligible[t_chk]
    if ok.sum() < 40:
        continue
    top = np.argsort(-np.where(ok, row, -np.inf))[:40]
    mc = md.mcap[t_chk, top] / 1e8
    v = vol20[t_chk, top]
    print(f"  {tag:<14} 市值中位 {np.nanmedian(mc):8.1f}亿 | "
          f"20日波动中位 {np.nanmedian(v)*100:5.2f}%")
