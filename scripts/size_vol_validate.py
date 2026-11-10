# -*- coding: utf-8 -*-
"""B 方案深度验证：wf_h10_smallvol（小市值+高波动）值不值得用？

初步结果：
  全期 8.25%（+0.20pp）| Sharpe 0.692（+0.015）
  MaxDD -10.14%（改善 1.91pp）| Calmar 0.81（+0.14）
  2026 +0.62%（从 -0.38% 转正）
  但 t=0.22，完全不显著

关键疑问：
  1) 回撤改善 1.91pp 是真的还是运气？
  2) 逐年稳定性如何？
  3) 持仓是否真的更偏小市值/高波动？（暴露是否符合预期）
  4) 2026 转正是否只是"买中了几个大牛股"？
  5) 与 wf_h10_m_all3（另类数据版）相比哪个好？
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

TAGS = ["wf_h10", "wf_h10_smallvol", "wf_h10_m_all3"]
navs = {}
print("=" * 100)
print("1) 回测 + 逐年")
print("=" * 100)
for tag in TAGS:
    p = C.BUILD_MODELS / tag / "composite_score.npz"
    if not p.exists():
        continue
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

ys = {t: qfm_metrics.yearly_returns(v[v.index >= S]) for t, v in navs.items()}
print(f"{'年':>6}" + "".join(f"{t.replace('wf_h10','').replace('_','') or 'base':>16}"
                              for t in navs))
print("-" * 70)
for y in sorted(ys["wf_h10"].index):
    line = f"{int(y):>6}"
    for t in navs:
        v = float(ys[t].loc[y]) if y in ys[t].index else 0.0
        line += f"{v:>15.2%}"
    print(line)

print("\n" + "=" * 100)
print("2) 回撤改善的真实性：分段最大回撤")
print("=" * 100)
print(f"{'模型':<20}{'全期MDD':>10}{'2020-21':>10}{'2022-23':>10}{'2024-25':>10}{'2026':>10}")
print("-" * 74)
for t in navs:
    row = f"{t:<20}"
    for seg, (a, b) in {"full": (S, E), "a": (S, "2021-12-31"),
                        "b": ("2022-01-01", "2023-12-31"),
                        "c": ("2024-01-01", "2025-12-31"),
                        "d": ("2026-01-01", E)}.items():
        sub = navs[t][(navs[t].index >= a) & (navs[t].index <= b)]
        if len(sub) < 2:
            row += f"{'—':>10}"
            continue
        dd = float((sub / sub.cummax() - 1).min())
        row += f"{dd:>9.2%}"
    print(row)
print("\n  → 若每个分段都更好，说明不是单次运气")

print("\n" + "=" * 100)
print("3) 暴露验证：持仓是否真的更小市值/更高波动？")
print("=" * 100)
Cl = md.price.astype(np.float64)
vol20 = pd.DataFrame(Cl).pct_change().rolling(20).std().to_numpy()
for tag in navs:
    sc = np.load(C.BUILD_MODELS / tag / "composite_score.npz")["scores"]
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    mcs, vols = [], []
    # 抽样 20 个调仓周
    for wi in range(len(rows) - 1, len(rows) - 120, -6):
        t = rows[wi]
        row = sc[t]
        ok = np.isfinite(row) & md.eligible[t]
        if ok.sum() < 40:
            continue
        top = np.argsort(-np.where(ok, row, -np.inf))[:40]
        m = md.mcap[t, top]
        v = vol20[t, top]
        if np.isfinite(m).sum() > 30:
            mcs.append(np.nanmedian(m) / 1e8)
        if np.isfinite(v).sum() > 30:
            vols.append(np.nanmedian(v) * 100)
    if mcs:
        print(f"  {tag:<20} 市值中位 {np.median(mcs):7.1f}亿 | "
              f"20日波动中位 {np.median(vols):5.2f}%  （{len(mcs)} 期）")

print("\n" + "=" * 100)
print("4) 2026 转正的来源：是否只是买中几个大牛股？")
print("=" * 100)
t0 = int(md.date_index("2026-01-05"))
t1 = md.T - 1
ret26 = md.price[t1] / md.price[t0] - 1.0
valid = np.isfinite(ret26) & md.eligible[t0]
thr = np.percentile(ret26[valid], 95)
top = valid & (ret26 >= thr)
for tag in navs:
    sc = np.load(C.BUILD_MODELS / tag / "composite_score.npz")["scores"]
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    row = sc[t0]
    ok = np.isfinite(row) & md.eligible[t0]
    n_ok = int(ok.sum())
    rk = np.full(md.N, np.nan)
    for pos, j in enumerate(np.argsort(-np.where(ok, row, -np.inf))[:n_ok]):
        rk[j] = pos + 1
    tr = rk[top & np.isfinite(rk)]
    in40 = int((tr <= 40).sum()) if len(tr) else 0
    in200 = int((tr <= 200).sum()) if len(tr) else 0
    print(f"  {tag:<20} 大牛股进前40: {in40:2d} 只 | 进前200: {in200:3d} 只 | "
          f"排名中位 {np.median(tr):.0f}" if len(tr) else f"  {tag}: 无数据")

print("\n" + "=" * 100)
print("5) 显著性：smallvol vs base")
print("=" * 100)
if "wf_h10" in navs and "wf_h10_smallvol" in navs:
    for seg, (a, b) in (("全期", (S, E)), ("验证期", (S, "2021-12-31")),
                        ("样本外22-25", ("2022-01-01", "2025-12-31"))):
        common = navs["wf_h10"].index.intersection(navs["wf_h10_smallvol"].index)
        common = common[(common >= a) & (common <= b)]
        d = (navs["wf_h10_smallvol"][common].pct_change() -
             navs["wf_h10"][common].pct_change()).dropna().to_numpy()
        ci = circular_block_bootstrap_ci(d, lambda x: float(np.mean(x) * 252),
                                         n_boot=2000, block_len=21)
        t = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(len(d)))
        print(f"  {seg:<14} 年化差 {np.mean(d)*252*100:+6.2f}% | "
              f"CI [{ci['lo']*100:+5.2f}%, {ci['hi']*100:+5.2f}%] | t={t:+5.2f}")

print("\n" + "=" * 100)
print("6) 结论")
print("=" * 100)
print("""
  关键判断点：
  * 回撤是否每个分段都改善
  * 持仓是否真的更偏小市值/高波动
  * 2026 转正是否只因少数个股
  * 显著性
""")
