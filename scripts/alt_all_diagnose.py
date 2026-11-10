# -*- coding: utf-8 -*-
"""关键矛盾诊断：为什么另类数据"样本外大幅改善"但"全期下降"？

观察：
  wf_h10        : 全期 8.05% | 样本外22-25 5.43%
  wf_h10_m_all3 : 全期 6.97% | 样本外22-25 8.41%（+2.98pp！）

矛盾：样本外大幅改善，但全期反而更差 → 说明**验证期(2020-2021)大幅变差**。

必须查清：
  1) 验证期到底差多少
  2) 逐年表现
  3) 是不是"过度拟合 DEV 期"导致的（DEV 2017-2019 与 2020-21 风格不同）
  4) 换手率/成本是否暴增
  5) 另类因子在 2020-21 的 IC 表现
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

MODELS = ["wf_h10", "wf_h10_margin", "wf_h10_m_all2", "wf_h10_m_all3", "wf_h10_m_all4"]
navs, info = {}, {}
for tag in MODELS:
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
    info[tag] = {"n_trades": len(r.trades)}

print("=" * 104)
print("1) 分区间完整对比（含验证期）")
print("=" * 104)
print(f"{'模型':<20}{'验证20-21':>11}{'样本外22-25':>12}{'样本外22-26':>12}"
      f"{'全期':>9}{'交易笔数':>9}")
print("-" * 104)
segs = {"val": ("2020-01-08", "2021-12-31"),
        "os25": ("2022-01-01", "2025-12-31"),
        "oos": ("2022-01-01", E),
        "full": (S, E)}
rows_out = {}
for tag in MODELS:
    if tag not in navs:
        continue
    d = {}
    for k, (a, b) in segs.items():
        sub = navs[tag][(navs[tag].index >= a) & (navs[tag].index <= b)]
        p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        d[k] = {kk: float(p[kk]) for kk in ("cagr", "sharpe", "max_dd", "calmar")}
    rows_out[tag] = d
    print(f"{tag:<20}{d['val']['cagr']:>10.2%}{d['os25']['cagr']:>11.2%}"
          f"{d['oos']['cagr']:>11.2%}{d['full']['cagr']:>8.2%}"
          f"{info[tag]['n_trades']:>9}")

print("\n" + "=" * 104)
print("2) 逐年收益")
print("=" * 104)
ys = {t: qfm_metrics.yearly_returns(navs[t][navs[t].index >= S]) for t in navs}
print(f"{'年':>6}" + "".join(f"{t.replace('wf_h10','').replace('_m','') or 'base':>16}"
                              for t in navs))
print("-" * 104)
for y in sorted(ys["wf_h10"].index):
    line = f"{int(y):>6}"
    for t in navs:
        v = float(ys[t].loc[y]) if y in ys[t].index else 0.0
        line += f"{v:>15.2%}"
    print(line)

print("\n" + "=" * 104)
print("3) 关键：验证期(2020-2021)为什么变差？")
print("=" * 104)
for t in navs:
    if t == "wf_h10":
        continue
    dv = rows_out[t]["val"]["cagr"] - rows_out["wf_h10"]["val"]["cagr"]
    do = rows_out[t]["os25"]["cagr"] - rows_out["wf_h10"]["os25"]["cagr"]
    print(f"  {t:<20} 验证期 {dv*100:+6.2f}pp | 样本外 {do*100:+6.2f}pp")

print("\n" + "=" * 104)
print("4) 另类因子在验证期/样本外的 IC（检验是否只有 DEV 期有效）")
print("=" * 104)
zmw = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp("2019-12-31"))
val = np.asarray((dates_w >= "2020-01-01") & (dates_w <= "2021-12-31"))
oos = np.asarray(dates_w >= "2022-01-01")

# 从 pool 里找另类因子
for tag in ("wf_h10_m_all2", "wf_h10_m_all3", "wf_h10_m_all4"):
    if tag not in rows_out:
        continue
    try:
        res = qfm.load_result(C.BUILD_MODELS / tag)
    except Exception:
        continue
    alt = [k for k in res.pool if k.startswith(("mgn:", "lhb:", "alt:", "flow:"))]
    if not alt:
        continue
    print(f"\n  [{tag}] 池中另类因子 {len(alt)} 个:")
    # 从已保存的 IC 文件读
    for fn in ("margin_factor_ic.json", "billboard_factor_ic.json",
               "snapshot_factor_ic.json", "flow_factor_ic.json"):
        p = ROOT / "build" / "alt" / fn
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        for k in alt:
            if k in d:
                v = d[k]
                print(f"    {k:<20} DEV {v['dev_ic']:+.4f} | "
                      f"验证 {v.get('val_ic', float('nan')):+.4f} | "
                      f"样本外 {v.get('oos_ic', float('nan')):+.4f}")

print("\n" + "=" * 104)
print("5) 结论")
print("=" * 104)
print("""
  模式：另类数据越多 → 样本外越好，但验证期越差
  → 这是典型的「DEV 期(2017-2019)与验证期(2020-2021)风格不同」导致的
  需要判断：是"真信号"还是"DEV 期过拟合"。
""")
