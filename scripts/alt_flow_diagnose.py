# -*- coding: utf-8 -*-
"""资金流模型深度分析：为什么全期改善但样本外变差？

关键疑点：
  1) 只看全期会掩盖结构问题 —— 必须逐年看
  2) 资金流因子 IC 是负的（流入越多越跌），这意味着模型在"反向使用"
  3) 混合模型 pool 从 34 → 37，多了 4 个资金流因子，可能挤掉了原来有效的量价因子
  4) 需要检验：资金流因子是否只是"换手率的另一种表达"（与 TURN_Z 高相关）
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

ALT = ROOT / "build" / "alt"
S, E = "2020-01-08", "2026-10-14"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
n_w = len(rows)
phase = int(rows[0]) % 5

# ---------- 1) 逐年对比
print("=" * 96)
print("1) 逐年收益：基线 vs 纯资金流 vs 混合")
print("=" * 96)
start = int(md.date_index(S))
while start % 5 != phase:
    start += 1
end = int(md.date_index(E))
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)

navs = {}
for tag, d in (("基线", "wf_h10"), ("纯资金流", "wf_flow"), ("混合", "wf_h10_flow")):
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

ys = {k: qfm_metrics.yearly_returns(v[v.index >= S]) for k, v in navs.items()}
keys = ["基线", "纯资金流", "混合"]
print(f"{'年':>6}" + "".join(f"{k:>12}" for k in keys) + f"{'混合-基线':>12}")
print("-" * (6 + 12 * 3 + 12))
for y in sorted(ys["基线"].index):
    vals = [float(ys[k].loc[y]) if y in ys[k].index else 0.0 for k in keys]
    print(f"{int(y):>6}" + "".join(f"{v:>11.2%}" for v in vals)
          + f"{(vals[2]-vals[0])*100:>+11.2f}pp")

# ---------- 2) pool 差异：资金流挤掉了谁
print("\n" + "=" * 96)
print("2) 因子池差异：混合模型挤掉了哪些量价因子？")
print("=" * 96)
base_pool = set(qfm.load_result(C.BUILD_MODELS / "wf_h10").pool)
mix_pool = set(qfm.load_result(C.BUILD_MODELS / "wf_h10_flow").pool)
flow_in_mix = {k for k in mix_pool if k.startswith("flow:")}
print(f"  基线 pool {len(base_pool)} | 混合 pool {len(mix_pool)}")
print(f"  混合新增资金流: {sorted(flow_in_mix)}")
print(f"\n  基线有、混合没有的因子（被挤掉的）:")
for k in sorted(base_pool - mix_pool):
    print(f"    - {k}")
print(f"\n  混合新增的非资金流因子:")
for k in sorted(mix_pool - base_pool - flow_in_mix):
    print(f"    + {k}")

# ---------- 3) 资金流因子与换手率的相关性（是否只是换手的另一种表达）
print("\n" + "=" * 96)
print("3) 资金流因子是否只是「换手率」的另一种表达？")
print("=" * 96)
from qf import factors as qff
cat = qff.load_catalog()
inp = qff.FactorInputs(md)
zp = np.load(ALT / "flow_panel.npz")
net_pct, r0_pct, turn = zp["net_pct"], zp["r0_pct"], zp["turnover"]


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.sum()).to_numpy()


flow_panels = {
    "flow:FLOW_TURN": roll(net_pct, 20) / np.where(roll(turn, 20) > 0, roll(turn, 20), np.nan),
    "flow:R0_MA20": roll(r0_pct, 20),
    "flow:FLOW_MA20": roll(net_pct, 20),
}
# 找一个换手率/成交量类量价因子做对比
vol_keys = [k for k in cat["factor_key"] if "ts_rank_volume" in str(
    cat.loc[cat["factor_key"] == k, "base"].iloc[0])][:2]
print(f"  对比用的量价因子: {vol_keys}")

for fk, pan in flow_panels.items():
    f = qfm.build_feature_block(md, rows, {fk: np.asarray(pan, np.float32)[rows]}, fk, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    dev = np.asarray(dates_w <= pd.Timestamp("2019-12-31"))
    print(f"\n  {fk}: DEV IC {np.mean(ic[v & dev]):+.4f}")
    for vk in vol_keys:
        try:
            panv = qff.compute_factor(inp, cat.loc[vk].to_dict())
            fv = qfm.build_feature_block(md, rows, {vk: panv[rows]}, vk, mask)
            a, b = f.ravel(), fv.ravel()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 1000:
                print(f"    与 {vk} 相关: {np.corrcoef(a[m], b[m])[0,1]:+.3f}")
        except Exception:
            pass

# ---------- 4) 结论
print("\n" + "=" * 96)
print("4) 结论")
print("=" * 96)
print(f"""
  全期: 基线 8.05% → 混合 8.18%（+0.13pp），回撤 -12.05% → -11.01%（改善 1.04pp）
  样本外: 基线 4.45% → 混合 4.11%（-0.34pp）← **变差**
  t 值 0.16 → 不显著

  → 改善主要来自"回撤更小"而非"收益更高"，
    且样本外收益下降，说明资金流因子在近年失效或与量价因子冗余。
""")
