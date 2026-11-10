# -*- coding: utf-8 -*-
"""诊断：wf_style2 为什么失败？

假说：
  H1 篮子过于粗糙：只用 8 个因子 × 简单加权，信息量远小于 34 因子的 ICIR+ridge blend
  H2 风格切换追不上：滚动 IC 滞后 2 周 + 26 周窗口 = 反应太慢
  H3 等权兜底有害：两篮子都无效时各 50%，等于用一半"反转"去对冲一半"动量"
  H4 反转篮子选错：R_rev 用的是 reversal/breakdown 系列，但主模型最好的反转因子在 directional:R604 等
     （实际 R_rev 里就有 R604，所以可能不是这个）
  H5 平滑过度：0.6 平滑让权重停留在历史状态

检验方式：逐项拆解，看哪个环节损失最大。
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

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
n_w = len(rows)
phase = int(rows[0]) % 5

# ---------- 1) 各篮子单独的 IC 质量
print("=" * 96)
print("H1 各篮子单独表现（看篮子本身是否够强）")
print("=" * 96)
meta = json.load(open(C.BUILD_MODELS / "wf_style2" / "model_meta.json", encoding="utf-8"))
dev = np.asarray(dates_w <= pd.Timestamp(meta["dev_end"]))

zf = np.load(C.BUILD_FACTORS / "weekly_top.npz")
panels = {k: zf[k] for k in zf.files}
cat = qf_cat = None
from qf import factors as qff
cat = qff.load_catalog()
for ep, cp in ((C.BUILD_FACTORS / "momentum_extra.npz",
                C.BUILD_FACTORS / "momentum_catalog.parquet"),):
    if ep.exists():
        ze = np.load(ep)
        ec = pd.read_parquet(cp)
        for k in ze.files:
            if k not in panels and ze[k].shape[0] == n_w:
                panels[k] = ze[k]
panels = {k: v for k, v in panels.items() if k in set(cat["factor_key"])}

print(f"\n  篮子构成:")
for bk, lst in meta["baskets"].items():
    print(f"    {bk}: {lst}")

# ---------- 2) 风格权重时序 vs 实际最优风格
print("\n" + "=" * 96)
print("H2/H5 风格权重是否踩对点？")
print("=" * 96)
sc = np.load(C.BUILD_MODELS / "wf_style2" / "composite_score.npz")["scores"]
sc_w = sc[rows]
ic_style = qfm.weekly_ic(sc_w, label, mask)
s_ic = pd.Series(ic_style, index=dates_w)
print(f"\n  wf_style2 复合分滚动 52 周 IC:")
roll = s_ic.rolling(52, min_periods=26).mean()
for y in sorted(set(dates_w.year)):
    sub = roll[roll.index.year == y]
    if sub.notna().sum() > 3:
        print(f"    {y}: {sub.mean():+.4f}")

model_h = qfm.load_result(C.BUILD_MODELS / "wf_h10")
ic_h = qfm.weekly_ic(np.asarray(model_h.scores)[rows], label, mask)
roll_h = pd.Series(ic_h, index=dates_w).rolling(52, min_periods=26).mean()
print(f"\n  wf_h10 复合分滚动 52 周 IC（对照）:")
for y in sorted(set(dates_w.year)):
    sub = roll_h[roll_h.index.year == y]
    if sub.notna().sum() > 3:
        print(f"    {y}: {sub.mean():+.4f}")

# ---------- 3) 回测对比：只用一个篮子会怎样
print("\n" + "=" * 96)
print("H3 单篮子回测 vs 混合（看混合是否拖累）")
print("=" * 96)
t0 = int(md.date_index("2020-01-08"))
t_end = int(md.date_index("2026-10-14"))
start = t0
while start % 5 != phase:
    start += 1
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)


def run(sc_T, tag):
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc_T, start, t_end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=offset,
                              trade_band_rel=0.30, trade_band_abs=6000.0,
                              max_new_entries=6, cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    sub = r.nav[(r.nav.index >= "2020-01-08") & (r.nav.index <= "2026-10-14")]
    p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    print(f"  {tag:<26} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
          f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
    return r.nav


# 重建单篮子得分
def basket_T(lst):
    tot = np.zeros((n_w, md.N), np.float32)
    wsum = 0.0
    for k in lst:
        pan = panels[k]
        f = qfm.build_feature_block(md, rows, {k: pan}, k, mask)
        z = np.full((n_w, md.N), np.nan, np.float32)
        for wi, t in enumerate(rows):
            row_ = f[wi]
            m_ = mask[wi] & np.isfinite(row_)
            if m_.sum() > 50:
                r_ = qfm.neutralize_rows(row_[None, :], md.industry[[t]],
                                         md.logmcap[[t]], m_[None, :])
                z[wi] = qfm.zscore_rows(r_, m_[None, :])[0]
        fv = f[dev]
        ic = qfm.weekly_ic(fv, label[dev], mask[dev])
        vv = np.isfinite(ic)
        if vv.sum() < 20:
            continue
        m_ic = float(np.mean(ic[vv]))
        m_icir = m_ic / max(float(np.std(ic[vv])), 1e-9)
        sign = 1.0 if m_ic > 0 else -1.0
        tot += sign * abs(m_icir) * np.nan_to_num(z, nan=0.0)
        wsum += abs(m_icir)
    out = np.full((md.T, md.N), np.nan, np.float32)
    out[rows] = tot / max(wsum, 1e-9)
    return out


print()
navs = {}
navs["h10"] = run(np.asarray(model_h.scores) if np.asarray(model_h.scores).shape[0] == md.T
                  else np.vstack([np.asarray(model_h.scores),
                                  np.full((md.T - np.asarray(model_h.scores).shape[0],
                                           md.N), np.nan, np.float32)]), "wf_h10")
navs["style2"] = run(sc, "wf_style2")
for bk in ("M_mom", "R_rev", "V_vol", "Q_volu"):
    lst = meta["baskets"].get(bk) or []
    if lst:
        navs[bk] = run(basket_T(lst), f"纯 {bk}")

print("\n" + "=" * 96)
print("结论诊断")
print("=" * 96)
print(f"""
  H1 篮子太弱：单篮子纯 M_mom / R_rev 的表现见上。
     若单 R_rev 就远弱于 wf_h10（34 因子 blend），说明"8 因子简单加权"信息量不足。

  H2 风格切换有效吗：看 wf_style2 的滚动 IC 是否高于 wf_h10。
     若更低 → 自适应没有创造价值，反而引入噪声。

  H3 等权兜底：两篮子都无效时各 50% → 一半仓位被"错误风格"占用。
""")
