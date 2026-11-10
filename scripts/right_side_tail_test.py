# -*- coding: utf-8 -*-
"""关键追问：材料说"右侧因子平均收益为负、但能保留右尾暴露"。
   那么真正的"右侧价值"是否在**右尾/胜率结构**而非平均 IC？
   同时验证：把 breakdown 从"突破族"剥离后，纯向上突破到底能否用择时+止损救活。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, model as qfm, factors as qff, backtest as qfbt, metrics as qfm_metrics  # noqa

OUT = C.BUILD_EVAL / "right_side_tail_test.json"
md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dev = np.asarray(pd.to_datetime(md.dates[rows]) <= pd.Timestamp(C.DEV_END))
close = md.close.astype(np.float64)
vol = md.volume.astype(np.float64)
T, N = close.shape

# ---------- 1) 右尾测试：+20% 以上收益的比例（5 分组）
print("=" * 74)
print("1) 右尾检验：纯向上突破 vs 向下跌破，各分组的未来 10 日收益分布")
print("=" * 74)
cat = qff.load_catalog().set_index("factor_key")
scr = pd.read_parquet(C.BUILD_EVAL / "factor_metrics_screen.parquet")
r10 = md.fwd_open_ret(10)[rows]  # (475, N)

for base, name in (("breakout", "向上突破"), ("breakdown", "向下跌破")):
    keys = list(scr[(scr.lib == "directional-alpha") & (scr.base == base)]
                .sort_values("dev_icir_5d", ascending=False).head(3).factor_key)
    feats = []
    for k in keys:
        pan = qff.compute_factor(qff.FactorInputs(md), cat.loc[k].to_dict())
        feats.append(qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask))
    if not feats:
        continue
    F = np.nanmean(np.stack(feats, axis=0), axis=0)  # 族内均值
    print(f"\n  [{name}] 因子 {len(keys)} 个")
    for q in (0.2, 0.5, 0.8):
        thr = np.nanquantile(F[dev], q)
        top = F >= thr
        m = top & np.isfinite(r10)
        if m.sum() < 100:
            continue
        vals = r10[m]
        print(f"    分位>{q:.0%}: n={m.sum():6d} 均值 {vals.mean()*100:+.2f}% | "
              f"胜率 {(vals>0).mean():.1%} | +20%以上 {(vals>0.20).mean():.2%} | "
              f"-20%以下 {(vals<-0.20).mean():.2%}")

# ---------- 2) 纯向上突破 + 严格止损/择时，能否救活
print()
print("=" * 74)
print("2) 纯向上突破（剔除 breakdown）+ 择时/止损 的回测")
print("=" * 74)
brk_keys = list(scr[(scr.lib == "directional-alpha") & (scr.base == "breakout")]
                .sort_values("dev_icir_5d", ascending=False).head(6).factor_key)
feats = []
for k in brk_keys:
    pan = qff.compute_factor(qff.FactorInputs(md), cat.loc[k].to_dict())
    feats.append(qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask))
F = np.nanmean(np.stack(feats, axis=0), axis=0)
sc = np.full((T, N), np.nan, dtype=np.float32)
for wi, t in enumerate(rows):
    row = F[wi]
    mm = mask[wi][None, :]
    ok = np.isfinite(row) & mask[wi]
    if ok.sum() > 50:
        rr = qfm.neutralize_rows(row[None, :], md.industry[rows[wi]][None, :],
                                 md.logmcap[rows[wi]][None, :], mm)
        sc[t] = qfm.zscore_rows(rr, mm)[0]

phase = int(rows[0]) % 5
start = max(int(rows[0]), md.date_index(C.VALID_START))
while start % 5 != phase:
    start += 1
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
res = {}
# 基线（overlay + 止损）
for tag, rb, sl, overlay in (("rb10+止损+overlay", 10, True, True),
                             ("rb10 无止损", 10, False, True),
                             ("rb40+止损+overlay", 40, True, True)):
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc, start, md.T - 1, risk_overlay=overlay, stop_loss=sl,
                              rebalance_days=rb, rebalance_offset=(phase - start) % rb,
                              cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    m = r.metrics
    res[tag] = {k: float(m[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}
    print(f"  [{tag:20s}] CAGR {m['cagr']:7.2%} Sharpe {m['sharpe']:6.3f} "
          f"MaxDD {m['max_dd']:7.2%} Calmar {m['calmar']:5.2f}")

# ---------- 3) 与 ETF 趋势对比
print()
print("=" * 74)
print("3) 参照：ETF 趋势（同一套 Donchian55 逻辑，但标的为宽基 ETF）")
print("=" * 74)
print("  CAGR 10.57% / Sharpe 0.91 / MaxDD -10.74% / Calmar 0.98（见 ETF趋势策略报告）")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
