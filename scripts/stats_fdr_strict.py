# -*- coding: utf-8 -*-
"""严格版 FDR 实验：直接用 800 候选在 **h10 DEV 标签**下的 |t| 做 BH-FDR，
再把"显著集"作为候选约束传入选池（--pool-from 之外的新路径：pool_override 传显著集）。
对比：现行池(34) / 显著集约束池 → 回测差异。
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
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, model as qfm, stats_adjust as qsa, backtest as qfbt  # noqa: E402

OUT = C.BUILD_EVAL / "stats_fdr_strict.json"

print("[1/4] 载入 DEV 面板与 h10 标签 ...")
md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
dev = np.asarray(pd.to_datetime(md.dates[rows]) <= pd.Timestamp(C.DEV_END))
rows_dev = rows[dev]
mask_dev = mask[dev]

panels_all = dict(np.load(C.BUILD_FACTORS / "weekly_top.npz"))
for extra_npz, extra_cat in (("momentum_extra.npz", "momentum_catalog.parquet"),
                             ("trend_extra.npz", "trend_catalog.parquet")):
    p = C.BUILD_FACTORS / extra_npz
    if p.exists():
        z = np.load(p)
        for k in z.files:
            if z[k].shape[0] == len(rows):
                panels_all[k] = z[k]
cat = pd.read_parquet(C.BUILD_FACTORS / "catalog.parquet")
for c in ("momentum_catalog.parquet", "trend_catalog.parquet"):
    p = C.BUILD_FACTORS / c
    if p.exists():
        cat = pd.concat([cat, pd.read_parquet(p)], ignore_index=True)
cat = cat.drop_duplicates("factor_key")
print(f"    因子面板 {len(panels_all)} 个；DEV 周频 {int(dev.sum())} 行")

print("[2/4] h10 DEV 单因子 t 统计量 + BH-FDR ...")
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
label_dev = label[dev]
keys, tres = [], []
for k in panels_all:
    pan = panels_all[k][dev]
    f = qfm.build_feature_block(md, rows_dev, {k: pan}, k, mask_dev)
    ic = qfm.weekly_ic(f, label_dev, mask_dev)
    ic = ic[np.isfinite(ic)]
    if len(ic) < 40:
        continue
    m, s = float(np.mean(ic)), float(np.std(ic))
    if s <= 1e-12:
        continue
    keys.append(k)
    tres.append(m / s * np.sqrt(len(ic)))   # ICIR * sqrt(n)
tres = np.array(tres)
pvals = np.array([qsa.p_value_two_sided(t) for t in tres])
r = qsa.bh_fdr(pvals, q=0.05)
sig_keys = [k for k, rej in zip(keys, r["reject"]) if rej]
print(f"    DEV(h10) 有效因子 {len(keys)} 个；BH-FDR(q=0.05) 显著 {len(sig_keys)} 个")

meta = json.load(open(C.BUILD_MODELS / "wf_h10/model_meta.json", encoding="utf-8"))
pool_now = set(meta["pool"])
print(f"    现行池 34 个中，落在 h10-DEV 显著集里的: {len(pool_now & set(sig_keys))}")
print(f"    不显著但入池: {sorted(pool_now - set(sig_keys))}")

print("[3/4] 用显著集重训（同超参）...")
res = qfm.walk_forward(md, rows, panels_all, cat, cfg=qfm.ModelConfig(
    horizon=10, method="blend", max_factors=40, corr_max=0.70, train_weeks=156,
    score_smooth_alpha=0.6), start_date=C.VALID_START,
    pool_override=sig_keys, verbose=False)
print(f"    walk-forward 池 {len(res.pool)} 个；refit {res.weights['refit_date'].nunique()} 次")

print("[4/4] 回测对比 ...")
phase = int(rows[0]) % 5
start = max(int(rows[0]), md.date_index(C.VALID_START))
while start % 5 != phase:
    start += 1
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)


def bt(scores, tag):
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        rr = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True, stop_loss=True,
                               rebalance_days=10, rebalance_offset=(phase - start) % 10,
                               cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    m = rr.metrics
    print(f"    [{tag:22s}] CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.3f} "
          f"MaxDD={m['max_dd']:.2%} Calmar={m['calmar']:.2f} 交易={m['n_trades']}")
    return m


m_now = bt(np.asarray(qfm.load_result(C.BUILD_MODELS / "wf_h10").scores), "现行(34, 未筛选)")
m_fdr = bt(res.scores, "h10-DEV BH-FDR 池")

OUT.write_text(json.dumps({
    "dev_h10_valid_factors": len(keys),
    "bh_sig_q005": len(sig_keys),
    "pool_now": len(pool_now),
    "pool_now_in_sig": len(pool_now & set(sig_keys)),
    "pool_now_not_sig": sorted(pool_now - set(sig_keys)),
    "fdr_pool_size": len(res.pool),
    "fdr_backtest": {k: m_fdr[k] for k in ("cagr", "sharpe", "max_dd", "calmar", "n_trades")},
    "baseline_backtest": {k: m_now[k] for k in ("cagr", "sharpe", "max_dd", "calmar", "n_trades")},
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
