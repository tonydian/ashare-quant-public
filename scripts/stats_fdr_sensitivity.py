# -*- coding: utf-8 -*-
"""敏感性：只保留"800 候选 5日DEV BH-FDR 显著"的池因子（23 个），看剔除 11 个不显著因子的影响。"""
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

OUT = C.BUILD_EVAL / "stats_fdr_sensitivity.json"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
panels = dict(np.load(C.BUILD_FACTORS / "weekly_top.npz"))
for nz, c in (("momentum_extra.npz", "momentum_catalog.parquet"),
              ("trend_extra.npz", "trend_catalog.parquet")):
    p = C.BUILD_FACTORS / nz
    if p.exists():
        z = np.load(p)
        for k in z.files:
            if z[k].shape[0] == len(rows):
                panels[k] = z[k]
cat = pd.read_parquet(C.BUILD_FACTORS / "catalog.parquet")
for c in ("momentum_catalog.parquet", "trend_catalog.parquet"):
    p = C.BUILD_FACTORS / c
    if p.exists():
        cat = pd.concat([cat, pd.read_parquet(p)], ignore_index=True)
cat = cat.drop_duplicates("factor_key")

# 800 候选的 5 日 DEV BH-FDR（= 选池真正用的口径）
scr = pd.read_parquet(C.BUILD_EVAL / "factor_metrics_screen.parquet")
t = scr["dev_icir_5d"].to_numpy() * np.sqrt(scr["dev_ic_n"].to_numpy())
p = np.array([qsa.p_value_two_sided(x) for x in t])
r = qsa.bh_fdr(p, q=0.05)
sig = set(scr.loc[r["reject"], "factor_key"])

pool_now = json.load(open(C.BUILD_MODELS / "wf_h10/model_meta.json", encoding="utf-8"))["pool"]
keep = [k for k in pool_now if k in sig]
drop = [k for k in pool_now if k not in sig]
print(f"现行池 {len(pool_now)}；BH-FDR 显著保留 {len(keep)}；剔除 {len(drop)}: {drop}")

phase = int(rows[0]) % 5
start = max(int(rows[0]), md.date_index(C.VALID_START))
while start % 5 != phase:
    start += 1
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)


def run(scores, tag):
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        rr = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True, stop_loss=True,
                               rebalance_days=10, rebalance_offset=(phase - start) % 10,
                               cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    m = rr.metrics
    print(f"  [{tag:24s}] CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.2%} "
          f"Calmar={m['calmar']:.2f} 交易={m['n_trades']}")
    return {k: float(m[k]) for k in ("cagr", "sharpe", "max_dd", "calmar", "n_trades")}


m_base = run(np.asarray(qfm.load_result(C.BUILD_MODELS / "wf_h10").scores), "现行池(34)")

res = qfm.walk_forward(md, rows, panels, cat, cfg=qfm.ModelConfig(
    horizon=10, method="blend", max_factors=40, corr_max=0.70, train_weeks=156,
    score_smooth_alpha=0.6), start_date=C.VALID_START, pool_override=keep, verbose=False)
print(f"  重训池（仅 BH-FDR 显著）: {len(res.pool)} 个")
m_sig = run(res.scores, "仅FDR显著池")

OUT.write_text(json.dumps({
    "pool_now": len(pool_now), "fdr_sig_in_pool": len(keep),
    "dropped": drop, "baseline": m_base, "fdr_significant_only": m_sig,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
print(f"\n差异：CAGR {(m_sig['cagr']-m_base['cagr'])*100:+.2f}pp | "
      f"Sharpe {m_sig['sharpe']-m_base['sharpe']:+.3f} | MaxDD {(m_sig['max_dd']-m_base['max_dd'])*100:+.2f}pp")
