# -*- coding: utf-8 -*-
"""对比 wf_h10（主策略，10 日标签）与 wf_v1（旧模型，5 日标签）在 v2 口径下的回测。"""
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics

md = qfdata.load_market(verbose=False)
cash_ret = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
T = md.T
print(f"md: {md.T} 日 × {md.N} 只，最后 {md.dates[-1].date()}")
for name in ("wf_h10", "wf_v1"):
    res = qfm.load_result(C.BUILD_MODELS / name)
    scores = np.asarray(res.scores)
    if scores.shape[0] < T:
        scores = np.vstack([scores, np.full((T - scores.shape[0], scores.shape[1]), np.nan, np.float32)])
    elif scores.shape[0] > T:
        scores = scores[:T]
    start = max(int(np.flatnonzero(np.isfinite(scores).any(axis=1))[0]), md.date_index(C.VALID_START))
    phase = int(res.rows_weekly[0]) % 5
    while start % 5 != phase:
        start += 1
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True,
                              stop_loss=True, rebalance_days=10,
                              rebalance_offset=(phase - start) % 10,
                              cash_asset_returns=cash_ret, verbose=False)
    finally:
        C.N_HOLDINGS = old
    nav = r.nav
    row = {}
    for pn, (s, e) in {"full": (C.VALID_START, "2026-12-31"),
                       "val": ("2020-01-01", "2021-12-31"),
                       "oos": ("2022-01-01", "2026-12-31")}.items():
        sub = nav[(nav.index >= s) & (nav.index <= e)]
        p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        for k in ("cagr", "sharpe", "max_dd", "calmar"):
            row[f"{pn}_{k}"] = p.get(k)
    print(f"[{name}] horizon={res.cfg.horizon} pool={len(res.pool)} | "
          f"full CAGR={row['full_cagr']:.2%} SR={row['full_sharpe']:.2f} "
          f"MDD={row['full_max_dd']:.2%} Calmar={row['full_calmar']:.2f} | "
          f"val CAGR={row['val_cagr']:.2%} | oos CAGR={row['oos_cagr']:.2%} "
          f"MDD={row['oos_max_dd']:.2%} | trades={r.metrics.get('n_trades'):.0f} "
          f"turnover={r.metrics.get('turnover_ann'):.2f}x")
