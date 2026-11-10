# -*- coding: utf-8 -*-
"""模型集成测试：v1（量价+动量）与 v2（量价+动量+基本面）复合分等权集成。"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics

md = qfdata.load_market(verbose=False)


def rownorm(a):
    m = np.isfinite(a)
    cnt = m.sum(axis=1, keepdims=True)
    mean = np.where(m, a, 0).sum(axis=1, keepdims=True) / np.maximum(cnt, 1)
    sd = np.sqrt(np.where(m, (a - mean) ** 2, 0).sum(axis=1, keepdims=True) / np.maximum(cnt - 1, 1))
    out = np.full_like(a, np.nan)
    ok = m & (sd > 1e-9)
    out[ok] = ((a - mean) / sd)[ok]
    return out.astype(np.float32)


def run(scores, tag, n=40, rb=10):
    valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
    start = int(valid[0])
    while start % 5 != int(qfm.load_result('build/models/wf_h10').rows_weekly[0]) % 5:
        start += 1
    old = C.N_HOLDINGS
    C.N_HOLDINGS = n
    try:
        r = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True,
                              stop_loss=True, rebalance_days=rb, verbose=False)
    finally:
        C.N_HOLDINGS = old
    nav = r.nav

    def perf(s, e):
        sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
        return qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    row = {"tag": tag, "turnover": r.metrics.get("turnover_ann"), "cost": r.metrics.get("cost_ratio")}
    for pn, (s, e) in {"val": ("2020-01-01", "2021-12-31"),
                       "full": ("2020-01-01", "2026-12-31"),
                       "oos": ("2022-01-01", "2026-12-31")}.items():
        p = perf(s, e)
        for k in ("cagr", "sharpe", "max_dd", "calmar"):
            row[f"{pn}_{k}"] = p.get(k)
    print(f"[{tag:10s}] val CAGR={row['val_cagr']:7.2%} SR={row['val_sharpe']:5.2f} "
          f"MDD={row['val_max_dd']:7.2%} Calmar={row['val_calmar']:5.2f} | "
          f"full CAGR={row['full_cagr']:7.2%} MDD={row['full_max_dd']:7.2%} Calmar={row['full_calmar']:5.2f} | "
          f"oos CAGR={row['oos_cagr']:7.2%} SR={row['oos_sharpe']:5.2f} MDD={row['oos_max_dd']:7.2%}")
    return row


r1 = qfm.load_result('build/models/wf_h10')
r2 = qfm.load_result('build/models/wf_h10_fund')
s1 = rownorm(r1.scores)
s2 = rownorm(r2.scores)
ens = np.where(np.isfinite(s1) & np.isfinite(s2), 0.5 * s1 + 0.5 * s2,
               np.where(np.isfinite(s1), s1, np.where(np.isfinite(s2), s2, np.nan))).astype(np.float32)
# 只在 v1/v2 都有周频打分的行上有效
rows = []
rows.append(run(s1, "v1"))
rows.append(run(s2, "v2"))
rows.append(run(ens, "ensemble"))
pd.DataFrame(rows).to_csv('build/runs/ensemble_v1v2.csv', index=False, encoding='utf-8-sig')
