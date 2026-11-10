# -*- coding: utf-8 -*-
"""测试：h5 与 h10 复合分集成（等权 z-score）在统一配置下的表现。"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics

md = qfdata.load_market(verbose=False)
r5 = qfm.load_result('build/models/wf_v1')
r10 = qfm.load_result('build/models/wf_h10')
assert np.array_equal(r5.rows_weekly, r10.rows_weekly)
# 逐行 z 分数后再平均
def rownorm(a):
    out = np.full_like(a, np.nan)
    m = np.isfinite(a)
    cnt = m.sum(axis=1, keepdims=True)
    mean = np.where(m, a, 0).sum(axis=1, keepdims=True) / np.maximum(cnt, 1)
    sd = np.sqrt(np.where(m, (a - mean) ** 2, 0).sum(axis=1, keepdims=True) / np.maximum(cnt - 1, 1))
    out = np.where(m & (sd > 1e-9), (a - mean) / np.maximum(sd, 1e-9), np.nan)
    return out.astype(np.float32)

s5 = rownorm(r5.scores); s10 = rownorm(r10.scores)
ens = (0.5 * np.nan_to_num(s5) + 0.5 * np.nan_to_num(s10)).astype(np.float32)
ens[~np.isfinite(s5) | ~np.isfinite(s10)] = np.nan

valid = np.flatnonzero((np.isfinite(s5).any(1)) & (np.isfinite(s10).any(1)))
start = int(valid[0])
phase = int(r10.rows_weekly[0]) % 5
while start % 5 != phase:
    start += 1

def run(scores, tag, n=40, rb=10):
    old = C.N_HOLDINGS; C.N_HOLDINGS = n
    try:
        r = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True,
                              stop_loss=True, rebalance_days=rb, verbose=False)
    finally:
        C.N_HOLDINGS = old
    nav = r.nav
    def perf(s, e):
        sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
        return qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    out = {"tag": tag, "turnover": r.metrics.get("turnover_ann"), "cost": r.metrics.get("cost_ratio")}
    for pn, (s, e) in {"full": ("2020-01-01", "2026-12-31"), "val": ("2020-01-01", "2021-12-31"),
                       "oos": ("2022-01-01", "2026-12-31")}.items():
        p = perf(s, e)
        for k in ("cagr", "sharpe", "max_dd", "calmar"):
            out[f"{pn}_{k}"] = p.get(k)
    print(f"[{tag:10s}] full CAGR={out['full_cagr']:7.2%} SR={out['full_sharpe']:5.2f} MDD={out['full_max_dd']:7.2%} "
          f"Calmar={out['full_calmar']:5.2f} | val CAGR={out['val_cagr']:7.2%} SR={out['val_sharpe']:5.2f} | "
          f"oos CAGR={out['oos_cagr']:7.2%} SR={out['oos_sharpe']:5.2f} MDD={out['oos_max_dd']:7.2%}")
    return out

rows = [run(s5, "h5"), run(s10, "h10"), run(ens, "ensemble")]
pd.DataFrame(rows).to_csv("build/runs/ensemble_compare.csv", index=False, encoding="utf-8-sig")
