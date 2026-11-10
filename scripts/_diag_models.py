# -*- coding: utf-8 -*-
"""模型对比：h5 vs h10（均含 momentum-extra），在统一配置下比较验证期/样本外表现。"""
import sys, pathlib, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics

md = qfdata.load_market(verbose=False)


def run(model_dir, tag, n=40, rb=10, lo=0.6, hi=0.75):
    res = qfm.load_result(model_dir)
    scores = res.scores
    valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
    start = int(valid[0])
    phase = int(res.rows_weekly[0]) % 5
    while start % 5 != phase:
        start += 1
    old = C.N_HOLDINGS; old_lo, old_hi = C.TREND_MULT_LO, C.TREND_MULT_HI
    C.N_HOLDINGS = n; C.TREND_MULT_LO, C.TREND_MULT_HI = lo, hi
    try:
        t0 = time.time()
        r = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True,
                              stop_loss=True, rebalance_days=rb, verbose=False)
    finally:
        C.N_HOLDINGS = old; C.TREND_MULT_LO, C.TREND_MULT_HI = old_lo, old_hi
    nav = r.nav

    def perf(s, e):
        sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
        return qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    row = {"tag": tag, "n_factors": len(res.pool), "elapsed": round(time.time() - t0, 1),
           "pool_keys": ",".join([k for k in res.pool if k.startswith("momentum")][:6]),
           "turnover": r.metrics.get("turnover_ann"), "cost": r.metrics.get("cost_ratio")}
    for pname, (s, e) in {"full": ("2020-01-01", "2026-12-31"),
                          "val": ("2020-01-01", "2021-12-31"),
                          "oos": ("2022-01-01", "2026-12-31")}.items():
        p = perf(s, e)
        for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar"):
            row[f"{pname}_{k}"] = p.get(k)
    print(f"[{tag:20s}] val CAGR={row['val_cagr']:7.2%} SR={row['val_sharpe']:5.2f} "
          f"MDD={row['val_max_dd']:7.2%} Calmar={row['val_calmar']:5.2f} | "
          f"oos CAGR={row['oos_cagr']:7.2%} SR={row['oos_sharpe']:5.2f} MDD={row['oos_max_dd']:7.2%} "
          f"Calmar={row['oos_calmar']:5.2f} | TO={row['turnover']:.1f}x cost={row['cost']:.2%} "
          f"| mom={row['pool_keys']}")
    return row


rows = [
    run("build/models/wf_v1", "h5_n40_rb5", rb=5),
    run("build/models/wf_v1", "h5_n40_rb10", rb=10),
    run("build/models/wf_h10", "h10_n40_rb10", rb=10),
    run("build/models/wf_h10", "h10_n50_rb10", n=50, rb=10),
    run("build/models/wf_h10", "h10_n30_rb10", n=30, rb=10),
]
pd.DataFrame(rows).to_csv("build/runs/model_compare.csv", index=False, encoding="utf-8-sig")
print("\n→ build/runs/model_compare.csv")
