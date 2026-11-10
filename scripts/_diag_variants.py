# -*- coding: utf-8 -*-
"""快速变体回测（复用已加载 md + 模型分数）：持仓数 / 调仓频率 / overlay 组合。"""
import sys, pathlib, time, json
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics

md = qfdata.load_market(verbose=False)
res = qfm.load_result('build/models/wf_v1')
scores = res.scores
valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
start_row = int(valid[0])
phase = int(res.rows_weekly[0]) % 5
while start_row % 5 != phase:
    start_row += 1
end_row = md.T - 1

def run(tag, n=20, rb=10, risk=True, stop=True, band_rel=0.30, band_abs=6000, max_new=6):
    old = C.N_HOLDINGS
    C.N_HOLDINGS = n
    try:
        t0 = time.time()
        r = qfbt.run_backtest(md, scores, start_row, end_row, risk_overlay=risk,
                              stop_loss=stop, rebalance_days=rb,
                              trade_band_rel=band_rel, trade_band_abs=band_abs,
                              max_new_entries=max_new, verbose=False)
    finally:
        C.N_HOLDINGS = old
    m = dict(r.metrics)
    nav = r.nav
    def perf(s, e):
        sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
        return qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    row = {"tag": tag, "n": n, "rb": rb, "risk": risk, "stop": stop,
           "full_cagr": m.get("cagr"), "full_sharpe": m.get("sharpe"),
           "full_maxdd": m.get("max_dd"), "full_calmar": m.get("calmar"),
           "turnover": m.get("turnover_ann"), "cost": m.get("cost_ratio"),
           "n_trades": m.get("n_trades"), "avg_hold": m.get("avg_holdings"),
           "elapsed": round(time.time() - t0, 1)}
    for name, (s, e) in {"val20_21": ("2020-01-01", "2021-12-31"),
                         "oos22": ("2022-01-01", "2026-12-31"),
                         "y2025": ("2025-01-01", "2025-12-31")}.items():
        p = perf(s, e)
        for k in ("cagr", "sharpe", "max_dd", "calmar"):
            row[f"{name}_{k}"] = p.get(k)
    print(f"[{tag:22s}] full CAGR={row['full_cagr']:7.2%} SR={row['full_sharpe']:5.2f} "
          f"MDD={row['full_maxdd']:7.2%} Calmar={row['full_calmar']:5.2f} | "
          f"val CAGR={row['val20_21_cagr']:7.2%} MDD={row['val20_21_max_dd']:7.2%} | "
          f"oos CAGR={row['oos22_cagr']:7.2%} MDD={row['oos22_max_dd']:7.2%} | "
          f"TO={row['turnover']:.1f}x cost={row['cost']:.2%}")
    return row

rows = []
for n in (10, 20, 30, 40, 50):
    rows.append(run(f"n{n}_rb10", n=n, rb=10))
for n in (30, 40, 50):
    rows.append(run(f"n{n}_rb5", n=n, rb=5))
rows.append(run("n40_rb10_nooverlay", n=40, rb=10, risk=False))
rows.append(run("n40_rb10_nostop", n=40, rb=10, stop=False))
rows.append(run("n40_rb10_maxnew10", n=40, rb=10, max_new=10))
rows.append(run("n40_rb20", n=40, rb=20))
df = pd.DataFrame(rows)
out = 'build/runs/variant_scan.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')
print("\n→", out)
