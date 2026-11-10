# -*- coding: utf-8 -*-
"""对**所有可选模型**逐一做统计校正（NW t / bootstrap CI / BH-FDR），输出对比表。

对每个模型：
  1) 复合分 IC 的 naive t / NW t（lag=ceil(h/5)）/ 有效样本 t / bootstrap 95% CI；
  2) 回测（同参数）的 Sharpe 及其 bootstrap 95% CI、年化 Sharpe×√years 的 t；
  3) 池内因子的 BH-FDR 通过率（用 800 候选口径）。
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

MODELS = ["wf_h10", "wf_v1", "wf_h10_fund", "wf_rs10"]
OUT = C.BUILD_EVAL / "stats_all_models.json"

md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
phase = int(rows[0]) % 5
start = max(int(rows[0]), md.date_index(C.VALID_START))
while start % 5 != phase:
    start += 1
end = md.T - 1
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
scr = pd.read_parquet(C.BUILD_EVAL / "factor_metrics_screen.parquet")
t800 = scr["dev_icir_5d"].to_numpy() * np.sqrt(scr["dev_ic_n"].to_numpy())
p800 = np.array([qsa.p_value_two_sided(t) for t in t800])
fdr800 = qsa.bh_fdr(p800, q=0.05)
sig800 = set(scr.loc[fdr800["reject"], "factor_key"])

out = {}
print(f"{'model':14s}{'pool':>5}{'IC t':>8}{'NW t':>8}{'CI lo':>8}{'CI hi':>8}"
      f"{'Sharpe':>8}{'SR CI':>16}{'BH通过':>8}")
for name in MODELS:
    res = qfm.load_result(C.BUILD_MODELS / name)
    cfg = res.cfg
    h = int(cfg.horizon)
    lag = int(np.ceil(h / 5))
    ic = res.composite_ic["ic"].to_numpy(dtype=np.float64)
    ic = ic[np.isfinite(ic)]
    naive_t = float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))) if len(ic) > 2 else np.nan
    nw = qsa.newey_west_tstat(ic, lag=lag)
    ci_ic = qsa.circular_block_bootstrap_ci(ic, lambda s: float(np.mean(s)), n_boot=1500,
                                            block_len=max(3, lag))
    scores = np.asarray(res.scores)
    if scores.shape[0] < md.T:
        scores = np.vstack([scores, np.full((md.T - scores.shape[0], md.N), np.nan, np.float32)])
    elif scores.shape[0] > md.T:
        scores = scores[:md.T]
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, scores, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=(phase - start) % 10,
                              cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    nav = r.nav
    ret = nav.pct_change().dropna().to_numpy()
    sr = qsa.sharpe_ratio(ret, rf=C.CASH_YIELD)
    ci_sr = qsa.circular_block_bootstrap_ci(ret, lambda s: qsa.sharpe_ratio(s, rf=C.CASH_YIELD),
                                             n_boot=1500, block_len=21)
    pool_bh = sum(1 for k in res.pool if k in sig800)
    out[name] = {
        "pool": len(res.pool), "horizon": h,
        "ic_mean": float(ic.mean()), "ic_naive_t": naive_t,
        "ic_nw_t": nw["t"], "ic_nw_lag": lag,
        "ic_ci": [ci_ic["lo"], ci_ic["hi"]],
        "sharpe": sr, "sharpe_t": float(sr * np.sqrt(len(ret) / 252)) if np.isfinite(sr) else np.nan,
        "sharpe_ci": [ci_sr["lo"], ci_sr["hi"]],
        "bh_pass": int(pool_bh), "pool_size": len(res.pool),
        "backtest": {k: float(r.metrics[k]) for k in ("cagr", "max_dd", "calmar")},
    }
    print(f"{name:14s}{len(res.pool):>5}{naive_t:>8.2f}{nw['t']:>8.2f}"
          f"{ci_ic['lo']:>8.4f}{ci_ic['hi']:>8.4f}{sr:>8.2f}"
          f"  [{ci_sr['lo']:>5.2f},{ci_sr['hi']:>5.2f}]"
          f"{pool_bh:>4}/{len(res.pool)}")

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
