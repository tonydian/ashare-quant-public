# -*- coding: utf-8 -*-
"""对比多个 walk-forward 模型在同一回测配置下的表现（验证期/样本外/全期 + 逐年）。

用法:
    python scripts/compare_models.py --models build/models/wf_h10 build/models/wf_h10_nn \
        build/models/wf_h10_fund --out build/runs/model_compare_v2.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C
from qf import data as qfdata
from qf import model as qfm
from qf import backtest as qfbt
from qf import metrics as qfm_metrics

PERIODS = {"val": ("2020-01-01", "2021-12-31"),
           "oos": ("2022-01-01", "2026-12-31"),
           "full": ("2020-01-01", "2026-12-31")}


def perf_slice(nav, s, e):
    sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
    if len(sub) < 40:
        return {}
    return qfm_metrics.performance(sub, rf=C.CASH_YIELD)


def year_ret(nav, y):
    sy = nav[nav.index.year == y]
    spy = nav[nav.index.year == y - 1]
    if len(sy) < 2:
        return np.nan
    base = spy.iloc[-1] if len(spy) else nav.iloc[0]
    return sy.iloc[-1] / base - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=str(C.BUILD_RUNS / "model_compare_v2.csv"))
    ap.add_argument("--n-holdings", type=int, default=40)
    ap.add_argument("--rebalance-days", type=int, default=10)
    args = ap.parse_args()

    md = qfdata.load_market(verbose=False)
    rows = []
    for mdir in args.models:
        res = qfm.load_result(mdir)
        scores = res.scores
        valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
        start = int(valid[0])
        phase = int(res.rows_weekly[0]) % 5
        while start % 5 != phase:
            start += 1
        old = C.N_HOLDINGS
        C.N_HOLDINGS = args.n_holdings
        try:
            t0 = time.time()
            r = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True,
                                  stop_loss=True, rebalance_days=args.rebalance_days,
                                  verbose=False)
        finally:
            C.N_HOLDINGS = old
        nav = r.nav
        row = {"model": pathlib.Path(mdir).name, "n_pool": len(res.pool),
               "n_fund": sum(1 for k in res.pool if k.startswith("fundamental")),
               "n_mom": sum(1 for k in res.pool if k.startswith("momentum")),
               "turnover": r.metrics.get("turnover_ann"),
               "cost": r.metrics.get("cost_ratio"),
               "n_trades": r.metrics.get("n_trades"),
               "avg_hold": r.metrics.get("avg_holdings"),
               "elapsed": round(time.time() - t0, 1)}
        for pn, (s, e) in PERIODS.items():
            p = perf_slice(nav, s, e)
            for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar"):
                row[f"{pn}_{k}"] = p.get(k)
        for y in range(2020, 2027):
            row[f"y{y}"] = year_ret(nav, y)
        rows.append(row)
        print(f"[{row['model']:16s}] pool={row['n_pool']:2d}(fund {row['n_fund']}, mom {row['n_mom']}) | "
              f"val CAGR={row['val_cagr']:7.2%} SR={row['val_sharpe']:5.2f} MDD={row['val_max_dd']:7.2%} "
              f"Calmar={row['val_calmar']:5.2f} | full CAGR={row['full_cagr']:7.2%} "
              f"MDD={row['full_max_dd']:7.2%} Calmar={row['full_calmar']:5.2f} | "
              f"oos CAGR={row['oos_cagr']:7.2%} MDD={row['oos_max_dd']:7.2%} "
              f"Calmar={row['oos_calmar']:5.2f} | TO={row['turnover']:.2f}x cost={row['cost']:.2%}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
