# -*- coding: utf-8 -*-
"""配置扫描：持仓数 × 调仓周期 × 风控强度。用 **验证期 2020-2021** 选型，
2022+ 只作为样本外报告（不参与选择）。

用法:
    python scripts/scan_config.py --model build/models/wf_v1 --out build/runs/config_scan.csv
"""
from __future__ import annotations

import argparse
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

OVERLAYS = {
    "strong": (C.TREND_MULT_LO, C.TREND_MULT_HI),   # (0.6, 0.75)
    "mild": (0.80, 0.90),
    "off": None,
}


def perf_slice(nav, start, end):
    sub = nav[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
    if len(sub) < 40:
        return {}
    return qfm_metrics.performance(sub, rf=C.CASH_YIELD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(C.BUILD_MODELS / "wf_v1"))
    ap.add_argument("--out", default=str(C.BUILD_RUNS / "config_scan.csv"))
    args = ap.parse_args()

    md = qfdata.load_market(verbose=False)
    res = qfm.load_result(args.model)
    scores = res.scores
    valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
    start_row = int(valid[0])
    phase = int(res.rows_weekly[0]) % 5
    while start_row % 5 != phase:
        start_row += 1
    end_row = md.T - 1

    rows = []
    for n in (20, 30, 40, 50):
        for rb in (5, 10):
            for ov_name, ov in OVERLAYS.items():
                for ti in ("csi_all", "csi1000"):
                    old_n = C.N_HOLDINGS
                    old_lo, old_hi = C.TREND_MULT_LO, C.TREND_MULT_HI
                    old_ti = C.TREND_INDEX
                    try:
                        C.N_HOLDINGS = n
                        C.TREND_INDEX = ti
                        if ov is not None:
                            C.TREND_MULT_LO, C.TREND_MULT_HI = ov
                        t0 = time.time()
                        r = qfbt.run_backtest(md, scores, start_row, end_row,
                                              risk_overlay=ov is not None, stop_loss=True,
                                              rebalance_days=rb, verbose=False)
                    finally:
                        C.N_HOLDINGS = old_n
                        C.TREND_MULT_LO, C.TREND_MULT_HI = old_lo, old_hi
                        C.TREND_INDEX = old_ti
                    nav = r.nav
                    m = r.metrics
                    row = {"tag": f"n{n}_rb{rb}_{ov_name}_{ti}", "n": n, "rb": rb,
                           "overlay": ov_name, "trend_index": ti,
                           "turnover": m.get("turnover_ann"), "cost": m.get("cost_ratio"),
                           "n_trades": m.get("n_trades"), "avg_hold": m.get("avg_holdings"),
                           "elapsed": round(time.time() - t0, 1)}
                    for pname, (s, e) in {"full": ("2020-01-01", "2026-12-31"),
                                          "val": ("2020-01-01", "2021-12-31"),
                                          "oos": ("2022-01-01", "2026-12-31")}.items():
                        p = perf_slice(nav, s, e)
                        for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar"):
                            row[f"{pname}_{k}"] = p.get(k)
                    rows.append(row)
                    print(f"[{row['tag']:26s}] val CAGR={row['val_cagr']:7.2%} SR={row['val_sharpe']:5.2f} "
                          f"MDD={row['val_max_dd']:7.2%} Calmar={row['val_calmar']:5.2f} | "
                          f"oos CAGR={row['oos_cagr']:7.2%} SR={row['oos_sharpe']:5.2f} "
                          f"MDD={row['oos_max_dd']:7.2%} | cost={row['cost']:.2%} TO={row['turnover']:.1f}x")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("\n=== 按验证期 Calmar 排序（选型依据） ===")
    show = df.sort_values("val_calmar", ascending=False)[
        ["tag", "val_cagr", "val_sharpe", "val_max_dd", "val_calmar",
         "oos_cagr", "oos_sharpe", "oos_max_dd", "oos_calmar", "turnover", "cost"]]
    print(show.round(4).to_string(index=False))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
