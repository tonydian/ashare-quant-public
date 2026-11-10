# -*- coding: utf-8 -*-
"""对比多个回测运行（调仓频率/参数选择用）：验证期 2020-2021 选型，2022+ 作为样本外报告。

用法:
    python scripts/compare_runs.py build/runs/rebal5 build/runs/rebal10 build/runs/rebal20 \
        --out build/runs/rebalance_compare.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qf import metrics as qfmetrics
from qf import config as C

PERIODS = {
    "full(2020+)": ("2020-01-01", "2026-12-31"),
    "val(2020-2021)": ("2020-01-01", "2021-12-31"),
    "oos(2022+)": ("2022-01-01", "2026-12-31"),
    "recent(2024+)": ("2024-01-01", "2026-12-31"),
}


def perf(nav: pd.Series, start: str, end: str) -> dict:
    sub = nav[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
    if len(sub) < 40:
        return {}
    return qfmetrics.performance(sub, rf=C.CASH_YIELD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(C.BUILD_RUNS / "rebalance_compare.csv"))
    args = ap.parse_args()
    rows = []
    for d in args.runs:
        d = pathlib.Path(d)
        nav = pd.read_parquet(d / "nav.parquet")["nav"]
        nav.index = pd.to_datetime(nav.index)
        meta = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        row = {"run": d.name, "rebal_days": meta.get("rebalance_days"),
               "turnover": meta.get("turnover_ann"), "cost_ratio": meta.get("cost_ratio"),
               "n_trades": meta.get("n_trades"), "avg_holdings": meta.get("avg_holdings")}
        for pname, (s, e) in PERIODS.items():
            m = perf(nav, s, e)
            for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar"):
                row[f"{pname}:{k}"] = m.get(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    show_cols = ["run", "rebal_days", "turnover", "cost_ratio"]
    for pname in PERIODS:
        show_cols += [f"{pname}:cagr", f"{pname}:sharpe", f"{pname}:max_dd", f"{pname}:calmar"]
    print(df[[c for c in show_cols if c in df.columns]].round(4).to_string(index=False))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
