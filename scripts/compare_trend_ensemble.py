# -*- coding: utf-8 -*-
"""右侧趋势模型（wf_rs10）与左侧主模型（wf_h10）的集成测试。

要点：右侧模型单独回测显著弱于左侧（CAGR 1.30% vs 9.22%），但两者选股排名几乎不相关
（Rank corr ≈ -0.04），因此测试"小权重右侧 + 主模型"的集成是否改善风险调整收益。
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics


def row_z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    m = np.isfinite(a)
    cnt = m.sum(axis=1, keepdims=True)
    mean = np.where(m, a, 0).sum(axis=1, keepdims=True) / np.maximum(cnt, 1)
    sd = np.sqrt(np.where(m, (a - mean) ** 2, 0).sum(axis=1, keepdims=True) / np.maximum(cnt - 1, 1))
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = m & (sd > 1e-9)
    out[ok] = ((a - mean) / sd)[ok]
    return out


def pad_to(scores: np.ndarray, T: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape[0] < T:
        pad = np.full((T - scores.shape[0], scores.shape[1]), np.nan, dtype=np.float32)
        return np.vstack([scores, pad])
    return scores[:T]


def main():
    md = qfdata.load_market(verbose=False)
    T = md.T
    h10 = qfm.load_result(C.BUILD_MODELS / "wf_h10")
    rs = qfm.load_result(C.BUILD_MODELS / "wf_rs10")
    z_h = pad_to(row_z(h10.scores), T)
    z_r = pad_to(row_z(rs.scores), T)
    # 调仓相位与主模型一致
    phase = int(h10.rows_weekly[0]) % 5
    base_start = max(int(np.flatnonzero(np.isfinite(z_h).any(axis=1))[0]),
                     md.date_index(C.VALID_START))
    while base_start % 5 != phase:
        base_start += 1
    end = T - 1
    cash_ret = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
    rows = []
    for w in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0):
        sc = (1.0 - w) * np.where(np.isfinite(z_h), z_h, 0.0) + \
             w * np.where(np.isfinite(z_r), z_r, 0.0)
        # 两边都无分的行保持 NaN（不参与选股）
        sc[~np.isfinite(z_h) & ~np.isfinite(z_r)] = np.nan
        old = C.N_HOLDINGS
        C.N_HOLDINGS = 40
        try:
            r = qfbt.run_backtest(md, sc.astype(np.float32), base_start, end,
                                  risk_overlay=True, stop_loss=True, rebalance_days=10,
                                  cash_asset_returns=cash_ret, verbose=False)
        finally:
            C.N_HOLDINGS = old
        nav = r.nav
        row = {"w_right": w}
        for pn, (s, e) in {"full": (C.VALID_START, "2026-12-31"),
                           "val": ("2020-01-01", "2021-12-31"),
                           "oos": ("2022-01-01", "2026-12-31")}.items():
            sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
            p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
            for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar"):
                row[f"{pn}_{k}"] = p.get(k)
        row["turnover"] = r.metrics.get("turnover_ann")
        row["cost"] = r.metrics.get("cost_ratio")
        rows.append(row)
        print(f"w_right={w:4.2f} | full CAGR={row['full_cagr']:7.2%} SR={row['full_sharpe']:5.2f} "
              f"MDD={row['full_max_dd']:7.2%} Calmar={row['full_calmar']:5.2f} | "
              f"oos CAGR={row['oos_cagr']:7.2%} Calmar={row['oos_calmar']:5.2f}")
    df = pd.DataFrame(rows)
    out = C.BUILD_RUNS / "trend_ensemble.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
