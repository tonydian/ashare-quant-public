# -*- coding: utf-8 -*-
"""研究：先指数 → 再板块 → 再选股，板块层能否让选股更好？

在现有"行业中性化"的左侧主模型打分（wf_h10）之上测试三种板块层叠加：
  A. 行业动量过滤：只保留 60 日动量排名前 50% / 前 30% 的行业
  B. 行业动量倾斜：composite + λ × 行业动量 z
  C. 基线（不改打分）
同样的回测参数（N=40、10 日调仓、80/20 现金替代、overlay+止损、2020-01-08~最新）。
行业层不中立化时，"板块选择"的增量价值可以直接从对比看到；若样本外无改善则说明
当前行业中性 + 行业上限的风控已经足够，不必再加方向性板块押注。
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


def pad_to(scores, T):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape[0] < T:
        pad = np.full((T - scores.shape[0], scores.shape[1]), np.nan, dtype=np.float32)
        return np.vstack([scores, pad])
    return scores[:T]


def industry_momentum(md, t: int, window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """t 日各行业等权区间收益（(n_ind,) 与 (n_ind,) 计数）。"""
    close = md.close
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((t >= window) & (close[t - window] > 0), close[t] / close[t - window] - 1.0, np.nan)
    ids = np.asarray(md.industry[t])
    mask = md.eligible[t] & np.isfinite(r) & (ids >= 0)
    n_ind = len(md.industry_cats)
    cnt = np.bincount(ids[mask], minlength=n_ind)
    s = np.bincount(ids[mask], weights=r[mask], minlength=n_ind)
    ttl = np.full(n_ind, np.nan)
    ok = cnt >= 5
    ttl[ok] = s[ok] / cnt[ok]
    return ttl, cnt


def industry_z(mm: np.ndarray) -> np.ndarray:
    m = np.isfinite(mm)
    if m.sum() < 3:
        return np.full_like(mm, np.nan)
    mu, sd = mm[m].mean(), mm[m].std()
    out = np.full_like(mm, np.nan)
    if sd > 1e-9:
        out[m] = (mm[m] - mu) / sd
    return out


def main():
    md = qfdata.load_market(verbose=False)
    T = md.T
    res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
    base_scores = pad_to(res.scores, T)
    score_rows = np.flatnonzero(np.isfinite(base_scores).any(axis=1))
    ids = np.asarray(md.industry)  # (T,N) int16
    n_ind = len(md.industry_cats)

    variants: dict[str, np.ndarray] = {"baseline": base_scores.copy()}
    for keep_pct, tag in ((0.5, "ind_filter_top50"), (0.3, "ind_filter_top30")):
        sc = base_scores.copy()
        for t in score_rows:
            mm, cnt = industry_momentum(md, t)
            if not np.isfinite(mm).any():
                continue
            order = np.argsort(np.where(np.isfinite(mm), mm, -np.inf))[::-1]
            valid = order[np.isfinite(mm[order])]
            n_keep = max(3, int(np.ceil(len(valid) * keep_pct)))
            keep_ids = set(int(k) for k in valid[:n_keep])
            row = sc[t]
            drop = np.array([bool(np.isfinite(row[j])) and int(ids[t, j]) >= 0
                             and int(ids[t, j]) not in keep_ids for j in range(md.N)])
            row[drop] = np.nan
        variants[tag] = sc
    for lam in (0.5, 1.0):
        sc = base_scores.copy()
        for t in score_rows:
            mm, cnt = industry_momentum(md, t)
            z = industry_z(mm)
            if not np.isfinite(z).any():
                continue
            stock_z = np.where(ids[t] >= 0, z[np.clip(ids[t], 0, n_ind - 1)], np.nan)
            add = np.where(np.isfinite(stock_z), lam * np.nan_to_num(stock_z), 0.0)
            sc[t] = np.where(np.isfinite(sc[t]), sc[t] + add, sc[t])
        variants[f"ind_tilt_{lam}"] = sc

    phase = int(res.rows_weekly[0]) % 5
    start = max(int(score_rows[0]), md.date_index(C.VALID_START))
    while start % 5 != phase:
        start += 1
    end = T - 1
    cash_ret = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
    rows = []
    for tag, sc in variants.items():
        old = C.N_HOLDINGS
        C.N_HOLDINGS = 40
        try:
            r = qfbt.run_backtest(md, sc.astype(np.float32), start, end,
                                  risk_overlay=True, stop_loss=True, rebalance_days=10,
                                  rebalance_offset=(phase - start) % 10,
                                  cash_asset_returns=cash_ret, verbose=False)
        finally:
            C.N_HOLDINGS = old
        nav = r.nav
        row = {"variant": tag}
        for pn, (s, e) in {"full": (C.VALID_START, "2026-12-31"),
                           "val": ("2020-01-01", "2021-12-31"),
                           "oos": ("2022-01-01", "2026-12-31")}.items():
            sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
            p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
            for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar"):
                row[f"{pn}_{k}"] = p.get(k)
        row["turnover"] = r.metrics.get("turnover_ann")
        row["cost"] = r.metrics.get("cost_ratio")
        row["avg_holdings"] = r.metrics.get("avg_holdings")
        row["n_trades"] = r.metrics.get("n_trades")
        rows.append(row)
        print(f"[{tag:16s}] full CAGR={row['full_cagr']:7.2%} SR={row['full_sharpe']:5.2f} "
              f"MDD={row['full_max_dd']:7.2%} Calmar={row['full_calmar']:5.2f} | "
              f"val CAGR={row['val_cagr']:7.2%} Calmar={row['val_calmar']:5.2f} | "
              f"oos CAGR={row['oos_cagr']:7.2%} SR={row['oos_sharpe']:5.2f} "
              f"MDD={row['oos_max_dd']:7.2%} Calmar={row['oos_calmar']:5.2f}")
    df = pd.DataFrame(rows)
    out = C.BUILD_RUNS / "industry_rotation_test.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
