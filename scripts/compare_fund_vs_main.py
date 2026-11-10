# -*- coding: utf-8 -*-
"""受控对比：wf_h10（纯量价+动量）vs wf_h10_fund（+4 个基本面因子）。

同区间、同参数、同成本，逐项比较；并检验差异是否在噪声范围内。
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

from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics  # noqa
from qf.stats_adjust import circular_block_bootstrap_ci, sharpe_ratio  # noqa

OUT = C.BUILD_EVAL / "fund_vs_main_controlled.json"

md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
phase = int(rows[0]) % 5
start = max(int(rows[0]), md.date_index(C.VALID_START))
while start % 5 != phase:
    start += 1
end = md.T - 1
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
print(f"区间 {md.dates[start].date()} ~ {md.dates[end].date()}（{end-start+1} 日）")

res = {}
navs = {}
for name in ("wf_h10", "wf_h10_fund"):
    m = qfm.load_result(C.BUILD_MODELS / name)
    sc = np.asarray(m.scores)
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    elif sc.shape[0] > md.T:
        sc = sc[:md.T]
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=(phase - start) % 10,
                              cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    navs[name] = r.nav
    row = {"pool": len(m.pool),
           "fund_factors": [k for k in m.pool if k.startswith("fundamental")],
           "turnover": float(r.metrics["turnover_ann"]),
           "n_trades": int(r.metrics["n_trades"])}
    for seg, (s, e) in {"full": ("2020-01-01", "2026-12-31"),
                        "val": ("2020-01-01", "2021-12-31"),
                        "oos": ("2022-01-01", "2026-12-31")}.items():
        sub = navs[name][(navs[name].index >= s) & (navs[name].index <= e)]
        p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        for k in ("cagr", "sharpe", "max_dd", "calmar"):
            row[f"{seg}_{k}"] = float(p[k])
    res[name] = row
    print(f"\n[{name}] pool={row['pool']} 基本面因子={row['fund_factors']}")
    print(f"  全期 CAGR {row['full_cagr']:.2%} | Sharpe {row['full_sharpe']:.3f} | "
          f"MaxDD {row['full_max_dd']:.2%} | Calmar {row['full_calmar']:.2f} | "
          f"换手 {row['turnover']:.2f}x")
    print(f"  验证期 {row['val_cagr']:.2%} / {row['val_sharpe']:.3f} | "
          f"样本外 {row['oos_cagr']:.2%} / {row['oos_sharpe']:.3f}")

# 差异的统计显著性：日收益差序列的 bootstrap
a = navs["wf_h10"].pct_change().dropna()
b = navs["wf_h10_fund"].pct_change().dropna()
common = a.index.intersection(b.index)
diff = (b[common] - a[common]).to_numpy()
ci = circular_block_bootstrap_ci(diff, lambda s: float(np.mean(s) * 252),
                                 n_boot=2000, block_len=21)
print(f"\n日收益差（fund - main）年化均值点估计: {diff.mean()*252*100:+.2f}%")
print(f"  block bootstrap 95% CI: [{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%] / 年")
print(f"  差序列 Sharpe 的 t 统计量: {diff.mean()/diff.std(ddof=1)*np.sqrt(len(diff)):.2f}")

# 逐因子贡献：fund 模型里 4 个基本面因子的实际权重
w = pd.read_parquet(C.BUILD_MODELS / "wf_h10_fund/model_weights.parquet")
fw = w[w.factor_key.str.startswith("fundamental")]
if len(fw):
    g = fw.groupby("factor_key")["weight"].agg(["mean", "std", "count"])
    print("\n4 个基本面因子的权重（23 次 refit）：")
    for k, r in g.iterrows():
        print(f"  {k:22s} 均值 {r['mean']:+.4f} 标准差 {r['std']:.4f} | 非零 {int((fw[fw.factor_key==k].weight.abs()>0).sum())}/{int(r['count'])}")

res["diff_bootstrap"] = {"annual_mean": float(diff.mean() * 252),
                         "ci_lo": float(ci["lo"]), "ci_hi": float(ci["hi"]),
                         "t": float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff)))}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
