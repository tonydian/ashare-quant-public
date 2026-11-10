# -*- coding: utf-8 -*-
"""闲置现金替代资产（国债/黄金）对主策略的影响测试。

在 v1 主模型打分（wf_h10, N=40, 10日调仓, strong overlay）下，把"未投资的现金"
按不同权重投入 国债指数(000012) + 黄金 ETF(518880) 的组合，对比收益/回撤。
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics

md = qfdata.load_market(verbose=False)
res = qfm.load_result('build/models/wf_h10')
scores = res.scores
valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
start = int(valid[0])
while start % 5 != int(res.rows_weekly[0]) % 5:
    start += 1

VARIANTS = {
    "cash": None,
    "bond100": (1.0, 0.0),
    "bond80_gold20": (0.8, 0.2),
    "bond60_gold40": (0.6, 0.4),
    "bond40_gold60": (0.4, 0.6),
    "gold100": (0.0, 1.0),
}
rows = []
for tag, bw in VARIANTS.items():
    car = None if bw is None else qfbt.load_cash_asset_returns(md, bw[0], bw[1])
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True,
                              stop_loss=True, rebalance_days=10, verbose=False,
                              cash_asset_returns=car)
    finally:
        C.N_HOLDINGS = old
    nav = r.nav

    def perf(s, e):
        sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
        return qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    row = {"variant": tag, "turnover": r.metrics.get("turnover_ann"), "cost": r.metrics.get("cost_ratio")}
    for pn, (s, e) in {"val": ("2020-01-01", "2021-12-31"),
                       "full": ("2020-01-01", "2026-12-31"),
                       "oos": ("2022-01-01", "2026-12-31")}.items():
        p = perf(s, e)
        for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar"):
            row[f"{pn}_{k}"] = p.get(k)
    for y in range(2020, 2027):
        sy = nav[nav.index.year == y]
        spy = nav[nav.index.year == y - 1]
        base = spy.iloc[-1] if len(spy) else nav.iloc[0]
        row[f"y{y}"] = sy.iloc[-1] / base - 1.0 if len(sy) > 1 else np.nan
    rows.append(row)
    print(f"[{tag:14s}] val CAGR={row['val_cagr']:7.2%} MDD={row['val_max_dd']:7.2%} Calmar={row['val_calmar']:5.2f} | "
          f"full CAGR={row['full_cagr']:7.2%} vol={row['full_ann_vol']:6.2%} "
          f"SR={row['full_sharpe']:5.2f} MDD={row['full_max_dd']:7.2%} Calmar={row['full_calmar']:5.2f} | "
          f"oos CAGR={row['oos_cagr']:7.2%} MDD={row['oos_max_dd']:7.2%}")
df = pd.DataFrame(rows)
df.to_csv('build/runs/cash_sleeve_test.csv', index=False, encoding='utf-8-sig')
print("\n逐年收益:")
print(df.set_index('variant')[[f"y{y}" for y in range(2020, 2027)]].applymap(lambda x: f"{x*100:.1f}%").to_string())
