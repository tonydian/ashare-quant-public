# -*- coding: utf-8 -*-
"""主策略（个股左侧）+ 指数趋势（右侧 ETF）组合的快速验证。"""
import sys
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.test_index_trend import load_data, backtest_rule
from qf import metrics as qfm, config as C

idx, cash = load_data()
main = pd.read_parquet(ROOT / "build/runs/main/nav.parquet")["nav"]
main.index = pd.to_datetime(main.index)
s, e = "2020-01-08", "2026-10-14"
m_ret = main.loc[s:e].pct_change().fillna(0)

for code in ["csi1000", "hs300", "csi500"]:
    close = idx[code].dropna()
    cash_ = cash.reindex(close.index).fillna(0)
    sig = close > close.rolling(55).max().shift(1)
    tnav, pos = backtest_rule(close, cash_, sig)
    t_ret = tnav.loc[s:e].pct_change().fillna(0)
    corr = float(m_ret.corr(t_ret))
    print(f"\n=== {code} 趋势策略（donchian55）与主策略日收益相关={corr:.3f} ===")
    print(f"{'w_main':>7} {'CAGR':>8} {'vol':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}")
    for w in (0.0, 0.5, 0.6, 0.7, 0.8, 1.0):
        nav = (1 + w * m_ret + (1 - w) * t_ret).cumprod()
        p = qfm.performance(nav, rf=C.CASH_YIELD)
        print(f"{w:7.1f} {p['cagr']*100:7.2f}% {p['ann_vol']*100:7.2f}% "
              f"{p['sharpe']:7.2f} {p['max_dd']*100:7.2f}% {p['calmar']:7.2f}")
