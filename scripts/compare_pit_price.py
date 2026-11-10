# -*- coding: utf-8 -*-
"""修复前后对比（最终正确版）：仅价格带过滤的 legacy vs PIT 切换，其余逻辑完全一致。"""
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
os.environ["QF_DATA_RAW"] = str(REPO / "data_extract" / "was_stock_data")
sys.path.insert(0, str(ROOT))

import qf.config as C  # noqa: E402
from qf import data as qfdata, model as qfm, backtest as qfbt  # noqa: E402

res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
scores = np.asarray(res.scores)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
phase = int(rows[0]) % 5


def run(tag, source: str):
    md = qfdata.MarketData(verbose=False)
    md.price_band_source = source
    md._build_universe()          # 用指定 source 重建选股域
    start = max(int(rows[0]), md.date_index(C.VALID_START))
    while start % 5 != phase:
        start += 1
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, scores, start, md.T - 1, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=(phase - start) % 10,
                              cash_asset_returns=qfbt.load_cash_asset_returns(md, 0.8, 0.2),
                              verbose=False)
    finally:
        C.N_HOLDINGS = old
    m = r.metrics
    print(f"[{tag:14s}] CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']:.2%} "
          f"Calmar={m['calmar']:.2f} 波动={m['ann_vol']:.2%} 交易={m['n_trades']} "
          f"换手={m['turnover_ann']:.2f}x 持仓={m['avg_holdings']:.1f}")
    return m


a = run("legacy(修前)", "legacy")
b = run("PIT(修后)", "pit")
print(f"\n差异：CAGR {b['cagr']-a['cagr']:+.2%} | Sharpe {b['sharpe']-a['sharpe']:+.3f} | "
      f"MaxDD {b['max_dd']-a['max_dd']:+.2%} | Calmar {b['calmar']-a['calmar']:+.2f}")
