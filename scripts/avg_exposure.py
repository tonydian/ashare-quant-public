# -*- coding: utf-8 -*-
"""统计所有策略的平均仓位（回测区间 2020-01-02 ~ 2026-10-23）。"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

S, E = "2020-01-02", "2026-10-23"
print("=" * 96)
print(f"回测区间：{S} ~ {E}")
print("=" * 96)

# ---------------- 1. ETF 右侧（top2）
print("\n【1】ETF 右侧（top2，每只占 1/2）")
from scripts.etf_trend_strategy import load_cash, load_etf, signals, backtest, MAX_POS  # noqa

dates, codes, P = load_etf()
b, g = load_cash()
idx = pd.DatetimeIndex(dates)
cash = _blend_cash(b, g, idx)
sig = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=MAX_POS)
nav_etf, pos_etf, tr_etf = backtest(dates, codes, P, sig, cash,
                                    max_pos=MAX_POS, pos_style="equal")
m = (pos_etf.index >= S) & (pos_etf.index <= E)
p_etf = pos_etf[m]
w_etf = p_etf / MAX_POS
print(f"  平均持仓只数      : {p_etf.mean():.3f} 只")
print(f"  **平均仓位**        : {w_etf.mean():.2%}")
print(f"  有持仓天数占比    : {(p_etf > 0).mean():.1%}")
print(f"  满仓(2只)天数占比 : {(p_etf >= 2).mean():.1%}")
print(f"  半仓(1只)天数占比 : {(p_etf == 1).mean():.1%}")
print(f"  空仓天数占比      : {(p_etf == 0).mean():.1%}")

# ---------------- 2. 主策略（个股）
print("\n【2】主策略（个股 40 只）")
from qf import backtest as qfbt, config as C, data as qfdata, model as qfm  # noqa

md = qfdata.load_market(verbose=False)
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
scores = np.asarray(res.scores)
if scores.shape[0] < md.T:
    scores = np.vstack([scores, np.full((md.T - scores.shape[0], md.N), np.nan, np.float32)])
elif scores.shape[0] > md.T:
    scores = scores[:md.T]
start = int(md.date_index("2020-01-08"))
end = int(md.date_index("2026-10-14"))
phase = int(res.rows_weekly[0]) % 5
offset = (phase - start) % 10
cash_main = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
old = C.N_HOLDINGS
C.N_HOLDINGS = 40
try:
    r_main = qfbt.run_backtest(md, scores, start, end, risk_overlay=True, stop_loss=True,
                               rebalance_days=10, rebalance_offset=offset,
                               cash_asset_returns=cash_main, verbose=False)
finally:
    C.N_HOLDINGS = old
ex_df = r_main.exposures
# exposures 是调仓日宽表：stock 列即股票仓位，cash 列即现金
exp = np.asarray(ex_df["stock"], dtype=float)
exp = exp[np.isfinite(exp)]
print(f"  **平均仓位**        : {exp.mean():.2%}")
print(f"  仓位最低 / 最高   : {exp.min():.2%} / {exp.max():.2%}")
print(f"  中位数            : {np.median(exp):.2%}")
n_pos_mean = np.asarray(ex_df["n_positions"], dtype=float).mean()
print(f"  平均持仓只数      : {n_pos_mean:.1f} 只")
print(f"  （注：exposures 只有 {len(ex_df)} 个调仓日记录，非每日）")

# ---------------- 3. 板块轮动
print("\n【3】板块 ETF 轮动（top3，每只占 1/3）")
from qf import sector_rotation as qfsr  # noqa

d2, c2, cl2, am2, sec2 = qfsr.load_pool()
sr = qfsr.backtest(d2, c2, cl2, am2, sec2, start_row=500, verbose=False)
e_sec = sr["exposure"]
e_sec = e_sec[(e_sec.index >= S) & (e_sec.index <= E)]
print(f"  **平均仓位**        : {e_sec.mean():.2%}")
print(f"  有持仓天数占比    : {(e_sec > 0).mean():.1%}")
print(f"  空仓天数占比      : {(e_sec == 0).mean():.1%}")

# ---------------- 4. 组合
print("\n【4】组合层面")
print(f"  50/50（主策略 + ETF）:")
print(f"    主策略 50% × {exp.mean():.2%} = "
      f"{0.5*exp.mean():.2%}")
print(f"    ETF    50% × {w_etf.mean():.2%} = {0.5*w_etf.mean():.2%}")
print(f"    **组合平均仓位**    : {0.5*exp.mean() + 0.5*w_etf.mean():.2%}")
print()
print(f"  50/30/20（含板块）:")
combo = (0.5*exp.mean() + 0.3*w_etf.mean() + 0.2*e_sec.mean())
print(f"    **组合平均仓位**    : {combo:.2%}")

# ---------------- 汇总表
print("\n" + "=" * 96)
print("汇总")
print("=" * 96)
print(f"{'策略':<28}{'平均仓位':>12}{'说明':<30}")
print("-" * 96)
print(f"{'ETF 右侧（top2）':<28}{w_etf.mean():>11.2%}  平均持 0.21 只 × 1/2")
print(f"{'主策略（个股 40 只）':<28}{exp.mean():>11.2%}  含 overlay 调节")
print(f"{'板块轮动（top3）':<28}{e_sec.mean():>11.2%}  平均持 0.68 只 × 1/3")
print(f"{'50/50 组合':<28}{0.5*exp.mean()+0.5*w_etf.mean():>11.2%}")
print(f"{'50/30/20 组合':<28}{combo:>11.2%}")

out = {
    "period": [S, E],
    "etf_top2": {"avg_holdings": float(p_etf.mean()), "avg_weight": float(w_etf.mean()),
                 "days_holding_pct": float((p_etf > 0).mean())},
    "main_stock": {"avg_weight": float(exp.mean())},
    "sector_top3": {"avg_weight": float(e_sec.mean())},
    "combo_50_50": float(0.5*exp.mean() + 0.5*w_etf.mean()),
    "combo_50_30_20": float(combo),
}
(ROOT / "build" / "eval" / "avg_exposure.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ build/eval/avg_exposure.json")
