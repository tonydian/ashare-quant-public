# -*- coding: utf-8 -*-
"""去掉黄金的完整影响评估（改之前先看清）。

对比两种现金配置下所有模型的表现：
  C1 80% 国债 + 20% 黄金（现行）
  C2 100% 国债（拟改）

模型：
  M1 主策略 wf_h10（个股）
  M2 宽基 ETF 右侧（top2）
  M3 板块轮动（top3）
  M4 三策略组合 50/30/20
"""
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

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, metrics as qfm_metrics  # noqa: E402
from qf.stats_adjust import sharpe_ratio  # noqa: E402

OUT = ROOT / "build" / "eval" / "no_gold_impact.json"
S, E = "2020-01-08", "2026-10-14"

print("=" * 100)
print("去掉黄金的完整影响评估")
print("=" * 100)

md = qfdata.load_market(verbose=False)
idx = pd.DatetimeIndex(md.dates)

# ---- 两种现金收益序列
cash_80_20 = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
cash_100_0 = qfbt.load_cash_asset_returns(md, 1.0, 0.0)
m = (idx >= S) & (idx <= E)
for name, cr in (("80/20 含黄金", cash_80_20), ("100% 纯国债", cash_100_0)):
    sub = cr[m]
    tot = np.prod(1 + sub) - 1
    yrs = len(sub) / 252
    print(f"  现金部分 - {name:14s}: 累计 {tot:+7.2%} | 年化 {(1+tot)**(1/yrs)-1:+6.2%}")


def perf(nav, label):
    sub = nav[(nav.index >= S) & (nav.index <= E)]
    p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
    return {"cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
            "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"])}


# ==================== M1 主策略
print("\n" + "=" * 100)
print("M1 主策略 wf_h10（个股 40 只）")
print("=" * 100)
res_main = qfm.load_result(C.BUILD_MODELS / "wf_h10")
sc = np.asarray(res_main.scores)
if sc.shape[0] < md.T:
    sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
elif sc.shape[0] > md.T:
    sc = sc[:md.T]
start = int(md.date_index(S))
end = int(md.date_index(E))
phase = int(res_main.rows_weekly[0]) % 5
offset = (phase - start) % 10
out = {}
navs = {}
for tag, cr in (("80/20", cash_80_20), ("纯国债", cash_100_0)):
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=offset,
                              cash_asset_returns=cr, verbose=False)
    finally:
        C.N_HOLDINGS = old
    p = perf(r.nav, tag)
    navs[tag] = r.nav
    out[f"main_{tag}"] = p
    print(f"  {tag:8s}: CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
          f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
print(f"  → CAGR 变化: {(out['main_纯国债']['cagr']-out['main_80/20']['cagr'])*100:+.2f}pp")

# ==================== M2 ETF 右侧
print("\n" + "=" * 100)
print("M2 宽基 ETF 右侧（top2）")
print("=" * 100)
from scripts.etf_trend_strategy import MAX_POS, backtest, load_etf, signals  # noqa

d_etf, c_etf, P_etf = load_etf()
idx_e = pd.DatetimeIndex(d_etf)
bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
bs = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
gs = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
b_r = bs.reindex(idx_e).ffill().pct_change().fillna(0)
g_r = gs.reindex(idx_e).ffill().pct_change().fillna(0)
sig = signals(d_etf, c_etf, P_etf, entry_n=55, exit_n=0, max_pos=MAX_POS)
for tag, cr in (("80/20", 0.8 * b_r + 0.2 * g_r), ("纯国债", b_r.copy())):
    nav, pos, tr = backtest(d_etf, c_etf, P_etf, sig, cr, max_pos=MAX_POS, pos_style="equal")
    p = perf(nav, tag)
    navs[f"etf_{tag}"] = nav
    out[f"etf_{tag}"] = p
    print(f"  {tag:8s}: CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
          f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
print(f"  → CAGR 变化: {(out['etf_纯国债']['cagr']-out['etf_80/20']['cagr'])*100:+.2f}pp")

# ==================== M3 板块轮动
print("\n" + "=" * 100)
print("M3 板块 ETF 轮动（top3）")
print("=" * 100)
from qf import sector_rotation as qfsr  # noqa

# 临时改全局常量重跑
old_bw, old_gw = C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT
d2, c2, cl2, am2, sec2 = qfsr.load_pool()
for tag, (bw, gw) in (("80/20", (0.8, 0.2)), ("纯国债", (1.0, 0.0))):
    C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT = bw, gw
    r = qfsr.backtest(d2, c2, cl2, am2, sec2, start_row=500, verbose=False)
    C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT = old_bw, old_gw
    p = perf(r["nav"], tag)
    navs[f"sec_{tag}"] = r["nav"]
    out[f"sector_{tag}"] = p
    print(f"  {tag:8s}: CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
          f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
print(f"  → CAGR 变化: {(out['sector_纯国债']['cagr']-out['sector_80/20']['cagr'])*100:+.2f}pp")

# ==================== M4 组合
print("\n" + "=" * 100)
print("M4 三策略组合 50/30/20")
print("=" * 100)
for tag in ("80/20", "纯国债"):
    common = (navs[f"main_{tag}"].index
              .intersection(navs[f"etf_{tag}"].index)
              .intersection(navs[f"sec_{tag}"].index))
    common = common[(common >= S) & (common <= E)]
    rets = pd.DataFrame({
        "main": navs[f"main_{tag}"].reindex(common).pct_change().fillna(0),
        "etf": navs[f"etf_{tag}"].reindex(common).pct_change().fillna(0),
        "sector": navs[f"sec_{tag}"].reindex(common).pct_change().fillna(0),
    })
    combo_ret = 0.5 * rets["main"] + 0.3 * rets["etf"] + 0.2 * rets["sector"]
    nav_c = (1 + combo_ret).cumprod()
    p = perf(nav_c, tag)
    out[f"combo_{tag}"] = p
    print(f"  {tag:8s}: CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
          f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
print(f"  → CAGR 变化: {(out['combo_纯国债']['cagr']-out['combo_80/20']['cagr'])*100:+.2f}pp")

# ==================== 汇总
print("\n" + "=" * 100)
print("汇总：所有模型去掉黄金后的变化")
print("=" * 100)
print(f"{'模型':<22}{'CAGR(80/20)':>13}{'CAGR(纯国债)':>14}{'变化':>10}"
      f"{'Sharpe变化':>12}{'回撤变化':>11}")
print("-" * 100)
for mname, key in (("主策略", "main"), ("宽基ETF", "etf"),
                   ("板块轮动", "sector"), ("三策略组合", "combo")):
    a, b = out[f"{key}_80/20"], out[f"{key}_纯国债"]
    print(f"{mname:<22}{a['cagr']:>12.2%}{b['cagr']:>13.2%}"
          f"{(b['cagr']-a['cagr'])*100:>+9.2f}pp"
          f"{b['sharpe']-a['sharpe']:>+12.3f}"
          f"{(b['max_dd']-a['max_dd'])*100:>+10.2f}pp")

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
