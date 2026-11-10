# -*- coding: utf-8 -*-
"""给出两种口径的完整对比，供最终决策。

口径 A：旧版（有同 bar 前视）→ 10.57%
口径 B：严格 T+1（改造后）   → 8.13% (top1) / 7.96% (top4)

用户选择了"保持 top1 不改造"，但前提是 10.57% 是真实的。
现已证明 10.57% 含前视 → 必须重新呈现真实选项。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci, sharpe_ratio  # noqa: E402
from scripts.etf_trend_strategy import load_cash, load_etf, signals, backtest  # noqa: E402

OUT = ROOT / "build" / "eval" / "etf_two_baselines.json"
S, E = "2020-01-02", "2026-10-23"

dates, codes, P = load_etf()
b, g = load_cash()
idx = pd.DatetimeIndex(dates)
cash = _blend_cash(b, g, idx)


def _blend_cash(b, g, idx):
    """按 C.CASH_BOND_WEIGHT / C.CASH_GOLD_WEIGHT 混合现金收益（黄金权重可为 0）。"""
    bw, gw = float(C.CASH_BOND_WEIGHT), float(C.CASH_GOLD_WEIGHT)
    tot = bw + gw
    if tot <= 0:
        return pd.Series(0.0, index=idx)
    r = (bw / tot) * b.reindex(idx).ffill().pct_change().fillna(0)
    if gw > 0:
        r = r + (gw / tot) * g.reindex(idx).ffill().pct_change().fillna(0)
    return r


def perf(nav, s=S, e=E):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino",
                                     "max_dd", "calmar")}


print("=" * 100)
print("两种口径对比（这是真实选项，不是假设）")
print("=" * 100)
res = {}

# 旧版结果（读已有净值）
old_nav = pd.read_parquet(ROOT / "build/runs/etf_trend/nav_donchian55.parquet")["nav"]
old_nav.index = pd.to_datetime(old_nav.index)
p_old = perf(old_nav)
res["旧版(有前视)"] = p_old
print(f"\n  【口径 A】旧版（含同 bar 前视）")
print(f"    CAGR {p_old['cagr']:6.2%} Sharpe {p_old['sharpe']:5.3f} "
      f"MaxDD {p_old['max_dd']:7.2%} Calmar {p_old['calmar']:5.2f}")
print(f"    ⚠️ 买入当日白拿 +1.538% 涨幅 → 虚高约 2.4pp")

# 严格 T+1 各配置
print(f"\n  【口径 B】严格 T+1（改造后，无前视）")
for k in (1, 2, 3, 4, 5):
    sig = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=k)
    nav, pos, tr = backtest(dates, codes, P, sig, cash, max_pos=k, pos_style="equal")
    p = perf(nav)
    key = f"严格T+1 top{k}"
    res[key] = p
    print(f"    top{k}: CAGR {p['cagr']:6.2%} Sharpe {p['sharpe']:5.3f} "
          f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")

print("\n" + "=" * 100)
print("决策矩阵（真实数字）")
print("=" * 100)
print(f"{'选项':<24}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}")
print("-" * 60)
order = ["旧版(有前视)", "严格T+1 top1", "严格T+1 top2", "严格T+1 top3",
         "严格T+1 top4", "严格T+1 top5"]
for k in order:
    if k in res:
        p = res[k]
        mark = " ⚠️前视" if "前视" in k else ""
        print(f"{k:<24}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}{p['max_dd']:>8.2%}"
              f"{p['calmar']:>8.2f}{mark}")

print("\n" + "=" * 100)
print("说明")
print("=" * 100)
print("  * '旧版(有前视)' 不可用于实盘预期 —— 它白拿了买入当日的涨幅")
print("  * 严格 T+1 的 top1 = 8.13%，这才是当前策略的真实水平")
print("  * top4 = 7.96% / Sharpe 1.082 / MaxDD -6.08% —— 风险调整最优")
print()
print(f"  top1 → top4：CAGR {res['严格T+1 top1']['cagr']:.2%} → "
      f"{res['严格T+1 top4']['cagr']:.2%}（-{abs(res['严格T+1 top1']['cagr']-res['严格T+1 top4']['cagr'])*100:.2f}pp）")
print(f"                Sharpe {res['严格T+1 top1']['sharpe']:.3f} → "
      f"{res['严格T+1 top4']['sharpe']:.3f}（+{(res['严格T+1 top4']['sharpe']/res['严格T+1 top1']['sharpe']-1)*100:.1f}%）")
print(f"                MaxDD {res['严格T+1 top1']['max_dd']:.2%} → "
      f"{res['严格T+1 top4']['max_dd']:.2%}（收窄 {abs(res['严格T+1 top1']['max_dd']-res['严格T+1 top4']['max_dd'])*100:.2f}pp）")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
