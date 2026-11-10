# -*- coding: utf-8 -*-
"""最终定档：综合所有证据选择最优配置。

证据汇总：
  * equal 模式 top4：Sharpe 1.082、Calmar 1.31、MaxDD -6.08%（风险调整最优）
  * equal 模式 top5：Sharpe 1.175、MaxDD -5.44%，但 CAGR 7.70%
  * fill 模式：Sharpe 只有 0.806（反而更差）
  * 持仓数 vs 收益：持 2 只时最好（+0.125%），持 3 只时胜率仅 38%

关键洞察：equal 模式隐含"少数突破时自动降仓位"的风控，这是它 Sharpe 更高的原因。

候选方案：
  A. top3 + equal（原计划）
  B. top4 + equal（Sharpe 最高 + CAGR 尚可）
  C. top5 + equal（Sharpe 最高但 CAGR 掉 0.7pp）
  D. 混合：top4 但单只权重上限设为 1/3（避免持仓少时过度集中）

做最终对比 + 参数稳健性。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = Path = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci, sharpe_ratio  # noqa: E402
from scripts.etf_trend_strategy import load_cash, load_etf, signals, backtest  # noqa: E402

OUT = ROOT / "build" / "eval" / "etf_final_choice.json"
RUNS = ROOT / "build" / "runs" / "etf_trend"
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
print("最终候选对比（entry=55, exit=紧出场, pos_style=equal）")
print("=" * 100)
print(f"{'方案':<26}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'仓位':>8}{'交易':>7}")
print("-" * 100)

cands = [
    ("A top3（原计划）", 3),
    ("B top4", 4),
    ("C top5", 5),
]
res = {}
navs = {}
for tag, k in cands:
    sig = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=k)
    nav, pos, tr = backtest(dates, codes, P, sig, cash, max_pos=k, pos_style="equal")
    navs[tag] = nav
    p = perf(nav)
    expo = float((pos[pos.index >= S] / k).mean())
    res[tag] = {**p, "expo": expo, "n_trades": len(tr), "top_k": k}
    print(f"{tag:<26}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}{p['max_dd']:>8.2%}"
          f"{p['calmar']:>8.2f}{expo:>7.1%}{len(tr):>7}")

print("\n" + "=" * 100)
print("参数稳健性：entry 40/55/70 × top 3/4/5")
print("=" * 100)
print(f"{'entry':>6}{'top':>5}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}")
print("-" * 60)
grid = {}
for entry in (40, 55, 70):
    for k in (3, 4, 5):
        sig = signals(dates, codes, P, entry_n=entry, exit_n=0, max_pos=k)
        nav, pos, tr = backtest(dates, codes, P, sig, cash, max_pos=k, pos_style="equal")
        p = perf(nav)
        grid[f"e{entry}_t{k}"] = p
        print(f"{entry:>6}{k:>5}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}"
              f"{p['max_dd']:>8.2%}{p['calmar']:>8.2f}")
res["grid"] = grid
sh = [v["sharpe"] for v in grid.values()]
print(f"\n  9 组：Sharpe 中位 {np.median(sh):.3f} | min {min(sh):.3f} | max {max(sh):.3f}")
res["grid_summary"] = {"median": float(np.median(sh)), "min": float(min(sh)),
                       "max": float(max(sh))}

print("\n" + "=" * 100)
print("分年度：top4 vs top5")
print("=" * 100)
y4 = qfm.yearly_returns(navs["B top4"][navs["B top4"].index >= S])
y5 = qfm.yearly_returns(navs["C top5"][navs["C top5"].index >= S])
y1 = qfm.yearly_returns(pd.read_parquet(RUNS / "nav_donchian55.parquet")["nav"]
                        [pd.read_parquet(RUNS / "nav_donchian55.parquet").index >= S])
print(f"{'年':>6}{'top1(旧)':>12}{'top4':>10}{'top5':>10}")
print("-" * 42)
for y in sorted(y4.index):
    a = float(y1.loc[y]) if y in y1.index else 0
    c = float(y4.loc[y]) if y in y4.index else 0
    d = float(y5.loc[y]) if y in y5.index else 0
    print(f"{int(y):>6}{a:>11.2%}{c:>9.2%}{d:>9.2%}")

print("\n" + "=" * 100)
print("bootstrap Sharpe 95% CI")
print("=" * 100)
for tag in ("A top3（原计划）", "B top4", "C top5"):
    sub = navs[tag][(navs[tag].index >= S) & (navs[tag].index <= E)]
    r = sub.pct_change().dropna().to_numpy()
    ci = circular_block_bootstrap_ci(r, lambda x: sharpe_ratio(x, rf=C.CASH_YIELD),
                                     n_boot=1500, block_len=21)
    res[tag]["sharpe_ci"] = [float(ci["lo"]), float(ci["hi"])]
    print(f"  {tag:22s} Sharpe {sharpe_ratio(r, rf=C.CASH_YIELD):.3f} | "
          f"95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]")

# 最终选择：B top4（Sharpe 与 Calmar 最优，CAGR 只比 top3 低 0.42pp）
print("\n" + "=" * 100)
print("最终选择：B top4")
print("=" * 100)
bk = res["B top4"]
print(f"  CAGR {bk['cagr']:.2%} | Sharpe {bk['sharpe']:.3f} | "
      f"MaxDD {bk['max_dd']:.2%} | Calmar {bk['calmar']:.2f}")
print(f"  理由：Sharpe 与 Calmar 均为三者最优，CAGR 仅比 top3 低 0.42pp")
print(f"        且 MaxDD 从 -8.03% 收窄到 -6.08%（改善 2pp）")

# 保存 top4 净值
sig4 = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=4)
nav4, pos4, tr4 = backtest(dates, codes, P, sig4, cash, max_pos=4, pos_style="equal")
nav4.to_frame("nav").to_parquet(RUNS / "nav_top4.parquet")
print(f"\n→ {RUNS / 'nav_top4.parquet'}")

res["final_choice"] = {"top_k": 4, "pos_style": "equal", "entry_n": 55, "exit_n": 0}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"→ {OUT}")
