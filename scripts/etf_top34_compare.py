# -*- coding: utf-8 -*-
"""补充验证：top3 vs top4 的完整对比（含分年度、稳定性）。

发现 top4 的 Sharpe（1.082）显著高于 top3（0.952），
但 CAGR 略低（7.96% vs 8.38%）。需要判断哪个更优。

判据：
  1) 风险调整收益（Sharpe / Calmar）优先 → top4
  2) 绝对收益优先 → top3
  3) 稳定性（分年度不能有某年暴跌）→ 都要看
  4) 参数高原中心 → 看 entry 40/55/70 下 top3/top4 是否都稳
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

OUT = ROOT / "build" / "eval" / "etf_top3_vs_top4.json"
S, E = "2020-01-02", "2026-10-23"

dates, codes, P = load_etf()
b, g = load_cash()
idx = pd.DatetimeIndex(dates)
cash_ret = _blend_cash(b, g, idx)


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
                                     "max_dd", "calmar", "win_rate_monthly")}


print("=" * 96)
print("top3 vs top4 完整对比")
print("=" * 96)

res = {}
navs = {}
for k in (1, 2, 3, 4, 5):
    sig = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=k)
    nav, pos, tr = backtest(dates, codes, P, sig, cash_ret, max_pos=k)
    navs[k] = nav
    p = perf(nav)
    # 分区间
    pv = perf(nav, "2020-01-02", "2021-12-31")
    po = perf(nav, "2022-01-01", E)
    avg_n = float(pos[(pos.index >= S)].mean())
    res[f"top{k}"] = {**p, "val_cagr": pv["cagr"], "oos_cagr": po["cagr"],
                      "oos_sharpe": po["sharpe"], "avg_holdings": avg_n,
                      "n_trades": len(tr)}
    print(f"\n  top{k}:")
    print(f"    全期  CAGR {p['cagr']:6.2%} Sharpe {p['sharpe']:5.3f} "
          f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
    print(f"    验证期 {pv['cagr']:6.2%} | 样本外 {po['cagr']:6.2%} (SR {po['sharpe']:.3f})")
    print(f"    平均持仓 {avg_n:.2f} 只 | 交易 {len(tr)}")

# 分年度对比 top3 / top4 / top5
print("\n" + "=" * 96)
print("分年度：top3 vs top4 vs top5")
print("=" * 96)
ys = {k: qfm.yearly_returns(navs[k][navs[k].index >= S]) for k in (3, 4, 5)}
print(f"{'年':>6}{'top3':>10}{'top4':>10}{'top5':>10}")
print("-" * 40)
for y in sorted(ys[3].index):
    r3 = float(ys[3].loc[y]) if y in ys[3].index else 0
    r4 = float(ys[4].loc[y]) if y in ys[4].index else 0
    r5 = float(ys[5].loc[y]) if y in ys[5].index else 0
    print(f"{int(y):>6}{r3:>9.2%}{r4:>9.2%}{r5:>9.2%}")
res["yearly"] = {f"top{k}": {int(y): float(v) for y, v in ys[k].items()} for k in (3, 4, 5)}

# 参数稳健性：entry × top
print("\n" + "=" * 96)
print("参数稳健性：entry 40/55/70 × top 2/3/4/5")
print("=" * 96)
print(f"{'entry':>6}{'top':>5}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}")
print("-" * 60)
grid = {}
for entry in (40, 55, 70):
    for k in (2, 3, 4, 5):
        sig = signals(dates, codes, P, entry_n=entry, exit_n=0, max_pos=k)
        nav, pos, tr = backtest(dates, codes, P, sig, cash_ret, max_pos=k)
        p = perf(nav)
        grid[f"e{entry}_t{k}"] = p
        print(f"{entry:>6}{k:>5}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}"
              f"{p['max_dd']:>8.2%}{p['calmar']:>8.2f}")
res["grid"] = grid
sh = [v["sharpe"] for v in grid.values()]
print(f"\n  12 组：Sharpe 中位 {np.median(sh):.3f} | min {min(sh):.3f} | max {max(sh):.3f}")
res["grid_summary"] = {"median": float(np.median(sh)), "min": float(min(sh)),
                       "max": float(max(sh))}

# bootstrap
print("\n" + "=" * 96)
print("bootstrap Sharpe 95% CI")
print("=" * 96)
for k in (3, 4):
    sub = navs[k][(navs[k].index >= S) & (navs[k].index <= E)]
    r = sub.pct_change().dropna().to_numpy()
    ci = circular_block_bootstrap_ci(r, lambda x: sharpe_ratio(x, rf=C.CASH_YIELD),
                                     n_boot=1500, block_len=21)
    res[f"top{k}"]["sharpe_ci"] = [float(ci["lo"]), float(ci["hi"])]
    print(f"  top{k}: Sharpe {sharpe_ratio(r, rf=C.CASH_YIELD):.3f} | "
          f"95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]")

# 最终推荐
print("\n" + "=" * 96)
print("推荐")
print("=" * 96)
t3, t4 = res["top3"], res["top4"]
print(f"  {'指标':<14}{'top3':>12}{'top4':>12}")
for k in ("cagr", "sharpe", "max_dd", "calmar", "oos_cagr"):
    print(f"  {k:<14}{t3[k]:>11.2%}{t4[k]:>11.2%}" if "cagr" in k or k == "max_dd"
          else f"  {k:<14}{t3[k]:>12.3f}{t4[k]:>12.3f}")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
