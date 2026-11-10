# -*- coding: utf-8 -*-
"""ETF 右侧 top3 改造的完整验证。

验证项（缺一不可）：
  V1 基准对比：top1（旧行为）vs top2 / top3 / top4
  V2 参数稳健性：entry 40/55/70 × top 1/2/3
  V3 分年度：top3 不能在任何一年明显变差
  V4 bootstrap Sharpe CI
  V5 严格 T+1 校验（首笔交易必须晚于首个信号日）
  V6 仓位与交易次数
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

OUT = ROOT / "build" / "eval" / "etf_top3_validation.json"
RUNS = ROOT / "build" / "runs" / "etf_trend"
RUNS.mkdir(parents=True, exist_ok=True)

dates, codes, P = load_etf()
b, g = load_cash()
idx = pd.DatetimeIndex(dates)
cash_ret = _blend_cash(b, g, idx)
S, E = "2020-01-02", "2026-10-23"


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


res = {}
print("=" * 100)
print("V1 基准对比：top_k 扫描（严格 T+1）")
print("=" * 100)
print(f"{'方案':<22}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'仓位':>8}{'交易':>7}")
print("-" * 100)

navs = {}
best = None
for k in (1, 2, 3, 4):
    sig = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=k)
    nav, pos, tr = backtest(dates, codes, P, sig, cash_ret, max_pos=k)
    p = perf(nav)
    expo = float((pos[(pos.index >= S)] > 0).mean())
    avg_n = float(pos[(pos.index >= S)].mean())
    tag = f"top{k}"
    navs[tag] = nav
    res[tag] = {**p, "expo_days": expo, "avg_holdings": avg_n, "n_trades": len(tr)}
    print(f"{tag:<22}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}{p['max_dd']:>8.2%}"
          f"{p['calmar']:>8.2f}{expo:>7.1%}{len(tr):>7}")
    if best is None or p["sharpe"] > best[1]:
        best = (tag, p["sharpe"])

print(f"\n  → Sharpe 最高：{best[0]}（{best[1]:.3f}）")

print("\n" + "=" * 100)
print("V2 参数稳健性：entry_n × top_k")
print("=" * 100)
print(f"{'entry':>6}{'top':>5}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}")
print("-" * 60)
grid = {}
for entry in (40, 55, 70):
    for k in (1, 2, 3):
        sig = signals(dates, codes, P, entry_n=entry, exit_n=0, max_pos=k)
        nav, pos, tr = backtest(dates, codes, P, sig, cash_ret, max_pos=k)
        p = perf(nav)
        grid[f"e{entry}_top{k}"] = p
        mark = " ◀" if (entry == 55 and k == 3) else ""
        print(f"{entry:>6}{k:>5}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}"
              f"{p['max_dd']:>8.2%}{p['calmar']:>8.2f}{mark}")
sh = [v["sharpe"] for v in grid.values()]
print(f"\n  9 组全部：Sharpe 中位 {np.median(sh):.3f} | min {min(sh):.3f} | "
      f"max {max(sh):.3f} | 为正 {(np.array(sh)>0).mean():.0%}")
res["grid"] = grid
res["grid_summary"] = {"n": len(grid), "median": float(np.median(sh)),
                       "min": float(min(sh)), "max": float(max(sh))}

print("\n" + "=" * 100)
print("V3 分年度：top1 vs top3")
print("=" * 100)
y1 = qfm.yearly_returns(navs["top1"][navs["top1"].index >= S])
y3 = qfm.yearly_returns(navs["top3"][navs["top3"].index >= S])
print(f"{'年':>6}{'top1':>10}{'top3':>10}{'差':>10}")
print("-" * 40)
diffs = []
for y in sorted(y1.index):
    v1, v3 = float(y1.loc[y]), float(y3.loc[y]) if y in y3.index else 0.0
    d = v3 - v1
    diffs.append(d)
    print(f"{int(y):>6}{v1:>9.2%}{v3:>9.2%}{d:>9.2%}")
res["yearly"] = {"top1": {int(k): float(v) for k, v in y1.items()},
                 "top3": {int(k): float(v) for k, v in y3.items()},
                 "mean_diff": float(np.mean(diffs))}

print("\n" + "=" * 100)
print("V4 bootstrap Sharpe 95% CI")
print("=" * 100)
for tag in ("top1", "top3"):
    sub = navs[tag][(navs[tag].index >= S) & (navs[tag].index <= E)]
    r = sub.pct_change().dropna().to_numpy()
    sr = sharpe_ratio(r, rf=C.CASH_YIELD)
    ci = circular_block_bootstrap_ci(r, lambda x: sharpe_ratio(x, rf=C.CASH_YIELD),
                                     n_boot=1500, block_len=21)
    res[tag]["sharpe_ci"] = [float(ci["lo"]), float(ci["hi"])]
    print(f"  {tag:6s} Sharpe {sr:.3f} | 95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]")

print("\n" + "=" * 100)
print("V5 严格 T+1 校验")
print("=" * 100)
sig3 = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=3)
nav3, pos3, tr3 = backtest(dates, codes, P, sig3, cash_ret, max_pos=3)
if len(tr3):
    first_sig_t = next(t for t in range(len(sig3)) if sig3[t])
    first_tr = pd.Timestamp(tr3.iloc[0]["date"])
    first_tr_t = int(idx.get_indexer([first_tr])[0])
    print(f"  首个信号日 index {first_sig_t}（{dates[first_sig_t].date()}）")
    print(f"  首笔交易日 index {first_tr_t}（{first_tr.date()}）")
    ok = first_tr_t > first_sig_t
    print(f"  → 首笔交易晚于信号日：{'✅ 通过' if ok else '❌ 失败'}")
    res["t1_check"] = {"first_signal_t": first_sig_t, "first_trade_t": first_tr_t,
                       "pass": bool(ok)}
    # 检查是否有同日买卖同一标的
    same_day = tr3.groupby(["date", "code"]).size()
    n_dup = int((same_day > 1).sum())
    print(f"  同日同标的重复交易：{n_dup} 次 {'✅' if n_dup == 0 else '⚠️'}")
    res["t1_check"]["dup_same_day"] = n_dup

print("\n" + "=" * 100)
print("V6 结论")
print("=" * 100)
t1s = res["top1"]["sharpe"]
t3s = res["top3"]["sharpe"]
print(f"  top1 → top3：Sharpe {t1s:.3f} → {t3s:.3f}（{(t3s/t1s-1)*100:+.1f}%）")
print(f"                CAGR {res['top1']['cagr']:.2%} → {res['top3']['cagr']:.2%}")
print(f"                MaxDD {res['top1']['max_dd']:.2%} → {res['top3']['max_dd']:.2%}")

# 保存 top3 净值供网页使用
nav3.to_frame("nav").to_parquet(RUNS / "nav_top3.parquet")
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
print(f"→ {RUNS / 'nav_top3.parquet'}")
