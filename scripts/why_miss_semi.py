# -*- coding: utf-8 -*-
"""复盘：2026 年板块轮动策略为何没抓住半导体行情。

逐项检查：
  1) 588200（半导体ETF）2026 年表现 vs 板块轮动策略表现
  2) 588200 是否在池内、是否满足流动性门槛
  3) 2026 年每个调仓日：588200 是否触发 55 日突破
  4) 若触发，是否被"动量前 3"挤掉（排第几）
  5) 策略 2026 年实际持有什么
  6) 对照：如果 2026 年一直满仓半导体，会怎样
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
os.environ.setdefault("QF_DATA_RAW", str(ROOT.parent / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm, sector_rotation as qfsr  # noqa: E402

OUT = ROOT / "build" / "eval" / "why_miss_semiconductor_2026.json"

dates, codes, close, amount, sector_of = qfsr.load_pool()
T, N = close.shape
print(f"板块池：{N} 只 / {len(set(sector_of.values()))} 个板块")
print(f"区间：{dates[0].date()} ~ {dates[-1].date()}")

# 半导体是否在池内
semi_j = None
for j, c in enumerate(codes):
    if sector_of.get(c) == "半导体":
        semi_j = j
        print(f"\n半导体标的：{c}（{sector_of[c]}）")
        break
if semi_j is None:
    print("\n⚠️ 板块池中没有半导体！")
    print("池内板块：", sorted(set(sector_of.values())))
    sys.exit(0)

semi = close[:, semi_j]
amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
fin = np.isfinite(semi)
print(f"  首个有效日：{dates[fin][0].date()}")
print(f"  2026 年有效天数：{int((fin & (dates.year == 2026)).sum())}")

# ---------- 1) 2026 年表现对比
print("\n" + "=" * 90)
print("1) 2026 年表现对比")
print("=" * 90)
nav = pd.read_parquet(ROOT / "build" / "runs" / "sector_rotation" / "nav.parquet")["nav"]
nav.index = pd.to_datetime(nav.index)

y26 = pd.Timestamp("2026-01-01")
seg_strategy = nav[nav.index >= y26]
r_strategy = seg_strategy.iloc[-1] / seg_strategy.iloc[0] - 1

m26 = (dates >= y26) & fin
idx26 = np.flatnonzero(m26)
if len(idx26) > 1:
    r_semi = semi[idx26[-1]] / semi[idx26[0]] - 1
    # 半导体在 2026 年的最大回撤
    seg = pd.Series(semi[idx26], index=dates[idx26])
    dd = (seg / seg.cummax() - 1).min()
    print(f"  板块轮动策略 2026 年收益：{r_strategy:+.2%}")
    print(f"  半导体ETF   2026 年收益：{r_semi:+.2%}")
    print(f"  差距：{(r_strategy - r_semi)*100:+.2f}pp")
    print(f"  半导体 2026 年最大回撤：{dd:.2%}")
    print(f"  半导体 2026 年区间：{dates[idx26[0]].date()} ~ {dates[idx26[-1]].date()}")

# ---------- 2) 流动性门槛
print("\n" + "=" * 90)
print("2) 流动性门槛检查（MIN_AMOUNT = {:.2e} = {:.0f} 万）".format(
    qfsr.MIN_AMOUNT, qfsr.MIN_AMOUNT / 1e4))
print("=" * 90)
a26 = amt20[(dates >= y26) & fin, semi_j]
print(f"  2026 年 20日均额：中位 {np.nanmedian(a26)/1e8:.2f} 亿 | "
      f"最低 {np.nanmin(a26)/1e8:.2f} 亿")
print(f"  达标天数：{int((a26 >= qfsr.MIN_AMOUNT).sum())} / {len(a26)}")

# ---------- 3) 2026 年每个调仓日：是否突破、排第几
print("\n" + "=" * 90)
print("3) 2026 年逐次调仓：半导体是否触发突破 / 排名第几")
print("=" * 90)
cs = pd.DataFrame(close)
hi = cs.rolling(qfsr.ENTRY_N).max().shift(1).to_numpy()
mom = (cs / cs.shift(qfsr.MOM_N) - 1.0).to_numpy()

rows = []
t0 = int(np.flatnonzero(dates >= y26)[0])
for t in range(t0, T):
    if t % qfsr.REBAL != 0:
        continue
    sig = qfsr.build_signal_at(dates, codes, close, amount, sector_of, t)
    # 半导体的状态
    c = close[t, semi_j]
    amt_ok = np.isfinite(amt20[t, semi_j]) and amt20[t, semi_j] >= qfsr.MIN_AMOUNT
    brk = np.isfinite(hi[t, semi_j]) and c > hi[t, semi_j]
    mv = mom[t, semi_j]
    # 所有突破板块按动量排序，看半导体位置
    cand = []
    best = {}
    for j in range(N):
        cc = close[t, j]
        if not np.isfinite(cc):
            continue
        if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= qfsr.MIN_AMOUNT):
            continue
        sec = sector_of.get(codes[j], codes[j])
        m = float(mom[t, j]) if np.isfinite(mom[t, j]) else -9.0
        if sec not in best or m > best[sec][1]:
            best[sec] = (j, m)
    for sec, (j, m) in best.items():
        if np.isfinite(hi[t, j]) and close[t, j] > hi[t, j]:
            cand.append((sec, j, m))
    cand.sort(key=lambda x: -x[2])
    rank = None
    for i, (sec, j, m) in enumerate(cand):
        if sec == "半导体":
            rank = i + 1
            break
    picks = [p["sector"] for p in sig["picks"]]
    rows.append({
        "date": str(dates[t].date()),
        "semi_amt_ok": bool(amt_ok),
        "semi_breakout": bool(brk),
        "semi_mom60": float(mv) if np.isfinite(mv) else None,
        "semi_rank_among_breakouts": rank,
        "n_breakout": len(cand),
        "picks": picks,
        "semi_picked": "半导体" in picks,
    })

for r in rows:
    mv = f"{r['semi_mom60']*100:+6.1f}%" if r["semi_mom60"] is not None else "   n/a"
    rk = f"第{r['semi_rank_among_breakouts']}" if r["semi_rank_among_breakouts"] else "未触发"
    print(f"  {r['date']} | 流动{'OK' if r['semi_amt_ok'] else 'NG'} "
          f"| 突破{'Y' if r['semi_breakout'] else 'N'} "
          f"| 动量 {mv} | 突破板块 {r['n_breakout']:2d} 个 | 半导体{rk:6s} "
          f"| 持仓 {r['picks']}")

n_rebal = len(rows)
n_break = sum(1 for r in rows if r["semi_breakout"])
n_pick = sum(1 for r in rows if r["semi_picked"])
n_inrank = sum(1 for r in rows if r["semi_rank_among_breakouts"] and r["semi_rank_among_breakouts"] <= 3)
print(f"\n  2026 年调仓 {n_rebal} 次：半导体突破 {n_break} 次，"
      f"进入前3名 {n_inrank} 次，实际被选中 {n_pick} 次")

# ---------- 4) 对照：一直满仓半导体
print("\n" + "=" * 90)
print("4) 对照：2026 年一直满仓半导体 vs 策略")
print("=" * 90)
if len(idx26) > 1:
    n_days = idx26[-1] - idx26[0] + 1
    cagr_semi = (1 + r_semi) ** (252 / max(n_days, 1)) - 1
    print(f"  一直持有半导体：收益 {r_semi:+.2%}（年化 {cagr_semi:+.2%}），"
          f"期间最大回撤 {dd:.2%}")
    print(f"  板块轮动策略  ：收益 {r_strategy:+.2%}")
    print(f"  → 差距 {(r_semi - r_strategy)*100:+.2f}pp，"
          f"但策略回撤远小于半导体（{dd:.1%} vs 策略参见整体指标）")

res = {
    "semi_code": codes[semi_j],
    "y2026": {
        "strategy_return": float(r_strategy),
        "semi_return": float(r_semi) if len(idx26) > 1 else None,
        "semi_max_dd": float(dd) if len(idx26) > 1 else None,
        "gap_pp": float((r_strategy - r_semi) * 100) if len(idx26) > 1 else None,
    },
    "liquidity": {
        "min_amount": float(qfsr.MIN_AMOUNT),
        "median_2026": float(np.nanmedian(a26)) if len(a26) else None,
        "days_ok": int((a26 >= qfsr.MIN_AMOUNT).sum()),
        "days_total": int(len(a26)),
    },
    "rebalances": rows,
    "summary": {"n_rebal": n_rebal, "n_breakout": n_break,
                "n_top3": n_inrank, "n_picked": n_pick},
}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
