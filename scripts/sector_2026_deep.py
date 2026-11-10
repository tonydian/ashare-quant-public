# -*- coding: utf-8 -*-
"""深挖：板块轮动 2026 年为何整体 -10.48%。

关键事实（上一脚本已确认）：
  * 半导体 588200 在 2026 年确实被选中过 1 次（2026-06-29，动量 +98.9% 排第1）
  * 但只持有到 07-13 就出场（紧出场）
  * 半导体 2026 年 +12.71%，但期间最大回撤 -45.96%
  * 策略 2026 年 -10.48%

本脚本回答：钱到底亏在哪？
  1) 逐笔交易盈亏（2026 年）
  2) 每笔持仓期间该 ETF 的涨跌
  3) 假突破统计：触发突破后 N 日的收益分布
  4) 出场后是否继续涨（错过多少）
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

from qf import sector_rotation as qfsr  # noqa: E402

OUT = ROOT / "build" / "eval" / "sector_2026_trades.json"
dates, codes, close, amount, sector_of = qfsr.load_pool()
T, N = close.shape
y26 = pd.Timestamp("2026-01-01")

# ---------- 重跑回测并记录每笔交易
res = qfsr.backtest(dates, codes, close, amount, sector_of, start_row=500, verbose=False)
trades = res["trades"]
nav = res["nav"]

print("=" * 92)
print("1) 2026 年逐笔交易与盈亏")
print("=" * 92)
t26 = trades[trades["date"] >= "2026-01-01"].reset_index(drop=True)
print(f"2026 年交易 {len(t26)} 笔\n")

# 配对买卖：按 code 配对
open_pos = {}
rows = []
for r in t26.itertuples():
    code = r.code
    d = pd.Timestamp(r.date)
    if r.side == "buy":
        open_pos[code] = (d, float(r.price))
    elif r.side == "sell" and code in open_pos:
        bd, bp = open_pos.pop(code)
        ret = float(r.price) / bp - 1.0
        rows.append({"code": code, "sector": r.sector,
                     "buy_date": str(bd.date()), "buy_px": bp,
                     "sell_date": str(d.date()), "sell_px": float(r.price),
                     "ret": ret, "days": int((d - bd).days)})
# 未平仓
for code, (bd, bp) in open_pos.items():
    j = codes.index(code)
    cur = close[-1, j]
    rows.append({"code": code, "sector": sector_of.get(code, ""),
                 "buy_date": str(bd.date()), "buy_px": bp,
                 "sell_date": "未平仓", "sell_px": float(cur),
                 "ret": float(cur) / bp - 1.0, "days": int((dates[-1] - bd).days)})

if rows:
    print(f"{'板块':<10}{'代码':<8}{'买入':<12}{'卖出':<12}{'持有天数':>8}{'收益':>9}")
    for r in sorted(rows, key=lambda x: x["ret"]):
        print(f"{r['sector']:<10}{r['code']:<8}{r['buy_date']:<12}"
              f"{r['sell_date']:<12}{r['days']:>8}{r['ret']*100:>8.2f}%")
    rets = np.array([r["ret"] for r in rows])
    print(f"\n  胜率 {(rets>0).mean():.0%} | 平均 {rets.mean()*100:+.2f}% | "
          f"最好的 {rets.max()*100:+.2f}% | 最差 {rets.min()*100:+.2f}%")
    print(f"  盈利笔数 {(rets>0).sum()} / {len(rets)}")

# ---------- 2) 假突破统计（全期 + 2026）
print("\n" + "=" * 92)
print("2) 假突破统计：触发 55 日突破后，后续 N 日收益")
print("=" * 92)
cs = pd.DataFrame(close)
hi = cs.rolling(qfsr.ENTRY_N).max().shift(1).to_numpy()
amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()

for label, mask_year in (("全期 2017-2026", None), ("仅 2026", 2026)):
    stats = {}
    for fwd in (5, 10, 20):
        vals = []
        for t in range(260, T - fwd):
            if mask_year and dates[t].year != mask_year:
                continue
            for j in range(N):
                if not np.isfinite(close[t, j]) or not np.isfinite(close[t + fwd, j]):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= qfsr.MIN_AMOUNT):
                    continue
                if not (np.isfinite(hi[t, j]) and close[t, j] > hi[t, j]):
                    continue
                vals.append(close[t + fwd, j] / close[t, j] - 1.0)
        if vals:
            v = np.array(vals)
            stats[f"fwd{fwd}d"] = {
                "n": int(len(v)), "mean": float(v.mean()),
                "median": float(np.median(v)), "win": float((v > 0).mean()),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            }
    print(f"\n  [{label}]")
    for k, s in stats.items():
        print(f"    {k:8s}: n={s['n']:5d} 均值 {s['mean']*100:+6.2f}% "
              f"中位 {s['median']*100:+6.2f}% 胜率 {s['win']:.0%} "
              f"p10 {s['p10']*100:+6.2f}% p90 {s['p90']*100:+6.2f}%")

# ---------- 3) 半导体：突破后走势与策略持有期对照
print("\n" + "=" * 92)
print("3) 半导体 588200 在 2026 年的突破点与后续走势")
print("=" * 92)
sj = codes.index("588200") if "588200" in codes else None
if sj is not None:
    semi = close[:, sj]
    brk_pts = []
    for t in range(260, T):
        if dates[t].year != 2026:
            continue
        if not (np.isfinite(hi[t, sj]) and semi[t] > hi[t, sj]):
            continue
        if t > 0 and np.isfinite(hi[t - 1, sj]) and semi[t - 1] > hi[t - 1, sj]:
            continue  # 只取连续突破的第一天
        brk_pts.append(t)
    print(f"  2026 年首次突破日：{[str(dates[t].date()) for t in brk_pts]}")
    for t in brk_pts:
        fwd = {}
        for n in (5, 10, 20, 40):
            if t + n < T and np.isfinite(semi[t + n]):
                fwd[n] = semi[t + n] / semi[t] - 1
        s = "  ".join(f"{n}日 {v*100:+.1f}%" for n, v in fwd.items())
        print(f"    {dates[t].date()} 价格 {semi[t]:.3f} → {s}")
    # 2026 年半导体的月度走势
    print("\n  2026 年半导体月度涨跌：")
    m = pd.Series(semi, index=dates)
    m26 = m[m.index >= y26]
    for month, grp in m26.groupby(m26.index.to_period("M")):
        if len(grp) > 1:
            print(f"    {month}: {grp.iloc[-1]/grp.iloc[0]-1:+.1%}")

res_out = {
    "trades_2026": rows,
    "win_rate": float((np.array([r["ret"] for r in rows]) > 0).mean()) if rows else None,
    "avg_ret": float(np.mean([r["ret"] for r in rows])) if rows else None,
    "semi_breakouts_2026": [str(dates[t].date()) for t in brk_pts] if sj is not None else [],
}
OUT.write_text(json.dumps(res_out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
