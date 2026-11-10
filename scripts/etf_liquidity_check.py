# -*- coding: utf-8 -*-
"""按次方量化文章的标准，检查我们 ETF 池的流动性。

文章判定标准（§6）：
  等级        近20日日均成交额        买卖价差
  🟢 良好     > 5000万              < 0.1%
  🟡 一般     1000万 ~ 5000万       0.1% ~ 0.3%
  🔴 差       < 1000万              > 0.3%

  两项取较差的一档。

另外文章给了单笔冲击成本的警戒线：
  "单只 ETF 买入金额占日均成交额比例超过 1% 时开始警惕"

我们用日均成交额做检查（价差需要盘口数据，我们没有历史盘口，
用"日内振幅"和"成交额"近似）。
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
OUT = ROOT / "build" / "eval" / "etf_liquidity_check.json"

z = np.load(ROOT / "build" / "etf" / "etf_ohlc_panel.npz", allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
meta = json.load(open(ROOT / "build" / "etf" / "etf_ohlc_meta.json", encoding="utf-8"))
amt = z["amount"].astype(np.float64)     # 千元? 需确认
close = z["close"].astype(np.float64)
high, low = z["high"], z["low"]

# 腾讯返回的成交额单位是"万元"（第9个字段），检查量级
recent = amt[-20:]
print("=" * 96)
print("ETF 池流动性检查（按次方量化标准）")
print("=" * 96)
print(f"  数据 {len(dates)} 日 × {len(codes)} 只")
print(f"  最近日期: {dates[-1].date()}")

# 判断成交额单位
sample = np.nanmedian(recent)
print(f"  成交额中位数（原始值）: {sample:,.1f}")
# 宽基 ETF 日均成交额通常数亿~数十亿
# 若原始值 ~10000，可能是万元；若 ~1e8，是元
print()

CAP = 300000.0
rows = []
for j, c in enumerate(codes):
    a = amt[:, j]
    ok = np.isfinite(a)
    if ok.sum() < 20:
        continue
    a20 = float(np.nanmean(a[-20:]))
    # 假设单位是万元 → 转元
    a20_yuan = a20 * 1e4 if a20 < 1e6 else a20

    # 用日内振幅近似价差（无盘口数据）
    h, l, cl = high[:, j], low[:, j], close[:, j]
    m = np.isfinite(h) & np.isfinite(l) & np.isfinite(cl) & (cl > 0)
    spread_proxy = float(np.nanmedian((h[m] - l[m]) / cl[m])) if m.sum() > 20 else np.nan

    # 单笔占日均成交额比例（top2 时每只 15 万）
    per_trade = CAP / 2 if "宽基" in str(meta.get(c, "")) else CAP / 2
    ratio = per_trade / a20_yuan if a20_yuan > 0 else np.inf

    # 判定（用成交额）
    if a20_yuan > 5e7:
        lvl = "🟢 良好"
    elif a20_yuan > 1e7:
        lvl = "🟡 一般"
    else:
        lvl = "🔴 差"

    # 冲击成本警戒
    warn = "⚠️" if ratio > 0.01 else ""

    rows.append({
        "code": c, "name": meta.get(c, c),
        "amt20_yuan": a20_yuan, "amt20_yi": a20_yuan / 1e8,
        "spread_proxy": spread_proxy,
        "trade_ratio": ratio, "level": lvl, "warn": warn,
    })

df = pd.DataFrame(rows).sort_values("amt20_yuan", ascending=False)
print(f"\n{'代码':<8}{'名称':<12}{'20日均额(亿)':>14}{'振幅代理':>10}"
      f"{'单笔占比':>10}{'等级':>10}")
print("-" * 96)
for _, r in df.iterrows():
    print(f"{r['code']:<8}{r['name']:<12}{r['amt20_yi']:>13.2f}"
          f"{r['spread_proxy']*100:>9.2f}%{r['trade_ratio']*100:>9.3f}%"
          f"{r['level']:>10}{r['warn']}")

print("\n" + "=" * 96)
print("汇总")
print("=" * 96)
cnt = df["level"].value_counts()
for k, v in cnt.items():
    print(f"  {k}: {v} 只")
n_bad = int((df["level"] == "🔴 差").sum())
n_warn = int((df["trade_ratio"] > 0.01).sum())
print(f"\n  流动性差（应移出）: {n_bad} 只")
print(f"  单笔占比 >1%（有冲击风险）: {n_warn} 只")

if n_bad:
    print(f"\n  🔴 流动性差（建议移出）:")
    for _, r in df[df["level"] == "🔴 差"].iterrows():
        print(f"    {r['code']} {r['name']}: 日均 {r['amt20_yi']*1e4:.0f} 万")

print("\n" + "=" * 96)
print("对照：我们的 ETF 趋势策略用了哪些标的")
print("=" * 96)
# 读交易记录看实际用了哪些
tr = pd.read_csv(ROOT / "build" / "runs" / "etf_trend" / "trades_donchian55.csv")
used = tr[tr["date"] >= "2020-01-08"]["code"].astype(str).unique()
print(f"  策略实际交易过 {len(used)} 只: {sorted(used)}")
print()
for c in sorted(used):
    row = df[df["code"] == c]
    if len(row):
        r = row.iloc[0]
        print(f"    {c} {r['name']:<10} 日均 {r['amt20_yi']:>6.2f} 亿  {r['level']}")
    else:
        print(f"    {c} -- 无数据")

json.dump({"n_etf": len(df), "liquidity_counts": cnt.to_dict(),
           "bad": int(n_bad), "warn": int(n_warn),
           "detail": df.to_dict("records")},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
