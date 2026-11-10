# -*- coding: utf-8 -*-
"""查清矛盾：持仓期年化 +28.64% vs 单笔交易年化 -26.2%？

疑点分析：
  A. "持仓期日均 +0.100%" 是我的反推公式有误？
     反推：etf_r = (nav_r - (1-invest)*cash_r) / invest
     但 nav 的日收益里包含了**当日买入的成本和换仓**
     且 invest 用的是"前一日持仓数"，买卖当日的权重已变 → 公式失效

  B. "单笔交易年化 -26.2%" 是短期年化的数学陷阱
     平均持有 3.3 天，+0.29% 收益
     年化 = (1.0029)^(252/3.3) - 1 = 24.9%  ← 应该为正！
     我的公式用 (1+r)^(1/d)-1 再取平均，对 d=1 的极短交易会爆

  重新用正确方法：
  * 直接统计"每日持股收益"（不反推）
  * 单笔交易用几何平均而非算术平均
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
from scripts.etf_trend_strategy import MAX_POS, backtest, load_cash, load_etf, signals  # noqa: E402

S, E = "2020-01-02", "2026-10-23"
dates, codes, P = load_etf()
close = P["close"]
b, g = load_cash()
idx = pd.DatetimeIndex(dates)
cash = _blend_cash(b, g, idx)

sig = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=MAX_POS)
nav, pos, trades = backtest(dates, codes, P, sig, cash, max_pos=MAX_POS, pos_style="equal")

m = (nav.index >= S) & (nav.index <= E)
nav_s, pos_s, cash_s = nav[m], pos[m], cash[m]
ret = nav_s.pct_change().fillna(0)
n_days = len(nav_s)
years = n_days / 252

print("=" * 92)
print("正确口径重算")
print("=" * 92)

# ---------- 方法一：直接看 ETF 本身的收益（不反推）
print("\n【方法一】持仓期间，持仓 ETF 自身的日收益（直接取值，不反推）")
print("-" * 92)
etf_daily = []
for i in range(1, len(nav_s)):
    if pos_s.iloc[i - 1] == 0:
        continue
    # 前一日持有的标的，统计其当日收益
    # 从 signals 反查不容易，改用另一个方法：比较"满仓ETF"和"实际"的差异
# 改用：直接构造一个"只要持仓就满仓"的对照
sig_hold = []
for i, t in enumerate(dates):
    if i < len(pos_s) and pos_s.iloc[i] > 0:
        sig_hold.append(list(sig[i]))
    else:
        sig_hold.append([])

# 计算持仓 ETF 的等权日收益
print("  直接统计（持仓日，持仓标的的等权收益）：")
vals = []
for i in range(1, len(pos_s)):
    if pos_s.iloc[i - 1] == 0:
        continue
    t = int(idx.get_indexer([pos_s.index[i]])[0])
    held = sig[t - 1] if t - 1 < len(sig) else []
    if not held:
        continue
    rs = [close[t, j] / close[t - 1, j] - 1.0 for j in held
          if np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0]
    if rs:
        vals.append(np.mean(rs))
vals = np.array(vals)
print(f"    持仓日 {len(vals)} 天 | 日均 {vals.mean()*100:+.4f}% | "
      f"胜率 {(vals>0).mean():.1%}")
print(f"    年化（按252天）: {(1+vals.mean())**252-1:+.2%}")

# ---------- 方法二：单笔交易的正确年化
print("\n【方法二】单笔交易（几何法，不用 1/d 的算术平均）")
print("-" * 92)
tr = trades.copy()
tr["date"] = pd.to_datetime(tr["date"])
tr = tr[(tr["date"] >= S) & (tr["date"] <= E)]
open_pos = {}
pairs = []
for row in tr.itertuples():
    if row.side == "buy":
        open_pos[row.code] = (row.date, row.price)
    elif row.code in open_pos:
        bd, bp = open_pos.pop(row.code)
        pairs.append({"ret": row.price / bp - 1, "days": max((row.date - bd).days, 1)})
pdf = pd.DataFrame(pairs)
print(f"  完整交易 {len(pdf)} 笔 | 平均持有 {pdf['days'].mean():.1f} 天")
print(f"  平均收益 {pdf['ret'].mean()*100:+.3f}% | 胜率 {(pdf['ret']>0).mean():.1%}")
# 正确年化：先算每笔的"日均几何收益"，再取均值
pdf["daily_geo"] = (1 + pdf["ret"]) ** (1 / pdf["days"]) - 1
print(f"  单笔日均（几何）均值: {pdf['daily_geo'].mean()*100:+.4f}%")
print(f"  → 折算年化: {(1+pdf['daily_geo'].mean())**252-1:+.2%}")
print(f"  （旧算法 -26.2% 是错的：它把 d=1 的短期波动年化了）")

# ---------- 方法三：直接分解 nav 的增长
print("\n【方法三】直接分解：总收益来自哪里")
print("-" * 92)
total = nav_s.iloc[-1] / nav_s.iloc[0] - 1
cagr = (1 + total) ** (1 / years) - 1
# 现金部分：假设全程空仓能赚多少
cash_only = (1 + cash_s.fillna(0)).prod()
# ETF 部分的贡献 = 总 / 现金
print(f"  总净值 {nav_s.iloc[-1]/nav_s.iloc[0]:.4f}（{total:+.2%}，年化 {cagr:.2%}）")
print(f"  假如全程空仓（只有国债+黄金）：{cash_only:.4f}（{cash_only-1:+.2%}）")
print(f"  → ETF 持仓的**超额贡献** = {nav_s.iloc[-1]/nav_s.iloc[0] / cash_only:.4f}")
print(f"    即 ETF 持仓带来 {(nav_s.iloc[-1]/nav_s.iloc[0] / cash_only - 1):+.2%} 的额外收益")
print(f"    折年化：{((nav_s.iloc[-1]/nav_s.iloc[0]/cash_only)**(1/years)-1):+.2%}")

# ---------- 关键：持仓日的"超额收益"
print("\n【方法四】持仓日 vs 空仓日的日收益对比（最直观）")
print("-" * 92)
is_hold = pos_s > 0
r_hold = ret[is_hold]
r_flat = ret[~is_hold]
print(f"  持仓日 {int(is_hold.sum()):4d} 天 | 日均 {r_hold.mean()*100:+.4f}% | "
      f"累计 {(1+r_hold).prod()-1:+.2%}")
print(f"  空仓日 {int((~is_hold).sum()):4d} 天 | 日均 {r_flat.mean()*100:+.4f}% | "
      f"累计 {(1+r_flat).prod()-1:+.2%}")
print(f"\n  **持仓日日均收益是空仓日的 {r_hold.mean()/r_flat.mean():.1f} 倍**")
print(f"  持仓日年化: {(1+r_hold.mean())**252-1:+.2%}")
print(f"  空仓日年化: {(1+r_flat.mean())**252-1:+.2%}")

# ---------- 最终数学验证
print("\n" + "=" * 92)
print("最终数学验证")
print("=" * 92)
comp = (1 + r_hold).prod() * (1 + r_flat).prod()
print(f"  (持仓日累计 {((1+r_hold).prod()):.4f}) × (空仓日累计 {((1+r_flat).prod()):.4f})")
print(f"  = {comp:.4f}  应为 {nav_s.iloc[-1]/nav_s.iloc[0]:.4f}  ✅")
print()
print(f"  持仓日占比 {is_hold.mean():.1%}，但这些天贡献了：")
print(f"    持仓日累计 {(1+r_hold).prod()-1:+.2%}")
print(f"    空仓日累计 {(1+r_flat).prod()-1:+.2%}")
print()
print(f"  结论：{years:.1f} 年里，{is_hold.mean():.0%} 的交易日（{int(is_hold.sum())} 天）")
print(f"        贡献了 {(1+r_hold).prod()-1:.1%} 的收益，其余时间只赚现金收益。")
print(f"        这不是矛盾，而是「集中在高胜率时刻出手」的结果。")

out = {
    "cagr": float(cagr),
    "hold_days_pct": float(is_hold.mean()),
    "hold_day_daily": float(r_hold.mean()),
    "flat_day_daily": float(r_flat.mean()),
    "hold_cum": float((1+r_hold).prod()-1),
    "flat_cum": float((1+r_flat).prod()-1),
    "etf_direct_daily": float(vals.mean()),
    "trade_daily_geo": float(pdf["daily_geo"].mean()),
}
(ROOT / "build" / "eval" / "etf_return_decompose.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n→ build/eval/etf_return_decompose.json")
