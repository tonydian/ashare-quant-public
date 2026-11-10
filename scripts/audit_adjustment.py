# -*- coding: utf-8 -*-
"""核查：我们的复权处理是否符合文章标准？

文章核心结论（《复权对 ETF 轮动策略的影响》）：
  1. 动量因子计算 → 必须用【等比复权价】
  2. 调仓成交价     → 用【真实原始价】
  3. 持仓净值计算   → 用【动态复权】（分红入账、送股增持）

关键问题：
  · 我们的 ETF 数据是 qfq（前复权）—— 是"普通前复权"还是"等比前复权"？
  · 主策略的个股数据是"后复权"—— 是普通还是等比？
  · 我们的净值计算有没有正确处理分红？

普通复权 vs 等比复权的区别：
  普通前复权：P_adj = P_raw - 分红  （减法，涨幅失真，可能为负）
  等比前复权：P_adj = P_raw × factor （乘法，涨幅准确）
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent

print("=" * 96)
print("复权处理核查")
print("=" * 96)

# ---------- 1) ETF 数据
print("\n【1】ETF 数据（腾讯 qfq）")
z = np.load(ROOT / "build" / "etf" / "etf_ohlc_panel.npz", allow_pickle=True)
close = z["close"].astype(np.float64)
codes = [str(c) for c in z["codes"]]
dates = pd.DatetimeIndex(z["dates"])

# 检查是否有负价或极低价（普通前复权的典型症状）
print(f"  价格范围: {np.nanmin(close):.4f} ~ {np.nanmax(close):.4f}")
print(f"  负价数量: {(close < 0).sum()}")
print(f"  极低价(<0.01)数量: {(close < 0.01).sum()}")

# 检查分红率高的 ETF（红利 510880）—— 普通前复权最容易出问题
for target in ("510880", "512690", "159928"):
    j = codes.index(target) if target in codes else None
    if j is None:
        continue
    s = close[:, j]
    ok = np.isfinite(s)
    print(f"\n  {target} (分红高): 价格 {np.nanmin(s):.3f} ~ {np.nanmax(s):.3f}")
    print(f"    首值 {s[ok][0]:.4f} → 末值 {s[ok][-1]:.4f}")
    # 检查是否有异常跳空（超过 ±20% 单日）
    r = pd.Series(s).pct_change()
    big = r[(r.abs() > 0.20) & r.notna()]
    if len(big):
        print(f"    单日 |涨跌|>20% 的天数: {len(big)}（可能是除权跳空）")
        for idx in big.index[:3]:
            print(f"      {dates[idx].date()}: {r[idx]*100:+.2f}%")
    else:
        print(f"    无异常跳空 ✅")

# ---------- 2) 主策略个股数据
print("\n【2】主策略个股数据（panel_ohlcv.npz）")
import os
os.environ.pop("QF_DATA_RAW", None)
import sys
sys.path.insert(0, str(ROOT))
from qf import config as C, data as qfdata

md = qfdata.load_market(verbose=False)
print(f"  形状: {md.price.shape}")
print(f"  价格范围: {np.nanmin(md.price):.4f} ~ {np.nanmax(md.price):.2f}")
print(f"  负价: {(md.price < 0).sum()}")

# 检查高分红股
for code in ("600153", "601398", "600900"):
    try:
        j = list(md.codes).index(code)
    except ValueError:
        continue
    s = md.price[:, j]
    ok = np.isfinite(s)
    r = pd.Series(s).pct_change()
    big = r[(r.abs() > 0.25) & r.notna()]
    print(f"\n  {code}: {np.nanmin(s):.3f} ~ {np.nanmax(s):.2f}")
    print(f"    单日 |涨跌|>25%: {len(big)} 天")
    if len(big):
        for idx in big.index[:3]:
            print(f"      {md.dates[idx].date()}: {r[idx]*100:+.2f}%")

# ---------- 3) 净值计算方式
print("\n【3】净值计算方式（是否用真实价 + 动态复权？）")
bt = (ROOT / "qf" / "backtest.py").read_text(encoding="utf-8")
checks = {
    "用 md.price（复权价）算净值": "md.price[t" in bt or "price[t, j]" in bt,
    "有分红入账逻辑": "dividend" in bt.lower(),
    "有送股/拆股处理": "split" in bt.lower() or "share_change" in bt.lower(),
}
for k, v in checks.items():
    print(f"  {'✅' if v else '❌'} {k}")

# 详细看净值计算
i = bt.find("nav[")
print(f"\n  净值计算代码片段:")
seg = bt[max(0, i-300):i+400]
for ln in seg.splitlines()[:15]:
    if ln.strip():
        print(f"    {ln.strip()[:100]}")

# ---------- 4) 结论
print("\n" + "=" * 96)
print("结论")
print("=" * 96)
print(f"""
  文章的三个标准：
    ① 动量因子 → 等比复权价
    ② 成交价   → 真实原始价
    ③ 净值     → 动态复权（分红入账）

  我们的情况：
    · ETF 数据：腾讯 qfq（前复权）—— 腾讯的 qfq 是"等比前复权"还是"普通前复权"？
      需要验证：如果除权日附近涨幅连续，说明是等比；如果跳空，是普通。
    · 个股数据：panel_ohlcv.npz 的 close 是"后复权"
      后复权通常是等比（按比例调整），需要确认。
    · 净值计算：见上

  ⚠️ 最关键的问题：
     如果我们的净值计算直接用复权价算涨跌，而**没有单独处理分红现金**，
     那净值是对的（因为复权价已含分红），但**成交价与实盘不符**。
     
     文章说：动量用复权价、成交用真实价、净值用动态复权。
     我们如果全用复权价 → 净值正确，但成交价偏乐观（因为复权价可能高于真实价）。
""")
