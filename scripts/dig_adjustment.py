# -*- coding: utf-8 -*-
"""深挖：复权因子为何下降？这对回测有何影响？

现象：
  · 复权因子 = md.price / raw_close
  · 600153: 0.6182 → 1.0545，但中间有 1166 次下降
  · 分红事件对照：60 个里只有 14 个完全一致

可能的解释：
  H1 raw_close 本身也是复权价（不是真原始价）→ 因子无意义
  H2 md.price 是"等比后复权"，raw_close 是"等比前复权" → 两者比值会变
  H3 数据源混用（不同来源拼接）

关键：判断哪一个价格权威

方法：
  1. 检查 raw_close 的定义和来源
  2. 用"分红事件的连续性"判断哪个是等比复权
  3. 看 update_data.py 的注释
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
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata  # noqa: E402

print("=" * 96)
print("复权因子的来源与影响")
print("=" * 96)

# ---------- 1) 看 update_data.py 怎么定义 raw_close
up = (ROOT / "scripts" / "update_data.py").read_text(encoding="utf-8")
print("\n【1】update_data.py 中 raw_close 的定义")
for ln in up.splitlines():
    low = ln.lower()
    if "raw_close" in low or "不复权" in ln or "缩放价" in ln:
        print(f"  {ln.strip()[:110]}")

# ---------- 2) 原始数据包的 README / 元信息
print("\n【2】原始数据包说明")
for fn in ("UPDATE_INFO.txt", "meta.csv"):
    p = C.DATA_RAW / fn
    if p.exists():
        if fn.endswith(".txt"):
            print(p.read_text(encoding="utf-8", errors="replace")[:600])
        break

# ---------- 3) 关键判断：哪个价格在做动量/净值？
print("\n【3】回测实际用哪个价格？")
bt = (ROOT / "qf" / "backtest.py").read_text(encoding="utf-8")
# 找 md.price 的赋值来源
for ln in bt.splitlines():
    if "md.price" in ln and ("=" in ln or "nav" in ln or "shares" in ln):
        print(f"  {ln.strip()[:110]}")

# ---------- 4) 数学验证：raw_close 是什么
print("\n【4】数学验证：raw_close 的性质")
md = qfdata.load_market(verbose=False)
raw = np.load(C.DATA_RAW / "raw_close.npz")
key = "raw_close" if "raw_close" in raw.files else raw.files[0]
raw = raw[key].astype(np.float64)
price = md.price
dates = pd.DatetimeIndex(md.dates)
code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}

# 取一只股票看两个序列的日收益差异
for code in ("600153", "601398"):
    j = code_idx.get(code)
    if j is None:
        continue
    p = pd.Series(price[:, j]).pct_change()
    r = pd.Series(raw[:, j]).pct_change()
    both = pd.DataFrame({"adj": p, "raw": r}).dropna()
    diff = (both["adj"] - both["raw"]).abs()
    print(f"\n  {code}:")
    print(f"    adj 收益 std: {both['adj'].std()*100:.3f}%")
    print(f"    raw 收益 std: {both['raw'].std()*100:.3f}%")
    print(f"    两者日收益差异 >0.5% 的天数: {(diff > 0.005).sum()} / {len(diff)}")
    # 差异大的日期
    big = diff[diff > 0.005]
    if len(big):
        print(f"    最大差异: {diff.max()*100:.2f}%")
        for idx in big.index[:3]:
            print(f"      {dates[idx].date()}: adj {both['adj'][idx]*100:+.2f}% "
                  f"vs raw {both['raw'][idx]*100:+.2f}%")

# ---------- 5) 结论
print("\n" + "=" * 96)
print("判断")
print("=" * 96)
print("""
  raw_close 在 update_data.py 里被描述为"不复权收盘（用缩放价近似）"。
  
  如果 raw_close 是"用缩放价近似"的，那它本身就不是真正的不复权价，
  而是某种近似 —— 所以 md.price / raw_close 的比值不代表真实复权因子。

  关键问题不是这个比值，而是：**md.price 本身是否等比复权？**

  判断方法（更直接）：
    如果 md.price 是等比后复权，那它相邻两日的涨幅
    应该等于"真实持有收益"（含分红），且**不会出现分红日的假跳空**。
    
    但我们的表里，很多分红日的复权涨幅与"原始涨幅+分红"不一致
    → 这可能是因为 raw_close 不准确，而不是 md.price 有问题。
""")
