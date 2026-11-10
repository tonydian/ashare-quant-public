# -*- coding: utf-8 -*-
"""验证：RSI<50 买入为什么是错的？

数据（OOS 期）：
  有 RSI：-6.08%
  无 RSI：+4.38%
  RSI 反向（>50 买）：+4.18%

差异 ~10pp。需要搞清楚：
  H1 RSI<50 = 下跌趋势中买入 → 接飞刀
  H2 RSI>70 = 卖出强势股 → 卖飞
  H3 两者都有

方法：分别测"只改买入规则"和"只改卖出规则"
  · 原规则：RSI<50 买，RSI>70 卖
  · 只改买：RSI 任意买，RSI>70 卖
  · 只改卖：RSI<50 买，不卖（只换仓）
  · 都不改：任意买，不卖

同时修基准数据 bug（bench 指数有 NaN）
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fund_rsi_strategy as S  # noqa: E402
from qf import config as C, metrics as qfm  # noqa: E402

md, dates, code_idx, fund = S.load_all()
rsi = S.rsi_panel(md.price, S.PARAMS["rsi_n"])

base = dict(S.PARAMS)

print("=" * 96)
print("RSI 买/卖规则分解")
print("=" * 96)
print(f"{'买入条件':<20}{'卖出条件':<18}{'DEV':>10}{'OOS':>10}{'OOS Sharpe':>12}")
print("-" * 72)

cases = [
    ("RSI<50（原）", 50, "RSI>70（原）", 70),
    ("RSI<50", 50, "无卖出", 999),
    ("不限制", 999, "RSI>70", 70),
    ("不限制", 999, "无卖出", 999),
    ("RSI>50（反向）", -1, "RSI>70", 70),
    ("RSI>50（反向）", -1, "无卖出", 999),
]
for bl, bv, sl, sv in cases:
    S.PARAMS.update(base)
    S.PARAMS["rsi_buy"] = bv
    S.PARAMS["rsi_sell"] = sv
    _, dv, _ = S.run(md, dates, code_idx, fund, rsi,
                     S.DEV[0], S.DEV[1], "d", verbose=False)
    _, ov, _ = S.run(md, dates, code_idx, fund, rsi,
                     "2022-01-01", "2026-10-31", "o", verbose=False)
    if dv and ov:
        print(f"{bl:<20}{sl:<18}{dv['cagr']:>9.2%}{ov['cagr']:>10.2%}"
              f"{ov['sharpe']:>12.3f}")
S.PARAMS.update(base)

print("\n" + "=" * 96)
print("RSI 分布：命中池内的股票 RSI 是多少？")
print("=" * 96)
# 抽样看池内股票的 RSI 分布
samples = []
t0 = int(dates.searchsorted(pd.Timestamp("2022-01-01")))
t1 = int(dates.searchsorted(pd.Timestamp("2026-10-31")))
fom = (pd.Series(dates).dt.month != pd.Series(dates).dt.month.shift(1)).to_numpy()
for t in np.flatnonzero(fom):
    if not (t0 <= t <= t1):
        continue
    avail = [(p, v["usable"]) for p, v in fund.items() if v["usable"] <= dates[t]]
    if not avail:
        continue
    pid = max(avail, key=lambda x: x[1])[0]
    pool = S.fundamental_mask(md, code_idx, fund, t, pid)
    if len(pool):
        rr = rsi[t][pool["j"].astype(int).to_numpy()]
        rr = rr[np.isfinite(rr)]
        if len(rr):
            samples.append(rr)

if samples:
    allr = np.concatenate(samples)
    print(f"  样本 {len(allr)} 个（池内股票的 RSI14）")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"    {q:>3}% 分位: {np.percentile(allr, q):>6.1f}")
    print(f"\n  RSI < 50 的比例: {(allr < 50).mean()*100:.1f}%")
    print(f"  RSI > 70 的比例: {(allr > 70).mean()*100:.1f}%")
    print(f"\n  → 说明：如果一半以上时间 RSI>50，那'RSI<50 才买'会错过大部分机会")

print("\n" + "=" * 96)
print("修正基准：指数原始数据检查")
print("=" * 96)
for key in ("hs300", "csi500", "csi1000", "csi_all"):
    b = md.bench.get(key)
    if b is None:
        print(f"  {key}: 缺失")
        continue
    ok = np.isfinite(b.to_numpy())
    print(f"  {key}: {len(b)} 行, 有效 {ok.sum()}, "
          f"NaN {len(b)-ok.sum()}, 首 {b.index[0].date()}, "
          f"末 {b.index[-1].date()}")
    # 检查是否有异常值
    v = b[ok]
    if len(v) > 1:
        r = v.pct_change().dropna()
        big = r[r.abs() > 0.2]
        if len(big):
            print(f"    异常日收益 >20%: {len(big)} 次")
