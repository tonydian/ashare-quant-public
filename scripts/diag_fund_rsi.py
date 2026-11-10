# -*- coding: utf-8 -*-
"""诊断：稳健版为何 DEV 13.47% 但 OOS -6.08%？

可能原因：
  H1 RSI 择时在中长期是反向指标（买了下跌的、卖了上涨的）
  H2 小市值因子在 2022-2024 失效（小盘股连续下跌）
  H3 基本面筛选在 OOS 期选到的都是"价值陷阱"
  H4 DEV 的好成绩来自 2019-2020 的特定行情

关键检验：
  · 分年度表现
  · RSI 择时的贡献（关掉 RSI 会怎样）
  · 小市值因子的表现
  · 持仓股的画像（OOS 期买的是什么）
"""
from __future__ import annotations

import json
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

print("=" * 96)
print("加载数据")
print("=" * 96)
md, dates, code_idx, fund = S.load_all()
rsi = S.rsi_panel(md.price, S.PARAMS["rsi_n"])

# ---------------------------------------------------------------- 1) 分年度
print("\n" + "=" * 96)
print("1) 分年度表现（逐年跑）")
print("=" * 96)
print(f"{'年份':>6}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>10}{'交易':>7}")
print("-" * 44)
yearly = {}
for y in range(2018, 2027):
    nav, res, dg = S.run(md, dates, code_idx, fund, rsi,
                         f"{y}-01-01", f"{y}-12-31", f"{y}", verbose=False)
    if res:
        yearly[y] = res
        print(f"{y:>6}{res['cagr']:>9.2%}{res['sharpe']:>9.3f}"
              f"{res['max_dd']:>9.2%}")

# ---------------------------------------------------------------- 2) RSI 的作用
print("\n" + "=" * 96)
print("2) RSI 择时的贡献（对照实验）")
print("=" * 96)
base = dict(S.PARAMS)

configs = [
    ("有 RSI（当前）", dict(rsi_buy=50, rsi_sell=70)),
    ("不设 RSI 门槛", dict(rsi_buy=101, rsi_sell=101)),
    ("RSI 放宽买 70/卖 90", dict(rsi_buy=70, rsi_sell=90)),
    ("RSI 反向（>50 买）", dict(rsi_buy=-1, rsi_sell=30)),
]
print(f"{'配置':<22}{'DEV CAGR':>11}{'OOS CAGR':>11}{'OOS Sharpe':>12}")
print("-" * 60)
for label, ov in configs:
    S.PARAMS.update(base)
    S.PARAMS.update(ov)
    _, dv, _ = S.run(md, dates, code_idx, fund, rsi,
                     S.DEV[0], S.DEV[1], "d", verbose=False)
    _, ov_, _ = S.run(md, dates, code_idx, fund, rsi,
                      "2022-01-01", "2026-10-31", "o", verbose=False)
    if dv and ov_:
        print(f"{label:<22}{dv['cagr']:>10.2%}{ov_['cagr']:>11.2%}"
              f"{ov_['sharpe']:>12.3f}")
S.PARAMS.update(base)

# ---------------------------------------------------------------- 3) 小市值的作用
print("\n" + "=" * 96)
print("3) 小市值因子的影响")
print("=" * 96)
S.PARAMS.update(base)
S.PARAMS["pool_top"] = 0     # 0 = 不限制，取全部合格
_, dv_all, _ = S.run(md, dates, code_idx, fund, rsi,
                     S.DEV[0], S.DEV[1], "d", verbose=False)
_, ov_all, _ = S.run(md, dates, code_idx, fund, rsi,
                     "2022-01-01", "2026-10-31", "o", verbose=False)
print(f"  小市值前 50（当前）: DEV {base and ''}"
      f"{yearly.get(2019, {}).get('cagr', 0):.0%}...")
S.PARAMS.update(base)
print(f"  限制前 50: 需要与不限制对比")
S.PARAMS["pool_top"] = 9999
_, dv_n, _ = S.run(md, dates, code_idx, fund, rsi,
                   S.DEV[0], S.DEV[1], "d", verbose=False)
_, ov_n, _ = S.run(md, dates, code_idx, fund, rsi,
                   "2022-01-01", "2026-10-31", "o", verbose=False)
if dv_n and ov_n:
    print(f"  不限制（全部合格）: DEV {dv_n['cagr']:.2%} / OOS {ov_n['cagr']:.2%}")
S.PARAMS.update(base)

# ---------------------------------------------------------------- 4) 全市场基准
print("\n" + "=" * 96)
print("4) 基准（指数）")
print("=" * 96)


def idx_perf(key, a, b):
    s = md.bench.get(key)
    if s is None:
        return None
    s = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
    if len(s) < 100:
        return None
    return qfm.performance(s, rf=C.CASH_YIELD)


for key, name in (("hs300", "沪深300"), ("csi500", "中证500"),
                  ("csi1000", "中证1000"), ("csi_all", "中证全指")):
    for per, (a, b) in (("DEV", S.DEV), ("OOS", ("2022-01-01", "2026-10-31"))):
        p = idx_perf(key, a, b)
        if p:
            print(f"  {name:<8}{per:<5} CAGR {p['cagr']:8.2%} "
                  f"Sharpe {p['sharpe']:6.3f} MaxDD {p['max_dd']:8.2%}")

# ---------------------------------------------------------------- 5) 持仓画像
print("\n" + "=" * 96)
print("5) OOS 期持仓画像（买的是什么）")
print("=" * 96)
pool_log = []
for t in np.flatnonzero((pd.Series(dates).dt.month !=
                         pd.Series(dates).dt.month.shift(1)).to_numpy()):
    if not (int(dates.searchsorted(pd.Timestamp("2022-01-01"))) <= t
            <= int(dates.searchsorted(pd.Timestamp("2026-10-31")))):
        continue
    avail = [(p, v["usable"]) for p, v in fund.items()
             if v["usable"] <= dates[t]]
    if not avail:
        continue
    pid = max(avail, key=lambda x: x[1])[0]
    pool = S.fundamental_mask(md, code_idx, fund, t, pid)
    if len(pool):
        pool["date"] = dates[t]
        pool_log.append(pool)

if pool_log:
    P = pd.concat(pool_log, ignore_index=True)
    print(f"  样本: {len(P)} 个（股票-月），{P['code'].nunique()} 只不同股票")
    print(f"\n  {'分位':>6}{'PB':>8}{'市现率':>9}{'ROE':>8}{'毛利':>8}{'市值(亿)':>10}")
    print("  " + "-" * 52)
    for q in (10, 25, 50, 75, 90):
        print(f"  {q:>5}%{np.nanpercentile(P['pb'].dropna(), q):>8.1f}"
              f"{np.nanpercentile(P['pcf'].dropna(), q):>9.1f}"
              f"{np.nanpercentile(P['roe'].dropna(), q):>8.1f}"
              f"{np.nanpercentile(P['gpm'].dropna(), q):>8.1f}"
              f"{np.nanpercentile(P['mcap_yi'].dropna(), q):>10.1f}")

    print(f"\n  出现最多的 10 只:")
    for (code, name), n in P.groupby(["code", "name"]).size() \
            .sort_values(ascending=False).head(10).items():
        print(f"    {code} {str(name)[:10]:<12} {n:>3} 次")

print("\n" + "=" * 96)
print("结论")
print("=" * 96)
print("""
  待填（由脚本上方结果决定）
""")
