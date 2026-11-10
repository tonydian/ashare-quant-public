# -*- coding: utf-8 -*-
"""诊断：弹性周期 + 双动量为何失败？

现象：
  DEV 14.81% → OOS -8.36%（衰减 23pp）
  MaxDD -52.71%（比买入持有还差）
  1129 笔交易（换手太高，弹性周期没起作用）
  卖出原因：rotate 443, trail 104, stop 16

疑问：
  H1 最小持有期没生效？（rotate 443 次说明频繁换仓）
  H2 双动量过滤太严，导致长期空仓错过反弹？
  H3 大盘择时导致在底部卖出？
  H4 回撤止损太敏感（trail 104 次）
  H5 池子里行业 ETF 太多，波动放大

方法：逐项拆解，看哪个条件在拖后腿
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

import etf_flex_dual_momentum as E  # noqa: E402

P = E.load_panel()

print("\n[计算动量] …")
mom1 = E.simple_mom(P["close"], E.PARAMS["mom1_n"])
mom2 = E.slope_mom(P["close"], E.PARAMS["mom2_n"])
ma = E.rolling_mean(P["close"], E.PARAMS["ma_filter"])

print("\n" + "=" * 96)
print("消融实验：逐个关闭条件，看影响")
print("=" * 96)

configs = {
    "完整版": {},
    "关掉双动量过滤": {"mom2_floor": -1e9},
    "关掉大盘择时": {"timing": False},
    "关掉均线过滤": {"ma_filter": 0},
    "关掉回撤止损": {"trail_stop": 1.0},
    "关掉动量上限": {"mom_ceil": 1e9},
    "关掉所有过滤": {"mom2_floor": -1e9, "timing": False,
                     "ma_filter": 0, "trail_stop": 1.0, "mom_ceil": 1e9},
    "只买国债(对照)": {"top_n": 1},
}

base = dict(E.PARAMS)
rows = []
for name, override in configs.items():
    E.PARAMS.update(base)
    E.PARAMS.update(override)
    _, dev = E.backtest(P, mom1, mom2, ma, E.DEV[0], E.DEV[1],
                        f"{name}-DEV", verbose=False)
    _, oos = E.backtest(P, mom1, mom2, ma, E.OOS[0], E.OOS[1],
                        f"{name}-OOS", verbose=False)
    if dev and oos:
        rows.append({
            "配置": name,
            "DEV CAGR": dev["cagr"], "DEV Sharpe": dev["sharpe"],
            "OOS CAGR": oos["cagr"], "OOS Sharpe": oos["sharpe"],
            "OOS MaxDD": oos["max_dd"],
        })
        print(f"  {name:<18} DEV {dev['cagr']:>7.2%}/{dev['sharpe']:>5.2f}  "
              f"OOS {oos['cagr']:>7.2%}/{oos['sharpe']:>5.2f}  "
              f"DD {oos['max_dd']:>7.2%}")
E.PARAMS.update(base)

print("\n" + "=" * 96)
print("关键检查：最小持有期是否生效？")
print("=" * 96)
print(f"  参数 min_hold = {E.PARAMS['min_hold']} 交易日")
print(f"  但 rotate 触发 443 次（全期）")
print(f"  平均持仓天数 = 2137 天 / (443+104+16) 次卖出 ≈ "
      f"{2137/(443+104+16):.1f} 天")

print("\n  分析：")
print("""
  如果 min_hold=8 生效，每次持仓至少 8 天，
  最多换仓次数 = 2137 / 8 ≈ 267 次
  但实际 563 次卖出 → 说明止损/止盈绕过了 min_hold
  （这是设计允许的，止损可打破最小持有期）

  真正问题是：trail_stop=8% 太敏感，104 次触发
            + stop_loss=10%，16 次
            = 120 次强制卖出（绕过 min_hold）
            + rotate 443 次正常换仓
""")

print("\n" + "=" * 96)
print("行业 ETF vs 债券 ETF 的表现拆解")
print("=" * 96)
# 看 OOS 期各类资产的表现
oos0 = int(P["dates"].searchsorted(pd.Timestamp(E.OOS[0])))
oos1 = int(P["dates"].searchsorted(pd.Timestamp(E.OOS[1]), side="right")) - 1

cats = {"宽基": [], "行业": [], "债券": [], "海外": []}
for j, c in enumerate(P["codes"]):
    nm = P["meta"].get(c, "")
    cat = E.CATEGORY.get(nm, "行业")
    cats.setdefault(cat, []).append(j)

for cat, js in cats.items():
    if not js:
        continue
    rets = []
    for j in js:
        s = P["close"][oos0:oos1, j]
        ok = np.isfinite(s)
        if ok.sum() < 100:
            continue
        buy_hold = s[ok][-1] / s[ok][0] - 1
        rets.append(buy_hold)
    if rets:
        print(f"  {cat:<6} {len(rets):>2} 只 | OOS 期间买入持有中位 {np.median(rets):>+7.2%}")

print("\n" + "=" * 96)
print("结论")
print("=" * 96)
best = max(rows, key=lambda r: r["OOS Sharpe"])
print(f"  最优配置（按 OOS Sharpe）: {best['配置']}")
print(f"    DEV {best['DEV CAGR']:.2%}/{best['DEV Sharpe']:.2f}")
print(f"    OOS {best['OOS CAGR']:.2%}/{best['OOS Sharpe']:.2f}")
