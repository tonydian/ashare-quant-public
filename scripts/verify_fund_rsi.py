# -*- coding: utf-8 -*-
"""稳健性检验：6.01% 是真信号还是运气？

必须检查：
  T1 参数敏感性（ROE/净利/市现率/持仓数）
  T2 逐年稳定性
  T3 滚动窗口
  T4 与"纯小市值"对照（剥离基本面是否还有增量）
  T5 与基准对比
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

md, dates, code_idx, fund = S.load_all()
rsi = S.rsi_panel(md.price, S.PARAMS["rsi_n"])
base = dict(S.PARAMS)

DEV = S.DEV
OOS = ("2022-01-01", "2026-10-31")


def go(**ov):
    S.PARAMS.update(base)
    S.PARAMS.update(ov)
    _, d, _ = S.run(md, dates, code_idx, fund, rsi, DEV[0], DEV[1], "d",
                    verbose=False)
    _, o, _ = S.run(md, dates, code_idx, fund, rsi, OOS[0], OOS[1], "o",
                    verbose=False)
    S.PARAMS.update(base)
    return d, o


print("=" * 96)
print("T1 参数敏感性")
print("=" * 96)

print("\n【ROE 门槛】")
print(f"{'ROE>':>8}{'DEV':>10}{'OOS':>10}{'OOS Sharpe':>12}")
print("-" * 42)
roe_sweep = {}
for v in (10, 12, 15, 18, 20, 25):
    d, o = go(roe_min=v)
    if d and o:
        roe_sweep[v] = (d["cagr"], o["cagr"])
        print(f"{v:>7}%{d['cagr']:>10.2%}{o['cagr']:>10.2%}{o['sharpe']:>12.3f}")

print("\n【净利增速门槛】")
print(f"{'增速>':>8}{'DEV':>10}{'OOS':>10}{'OOS Sharpe':>12}")
print("-" * 42)
yoy_sweep = {}
for v in (0, 10, 20, 30, 50):
    d, o = go(yoy_min=v)
    if d and o:
        yoy_sweep[v] = (d["cagr"], o["cagr"])
        print(f"{v:>7}%{d['cagr']:>10.2%}{o['cagr']:>10.2%}{o['sharpe']:>12.3f}")

print("\n【市现率门槛】")
print(f"{'市现率<':>9}{'DEV':>10}{'OOS':>10}{'OOS Sharpe':>12}")
print("-" * 43)
pcf_sweep = {}
for v in (10, 20, 30, 50, 100):
    d, o = go(pcf_max=v)
    if d and o:
        pcf_sweep[v] = (d["cagr"], o["cagr"])
        print(f"{v:>8}{d['cagr']:>10.2%}{o['cagr']:>10.2%}{o['sharpe']:>12.3f}")

print("\n【持仓数】")
print(f"{'持仓':>6}{'DEV':>10}{'OOS':>10}{'OOS Sharpe':>12}")
print("-" * 40)
n_sweep = {}
for v in (5, 10, 15, 20):
    d, o = go(n_hold=v)
    if d and o:
        n_sweep[v] = (d["cagr"], o["cagr"])
        print(f"{v:>6}{d['cagr']:>10.2%}{o['cagr']:>10.2%}{o['sharpe']:>12.3f}")

print("\n【小市值池大小】")
print(f"{'池大小':>7}{'DEV':>10}{'OOS':>10}{'OOS Sharpe':>12}")
print("-" * 41)
pool_sweep = {}
for v in (20, 50, 100, 9999):
    d, o = go(pool_top=v)
    if d and o:
        pool_sweep[v] = (d["cagr"], o["cagr"])
        lbl = str(v) if v < 9999 else "全市场"
        print(f"{lbl:>7}{d['cagr']:>10.2%}{o['cagr']:>10.2%}{o['sharpe']:>12.3f}")

print("\n" + "=" * 96)
print("T2 逐年表现")
print("=" * 96)
print(f"{'年份':>6}{'策略':>10}{'沪深300':>10}{'超额':>10}")
print("-" * 38)


def idx_year(key, y):
    b = md.bench.get(key)
    rr = b[(b.index >= pd.Timestamp(f"{y}-01-01"))
           & (b.index <= pd.Timestamp(f"{y}-12-31"))]
    if len(rr) < 50:
        return np.nan
    nav = (1 + rr.fillna(0)).cumprod()
    return float(nav.iloc[-1] - 1)


for y in range(2018, 2027):
    S.PARAMS.update(base)
    _, r, _ = S.run(md, dates, code_idx, fund, rsi,
                    f"{y}-01-01", f"{y}-12-31", str(y), verbose=False)
    bi = idx_year("hs300", y)
    if r and np.isfinite(bi):
        print(f"{y:>6}{r['cagr']:>9.2%}{bi:>10.2%}"
              f"{(r['cagr']-bi)*100:>+9.2f}pp")
S.PARAMS.update(base)

print("\n" + "=" * 96)
print("T3 滚动窗口")
print("=" * 96)
print(f"{'起点':>8}{'策略':>10}{'沪深300':>10}{'超额':>10}")
print("-" * 38)
win = 0
tot = 0
for sy in (2018, 2019, 2020, 2021, 2022):
    S.PARAMS.update(base)
    _, r, _ = S.run(md, dates, code_idx, fund, rsi,
                    f"{sy}-01-01", "2026-10-31", "w", verbose=False)
    b = md.bench.get("hs300")
    rr = b[(b.index >= pd.Timestamp(f"{sy}-01-01"))
           & (b.index <= pd.Timestamp("2026-10-31"))]
    if not r or len(rr) < 100:
        continue
    nav = (1 + rr.fillna(0)).cumprod()
    bc = float(nav.iloc[-1] ** (252 / len(rr)) - 1)
    d = (r["cagr"] - bc) * 100
    tot += 1
    if d > 0:
        win += 1
    print(f"{sy:>8}{r['cagr']:>9.2%}{bc:>10.2%}{d:>+9.2f}pp")
S.PARAMS.update(base)
print(f"\n  跑赢基准: {win}/{tot}")

print("\n" + "=" * 96)
print("T4 剥离测试：纯小市值 vs 基本面筛选")
print("=" * 96)
# 纯小市值（不做基本面，只取最小市值前 10）
S.PARAMS.update(base)
S.PARAMS["pb_range"] = (0, 999)
S.PARAMS["pcf_max"] = 1e9
S.PARAMS["roe_min"] = -1e9
S.PARAMS["gpm_min"] = -1e9
S.PARAMS["yoy_min"] = -1e9
S.PARAMS["pool_top"] = 10
_, d_pure, _ = S.run(md, dates, code_idx, fund, rsi, DEV[0], DEV[1], "d",
                     verbose=False)
_, o_pure, _ = S.run(md, dates, code_idx, fund, rsi, OOS[0], OOS[1], "o",
                     verbose=False)
S.PARAMS.update(base)
if d_pure and o_pure:
    print(f"  纯小市值前10: DEV {d_pure['cagr']:.2%} / OOS {o_pure['cagr']:.2%}")
_, d_full, _ = S.run(md, dates, code_idx, fund, rsi, DEV[0], DEV[1], "d",
                     verbose=False)
_, o_full, _ = S.run(md, dates, code_idx, fund, rsi, OOS[0], OOS[1], "o",
                     verbose=False)
if d_full and o_full:
    print(f"  基本面+小市值: DEV {d_full['cagr']:.2%} / OOS {o_full['cagr']:.2%}")
    if d_pure and o_pure:
        print(f"  → 基本面增量: DEV {(d_full['cagr']-d_pure['cagr'])*100:+.2f}pp, "
              f"OOS {(o_full['cagr']-o_pure['cagr'])*100:+.2f}pp")

# 保存
out = {"roe": roe_sweep, "yoy": yoy_sweep, "pcf": pcf_sweep,
       "hold": n_sweep, "pool": pool_sweep}
(ROOT / "build" / "eval").mkdir(parents=True, exist_ok=True)
(ROOT / "build" / "eval" / "fund_rsi_robustness.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1, default=float),
    encoding="utf-8")
print(f"\n→ build/eval/fund_rsi_robustness.json")
