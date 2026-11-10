# -*- coding: utf-8 -*-
"""按"次方量化"文章的滑点标尺，重算各策略成本。

文章给出的参考区间（§5.1）：
  主流宽基ETF   0.5‰ - 1‰    =  5 ~ 10 bp
  行业/主题ETF  1‰ - 2‰      = 10 ~ 20 bp
  小众ETF       2‰ - 5‰      = 20 ~ 50 bp
  大盘蓝筹股     1‰ - 2‰      = 10 ~ 20 bp
  中小市值股票   2‰ - 5‰      = 20 ~ 50 bp

按交易方式（§5.2）：
  市价单        1.5‰ - 3‰    = 15 ~ 30 bp
  限价单        0.5‰ - 1‰    =  5 ~ 10 bp

对照之前我的估算：
  统一用 40bp（4‰）—— 对小盘股是合理的，对 ETF 明显高估
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "slippage_calibrated.json"
CAP = 300000.0
START = "2020-01-08"

# 文章标尺（bp，单边）
SLIP_TABLE = {
    "宽基ETF": (5, 10, 20),          # (乐观, 中性, 保守)
    "行业ETF": (10, 15, 20),
    "小盘股": (20, 35, 50),
    "混合(按持仓)": (25, 40, 50),     # 之前用的 40bp 落在保守端
}

print("=" * 96)
print("按文章标尺重算（滑点单位 bp，单边）")
print("=" * 96)
print(f"{'标的类型':<16}{'乐观':>8}{'中性':>8}{'保守':>8}")
print("-" * 44)
for k, (a, b, c) in SLIP_TABLE.items():
    print(f"{k:<16}{a:>8}{b:>8}{c:>8}")


def comm_bp(per_yuan: float) -> float:
    return max(C.MIN_COMMISSION, per_yuan * C.COMMISSION_RATE) / per_yuan * 1e4


def analyze(label, csv_rel, nav_rel, n_hold, slip_bp, asset="股票"):
    cp, npth = ROOT / csv_rel, ROOT / nav_rel
    if not (cp.exists() and npth.exists()):
        return None
    tr = pd.read_csv(cp)
    nav = pd.read_parquet(npth)["nav"].astype(float)
    nav.index = pd.to_datetime(nav.index)
    sub = nav[nav.index >= pd.Timestamp(START)]
    if len(sub) < 100:
        return None

    tr["dt"] = pd.to_datetime(tr["date"])
    tr = tr[tr["dt"] >= pd.Timestamp(START)]
    yrs = len(sub) / 252

    per_amt = CAP / n_hold
    turnover = (len(tr) / yrs) * per_amt / CAP

    cb = comm_bp(per_amt)
    # ETF 免印花税
    stamp = 0.0 if asset == "ETF" else C.STAMP_TAX_SELL * 1e4
    rt = cb * 2 + stamp + slip_bp * 2
    ann_cost = turnover * rt * 1e-4

    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {
        "label": label, "n_hold": n_hold, "per_trade": float(per_amt),
        "turnover": float(turnover), "n_trades": int(len(tr)),
        "trades_per_year": float(len(tr) / yrs),
        "slip_bp": slip_bp, "comm_bp": float(cb),
        "roundtrip_bp": float(rt), "ann_cost": float(ann_cost),
        "cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
        "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
        "net_cagr": float(p["cagr"]) - float(ann_cost),
    }


STRATS = [
    ("主策略", "build/runs/main/trades.csv", "build/runs/main/nav.parquet", 40, "股票"),
    ("ETF趋势", "build/runs/etf_trend/trades_donchian55.csv",
     "build/runs/etf_trend/nav_donchian55.parquet", 2, "ETF"),
    ("板块轮动", "build/runs/sector_rotation/trades.csv",
     "build/runs/sector_rotation/nav.parquet", 3, "ETF"),
]

results = {}
for scen, idx in (("乐观", 0), ("中性", 1), ("保守", 2)):
    print("\n" + "=" * 96)
    print(f"情景：{scen}滑点")
    print("=" * 96)
    print(f"{'策略':<12}{'标的':<6}{'滑点':>6}{'换手':>8}{'往返bp':>8}"
          f"{'年化成本':>10}{'CAGR':>9}{'净CAGR':>9}")
    print("-" * 96)

    for label, csv, navf, nh, asset in STRATS:
        slip = SLIP_TABLE["宽基ETF" if asset == "ETF" and nh <= 2 else
                          ("行业ETF" if asset == "ETF" else "混合(按持仓)")][idx]
        r = analyze(label, csv, navf, nh, slip, asset)
        if r is None:
            print(f"{label:<12}  数据缺失")
            continue
        results[f"{label}_{scen}"] = r
        print(f"{label:<12}{asset:<6}{slip:>5.0f}{r['turnover']:>7.1f}x"
              f"{r['roundtrip_bp']:>7.0f}{r['ann_cost']:>9.2%}"
              f"{r['cagr']:>8.2%}{r['net_cagr']:>8.2%}")

# ---- 主策略 40 日（用前面算的换手 1.07x）
print("\n" + "=" * 96)
print("主策略 40 日调仓（换手 1.07x）在各滑点情景下")
print("=" * 96)
m40_cagr = 0.0621
for scen, slip in (("乐观", 25), ("中性", 40), ("保守", 50)):
    per = CAP / 40
    cb = comm_bp(per)
    rt = cb * 2 + C.STAMP_TAX_SELL * 1e4 + slip * 2
    cost = 1.07 * rt * 1e-4
    print(f"  {scen}（{slip}bp）: 往返 {rt:.0f}bp, 年化成本 {cost:.2%}, "
          f"净 CAGR {m40_cagr - cost:.2%}")
    results[f"主策略40日_{scen}"] = {
        "net_cagr": m40_cagr - cost, "ann_cost": cost,
        "roundtrip_bp": rt, "turnover": 1.07}

# ---- 关键对比表
print("\n" + "=" * 96)
print("关键对比：改善前后")
print("=" * 96)
def g(k):
    return results.get(k, {}).get("net_cagr")

print(f"\n  {'策略':<20}{'乐观净':>10}{'中性净':>10}{'保守净':>10}")
print("-" * 52)
for label in ("主策略", "主策略40日", "ETF趋势", "板块轮动"):
    vals = [g(f"{label}_{s}") for s in ("乐观", "中性", "保守")]
    if any(v is None for v in vals):
        continue
    print(f"  {label:<20}{vals[0]:>9.2%}{vals[1]:>9.2%}{vals[2]:>9.2%}")

print("\n" + "=" * 96)
print("结论")
print("=" * 96)
n_etf = g("ETF趋势_中性")
n_m40 = g("主策略40日_中性")
print(f"""
  1) 我之前统一用 40bp 滑点：
     · 对小盘股（主策略）合理 —— 文章说中小市值股 20~50bp
     · 对 ETF 明显高估 —— 文章说宽基 ETF 只有 5~10bp

  2) ETF 趋势在乐观情景（5bp）下：
     净 CAGR = {g('ETF趋势_乐观'):.2%}（之前估算是 -15.65%）
     → 差异来自滑点假设，不是策略本身

  3) 但 ETF 趋势换手 25x 仍是硬伤：
     即使 5bp 滑点，成本也占 {results['ETF趋势_乐观']['ann_cost']:.2%}

  4) 主策略 40 日调仓在所有情景都可用：
     乐观 {g('主策略40日_乐观'):.2%} / 中性 {g('主策略40日_中性'):.2%} / 保守 {g('主策略40日_保守'):.2%}
""")

json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
