# -*- coding: utf-8 -*-
"""最终版：修正净值归一化问题后的成本分析。

关键修正：
  各策略的 nav.parquet 是**归一化净值**（起点=1 或 30 万，但数值量级不同）
  → 换手率必须用"年成交额 / 期初本金"，不能用 nav 均值

正确公式：
  年换手 = (年交易笔数 × 单笔金额) / 本金

  其中单笔金额 = 本金 / 持仓数
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

OUT = ROOT / "build" / "eval" / "other_cost_final.json"
CAP = 300000.0          # 本金
SLIP = 40.0
START = "2020-01-08"


def comm_bp(per_yuan: float) -> float:
    return max(C.MIN_COMMISSION, per_yuan * C.COMMISSION_RATE) / per_yuan * 1e4


def analyze(label, csv_rel, nav_rel, n_hold):
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
    ann_turnover = (len(tr) / yrs) * per_amt / CAP     # 年换手倍数

    cb = comm_bp(per_amt)
    rt = cb * 2 + C.STAMP_TAX_SELL * 1e4 + SLIP * 2
    ann_cost = ann_turnover * rt * 1e-4

    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {
        "label": label, "n_trades": int(len(tr)), "years": float(yrs),
        "trades_per_year": float(len(tr) / yrs),
        "per_trade": float(per_amt), "comm_bp": float(cb),
        "roundtrip_bp": float(rt), "turnover": float(ann_turnover),
        "ann_cost": float(ann_cost),
        "cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
        "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
        "net_cagr": float(p["cagr"]) - float(ann_cost),
    }


print("=" * 96)
print("各策略：真实成本下的净收益")
print("=" * 96)
print(f"  本金 {CAP:,.0f} | 佣金万0.1(最低5元) | 印花5bp | 滑点{SLIP:.0f}bp/边")
print(f"  换手 = (年笔数 × 单笔金额) / 本金\n")

print(f"{'策略':<14}{'持仓':>5}{'单笔金额':>11}{'笔/年':>8}{'换手':>8}"
      f"{'佣金bp':>8}{'往返bp':>8}{'年化成本':>10}{'CAGR':>9}{'净CAGR':>9}")
print("-" * 96)

results = {}
for label, csv, navf, nh in (
        ("主策略", "build/runs/main/trades.csv", "build/runs/main/nav.parquet", 40),
        ("ETF趋势", "build/runs/etf_trend/trades_donchian55.csv",
         "build/runs/etf_trend/nav_donchian55.parquet", 2),
        ("板块轮动", "build/runs/sector_rotation/trades.csv",
         "build/runs/sector_rotation/nav.parquet", 3),
):
    r = analyze(label, csv, navf, nh)
    if r is None:
        print(f"{label:<14}  数据缺失")
        continue
    results[label] = r
    print(f"{label:<14}{nh:>5}{r['per_trade']:>10,.0f}{r['trades_per_year']:>8.1f}"
          f"{r['turnover']:>7.1f}x{r['comm_bp']:>7.2f}{r['roundtrip_bp']:>7.0f}"
          f"{r['ann_cost']:>9.2%}{r['cagr']:>8.2%}{r['net_cagr']:>8.2%}")

# 主策略 40 日版本（已知结果）
print(f"\n  【主策略改 40 日调仓】")
print(f"    换手 4.76x → 1.07x（-78%）")
print(f"    年化成本 4.69% → 1.06%（省 3.63pp）")
print(f"    净 CAGR 5.48% - 4.69% = {0.0548-0.0469:.2%} → 6.21% - 1.06% = {0.0621-0.0106:.2%}")

# ================================================================
print("\n" + "=" * 96)
print("六种组合配置重估")
print("=" * 96)

M10 = results.get("主策略", {}).get("net_cagr")
W = results.get("ETF趋势", {}).get("net_cagr")
S = results.get("板块轮动", {}).get("net_cagr")

if None in (M10, W, S):
    print("  数据不全，跳过")
else:
    M40 = 0.0621 - 0.0106
    print(f"\n  各策略净 CAGR:")
    print(f"    M 主策略(10日)  {M10:>7.2%}")
    print(f"    M 主策略(40日)  {M40:>7.2%}  ← 推荐")
    print(f"    W ETF趋势       {W:>7.2%}")
    print(f"    S 板块轮动      {S:>7.2%}")

    combos = {
        "等权 33/33/33": (1/3, 1/3, 1/3),
        "50/50 主+宽基": (0.5, 0.5, 0.0),
        "50/30/20": (0.5, 0.3, 0.2),
        "IVP 逆波动": (0.27, 0.56, 0.17),
        "ERC 风险平价": (0.28, 0.56, 0.17),
        "MDP 最大分散": (0.29, 0.57, 0.14),
    }
    print(f"\n  {'配置':<18}{'M':>6}{'W':>6}{'S':>6}"
          f"{'10日净':>10}{'40日净':>10}")
    print("-" * 68)
    combo_out = {}
    for name, w in combos.items():
        c10 = w[0] * M10 + w[1] * W + w[2] * S
        c40 = w[0] * M40 + w[1] * W + w[2] * S
        combo_out[name] = {"weights": list(w), "cagr_10d": float(c10),
                           "cagr_40d": float(c40)}
        print(f"  {name:<18}{w[0]:>5.0%}{w[1]:>6.0%}{w[2]:>6.0%}"
              f"{c10:>9.2%}{c40:>9.2%}")

    b10 = max(combo_out.items(), key=lambda kv: kv[1]["cagr_10d"])
    b40 = max(combo_out.items(), key=lambda kv: kv[1]["cagr_40d"])
    print(f"\n  10 日最优: {b10[0]} → {b10[1]['cagr_10d']:.2%}")
    print(f"  40 日最优: {b40[0]} → {b40[1]['cagr_40d']:.2%}")

    # 相对国债
    print(f"\n  相对国债(2.5%)超额:")
    for name, v in combo_out.items():
        print(f"    {name:<18} {v['cagr_40d']-0.025:>+6.2%}")

    json.dump({"strategies": results, "combos": combo_out,
               "M10": M10, "M40": M40, "W": W, "S": S},
              open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {OUT}")
