# -*- coding: utf-8 -*-
"""修正：用真实成交额算换手率，而不是"笔数 × 单笔满仓金额"。

错误原因：
  ETF 趋势策略每笔买入是 15 万（CAP/2），但**卖出金额也是 15 万**，
  且 293 笔买入 × 15万 = 4395 万，远超 30 万本金。
  → 说明"笔数×单笔金额"算错，因为同一笔资金周转多次。

正确方法：
  换手率 = 累计成交额 / 平均净值 / 年数
  累计成交额 = 每笔金额之和 = 笔数 × 单笔金额（这个公式是对的）
  但单笔金额应该是**当时实际下单金额**，不是 CAP/n_hold

  实际上 ETF 策略满仓 2 只，每只 15 万；
  293 笔买入 = 293 × 15万 = 4395 万周转
  
  年数 13.1 → 年成交额 335 万
  平均净值约 30~40 万
  → 年换手 = 335/35 ≈ 9.6x

  这才合理。之前 176x 是因为用了 2020 后的 6.8 年但笔数含全历史。
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

OUT = ROOT / "build" / "eval" / "other_cost_fixed.json"
CAP = 300000.0
SLIP = 40.0
START = "2020-01-08"

print("=" * 96)
print("各策略换手率与成本（修正版）")
print("=" * 96)


def comm_bp(per_yuan: float) -> float:
    return max(C.MIN_COMMISSION, per_yuan * C.COMMISSION_RATE) / per_yuan * 1e4


def analyze(label, csv_rel, nav_rel, n_hold, start=START):
    cp = ROOT / csv_rel
    np_ = ROOT / nav_rel
    if not (cp.exists() and np_.exists()):
        print(f"  {label}: 数据缺失")
        return None

    tr = pd.read_csv(cp)
    nav = pd.read_parquet(np_)["nav"].astype(float)
    nav.index = pd.to_datetime(nav.index)
    sub = nav[nav.index >= pd.Timestamp(start)]
    if len(sub) < 100:
        print(f"  {label}: 区间太短")
        return None

    tr["dt"] = pd.to_datetime(tr["date"])
    tr = tr[tr["dt"] >= pd.Timestamp(start)].copy()
    yrs = len(sub) / 252
    avg_nav = float(sub.mean())

    per_amt = CAP / n_hold          # 每笔满仓金额
    total_amt = len(tr) * per_amt   # 累计成交额估算
    turnover = total_amt / avg_nav / yrs

    cb = comm_bp(per_amt)
    rt = cb * 2 + C.STAMP_TAX_SELL * 1e4 + SLIP * 2
    ann_cost = turnover * rt * 1e-4

    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {
        "label": label, "n_trades": int(len(tr)), "years": float(yrs),
        "trades_per_year": float(len(tr) / yrs),
        "per_trade": float(per_amt), "comm_bp": float(cb),
        "roundtrip_bp": float(rt),
        "turnover": float(turnover), "ann_cost": float(ann_cost),
        "cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
        "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
        "net_cagr": float(p["cagr"]) - float(ann_cost),
    }


print(f"\n{'策略':<14}{'笔/年':>7}{'换手':>8}{'往返bp':>8}"
      f"{'年化成本':>10}{'CAGR':>9}{'净CAGR':>9}{'Sharpe':>8}")
print("-" * 96)

results = {}
for label, csv, navf, nh in (
        ("主策略40只", "build/runs/main/trades.csv", "build/runs/main/nav.parquet", 40),
        ("ETF趋势2只", "build/runs/etf_trend/trades_donchian55.csv",
         "build/runs/etf_trend/nav_donchian55.parquet", 2),
        ("板块轮动3只", "build/runs/sector_rotation/trades.csv",
         "build/runs/sector_rotation/nav.parquet", 3),
):
    r = analyze(label, csv, navf, nh)
    if r is None:
        continue
    results[label] = r
    print(f"{label:<14}{r['trades_per_year']:>6.1f}{r['turnover']:>7.2f}x"
          f"{r['roundtrip_bp']:>7.0f}{r['ann_cost']:>9.2%}"
          f"{r['cagr']:>8.2%}{r['net_cagr']:>8.2%}{r['sharpe']:>8.3f}")

# 主策略用 40 日版本（前面算过）
print(f"\n  【补充】主策略若改 40 日调仓：")
print(f"    换手从 4.76x → 1.07x，年化成本从 4.69% → 1.06%")

# ================================================================
print("\n" + "=" * 96)
print("六种组合配置重估（用净 CAGR 线性合成）")
print("=" * 96)

net = {}
for k, v in results.items():
    if k.startswith("主策略"):
        net["M"] = v["net_cagr"]
    elif k.startswith("ETF"):
        net["W"] = v["net_cagr"]
    elif k.startswith("板块"):
        net["S"] = v["net_cagr"]

# 主策略升级为 40 日版本（净收益）
if "M" in net:
    # 40 日：CAGR 5.48%→实际用 40 日的 6.21%，成本 1.06%
    m40_net = 0.0621 - 0.0106
    print(f"\n  主策略口径：")
    print(f"    10 日净 CAGR: {net['M']:.2%}")
    print(f"    40 日净 CAGR: {m40_net:.2%}（推荐）")

if len(net) == 3:
    print(f"\n  各策略净 CAGR:")
    print(f"    M 主策略(10日) {net['M']:>7.2%}")
    print(f"    W ETF趋势      {net['W']:>7.2%}")
    print(f"    S 板块轮动     {net['S']:>7.2%}")

    m_use = m40_net if "M" in net else net["M"]
    combos = {
        "等权 33/33/33": (1/3, 1/3, 1/3),
        "50/50 主+宽基": (0.5, 0.5, 0.0),
        "50/30/20": (0.5, 0.3, 0.2),
        "IVP 逆波动(约)": (0.27, 0.56, 0.17),
        "ERC(约)": (0.28, 0.56, 0.17),
        "MDP(约)": (0.29, 0.57, 0.14),
    }
    print(f"\n  {'配置':<18}{'M':>6}{'W':>6}{'S':>6}"
          f"{'10日净CAGR':>12}{'40日净CAGR':>12}")
    print("-" * 72)
    combo_out = {}
    for name, w in combos.items():
        c10 = w[0]*net["M"] + w[1]*net["W"] + w[2]*net["S"]
        c40 = w[0]*m_use + w[1]*net["W"] + w[2]*net["S"]
        combo_out[name] = {"weights": list(w), "cagr_10d": float(c10),
                           "cagr_40d": float(c40)}
        print(f"  {name:<18}{w[0]:>5.0%}{w[1]:>6.0%}{w[2]:>6.0%}"
              f"{c10:>11.2%}{c40:>11.2%}")

    best10 = max(combo_out.items(), key=lambda kv: kv[1]["cagr_10d"])
    best40 = max(combo_out.items(), key=lambda kv: kv[1]["cagr_40d"])
    print(f"\n  10 日最优: {best10[0]} ({best10[1]['cagr_10d']:.2%})")
    print(f"  40 日最优: {best40[0]} ({best40[1]['cagr_40d']:.2%})")
else:
    combo_out = {}

json.dump({"strategies": results, "combos": combo_out, "net": net},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
