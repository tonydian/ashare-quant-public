# -*- coding: utf-8 -*-
"""其他策略的成本/换手分析 + 组合配置重估。

三个问题：
  Q1 ETF 趋势策略（44.7 笔/年）成本敏感吗？
  Q2 板块轮动（REBAL=10）能降频吗？
  Q3 六种组合配置在真实成本下排序会变吗？

原则：
  · 不改策略逻辑，只算真实成本下的净收益
  · 组合配置用已有的权重方案，重算净收益
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

OUT = ROOT / "build" / "eval" / "other_cost_analysis.json"
CAP = 300000.0
MIN_COMM = 5.0
COMM_RATE = C.COMMISSION_RATE      # 万0.1
STAMP = C.STAMP_TAX_SELL
SLIP = 40.0

print("=" * 96)
print("一、各策略的交易成本估算")
print("=" * 96)


def real_roundtrip_bp(per_trade_yuan: float) -> dict:
    """按单笔金额算真实往返成本（bp）。"""
    comm = max(MIN_COMM, per_trade_yuan * COMM_RATE)
    comm_bp = comm / per_trade_yuan * 1e4 if per_trade_yuan > 0 else 0
    return {"comm_side": comm_bp, "comm_roundtrip": comm_bp * 2,
            "stamp": STAMP * 1e4, "slip_roundtrip": SLIP * 2,
            "total": comm_bp * 2 + STAMP * 1e4 + SLIP * 2 + 0.2}


rows = []
for name, n_hold, per_amt in (
        ("主策略(40只)", 40, CAP / 40),
        ("ETF趋势(2只)", 2, CAP / 2),
        ("板块轮动(3只)", 3, CAP / 3),
):
    r = real_roundtrip_bp(per_amt)
    rows.append({"strategy": name, "n_hold": n_hold, "per_trade": per_amt, **r})
    print(f"\n  【{name}】单笔 {per_amt:,.0f} 元")
    print(f"    佣金 {r['comm_side']:.2f}bp/边（往返 {r['comm_roundtrip']:.2f}bp）")
    print(f"    印花 {r['stamp']:.0f}bp | 滑点 {r['slip_roundtrip']:.0f}bp")
    print(f"    **往返合计 {r['total']:.1f}bp**")

# ================================================================
print("\n" + "=" * 96)
print("二、各策略的换手率与年化成本")
print("=" * 96)


def analyze_trades(csv_path, nav_path, label, n_hold):
    p = ROOT / csv_path
    n = ROOT / nav_path
    if not (p.exists() and n.exists()):
        return None
    tr = pd.read_csv(p)
    nav = pd.read_parquet(n)["nav"].astype(float)
    nav.index = pd.to_datetime(nav.index)
    sub = nav[nav.index >= pd.Timestamp("2020-01-08")]
    if len(sub) < 100:
        return None

    tr["dt"] = pd.to_datetime(tr["date"])
    tr = tr[tr["dt"] >= pd.Timestamp("2020-01-08")]
    yrs = len(sub) / 252

    # 成交额：用 price × 假设股数（未知）→ 改用"笔数 × 单笔金额"估算
    per_amt = CAP / n_hold
    total_amt = len(tr) * per_amt
    avg_nav = float(sub.mean())
    turnover = total_amt / avg_nav / yrs if avg_nav > 0 else 0

    rt = real_roundtrip_bp(per_amt)["total"]
    ann_cost = turnover * rt * 1e-4

    perf = qfm.performance(sub, rf=C.CASH_YIELD)
    return {
        "label": label, "n_trades": len(tr), "years": yrs,
        "trades_per_year": len(tr) / yrs, "turnover": turnover,
        "roundtrip_bp": rt, "ann_cost": ann_cost,
        "cagr": float(perf["cagr"]), "sharpe": float(perf["sharpe"]),
        "max_dd": float(perf["max_dd"]), "calmar": float(perf["calmar"]),
        "net_cagr": float(perf["cagr"]) - ann_cost,
    }


print(f"\n{'策略':<16}{'笔/年':>8}{'换手':>8}{'往返bp':>9}"
      f"{'年化成本':>10}{'CAGR(含费)':>12}{'CAGR(扣费)':>12}")
print("-" * 96)

results = {}
for label, csv, nav, nh in (
        ("ETF趋势", "build/runs/etf_trend/trades_donchian55.csv",
         "build/runs/etf_trend/nav_donchian55.parquet", 2),
        ("板块轮动", "build/runs/sector_rotation/trades.csv",
         "build/runs/sector_rotation/nav.parquet", 3),
        ("主策略(10日)", "build/runs/main/trades.csv",
         "build/runs/main/nav.parquet", 40),
):
    r = analyze_trades(csv, nav, label, nh)
    if r is None:
        print(f"{label:<16} 数据缺失")
        continue
    results[label] = r
    print(f"{label:<16}{r['trades_per_year']:>7.1f}{r['turnover']:>7.2f}x"
          f"{r['roundtrip_bp']:>8.0f}{r['ann_cost']:>9.2%}"
          f"{r['cagr']:>11.2%}{r['net_cagr']:>11.2%}")

# ================================================================
print("\n" + "=" * 96)
print("三、六种组合配置在真实成本下的重估")
print("=" * 96)
print("  方法：用各策略的净 CAGR（扣费后）按权重合成，")
print("       组合自身不产生额外成本（只按季度再平衡）")

# 各策略净收益（用上面算的）
net = {}
if "主策略(10日)" in results:
    net["M"] = results["主策略(10日)"]["net_cagr"]
if "ETF趋势" in results:
    net["W"] = results["ETF趋势"]["net_cagr"]
if "板块轮动" in results:
    net["S"] = results["板块轮动"]["net_cagr"]

if len(net) == 3:
    print(f"\n  各策略净 CAGR（扣真实成本）:")
    print(f"    主策略(M)   {net['M']:.2%}")
    print(f"    ETF趋势(W)  {net['W']:.2%}")
    print(f"    板块轮动(S) {net['S']:.2%}")

    combos = {
        "等权 33/33/33": (1/3, 1/3, 1/3),
        "50/50 主+宽基": (0.5, 0.5, 0.0),
        "50/30/20": (0.5, 0.3, 0.2),
        "逆波动 IVP": None,
        "ERC": None,
        "MDP": None,
    }

    # 尝试从已有结果读动态权重
    erc_p = ROOT / "build" / "eval" / "erc_deep.json"
    dyn = {}
    if erc_p.exists():
        try:
            d = json.load(open(erc_p, encoding="utf-8"))
            for k, v in (d.get("methods") or {}).items():
                w = v.get("avg_weights") or {}
                if w:
                    dyn[k] = (w.get("M", 1/3), w.get("W", 1/3), w.get("S", 1/3))
        except Exception:
            pass

    print(f"\n  {'配置':<18}{'M':>7}{'W':>7}{'S':>7}{'合成CAGR':>12}")
    print("-" * 60)

    combo_out = {}
    for name, w in combos.items():
        if w is None:
            # 用 erc_deep 的动态权重（简化：取平均值）
            for k in ("ERC 真等风险", "IVP 逆波动", "MDP 最大分散"):
                if k in dyn:
                    w = dyn[k]
                    break
            if w is None:
                continue
        cagr = w[0] * net["M"] + w[1] * net["W"] + w[2] * net["S"]
        combo_out[name] = {"weights": list(w), "cagr": float(cagr)}
        print(f"  {name:<18}{w[0]:>6.0%}{w[1]:>6.0%}{w[2]:>6.0%}{cagr:>11.2%}")

    print(f"\n  ⚠️ 注意：这是**线性合成**，忽略再平衡收益和波动率拖累")
    print(f"     实际组合表现会因再平衡和分散效应略有不同")

else:
    print("  策略数据不全，跳过组合重估")
    combo_out = {}

json.dump({"roundtrip": rows, "strategies": results,
           "combos": combo_out, "net": net},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
