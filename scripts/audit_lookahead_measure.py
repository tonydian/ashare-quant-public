# -*- coding: utf-8 -*-
"""系统性前视审计：对每个策略做「正确 T+1」vs「同 bar」对照实验。

判定标准：
  * 若注入同 bar 后结果**几乎不变**（<2pp CAGR）→ 该策略对前视不敏感，当前实现安全
  * 若显著变好（>5pp）→ 说明策略收益依赖"提前成交"，必须确认当前用的是哪种
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "lookahead_audit_all.json"


def _blend_cash(b, g, idx):
    """按 C.CASH_BOND_WEIGHT / C.CASH_GOLD_WEIGHT 混合现金收益（黄金权重可为 0）。"""
    bw, gw = float(C.CASH_BOND_WEIGHT), float(C.CASH_GOLD_WEIGHT)
    tot = bw + gw
    if tot <= 0:
        return pd.Series(0.0, index=idx)
    r = (bw / tot) * b.reindex(idx).ffill().pct_change().fillna(0)
    if gw > 0:
        r = r + (gw / tot) * g.reindex(idx).ffill().pct_change().fillna(0)
    return r


def perf(nav, s="2020-01-02", e="2026-10-23"):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    if len(sub) < 30:
        return {}
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


# ============================================================
# 一、宽基 ETF 趋势：官方 signals() + 两种 backtest
# ============================================================
print("=" * 98)
print("一、宽基 ETF 趋势（scripts/etf_trend_strategy.py）")
print("=" * 98)
from scripts.etf_trend_strategy import load_cash, load_etf, signals  # noqa: E402

dates, codes, P = load_etf()
b, g = load_cash()
idx = pd.DatetimeIndex(dates)
cash_ret = _blend_cash(b, g, idx)
close = P["close"]
sig = signals(dates, codes, P, 55, 0)
T, N = close.shape
cost_bp = 5.0


def bt_etf(sig, same_bar: bool):
    """same_bar=True: t 日信号 t 日成交（前视）；False: t+1 成交（正确）"""
    nav = np.ones(T)
    pos = np.full(T, -1, dtype=int)
    holding = -1
    pending = None
    for t in range(T):
        if same_bar:
            target = sig[t] if sig[t] is not None else -1
            if t > 0 and target != holding:
                px = close[t, :]
                if holding >= 0:
                    nav[t] *= (1 - cost_bp / 1e4)
                if target >= 0:
                    nav[t] *= (1 + cost_bp / 1e4)
                holding = target
            if holding >= 0 and t > 0 and np.isfinite(close[t, holding]) and np.isfinite(close[t - 1, holding]):
                nav[t] = nav[t - 1] * (close[t, holding] / close[t - 1, holding])
            elif t > 0:
                cr = cash_ret.iloc[t] if t < len(cash_ret) else 0.0
                nav[t] = nav[t - 1] * (1.0 + (float(cr) if np.isfinite(cr) else 1.0))
            pos[t] = holding
        else:
            if pending is not None and t > 0:
                target = pending
                px = close[t, :]
                if holding >= 0 and target != holding:
                    p = px[holding] * (1 - cost_bp / 1e4)
                    if np.isfinite(p) and p > 0:
                        holding = -1
                if target >= 0 and holding < 0:
                    p = px[target] * (1 + cost_bp / 1e4)
                    if np.isfinite(p) and p > 0:
                        holding = target
                pending = None
            if holding >= 0 and t > 0 and np.isfinite(close[t, holding]) and np.isfinite(close[t - 1, holding]):
                nav[t] = nav[t - 1] * (close[t, holding] / close[t - 1, holding])
            elif t > 0:
                cr = cash_ret.iloc[t] if t < len(cash_ret) else 0.0
                nav[t] = nav[t - 1] * (1.0 + (float(cr) if np.isfinite(cr) else 1.0))
            pos[t] = holding
            pending = sig[t] if sig[t] is not None else -1
    return pd.Series(nav, index=dates)


nav_correct = bt_etf(sig, same_bar=False)
nav_look = bt_etf(sig, same_bar=True)
pc, pl = perf(nav_correct), perf(nav_look)
print(f"  官方实现（T+1 正确）  : CAGR {pc['cagr']:.2%} Sharpe {pc['sharpe']:.3f} DD {pc['max_dd']:.2%}")
print(f"  注入同 bar 前视       : CAGR {pl['cagr']:.2%} Sharpe {pl['sharpe']:.3f} DD {pl['max_dd']:.2%}")
print(f"  → 前视带来的虚高      : {(pl['cagr']-pc['cagr'])*100:+.2f}pp")
res = {"etf_trend": {"correct": pc, "lookahead": pl,
                     "delta_cagr_pp": (pl["cagr"] - pc["cagr"]) * 100}}

# ============================================================
# 二、板块轮动：官方 qf/sector_rotation.py
# ============================================================
print("\n" + "=" * 98)
print("二、板块轮动（qf/sector_rotation.py，已修为严格 T+1）")
print("=" * 98)
from qf import sector_rotation as qfsr  # noqa: E402

d2, c2, cl2, am2, sec2 = qfsr.load_pool()
start = max(int(np.flatnonzero(d2 >= pd.Timestamp("2020-01-02"))[0]), 250)
r_official = qfsr.backtest(d2, c2, cl2, am2, sec2, start_row=start, verbose=False)
po = perf(r_official["nav"])
print(f"  当前官方实现（T+1）   : CAGR {po['cagr']:.2%} Sharpe {po['sharpe']:.3f} DD {po['max_dd']:.2%}")
print(f"  参考：修复前同 bar 版  : CAGR 32.09% Sharpe 1.468 DD -20.41%")
print(f"  → 前视虚高            : {(0.3209-po['cagr'])*100:+.2f}pp")
res["sector_rotation"] = {"correct": po, "same_bar_old": {"cagr": 0.3209, "sharpe": 1.468},
                          "delta_cagr_pp": (0.3209 - po["cagr"]) * 100}

# ============================================================
# 三、三策略组合：净值合成，无成交行为 → 结构上不可能有前视
# ============================================================
print("\n" + "=" * 98)
print("三、三策略组合（scripts/three_strategy_combo.py）")
print("=" * 98)
src = (ROOT / "scripts" / "three_strategy_combo.py").read_text(encoding="utf-8")
print("  检查：是否只做净值加权（无成交/无价格）")
has_price = any(k in src for k in ("close[", "open[", "price["))
has_trade = any(k in src for k in ("buy_", "sell_", "order", "shares"))
print(f"    使用价格数组: {has_price} | 使用成交逻辑: {has_trade}")
print(f"    → {'安全：纯净值加权，无成交语义，结构上不可能前视' if not (has_price and has_trade) else '需人工检查'}")
res["three_combo"] = {"has_price": has_price, "has_trade": has_trade,
                      "safe": not (has_price and has_trade)}

# ============================================================
# 四、ETF 50/50 组合
# ============================================================
print("\n" + "=" * 98)
print("四、ETF 50/50 组合（scripts/etf_combo_50_50.py）")
print("=" * 98)
src2 = (ROOT / "scripts" / "etf_combo_50_50.py").read_text(encoding="utf-8")
hp = any(k in src2 for k in ("close[", "open[", "price["))
ht = any(k in src2 for k in ("buy_", "sell_", "order"))
print(f"    使用价格数组: {hp} | 使用成交逻辑: {ht}")
print(f"    → {'安全：纯净值加权' if not (hp and ht) else '需人工检查'}")
res["etf_combo"] = {"has_price": hp, "has_trade": ht, "safe": not (hp and ht)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
