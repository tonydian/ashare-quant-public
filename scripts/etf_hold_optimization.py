# -*- coding: utf-8 -*-
"""关键诊断 + 优化：现行"紧出场"持仓中位仅 2 天，ATR 止损永远触发不了。
   真正的问题是**入场太尖、出场太早**，不是止损参数。

本脚本测试"让趋势跑起来"的几种组合：
  1) 入场放宽（不要求创新高，而是站上 MA20/MA60）+ ATR 跟踪止损 → 能否吃到趋势主升段；
  2) 入场用 Donchian55，出场改用 ATR 止损 + 最长持有期；
  3) 入场用 Donchian55，但允许**加仓/再入场**（趋势延续时不清空）。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from qf import config as C, metrics as qfm  # noqa: E402
from scripts.etf_trend_strategy import MIN_AMOUNT, SLIPPAGE, load_cash, load_etf  # noqa: E402

OUT = ROOT / "build" / "eval" / "etf_hold_optimization.json"


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


def atr14(P):
    close, high, low = P["close"], P["high"], P["low"]
    prev_c = np.roll(close, 1, axis=0)
    prev_c[0] = np.nan
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    return pd.DataFrame(tr).rolling(14, min_periods=10).mean().to_numpy()


def make_signals(P, entry, exit_rule, atr, atr_k=2.5, entry_n=55, ma_n=20, max_hold=None):
    """entry: 'breakout' | 'ma' ；exit_rule: 'tight' | 'atr' | 'atr_or_ma'"""
    close = P["close"]
    amount = P["amount"]
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    ma = cs.rolling(ma_n, min_periods=ma_n).mean().to_numpy()
    mom60 = (cs / cs.shift(60) - 1.0).to_numpy()

    sig = []
    holding = None
    stop = peak = np.nan
    held = 0
    for t in range(T):
        if holding is not None:
            held += 1
            c = close[t, holding]
            if not np.isfinite(c):
                holding = None; stop = peak = np.nan; held = 0
            else:
                if not np.isfinite(peak) or c > peak:
                    peak = c
                a = atr[t, holding]
                if np.isfinite(a) and a > 0:
                    ns = peak - atr_k * a
                    stop = ns if not np.isfinite(stop) else max(stop, ns)
                out = False
                if exit_rule == "tight":
                    if np.isfinite(hi[t, holding]) and c <= hi[t, holding]:
                        out = True
                elif exit_rule == "atr":
                    if np.isfinite(stop) and c < stop:
                        out = True
                elif exit_rule == "atr_or_ma":
                    if np.isfinite(stop) and c < stop:
                        out = True
                    if np.isfinite(ma[t, holding]) and c < ma[t, holding]:
                        out = True
                if max_hold and held >= max_hold:
                    out = True
                if out:
                    holding = None; stop = peak = np.nan; held = 0
        if holding is None:
            cand = []
            for j in range(N):
                c = close[t, j]
                if not np.isfinite(c):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= MIN_AMOUNT):
                    continue
                if entry == "breakout":
                    if not (np.isfinite(hi[t, j]) and c > hi[t, j]):
                        continue
                else:  # ma
                    if not (np.isfinite(ma[t, j]) and c > ma[t, j]):
                        continue
                cand.append((j, mom60[t, j] if np.isfinite(mom60[t, j]) else -9))
            if cand:
                holding = max(cand, key=lambda x: x[1])[0]
                peak = close[t, holding]
                a = atr[t, holding]
                stop = peak - atr_k * a if (np.isfinite(a) and a > 0) else np.nan
                held = 0
        sig.append(holding)
    return sig


def backtest(dates, codes, P, sig, cash_ret, cost_bp=SLIPPAGE * 1e4):
    close = P["close"]
    T, N = close.shape
    nav = np.ones(T); pos = np.full(T, -1, dtype=int)
    holding = -1; pending = None; trades = []
    for t in range(T):
        if pending is not None and t > 0:
            target = pending; px = close[t, :]
            if holding >= 0 and target != holding:
                p = px[holding] * (1 - cost_bp / 1e4)
                if np.isfinite(p) and p > 0:
                    trades.append({"date": dates[t], "code": codes[holding], "side": "sell", "price": float(p)})
                    holding = -1
            if target >= 0 and holding < 0:
                p = px[target] * (1 + cost_bp / 1e4)
                if np.isfinite(p) and p > 0:
                    holding = target
                    trades.append({"date": dates[t], "code": codes[target], "side": "buy", "price": float(p)})
            pending = None
        if holding >= 0 and t > 0 and np.isfinite(close[t, holding]) and np.isfinite(close[t - 1, holding]):
            nav[t] = nav[t - 1] * (close[t, holding] / close[t - 1, holding])
        elif t > 0:
            cr = cash_ret.iloc[t] if t < len(cash_ret) else 0.0
            nav[t] = nav[t - 1] * (1.0 + float(cr) if np.isfinite(cr) else 1.0)
        pos[t] = holding
        pending = sig[t] if sig[t] is not None else -1
    return pd.Series(nav, index=dates), pd.Series(pos, index=dates), pd.DataFrame(trades)


def stats(nav, pos, trades, tag):
    p = qfm.performance(nav, rf=C.CASH_YIELD)
    out = {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}
    out.update({"exposure": float((pos >= 0).mean()), "n_trades": int(len(trades))})
    out["yearly"] = {int(k): float(v) for k, v in qfm.yearly_returns(nav).items()}
    print(f"  {tag:38s} CAGR {out['cagr']:7.2%} Sharpe {out['sharpe']:6.3f} "
          f"MaxDD {out['max_dd']:7.2%} Calmar {out['calmar']:5.2f} 暴露 {out['exposure']:5.1%}")
    return out


def main():
    dates, codes, P = load_etf()
    b, g = load_cash()
    idx = pd.DatetimeIndex(dates)
    cash_ret = _blend_cash(b, g, idx)
    start = int(np.flatnonzero(idx >= pd.Timestamp(C.VALID_START))[0])
    atr = atr14(P)
    res = {}
    print(f"起点 {idx[start].date()}；ATR/close 均值 {np.nanmean(atr/P['close']):.2%}\n")

    tests = [
        ("基准：突破55 + 紧出场", dict(entry="breakout", exit_rule="tight")),
        ("突破55 + ATR2.5", dict(entry="breakout", exit_rule="atr")),
        ("突破55 + ATR2.5 + 最长20日", dict(entry="breakout", exit_rule="atr", max_hold=20)),
        ("突破55 + ATR2.5 + 最长40日", dict(entry="breakout", exit_rule="atr", max_hold=40)),
        ("MA20上方 + ATR2.5", dict(entry="ma", exit_rule="atr", ma_n=20)),
        ("MA60上方 + ATR2.5", dict(entry="ma", exit_rule="atr", ma_n=60)),
        ("MA20上方 + ATR2.5 + 最长40日", dict(entry="ma", exit_rule="atr", ma_n=20, max_hold=40)),
        ("MA20上方 + ATR_or_MA20", dict(entry="ma", exit_rule="atr_or_ma", ma_n=20)),
    ]
    for tag, kw in tests:
        sig = make_signals(P, atr=atr, **kw)
        nav, pos, tr = backtest(dates, codes, P, sig, cash_ret)
        res[tag] = stats(nav.iloc[start:], pos.iloc[start:], tr, tag)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
