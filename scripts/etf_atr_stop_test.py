# -*- coding: utf-8 -*-
"""ATR 跟踪止损 A/B 测试（材料建议的最后一个未验证项）。

对比四组出场规则（入场规则完全相同：突破 55 日新高 + 成交额过滤）：
  A. 紧出场（现行）：收盘 <= 过去 55 日最高（即不再创新高）→ 次日卖
  B. 20 日低出场（海龟式）：收盘 < 过去 20 日最低 → 次日卖
  C. 紧出场 + ATR 跟踪止损：A 或 触发 ATR 止损，任一成立即卖
  D. 仅 ATR 跟踪止损：入场后只靠 ATR 止损离场（不给趋势反转出场）

ATR 止损定义（只上移、不下移）：
  stop_t = max(stop_{t-1}, 入场后最高收盘 - k × ATR(14)_t)
  收盘 < stop_t → 次日卖出
k 做敏感性扫描（1.5/2.0/2.5/3.0/4.0），看是否高原。
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
from scripts.etf_trend_strategy import (  # noqa: E402
    MIN_AMOUNT, SLIPPAGE, load_cash, load_etf,
)

OUT = ROOT / "build" / "eval" / "etf_atr_stop_test.json"
OUT_RUN = ROOT / "build" / "runs" / "etf_atr"


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


def atr14(P) -> np.ndarray:
    """ATR(14)：TR = max(H-L, |H-C_prev|, |L-C_prev|)，简单均值。"""
    close, high, low = P["close"], P["high"], P["low"]
    prev_c = np.roll(close, 1, axis=0)
    prev_c[0] = np.nan
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    return pd.DataFrame(tr).rolling(14, min_periods=10).mean().to_numpy()


def signals_atr(P, entry_n=55, exit_mode="tight", atr_k=2.5, atr=None):
    """带 ATR 跟踪止损的信号生成。

    exit_mode: 'tight'（紧出场）/ 'low20'（20日低）/ 'tight_atr' / 'atr_only'
    """
    close = P["close"]
    amount = P["amount"]
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    lo20 = cs.rolling(20).min().shift(1).to_numpy()
    mom60 = (cs / cs.shift(60) - 1.0).to_numpy()
    if atr is None:
        atr = atr14(P)

    sig = []
    holding = None
    stop = np.nan          # 当前持仓的跟踪止损位
    peak = np.nan          # 入场后最高收盘

    for t in range(T):
        # ---------- 出场判断（用 t 日收盘）
        if holding is not None:
            c = close[t, holding]
            if not np.isfinite(c):
                holding = None
                stop = peak = np.nan
            else:
                # 更新 peak 与止损位（只上移）
                if not np.isfinite(peak) or c > peak:
                    peak = c
                a = atr[t, holding]
                if np.isfinite(a) and a > 0:
                    new_stop = peak - atr_k * a
                    stop = new_stop if not np.isfinite(stop) else max(stop, new_stop)

                out = False
                if exit_mode in ("tight", "tight_atr"):
                    if np.isfinite(hi[t, holding]) and c <= hi[t, holding]:
                        out = True
                elif exit_mode == "low20":
                    if np.isfinite(lo20[t, holding]) and c < lo20[t, holding]:
                        out = True
                if exit_mode in ("atr_only", "tight_atr"):
                    if np.isfinite(stop) and c < stop:
                        out = True
                if out:
                    holding = None
                    stop = peak = np.nan

        # ---------- 入场判断
        if holding is None:
            cand = []
            for j in range(N):
                c = close[t, j]
                if not (np.isfinite(c) and np.isfinite(hi[t, j]) and c > hi[t, j]):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= MIN_AMOUNT):
                    continue
                cand.append((j, mom60[t, j] if np.isfinite(mom60[t, j]) else -9))
            if cand:
                holding = max(cand, key=lambda x: x[1])[0]
                peak = close[t, holding]
                a = atr[t, holding]
                stop = peak - atr_k * a if (np.isfinite(a) and a > 0) else np.nan
        sig.append(holding)
    return sig


def backtest(dates, codes, P, sig, cash_ret, cost_bp=SLIPPAGE * 1e4):
    """与主 ETF 回测同口径（T+1，收盘成交 + 滑点）。"""
    close = P["close"]
    T, N = close.shape
    nav = np.ones(T, dtype=float)
    pos = np.full(T, -1, dtype=int)
    holding = -1
    pending = None
    trades = []
    for t in range(T):
        if pending is not None and t > 0:
            target = pending
            px = close[t, :]
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
    yearly = qfm.yearly_returns(nav)
    out["yearly"] = {int(k): float(v) for k, v in yearly.items()}
    print(f"  {tag:30s} CAGR {out['cagr']:7.2%} Sharpe {out['sharpe']:6.3f} "
          f"MaxDD {out['max_dd']:7.2%} Calmar {out['calmar']:5.2f} "
          f"暴露 {out['exposure']:5.1%} 交易 {out['n_trades']}")
    return out


def main():
    dates, codes, P = load_etf()
    b, g = load_cash()
    idx = pd.DatetimeIndex(dates)
    cash_ret = _blend_cash(b, g, idx)
    start_row = int(np.flatnonzero(idx >= pd.Timestamp(C.VALID_START))[0])
    atr = atr14(P)
    print(f"ETF 面板 {idx[0].date()} ~ {idx[-1].date()}；回测起点 {idx[start_row].date()}")
    print(f"ATR(14)/close 均值 {np.nanmean(atr/P['close']):.2%}\n")

    res = {}
    print("=" * 96)
    print("A/B 对比：出场规则（入场规则相同）")
    print("=" * 96)
    for mode, k, tag in (("tight", 0, "A 紧出场（现行）"),
                         ("low20", 0, "B 20日低出场（海龟）"),
                         ("tight_atr", 2.5, "C 紧出场 + ATR(2.5)"),
                         ("atr_only", 2.5, "D 仅 ATR(2.5) 止损")):
        sig = signals_atr(P, exit_mode=mode, atr_k=k if k else 2.5, atr=atr)
        nav, pos, tr = backtest(dates, codes, P, sig, cash_ret)
        res[tag] = stats(nav.iloc[start_row:], pos.iloc[start_row:], tr, tag)
        if mode == "tight":
            base_nav = nav
        elif mode == "tight_atr":
            res[tag]["_nav"] = nav
        elif mode == "atr_only":
            res[tag]["_nav"] = nav

    print()
    print("=" * 96)
    print("k 敏感性（ATR 倍数）")
    print("=" * 96)
    sweep = {}
    for k in (1.5, 2.0, 2.5, 3.0, 4.0):
        for mode, mtag in (("tight_atr", "紧+ATR"), ("atr_only", "仅ATR")):
            sig = signals_atr(P, exit_mode=mode, atr_k=k, atr=atr)
            nav, pos, tr = backtest(dates, codes, P, sig, cash_ret)
            s = stats(nav.iloc[start_row:], pos.iloc[start_row:], tr, f"{mtag} k={k}")
            sweep[f"{mode}_k{k}"] = s
            res[f"{mtag}_k{k}"] = s
    res["sweep"] = sweep

    # 分年对比（紧出场 vs 紧+ATR2.5）
    print()
    print("=" * 96)
    print("分年收益对比")
    print("=" * 96)
    ta = res["A 紧出场（现行）"]["yearly"]
    tb = res["C 紧出场 + ATR(2.5)"]["yearly"]
    for y in sorted(ta):
        print(f"  {y}: 紧出场 {ta[y]*100:6.2f}%  |  紧+ATR {tb.get(y,0)*100:6.2f}%  "
              f"({(tb.get(y,0)-ta[y])*100:+.2f}pp)")

    # 清理不可序列化字段
    for v in res.values():
        if isinstance(v, dict):
            v.pop("_nav", None)
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
