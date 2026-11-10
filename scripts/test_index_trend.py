# -*- coding: utf-8 -*-
"""右侧交易研究：指数/ETF 趋势跟踪 vs 个股趋势选股（wf_rs10）。

只用指数日线 + 债券/黄金现金替代，规则简单透明（长多）：
  - MA 规则：昨收 > MA(N) 则持有指数，否则持有现金替代（100% 国债）
  - Donchian：昨收 > 过去 N 日收盘最高（不含当日）则持有
  - TSMOM：过去 N 日收益 > 0 则持有
  - 组合规则：MA200 且 60 日动量为正
交易成本按每次切换 5bp 计；信号 t-1 收盘决定 t 日持仓（无未来函数）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
import qf.config as C  # noqa: E402
from qf import metrics as qfm_metrics  # noqa: E402


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


def load_data():
    idx_path = REPO / "data_extract" / "was_stock_data_ext" / "indices.csv"
    if not idx_path.exists():
        idx_path = REPO / "data_extract" / "was_stock_data" / "indices.csv"
    idx = pd.read_csv(idx_path, index_col=0, parse_dates=True)
    bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
    gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
    b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
    g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
    b = b[~b.index.duplicated()].reindex(idx.index).ffill().pct_change().fillna(0)
    g = g[~g.index.duplicated()].reindex(idx.index).ffill().pct_change().fillna(0)
    cash = _blend_cash(b, g, idx)  # 与主策略一致的闲置现金收益
    return idx, cash


def backtest_rule(close: pd.Series, cash: pd.Series, signal: pd.Series, cost_bp: float = 5.0):
    sig = signal.shift(1).fillna(False).astype(bool)  # t 日按 t-1 收盘信号持仓
    r = close.pct_change().fillna(0.0)
    pos = sig.astype(float)
    switch = pos.diff().abs().fillna(0.0)
    nav = (1 + pos * r + (1 - pos) * cash - switch * cost_bp / 1e4).cumprod()
    return nav, pos


def periodic(close, cash, signal, n=None):
    nav, pos = backtest_rule(close, cash, signal)
    if n is not None:
        nav = nav.iloc[-n:]
        close = close.iloc[-n:]
        cash = cash.iloc[-n:]
        pos = pos.iloc[-n:]
    p = qfm_metrics.performance(nav, rf=C.CASH_YIELD)
    p["avg_exposure"] = float(pos.mean())
    p["switches_per_year"] = float((pos.diff().abs() > 0).sum() / max((len(pos) - 1) / 252, 1e-9))
    return p, nav


def turtle_signal(close: pd.Series, entry_n: int = 55, exit_n: int = 20) -> pd.Series:
    """经典海龟：收盘突破过去 entry_n 日最高（不含当日）进场，跌破 exit_n 日最低离场。"""
    hi = close.rolling(entry_n).max().shift(1)
    lo = close.rolling(exit_n).min().shift(1)
    state = np.zeros(len(close), dtype=bool)
    holding = False
    for i in range(len(close)):
        c = close.iloc[i]
        if holding:
            if np.isfinite(lo.iloc[i]) and c < lo.iloc[i]:
                holding = False
        else:
            if np.isfinite(hi.iloc[i]) and c > hi.iloc[i]:
                holding = True
        state[i] = holding
    return pd.Series(state, index=close.index)


def main():
    idx, cash = load_data()
    start = "2017-01-01"
    out_rows = []
    for code, name in [("hs300", "沪深300"), ("csi500", "中证500"),
                       ("csi1000", "中证1000"), ("csi_all", "中证全指")]:
        close = idx[code].dropna()
        close = close[close.index >= start]
        cash_ = cash.reindex(close.index).fillna(0)
        rules = {"buy&hold": pd.Series(True, index=close.index)}
        for w in (20, 60, 120, 200):
            rules[f"MA{w}"] = close > close.rolling(w).mean()
        for w in (20, 55):
            rules[f"donchian{w}"] = close > close.rolling(w).max().shift(1)
        for w in (60, 120):
            rules[f"tsmom{w}"] = close.pct_change(w) > 0
        rules["ma200+tsmom60"] = (close > close.rolling(200).mean()) & (close.pct_change(60) > 0)
        for en, ex in ((55, 20), (20, 10), (100, 30)):
            rules[f"turtle{en}/{ex}"] = turtle_signal(close, en, ex)
        for rname, sig in rules.items():
            p, nav = periodic(close, cash_, sig)
            out_rows.append({"index": code, "index_name": name, "rule": rname,
                             "cagr": p.get("cagr"), "ann_vol": p.get("ann_vol"),
                             "sharpe": p.get("sharpe"), "max_dd": p.get("max_dd"),
                             "calmar": p.get("calmar"), "avg_exposure": p["avg_exposure"],
                             "switches_per_year": p["switches_per_year"]})
    df = pd.DataFrame(out_rows)
    out = C.BUILD_RUNS / "index_trend_test.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    cols = ["index_name", "rule", "cagr", "ann_vol", "sharpe", "max_dd", "calmar",
            "avg_exposure", "switches_per_year"]
    show = df[cols].copy()
    for c in ("cagr", "ann_vol", "max_dd"):
        show[c] = (show[c] * 100).round(2)
    show["sharpe"] = show["sharpe"].round(2)
    show["calmar"] = show["calmar"].round(2)
    show["avg_exposure"] = (show["avg_exposure"] * 100).round(0)
    show["switches_per_year"] = show["switches_per_year"].round(1)
    print(show.to_string(index=False))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
