# -*- coding: utf-8 -*-
"""可行性预验证：ETF 趋势（指数代理）+ 主策略动态切换规则是否成立。

在正式建 ETF 数据管道前，先用指数数据快速验证三个关键假设：
  H1: 趋势规则在指数上可复现（Donchian55），且对参数不敏感；
  H2: 主策略自身趋势（净值 vs MA60）作为切换信号，在验证期/样本外是否稳定；
  H3: 动态切换 50%+50%*signal 是否优于固定 50/50。

若 H2/H3 不成立 → 按用户约定退回固定 50/50。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402

OUT = C.BUILD_EVAL / "etf_trend_feasibility.json"

# ---------- 数据：指数日线
idx = pd.read_csv(REPO / "data_extract" / "was_stock_data_ext" / "indices.csv",
                  index_col=0, parse_dates=True)
bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
b = b[~b.index.duplicated()].reindex(idx.index).ffill().pct_change().fillna(0)
g = g[~g.index.duplicated()].reindex(idx.index).ffill().pct_change().fillna(0)
cash_ret = _blend_cash(b, g, idx)
print(f"指数数据 {idx.index[0].date()} ~ {idx.index[-1].date()}，{len(idx)} 日")


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


def trend_nav(close: pd.Series, cash: pd.Series, sig: pd.Series, cost_bp: float = 5.0):
    pos = sig.shift(1).fillna(False).astype(float)
    r = close.pct_change().fillna(0.0)
    switch = pos.diff().abs().fillna(0.0)
    return (1 + pos * r + (1 - pos) * cash - switch * cost_bp / 1e4).cumprod(), pos


def donchian(close: pd.Series, n: int = 55) -> pd.Series:
    return close > close.rolling(n).max().shift(1)


def stat(nav: pd.Series) -> dict:
    p = qfm.performance(nav, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}


S, E = "2020-01-01", "2026-10-23"
print("\n=== H1: Donchian 参数敏感性（沪深300 / 中证1000） ===")
h1 = {}
for code in ("hs300", "csi1000"):
    close = idx[code].dropna()
    close = close[close.index >= "2017-01-01"]
    cash_ = cash_ret.reindex(close.index).fillna(0)
    rows = {}
    for n in (20, 40, 55, 70, 100):
        nav, pos = trend_nav(close, cash_, donchian(close, n))
        rows[n] = stat(nav[nav.index >= S])
    h1[code] = rows
    print(f"  {code}: " + " | ".join(f"N={n}: {v['cagr']*100:.1f}%/{v['sharpe']:.2f}/{v['max_dd']*100:.1f}%"
                                     for n, v in rows.items()))

# ---------- 主策略净值（PIT 口径）
main_nav = pd.read_parquet(C.BUILD_RUNS / "main/nav.parquet")["nav"].astype(float)
main_nav.index = pd.to_datetime(main_nav.index)
main_ret = main_nav.pct_change().fillna(0)
print(f"\n主策略净值 {main_nav.index[0].date()} ~ {main_nav.index[-1].date()}，"
      f"{len(main_nav)} 日")

print("\n=== H2: 主策略自身趋势信号（净值 vs MA60）的稳定性 ===")
h2 = {}
for w in (40, 60, 90, 120):
    ma = main_nav.rolling(w, min_periods=w).mean()
    sig = (main_nav > ma).astype(float)
    for tag, (s, e) in (("验证期", ("2020-01-01", "2021-12-31")),
                        ("样本外", ("2022-01-01", "2026-10-23"))):
        sub = sig[(sig.index >= s) & (sig.index <= e)]
        h2[f"MA{w}_{tag}"] = {"exposure": float(sub.mean()),
                              "n_switch": int((sub.diff().abs() > 0).sum())}
    print(f"  MA{w}: 验证期暴露 {h2[f'MA{w}_验证期']['exposure']:.0%}"
          f"（切换 {h2[f'MA{w}_验证期']['n_switch']} 次） | "
          f"样本外暴露 {h2[f'MA{w}_样本外']['exposure']:.0%}"
          f"（切换 {h2[f'MA{w}_样本外']['n_switch']} 次）")

print("\n=== H3: 动态切换 vs 固定 50/50 ===")
h3 = {}
for code in ("hs300", "csi1000"):
    close = idx[code].dropna()
    cash_ = cash_ret.reindex(close.index).fillna(0)
    e_nav, _ = trend_nav(close, cash_, donchian(close, 55))
    for w_ma in (60, 90):
        ma = main_nav.rolling(w_ma, min_periods=w_ma).mean()
        sig = (main_nav > ma).astype(float).reindex(main_nav.index).fillna(0)
        for w_main, tag in ((0.5, "固定50/50"),
                            (None, f"动态(MA{w_ma})")):
            if w_main is not None:
                w = pd.Series(w_main, index=main_nav.index)
            else:
                w = 0.5 + 0.5 * sig          # 主策略强时 100%，弱时 0%
            common = main_ret.index.intersection(e_nav.index).intersection(w.index)
            combo = (1 + w[common] * main_ret[common]
                     + (1 - w[common]) * e_nav[common].pct_change().fillna(0)).cumprod()
            for seg, (s, e) in (("全期", (S, E)), ("验证期", ("2020-01-01", "2021-12-31")),
                                ("样本外", ("2022-01-01", "2026-10-23"))):
                sub = combo[(combo.index >= s) & (combo.index <= e)]
                h3[f"{code}_{tag}_{seg}"] = stat(sub)
        print(f"  {code} MA{w_ma}: " + " | ".join(
            f"{seg} {h3[f'{code}_动态(MA{w_ma})_{seg}']['cagr']*100:.1f}%/{h3[f'{code}_动态(MA{w_ma})_{seg}']['sharpe']:.2f}"
            for seg in ("验证期", "样本外")) +
            f"  vs 固定50/50 验证期 {h3[f'{code}_固定50/50_验证期']['cagr']*100:.1f}%/{h3[f'{code}_固定50/50_验证期']['sharpe']:.2f}"
            f" 样本外 {h3[f'{code}_固定50/50_样本外']['cagr']*100:.1f}%/{h3[f'{code}_固定50/50_样本外']['sharpe']:.2f}")

OUT.write_text(json.dumps({"h1_param_sensitivity": h1, "h2_signal_stability": h2,
                           "h3_dynamic_vs_fixed": h3}, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"\n→ {OUT}")
