# -*- coding: utf-8 -*-
"""动态切换的过拟合排除测试。

三个测试：
  T1 严格因果：切换信号只用 t-1 及以前的净值（实盘可行），与"用当天"对比差异；
  T2 噪声净值对照：把主策略净值替换成同均值同波动的随机游走（但保留 ETF 侧不变），
     若动态版仍然大赢 → 收益来自 ETF 侧与"日内信息"，切换规则没有真信息；
  T3 冷启动/样本外一致性：把规则参数在验证期选，只看样本外，和固定 50/50 对比。
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

OUT = C.BUILD_EVAL / "etf_switch_overfit_test.json"
idx = pd.read_csv(REPO / "data_extract" / "was_stock_data_ext" / "indices.csv",
                  index_col=0, parse_dates=True)
bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
b = b[~b.index.duplicated()].reindex(idx.index).ffill().pct_change().fillna(0)
g = g[~g.index.duplicated()].reindex(idx.index).ffill().pct_change().fillna(0)
cash = _blend_cash(b, g, idx)

main_nav = pd.read_parquet(C.BUILD_RUNS / "main/nav.parquet")["nav"].astype(float)
main_nav.index = pd.to_datetime(main_nav.index)
main_ret = main_nav.pct_change().fillna(0)


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


def etf_ret(code: str) -> pd.Series:
    close = idx[code].dropna()
    return (close > close.rolling(55).max().shift(1)).shift(1).fillna(False).astype(float) * \
        close.pct_change().fillna(0) + \
        (1 - (close > close.rolling(55).max().shift(1)).shift(1).fillna(False).astype(float)) * \
        cash.reindex(close.index).fillna(0)


def perf(nav: pd.Series, s: str, e: str) -> dict:
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


def combo(main_series: pd.Series, etf_series: pd.Series, w: pd.Series, s: str, e: str) -> dict:
    common = main_series.index.intersection(etf_series.index).intersection(w.index)
    nav = (1 + w[common] * main_series[common] + (1 - w[common]) * etf_series[common]).cumprod()
    return perf(nav, s, e)


SEGS = {"验证期": ("2020-01-01", "2021-12-31"), "样本外": ("2022-01-01", "2026-10-23")}
out = {}

print("=" * 84)
print("T1: 严格因果（信号只用 t-1 及以前净值）")
print("=" * 84)
for code in ("hs300", "csi1000"):
    e_ret = etf_ret(code)
    for w_ma in (60, 90):
        # 因果版：signal 用 shift(1)
        ma = main_nav.rolling(w_ma, min_periods=w_ma).mean()
        sig = ((main_nav > ma).astype(float)).shift(1).fillna(0.5)
        w = 0.5 + 0.5 * sig
        for seg, (s, e) in SEGS.items():
            r = combo(main_ret, e_ret, w, s, e)
            out[f"T1_{code}_MA{w_ma}_{seg}"] = r
        print(f"  {code} MA{w_ma}: 验证期 {out[f'T1_{code}_MA{w_ma}_验证期']['cagr']*100:6.2f}%/"
              f"{out[f'T1_{code}_MA{w_ma}_验证期']['sharpe']:5.2f} | "
              f"样本外 {out[f'T1_{code}_MA{w_ma}_样本外']['cagr']*100:6.2f}%/"
              f"{out[f'T1_{code}_MA{w_ma}_样本外']['sharpe']:5.2f}")

print()
print("=" * 84)
print("T2: 噪声净值对照（把主策略净值换成同均值/同波动的随机游走）")
print("=" * 84)
rng = np.random.default_rng(20261023)
mu, sd = main_ret.mean(), main_ret.std()
n = len(main_ret)
noise_nav = pd.Series(100.0 * np.cumprod(1 + rng.normal(mu, sd, n)), index=main_ret.index)
noise_ret = noise_nav.pct_change().fillna(0)
for code in ("hs300", "csi1000"):
    e_ret = etf_ret(code)
    for w_ma in (60,):
        ma = noise_nav.rolling(w_ma, min_periods=w_ma).mean()
        sig = ((noise_nav > ma).astype(float)).shift(1).fillna(0.5)
        w = 0.5 + 0.5 * sig
        for seg, (s, e) in SEGS.items():
            r = combo(noise_ret, e_ret, w, s, e)
            out[f"T2_noise_{code}_MA{w_ma}_{seg}"] = r
        # 对照：噪声净值 + 固定 50/50
        for seg, (s, e) in SEGS.items():
            r = combo(noise_ret, e_ret, pd.Series(0.5, index=noise_ret.index), s, e)
            out[f"T2_noise_fixed_{code}_{seg}"] = r
        print(f"  {code} 噪声+MA{w_ma}: 验证期 {out[f'T2_noise_{code}_MA{w_ma}_验证期']['sharpe']:5.2f} "
              f"vs 固定 {out[f'T2_noise_fixed_{code}_验证期']['sharpe']:5.2f} | "
              f"样本外 {out[f'T2_noise_{code}_MA{w_ma}_样本外']['sharpe']:5.2f} "
              f"vs 固定 {out[f'T2_noise_fixed_{code}_样本外']['sharpe']:5.2f}")

print()
print("=" * 84)
print("T3: 参数在验证期选，只看样本外（和固定 50/50 对照）")
print("=" * 84)
best = None
for code in ("hs300", "csi1000"):
    e_ret = etf_ret(code)
    for w_ma in (40, 60, 90, 120):
        ma = main_nav.rolling(w_ma, min_periods=w_ma).mean()
        sig = ((main_nav > ma).astype(float)).shift(1).fillna(0.5)
        w = 0.5 + 0.5 * sig
        v = combo(main_ret, e_ret, w, *SEGS["验证期"])
        if best is None or v["sharpe"] > best[3]["sharpe"]:
            best = (code, w_ma, e_ret, v)
    fv = combo(main_ret, e_ret, pd.Series(0.5, index=main_ret.index), *SEGS["验证期"])
    print(f"  {code} 固定50/50 验证期 Sharpe {fv['sharpe']:.2f}")
code, w_ma, e_ret, v = best
ma = main_nav.rolling(w_ma, min_periods=w_ma).mean()
sig = ((main_nav > ma).astype(float)).shift(1).fillna(0.5)
oos = combo(main_ret, e_ret, 0.5 + 0.5 * sig, *SEGS["样本外"])
oos_fixed = combo(main_ret, e_ret, pd.Series(0.5, index=main_ret.index), *SEGS["样本外"])
print(f"\n  验证期选出的最优：{code} MA{w_ma}（验证期 Sharpe {v['sharpe']:.2f}）")
print(f"  → 该规则样本外：CAGR {oos['cagr']*100:.2f}% Sharpe {oos['sharpe']:.2f} "
      f"MaxDD {oos['max_dd']*100:.2f}%")
print(f"  → 固定 50/50 样本外：CAGR {oos_fixed['cagr']*100:.2f}% Sharpe {oos_fixed['sharpe']:.2f} "
      f"MaxDD {oos_fixed['max_dd']*100:.2f}%")
out["T3_selected"] = {"code": code, "w_ma": w_ma, "val": v, "oos": oos, "oos_fixed": oos_fixed}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
