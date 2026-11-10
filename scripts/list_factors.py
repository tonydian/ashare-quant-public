# -*- coding: utf-8 -*-
"""列出 wf_h10 的 34 个因子：含义、类别、IC、权重。"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, model as qfm, factors as qff  # noqa: E402

res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
cat = qff.load_catalog().set_index("factor_key")
pm = pd.read_parquet(C.BUILD_MODELS / "wf_h10" / "pool_meta.parquet")

# 最新权重
try:
    w, refit = qfm.latest_weights(res)
    wmap = dict(zip(res.pool, np.asarray(w)))
except Exception:
    wmap = {}

print("=" * 110)
print(f"wf_h10 因子池：{len(res.pool)} 个")
print("=" * 110)

rows = []
for k in res.pool:
    base = cat.loc[k, "base"] if k in cat.index else "?"
    slug = cat.loc[k, "slug"] if k in cat.index else k
    transform = cat.loc[k, "transform"] if k in cat.index else "?"
    window = cat.loc[k, "window"] if k in cat.index else "?"
    # pool_meta 的 index 是 factor_key
    ic = pm.loc[k, "ic"] if k in pm.index else np.nan
    icir = pm.loc[k, "icir"] if k in pm.index else np.nan
    t = pm.loc[k, "t"] if k in pm.index else np.nan
    rows.append({
        "key": k, "base": base, "slug": slug, "transform": transform,
        "window": window, "ic": float(ic), "icir": float(icir), "t": float(t),
        "w": float(wmap.get(k, np.nan)),
    })

df = pd.DataFrame(rows)
df["abs_icir"] = df["icir"].abs()
df = df.sort_values("abs_icir", ascending=False).reset_index(drop=True)

print(f"\n{'#':>3} {'因子':<24}{'base':<18}{'变换':<8}{'窗口':>5}"
      f"{'IC':>9}{'ICIR':>8}{'t':>8}{'权重':>8}")
print("-" * 110)
for i, r in df.iterrows():
    print(f"{i+1:>3} {r['key']:<24}{str(r['base'])[:16]:<18}"
          f"{str(r['transform'])[:7]:<8}{str(r['window']):>5}"
          f"{r['ic']:>+9.4f}{r['icir']:>+8.3f}{r['t']:>+8.2f}"
          f"{(r['w']*100 if np.isfinite(r['w']) else np.nan):>7.2f}%")

print("\n" + "=" * 110)
print("按 base（因子族）归类")
print("=" * 110)
g = df.groupby("base").agg(n=("key", "size"), ic=("ic", "mean"),
                           icir=("icir", "mean")).sort_values("n", ascending=False)
print(f"{'base 因子族':<22}{'个数':>6}{'平均IC':>10}{'平均ICIR':>10}  含义")
print("-" * 110)

MEANING = {
    "price_volume_corr": "价量相关性（量价背离）",
    "realized_vol": "已实现波动率（低波溢价）",
    "gap_sum": "跳空缺口累积（隔夜情绪）",
    "atr_ratio": "ATR 振幅比（波动）",
    "range_ratio": "日内振幅比",
    "ts_rank_close": "收盘价时序排名（自相对位置）",
    "upper_wick": "上影线占比（抛压）",
    "lower_wick": "下影线占比（承接）",
    "volume_ratio": "量比",
    "volume_z": "成交量 z 分",
    "obv_slope": "OBV 斜率（能量潮）",
    "dollar_volume": "成交额（流动性）",
    "ret_skew": "收益偏度",
    "ret_kurt": "收益峰度",
    "return": "区间收益率（动量/反转）",
    "skip_return": "跳过最近 1 日收益（反转）",
    "breakout": "突破强度",
    "breakdown": "破位强度",
    "sma_gap": "价格/均线偏离",
    "ema_gap": "价格/EMA 偏离",
    "dual_ema_gap": "双 EMA 差",
    "sma_slope": "均线斜率",
    "trend_strength": "趋势强度",
    "efficiency": "价格效率（净变动/总变动）",
    "range_position": "区间位置（在高低点间的位置）",
    "reversal": "反转强度",
    "rsi": "RSI",
    "rsi_reversal": "RSI 反转",
    "stoch": "随机指标",
    "downside_vol": "下行波动",
    "drawdown": "回撤幅度",
    "intraday": "日内位置",
}
for base, r in g.iterrows():
    print(f"{str(base):<22}{int(r['n']):>6}{r['ic']:>+10.4f}{r['icir']:>+10.3f}  "
          f"{MEANING.get(base, '')}")

# 保存
df.to_csv(ROOT / "build" / "eval" / "wf_h10_factors.csv", index=False, encoding="utf-8-sig")
print(f"\n→ build/eval/wf_h10_factors.csv")

# 按大类统计
print("\n" + "=" * 110)
print("按大类统计")
print("=" * 110)
def big_class(b):
    if b in ("price_volume_corr", "volume_ratio", "volume_z", "obv_slope", "dollar_volume"):
        return "成交量类"
    if b in ("realized_vol", "atr_ratio", "range_ratio", "downside_vol", "ret_skew", "ret_kurt"):
        return "波动/分布类"
    if b in ("return", "skip_return", "breakout", "breakdown", "sma_gap", "ema_gap",
             "dual_ema_gap", "sma_slope", "trend_strength", "efficiency",
             "range_position", "reversal"):
        return "趋势/动量类"
    if b in ("upper_wick", "lower_wick", "gap_sum", "intraday", "drawdown"):
        return "K线形态类"
    if b in ("rsi", "rsi_reversal", "stoch", "ts_rank_close"):
        return "超买超卖类"
    return "其他"

df["class"] = df["base"].map(big_class)
gc = df.groupby("class").agg(n=("key", "size"), ic=("ic", "mean"),
                              icir_abs=("abs_icir", "mean")).sort_values("n", ascending=False)
for c, r in gc.iterrows():
    print(f"  {c:<14}{int(r['n']):>3} 个 | 平均 IC {r['ic']:+.4f} | "
          f"平均 |ICIR| {r['icir_abs']:.3f}")
