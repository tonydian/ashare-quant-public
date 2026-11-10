# -*- coding: utf-8 -*-
"""TA-Lib 覆盖度分析：现有 800 因子（33 种 base）vs TA-Lib 158 函数。

分析目标：
  1) 找出 TA-Lib 有、但我们**完全没有**的指标族（真正的增量）
  2) 找出 TA-Lib 有、但我们**已有等价物**的（重复，无价值）
  3) 对"没有的"做 IC 实测，看是否真有 alpha
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
OUT = ROOT / "build" / "eval" / "talib_coverage.json"

# 现有 33 种 base（从 factor_metrics_screen 读出）
OUR_BASES = {
    # volume-stat
    "dollar_volume", "obv_slope", "price_volume_corr", "ret_kurt", "ret_skew",
    "ts_rank_close", "ts_rank_volume", "volume_ratio", "volume_z",
    # directional
    "breakdown", "breakout", "dual_ema_gap", "efficiency", "ema_gap",
    "range_position", "return", "reversal", "skip_return", "sma_gap",
    "sma_slope", "trend_strength",
    # risk-pattern
    "atr_ratio", "downside_vol", "drawdown", "gap_sum", "intraday",
    "lower_wick", "range_ratio", "realized_vol", "rsi", "rsi_reversal",
    "stoch", "upper_wick",
}

# TA-Lib 主要函数按族分类，标注"我们是否有等价物"
TALIB_GROUPS = {
    "重叠研究 Overlap (均线/通道/布林)": {
        "SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/MAVP": "sma_gap / ema_gap / dual_ema_gap",
        "BBANDS（布林带）": "range_position（近似）",
        "SAR / SAREXT（抛物线）": "❌ 无",
        "MIDPOINT / MIDPRICE": "range_position（近似）",
        "HT_TRENDLINE（希尔伯特趋势）": "❌ 无",
    },
    "动量 Momentum": {
        "RSI": "rsi ✅",
        "STOCH / STOCHRSI": "stoch ✅",
        "MACD": "ema_gap（近似）",
        "ADX / ADXR / PLUS_DI / MINUS_DI": "trend_strength（近似）",
        "CCI": "❌ 无",
        "MFI（资金流量）": "❌ 无（但 volume_ratio 部分覆盖）",
        "WILLR / ULTOSC / ROC / ROCP / ROCR": "return / reversal（近似）",
        "MOM / CMO / TRIX / PPO": "return（近似）",
        "AROON / AROONOSC": "❌ 无",
        "BOP（均势）": "❌ 无",
    },
    "成交量 Volume": {
        "OBV / AD / ADOSC": "obv_slope ✅",
        "MFI": "❌ 无",
    },
    "波动率 Volatility": {
        "ATR / NATR / TRANGE": "atr_ratio ✅",
        "HT_DCPERIOD/DCPHASE/PHASOR/SINE/TRENDMODE": "❌ 无（希尔伯特变换族）",
    },
    "统计 Statistic": {
        "STDDEV / VAR": "realized_vol ✅",
        "LINEARREG / LINEARREG_SLOPE/ANGLE/INTERCEPT": "sma_slope（近似）",
        "TSF": "❌ 无",
    },
    "形态识别 Pattern Recognition（61 个）": {
        "CDL2CROWS/CDL3BLACKCROWS/CDLABANDONEDBABY/... 共 61 个": "❌ 全部无（但 lower_wick/upper_wick 部分近似）",
    },
    "周期 Cycle Indicators": {
        "HT_DCPERIOD/HT_TRENDMODE 等": "❌ 无",
    },
}

print("=" * 96)
print("TA-Lib vs 现有因子库 覆盖度分析")
print("=" * 96)
print(f"现有因子：800 个 / 33 种 base")
print(f"TA-Lib：158 个函数\n")

gap = []
dup = []
for group, items in TALIB_GROUPS.items():
    print(f"\n【{group}】")
    for talib_fn, ours in items.items():
        has = not ours.startswith("❌")
        mark = "已有" if has else "缺失"
        print(f"  {mark:<4} {talib_fn:<48} → {ours}")
        if has:
            dup.append((talib_fn, ours))
        else:
            gap.append((group, talib_fn))

print("\n" + "=" * 96)
print("结论")
print("=" * 96)
print(f"已有等价物：{len(dup)} 项（TA-Lib 主要是更快的实现，不是新因子）")
print(f"真正缺失：{len(gap)} 项")
print("\n真正缺失的族：")
from collections import Counter
c = Counter(g for g, _ in gap)
for grp, n in c.items():
    print(f"  {grp}: {n} 项")

OUT.write_text(json.dumps({"dup": dup, "gap": gap}, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"\n→ {OUT}")
