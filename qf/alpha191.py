# -*- coding: utf-8 -*-
"""国泰君安 Alpha191（短周期价量特征）—— 公式表 + 向量化实现。

来源
----
国泰君安证券 2017《基于短周期价量特征的多因子选股体系》，191 个日频量价因子。
公式取自聚宽 Alphas191 数据字典（jqdatasdk ``alpha191`` 模块的 docstring），
原文中的笔误（``YSRANK``/``LOSE``/``DELAT``/``HGIH``/全角逗号等）在
:data:`FORMULAS` 里已按语义修正，逐条差异见 ``docs/ALPHA191.md``。

口径
----
* 面板 ``(T, N)`` float64，行=交易日、列=股票；窗口算子为满窗口径（含 NaN → NaN）。
* ``VOLUME`` 缩放为「百万股」、``AMOUNT`` 为「亿元」，仅为数值稳定，不影响横截面排名。
* ``VWAP``：数据源无均价字段，由 ``amount / volume``（不复权价口径）还原，
  再按「复权因子分段恒定」估计出后复权均价，见 :func:`build_vwap`。
* ``RANK`` 为横截面百分位秩（居中到 [-0.5, 0.5]），与 JQ 的 ``RANK`` 同义。
* 需要 ``SELF``（alpha_143 递归）、Fama-French 三因子（alpha_030）、
  基准指数（alpha_075/181）、``SUMAC``（alpha_165/183）或原文残缺
  （alpha_101/123/128）的因子不在实现范围内，见 :data:`EXCLUDED`。
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import alpha_ops as op

__all__ = ["FORMULAS", "EXCLUDED", "build_inputs", "build_vwap", "compute", "compute_one", "available"]

# ------------------------------------------------------------------ 常量
SQRT = np.sqrt
ABS = np.abs
LOG = np.log
POW = np.power
NAN = np.nan


# ------------------------------------------------------------------ VWAP
def build_vwap(md, cache_path=None, verbose: bool = True, method: str = "rolling",
               window: int = 61) -> np.ndarray:
    """用 ``amount / volume`` 还原后复权 VWAP。

    数据源里 ``open/high/low/close`` 是后复权，``volume/amount`` 是不复权口径，
    两者相差一个 **逐股分段恒定** 的复权因子 k（仅在除权日跳变）。因此

    .. math:: k \\approx \\mathrm{median}\\big(\\text{typical\\_price}_{adj} / (amount/volume)\\big)

    ``method="rolling"``（默认，严格 PIT）：取**过去 window 天**的滚动中位数，
    任何时点只用历史数据。

    ``method="segment"``（快但**有前视**）：先用 2% 跳变切段、取**段内**中位数。
    段内中位数含段后半段信息，实测会把 ``alpha_013`` 的 IC 从 0.029 虚高到 0.099
    （见 ``docs/ALPHA191.md`` §6），仅供复现问题用，不要用于结论。
    """
    from . import config as C
    cache_path = Path(cache_path) if cache_path else (
        C.BUILD / "alpha191" / f"vwap_{method}{window if method == 'rolling' else ''}.npy")
    if cache_path.exists():
        v = np.load(cache_path)
        if v.shape == (md.T, md.N):
            return v

    amt = np.asarray(md.amount, dtype=np.float64)
    vol = np.asarray(md.volume, dtype=np.float64)
    close = np.asarray(md.close, dtype=np.float64)
    tp = (np.asarray(md.high, np.float64) + np.asarray(md.low, np.float64) + close) / 3.0

    with np.errstate(invalid="ignore", divide="ignore"):
        vp = np.where(vol > 0, amt / np.where(vol > 0, vol, 1.0), NAN)
    ok = np.isfinite(vp) & (vp > 0) & np.isfinite(tp)
    r = np.where(ok, tp / np.where(ok, vp, 1.0), NAN)

    T, N = r.shape
    if method == "rolling":
        k = pd.DataFrame(r).rolling(int(window), min_periods=max(5, int(window) // 4)).median().to_numpy()
    else:
        R = pd.DataFrame(r)
        jump = (R.pct_change().abs() > 0.02).to_numpy()
        jump = np.where(np.isfinite(jump), jump, False)
        seg = np.cumsum(jump.astype(np.int64), axis=0)
        key = seg.ravel() * N + np.tile(np.arange(N), T)
        flat = pd.DataFrame({"k": key, "v": r.ravel()})
        k = flat.groupby("k")["v"].transform("median").to_numpy().reshape(T, N)

    # 段内全 NaN → 用全样本中位数兜底，再不行用 1
    fallback = pd.DataFrame(r).median(axis=0).to_numpy()
    k = np.where(np.isfinite(k), k, fallback[None, :])
    k = np.where(np.isfinite(k) & (k > 0), k, 1.0)

    vwap = vp * k
    vwap = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close)

    ratio = vwap / np.where(tp > 0, tp, NAN)
    bad = ~np.isfinite(ratio) | (ratio < 0.5) | (ratio > 2.0)
    if verbose:
        print(f"[vwap:{method}] 与典型价偏离>50% 的比例 {bad.mean():.4%} → 退回典型价")
    vwap = np.where(bad, tp, vwap)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vwap.astype(np.float32))
    return vwap


# ------------------------------------------------------------------ 输入面板
def build_inputs(md, vwap=None, verbose: bool = True) -> dict:
    """构造 Alpha191 的表达式命名空间（基础面板 + 算子）。

    所有基础面板一律存 **float32**：2440×5281 的一份 float64 是 103MB，
    全流程要同时活十几个面板，16GB 内存的机器上 float64 会 OOM。
    算子内部再按需升 float64（``shift`` 保留入参 dtype，避免多余拷贝）。
    """
    f32 = np.float32
    vwap = build_vwap(md, verbose=verbose) if vwap is None else vwap
    ns = {
        "OPEN": np.asarray(md.open, dtype=f32),
        "HIGH": np.asarray(md.high, dtype=f32),
        "LOW": np.asarray(md.low, dtype=f32),
        "CLOSE": np.asarray(md.close, dtype=f32),
        "VWAP": np.asarray(vwap, dtype=f32),
        "VOL": np.asarray(md.volume, dtype=f32) * f32(1e-6),
        "AMT": np.asarray(md.amount, dtype=f32) * f32(1e-8),
    }
    del vwap
    o, h, l, c = ns["OPEN"], ns["HIGH"], ns["LOW"], ns["CLOSE"]
    with np.errstate(all="ignore"):
        dc1 = op.shift(c, 1)
        do1 = op.shift(o, 1)
        ns["RET"] = (c / dc1 - 1.0).astype(f32)
        ns.update({
            "DC1": dc1, "DO1": do1,
            "DH1": op.shift(h, 1), "DL1": op.shift(l, 1),
            "TR": np.maximum(np.maximum(h - l, ABS(h - dc1)), ABS(l - dc1)),
            "LD": op.shift(l, 1) - l,
            "HD": h - op.shift(h, 1),
            "DTM": IFF(o <= do1, f32(0.0), np.maximum(o - l, o - do1)),
            "DBM": IFF(o >= do1, f32(0.0), np.maximum(o - h, o - do1)),
            "TP": (h + l + c) / f32(3.0),
        })
        ns.update(_operators(ns))
        _build_helpers(ns, c, dc1)
    ns["IFF"] = IFF
    return ns


def _build_helpers(ns: dict, close: np.ndarray, dc1: np.ndarray) -> None:
    f32 = np.float32
    ns["RSV12"] = (op.sma(np.maximum(close - dc1, 0.0), 12, 1)
                   / op.sma(np.abs(close - dc1), 12, 1) * 100).astype(f32)
    ns["SMA3LOG"] = op.sma(op.sma(op.sma(np.log(close), 13, 2), 13, 2), 13, 2).astype(f32)
    _a = op.roll_sum(np.where((ns["LD"] > 0) & (ns["LD"] > ns["HD"]), ns["LD"], 0.0), 14) \
        * 100 / op.roll_sum(ns["TR"], 14)
    _b = op.roll_sum(np.where((ns["HD"] > 0) & (ns["HD"] > ns["LD"]), ns["HD"], 0.0), 14) \
        * 100 / op.roll_sum(ns["TR"], 14)
    ns["DMI_X"] = op.roll_mean(np.abs(_a - _b) / (_a + _b) * 100, 6).astype(f32)


def _operators(ns: dict) -> dict:
    """算子命名空间（表达式里直接调用大写函数名）。"""
    return {
        "DL": op.shift, "DLT": op.delta,
        "SUM": op.roll_sum, "MEAN": op.roll_mean, "STD": op.roll_std,
        "CORR": op.roll_corr, "COV": op.roll_cov,
        "TSMAX": op.ts_max, "TSMIN": op.ts_min,
        "TSRANK": op.ts_rank, "DECAY": op.decay, "WMA": op.wma, "SMA": op.sma,
        "RANK": op.rank_pct, "COUNT": op.count_true, "SUMIF": op.sum_where,
        "LOWDAY": op.ts_argmin, "HIGHDAY": op.ts_argmax,
        "PROD": op.prod, "REGBETA": op.sequence_beta,
        "IFF": IFF, "MAX": np.maximum, "MIN": np.minimum,
        "ABS": ABS, "LOG": LOG, "SIGN": np.sign, "SQRT": SQRT, "POW": POW,
        "NAN": NAN,
    }


def IFF(c, a, b):
    """条件表达式（替代原文的 ``cond?a:b``），支持标量与面板广播。"""
    return np.where(c, a, b)


# ------------------------------------------------------------------ 公式表
# 说明：键为因子号，值为 (表达式, 置信度 tip)。"low" 表示原文残缺/歧义，实现为推测口径。
FORMULAS: dict[int, str] = {
    1: "-CORR(RANK(DLT(LOG(VOL),1)), RANK((CLOSE-OPEN)/OPEN), 6)",
    2: "-DLT(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW), 1)",
    3: "SUM(IFF(CLOSE==DC1, 0.0, CLOSE-IFF(CLOSE>DC1, MIN(LOW,DC1), MAX(HIGH,DC1))), 6)",
    4: "IFF((SUM(CLOSE,8)/8+STD(CLOSE,8)) < SUM(CLOSE,2)/2, -1.0, "
       "IFF(SUM(CLOSE,2)/2 < (SUM(CLOSE,8)/8-STD(CLOSE,8)), 1.0, "
       "IFF(VOL/MEAN(VOL,20) >= 1, 1.0, -1.0)))",
    5: "-TSMAX(CORR(TSRANK(VOL,5), TSRANK(HIGH,5), 5), 3)",
    6: "-RANK(SIGN(DLT(OPEN*0.85+HIGH*0.15, 4)))",
    7: "(RANK(MAX(VWAP-CLOSE,3)) + RANK(MIN(VWAP-CLOSE,3))) * RANK(DLT(VOL,3))",
    8: "-RANK(DLT(((HIGH+LOW)/2)*0.2 + VWAP*0.8, 4))",
    9: "SMA(((HIGH+LOW)/2 - (DH1+DL1)/2)*(HIGH-LOW)/VOL, 7, 2)",
    10: "RANK(MAX(IFF(RET<0, STD(RET,20), CLOSE)**2, 5))",
    11: "SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOL, 6)",
    12: "RANK(OPEN - SUM(VWAP,10)/10) * (-RANK(ABS(CLOSE-VWAP)))",
    13: "SQRT(HIGH*LOW) - VWAP",
    14: "CLOSE - DL(CLOSE,5)",
    15: "OPEN/DC1 - 1",
    16: "-TSMAX(RANK(CORR(RANK(VOL), RANK(VWAP), 5)), 5)",
    17: "RANK(VWAP-MAX(VWAP,15)) ** DLT(CLOSE,5)",
    18: "CLOSE/DL(CLOSE,5)",
    19: "IFF(CLOSE<DL(CLOSE,5), (CLOSE-DL(CLOSE,5))/DL(CLOSE,5), "
        "IFF(CLOSE==DL(CLOSE,5), 0.0, (CLOSE-DL(CLOSE,5))/CLOSE))",
    20: "(CLOSE-DL(CLOSE,6))/DL(CLOSE,6)*100",
    21: "REGBETA(MEAN(CLOSE,6), 6)",
    22: "SMA((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6) "
        "- DL((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3), 12, 1)",
    23: "SMA(IFF(CLOSE>DC1, STD(CLOSE,20), 0.0),20,1) / "
        "(SMA(IFF(CLOSE>DC1, STD(CLOSE,20), 0.0),20,1) "
        "+ SMA(IFF(CLOSE<=DC1, STD(CLOSE,20), 0.0),20,1)) * 100",
    24: "SMA(CLOSE-DL(CLOSE,5), 5, 1)",
    25: "-RANK(DLT(CLOSE,7)*(1-RANK(DECAY(VOL/MEAN(VOL,20),9)))) * (1+RANK(SUM(RET,250)))",
    26: "(SUM(CLOSE,7)/7 - CLOSE) + CORR(VWAP, DL(CLOSE,5), 230)",
    27: "WMA((CLOSE-DL(CLOSE,3))/DL(CLOSE,3)*100 + (CLOSE-DL(CLOSE,6))/DL(CLOSE,6)*100, 12)",
    28: "3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1) "
        "- 2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)",
    29: "(CLOSE-DL(CLOSE,6))/DL(CLOSE,6)*VOL",
    31: "(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100",
    32: "-SUM(RANK(CORR(RANK(HIGH), RANK(VOL), 3)), 3)",
    33: "((-TSMIN(LOW,5)) + DL(TSMIN(LOW,5),5)) * RANK((SUM(RET,240)-SUM(RET,20))/220) * TSRANK(VOL,5)",
    34: "MEAN(CLOSE,12)/CLOSE",
    35: "-MIN(RANK(DECAY(DLT(OPEN,1),15)), RANK(DECAY(CORR(VOL, OPEN, 17), 7)))",
    36: "RANK(SUM(CORR(RANK(VOL), RANK(VWAP), 6), 2))",
    37: "-RANK(SUM(OPEN,5)*SUM(RET,5) - DL(SUM(OPEN,5)*SUM(RET,5), 10))",
    38: "IFF(SUM(HIGH,20)/20 < HIGH, -DLT(HIGH,2), 0.0)",
    39: "-(RANK(DECAY(DLT(CLOSE,2),8)) "
        "- RANK(DECAY(CORR(VWAP*0.3+OPEN*0.7, SUM(MEAN(VOL,180),37), 14), 12)))",
    40: "SUM(IFF(CLOSE>DC1, VOL, 0.0),26)/SUM(IFF(CLOSE<=DC1, VOL, 0.0),26)*100",
    41: "-RANK(MAX(DLT(VWAP,3), 5))",
    42: "-RANK(STD(HIGH,10)) * CORR(HIGH, VOL, 10)",
    43: "SUM(IFF(CLOSE>DC1, VOL, IFF(CLOSE<DC1, -VOL, 0.0)), 6)",
    44: "TSRANK(DECAY(CORR(LOW, MEAN(VOL,10), 7), 6), 4) "
        "+ TSRANK(DECAY(DLT(VWAP,3), 10), 15)",
    45: "RANK(DLT(CLOSE*0.6+OPEN*0.4, 1)) * RANK(CORR(VWAP, MEAN(VOL,150), 15))",
    46: "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)",
    47: "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 9, 1)",
    48: "-(RANK(SIGN(CLOSE-DC1) + SIGN(DL(CLOSE,1)-DL(CLOSE,2)) + SIGN(DL(CLOSE,2)-DL(CLOSE,3))) "
        "* SUM(VOL,5)/SUM(VOL,20))",
    49: "SUM(IFF(HIGH+LOW >= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12) / "
        "(SUM(IFF(HIGH+LOW >= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12) "
        "+ SUM(IFF(HIGH+LOW <= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12))",
    50: "SUM(IFF(HIGH+LOW <= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12) / "
        "(SUM(IFF(HIGH+LOW <= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12) "
        "+ SUM(IFF(HIGH+LOW >= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12)) "
        "- SUM(IFF(HIGH+LOW >= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12) / "
        "(SUM(IFF(HIGH+LOW >= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12) "
        "+ SUM(IFF(HIGH+LOW <= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12))",
    51: "SUM(IFF(HIGH+LOW <= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12) / "
        "(SUM(IFF(HIGH+LOW <= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12) "
        "+ SUM(IFF(HIGH+LOW >= DH1+DL1, 0.0, MAX(ABS(HIGH-DH1), ABS(LOW-DL1))), 12))",
    52: "SUM(MAX(0.0, HIGH-DL(TP,1)), 26)/SUM(MAX(0.0, DL(TP,1)-LOW), 26)*100",
    53: "COUNT(CLOSE>DC1, 12)/12*100",
    55: "SUM(16*(CLOSE-DC1 + (CLOSE-OPEN)/2 + DC1-DO1) / "
        "IFF((ABS(HIGH-DC1) > ABS(LOW-DC1)) & (ABS(HIGH-DC1) > ABS(HIGH-LOW)), "
        "ABS(HIGH-DC1) + ABS(LOW-DC1)/2 + ABS(DC1-DO1)/4, "
        "IFF((ABS(LOW-DC1) > ABS(HIGH-LOW)) & (ABS(LOW-DC1) > ABS(HIGH-DC1)), "
        "ABS(LOW-DC1) + ABS(HIGH-DC1)/2 + ABS(DC1-DO1)/4, "
        "ABS(HIGH-LOW) + ABS(DC1-DO1)/4)) "
        "* MAX(ABS(HIGH-DC1), ABS(LOW-DC1)), 20)",
    56: "RANK(OPEN-TSMIN(OPEN,12)) < RANK(CORR(SUM((HIGH+LOW)/2,19), SUM(MEAN(VOL,40),19), 13)**5)",
    57: "SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100, 3, 1)",
    58: "COUNT(CLOSE>DC1, 20)/20*100",
    59: "SUM(IFF(CLOSE==DC1, 0.0, CLOSE-IFF(CLOSE>DC1, MIN(LOW,DC1), MAX(HIGH,DC1))), 20)",
    60: "SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOL, 20)",
    61: "-MAX(RANK(DECAY(DLT(VWAP,1),12)), RANK(DECAY(RANK(CORR(LOW, MEAN(VOL,80), 8)),17)))",
    62: "-CORR(HIGH, RANK(VOL), 5)",
    63: "SMA(MAX(CLOSE-DC1,0.0),6,1)/SMA(ABS(CLOSE-DC1),6,1)*100",
    64: "-MAX(RANK(DECAY(CORR(RANK(VWAP), RANK(VOL), 4), 4)), "
        "RANK(DECAY(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOL,60)), 4), 13), 14)))",
    65: "MEAN(CLOSE,6)/CLOSE",
    66: "(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100",
    67: "SMA(MAX(CLOSE-DC1,0.0),24,1)/SMA(ABS(CLOSE-DC1),24,1)*100",
    68: "SMA(((HIGH+LOW)/2 - (DH1+DL1)/2)*(HIGH-LOW)/VOL, 15, 2)",
    69: "IFF(SUM(DTM,20) > SUM(DBM,20), (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20), "
        "IFF(SUM(DTM,20) == SUM(DBM,20), 0.0, (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))",
    70: "STD(AMT, 6)",
    71: "(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100",
    72: "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 15, 1)",
    73: "-(TSRANK(DECAY(DECAY(CORR(CLOSE, VOL, 10), 16), 4), 5) "
        "- RANK(DECAY(CORR(VWAP, MEAN(VOL,30), 4), 3)))",
    74: "RANK(CORR(SUM(LOW*0.35+VWAP*0.65,20), SUM(MEAN(VOL,40),20), 7)) "
        "+ RANK(CORR(RANK(VWAP), RANK(VOL), 6))",
    76: "STD(ABS(CLOSE/DC1-1)/VOL, 20)/MEAN(ABS(CLOSE/DC1-1)/VOL, 20)",
    77: "MIN(RANK(DECAY(((HIGH+LOW)/2 + HIGH) - (VWAP+HIGH), 20)), "
        "RANK(DECAY(CORR((HIGH+LOW)/2, MEAN(VOL,40), 3), 6)))",
    78: "(TP - MEAN(TP,12))/(0.015*MEAN(ABS(CLOSE-MEAN(TP,12)),12))",
    79: "SMA(MAX(CLOSE-DC1,0.0),12,1)/SMA(ABS(CLOSE-DC1),12,1)*100",
    80: "(VOL-DL(VOL,5))/DL(VOL,5)*100",
    81: "SMA(VOL,21,2)",
    82: "SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100, 20, 1)",
    83: "-RANK(COV(RANK(HIGH), RANK(VOL), 5))",
    84: "SUM(IFF(CLOSE>DC1, VOL, IFF(CLOSE<DC1, -VOL, 0.0)), 20)",
    85: "TSRANK(VOL/MEAN(VOL,20), 20) * TSRANK(-DLT(CLOSE,7), 8)",
    86: "IFF(0.25 < ((DL(CLOSE,20)-DL(CLOSE,10))/10 - (DL(CLOSE,10)-CLOSE)/10), -1.0, "
        "IFF(((DL(CLOSE,20)-DL(CLOSE,10))/10 - (DL(CLOSE,10)-CLOSE)/10) < 0, 1.0, -DLT(CLOSE,1)))",
    87: "-(RANK(DECAY(DLT(VWAP,4),7)) "
        "+ TSRANK(DECAY((LOW - VWAP)/(OPEN-(HIGH+LOW)/2), 11), 7))",
    88: "(CLOSE-DL(CLOSE,20))/DL(CLOSE,20)*100",
    89: "2*(SMA(CLOSE,13,2) - SMA(CLOSE,27,2) - SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))",
    90: "-RANK(CORR(RANK(VWAP), RANK(VOL), 5))",
    91: "-RANK(CLOSE-MAX(CLOSE,5)) * RANK(CORR(MEAN(VOL,40), LOW, 5))",
    92: "-MAX(RANK(DECAY(DLT(CLOSE*0.35+VWAP*0.65,2),3)), "
        "TSRANK(DECAY(ABS(CORR(MEAN(VOL,180), CLOSE, 13)), 5), 15))",
    93: "SUM(IFF(OPEN>=DO1, 0.0, MAX(OPEN-LOW, OPEN-DO1)), 20)",
    94: "SUM(IFF(CLOSE>DC1, VOL, IFF(CLOSE<DC1, -VOL, 0.0)), 30)",
    95: "STD(AMT, 20)",
    96: "SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100, 3, 1), 3, 1)",
    97: "STD(VOL, 10)",
    98: "IFF(DLT(MEAN(CLOSE,100),100)/DL(CLOSE,100) <= 0.05, "
        "-(CLOSE-TSMIN(CLOSE,100)), -DLT(CLOSE,3))",
    99: "-RANK(COV(RANK(CLOSE), RANK(VOL), 5))",
    100: "STD(VOL, 20)",
    102: "SMA(MAX(VOL-DL(VOL,1),0.0),6,1)/SMA(ABS(VOL-DL(VOL,1)),6,1)*100",
    103: "(20-LOWDAY(LOW,20))/20*100",
    104: "-DLT(CORR(HIGH, VOL, 5), 5) * RANK(STD(CLOSE,20))",
    105: "-CORR(RANK(OPEN), RANK(VOL), 10)",
    106: "CLOSE - DL(CLOSE,20)",
    107: "-RANK(OPEN-DH1) * RANK(OPEN-DC1) * RANK(OPEN-DL1)",
    108: "-(RANK(HIGH-MIN(HIGH,2)) ** RANK(CORR(VWAP, MEAN(VOL,120), 6)))",
    109: "SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)",
    110: "SUM(MAX(0.0, HIGH-DC1),20)/SUM(MAX(0.0, DC1-LOW),20)*100",
    111: "SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2) "
         "- SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)",
    112: "(SUM(IFF(CLOSE>DC1, CLOSE-DC1, 0.0),12) - SUM(IFF(CLOSE<DC1, ABS(CLOSE-DC1), 0.0),12)) / "
         "(SUM(IFF(CLOSE>DC1, CLOSE-DC1, 0.0),12) + SUM(IFF(CLOSE<DC1, ABS(CLOSE-DC1), 0.0),12)) * 100",
    113: "-RANK(SUM(DL(CLOSE,5),20)/20) * CORR(CLOSE, VOL, 2) * RANK(CORR(SUM(CLOSE,5), SUM(CLOSE,20), 2))",
    114: "RANK(DL((HIGH-LOW)/(SUM(CLOSE,5)/5),2)) * RANK(RANK(VOL)) / "
         "((HIGH-LOW)/(SUM(CLOSE,5)/5)/(VWAP-CLOSE))",
    115: "RANK(CORR(HIGH*0.9+CLOSE*0.1, MEAN(VOL,30), 10)) "
         "** RANK(CORR(TSRANK((HIGH+LOW)/2,4), TSRANK(VOL,10), 7))",
    116: "REGBETA(CLOSE, 20)",
    117: "TSRANK(VOL,32) * (1-TSRANK(CLOSE+HIGH-LOW,16)) * (1-TSRANK(RET,32))",
    118: "SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100",
    119: "RANK(DECAY(CORR(VWAP, SUM(MEAN(VOL,5),26), 5), 7)) "
         "- RANK(DECAY(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOL,15)), 21), 9), 7), 8))",
    120: "RANK(VWAP-CLOSE)/RANK(VWAP+CLOSE)",
    121: "-(RANK(VWAP-TSMIN(VWAP,12)) ** TSRANK(CORR(TSRANK(VWAP,20), TSRANK(MEAN(VOL,60),2), 18), 3))",
    122: "(SMA3LOG - DL(SMA3LOG,1))/DL(SMA3LOG,1)",
    124: "(CLOSE-VWAP)/DECAY(RANK(TSMAX(CLOSE,30)), 2)",
    125: "RANK(DECAY(CORR(VWAP, MEAN(VOL,80), 17), 20)) "
         "/ RANK(DECAY(DLT(CLOSE*0.5+VWAP*0.5, 3), 16))",
    126: "(CLOSE+HIGH+LOW)/3",
    127: "SQRT(MEAN((100*(CLOSE-MAX(CLOSE,12))/MAX(CLOSE,12))**2, 12))",
    128: "100 - 100/(1 + SUM(IFF(TP > DL(TP,1), TP*VOL, 0.0),14) "
         "/ SUM(IFF(TP < DL(TP,1), TP*VOL, 0.0),14))",
    129: "SUM(IFF(CLOSE<DC1, ABS(CLOSE-DC1), 0.0), 12)",
    130: "RANK(DECAY(CORR((HIGH+LOW)/2, MEAN(VOL,40), 9), 10)) "
         "/ RANK(DECAY(CORR(RANK(VWAP), RANK(VOL), 7), 3))",
    131: "RANK(DLT(VWAP,1)) ** TSRANK(CORR(CLOSE, MEAN(VOL,50), 18), 18)",
    132: "MEAN(AMT, 20)",
    133: "(20-HIGHDAY(HIGH,20))/20*100 - (20-LOWDAY(LOW,20))/20*100",
    134: "(CLOSE-DL(CLOSE,12))/DL(CLOSE,12)*VOL",
    135: "-SMA(DL(CLOSE/DL(CLOSE,20),1), 20, 1)",
    136: "-RANK(DLT(RET,3)) * CORR(OPEN, VOL, 10)",
    137: "16*(CLOSE-DC1 + (CLOSE-OPEN)/2 + DC1-DO1) / "
         "IFF((ABS(HIGH-DC1) > ABS(LOW-DC1)) & (ABS(HIGH-DC1) > ABS(HIGH-LOW)), "
         "ABS(HIGH-DC1) + ABS(LOW-DC1)/2 + ABS(DC1-DO1)/4, "
         "IFF((ABS(LOW-DC1) > ABS(HIGH-LOW)) & (ABS(LOW-DC1) > ABS(HIGH-DC1)), "
         "ABS(LOW-DC1) + ABS(HIGH-DC1)/2 + ABS(DC1-DO1)/4, "
         "ABS(HIGH-LOW) + ABS(DC1-DO1)/4)) * MAX(ABS(HIGH-DC1), ABS(LOW-DC1))",
    138: "-(RANK(DECAY(DLT(LOW*0.7+VWAP*0.3, 3), 20)) "
         "- TSRANK(DECAY(TSRANK(CORR(TSRANK(LOW,8), TSRANK(MEAN(VOL,60),17), 5), 19), 16), 7))",
    139: "-CORR(OPEN, VOL, 10)",
    140: "MIN(RANK(DECAY(RANK(OPEN)+RANK(LOW)-RANK(HIGH)-RANK(CLOSE), 8)), "
         "TSRANK(DECAY(CORR(TSRANK(CLOSE,8), TSRANK(MEAN(VOL,60),20), 8), 7), 3))",
    141: "-RANK(CORR(RANK(HIGH), RANK(MEAN(VOL,15)), 9))",
    142: "-RANK(TSRANK(CLOSE,10)) * RANK(DLT(DLT(CLOSE,1),1)) * RANK(TSRANK(VOL/MEAN(VOL,20), 5))",
    144: "SUMIF(ABS(CLOSE/DC1-1)/AMT, 20, CLOSE<DC1)/COUNT(CLOSE<DC1, 20)",
    145: "(MEAN(VOL,9)-MEAN(VOL,26))/MEAN(VOL,12)*100",
    146: "MEAN(RET-SMA(RET,61,2),20) * (RET-SMA(RET,61,2)) / SMA((RET-SMA(RET,61,2))**2, 60)",
    147: "REGBETA(MEAN(CLOSE,12), 12)",
    148: "-IFF(RANK(CORR(OPEN, SUM(MEAN(VOL,60),9), 6)) < RANK(OPEN-TSMIN(OPEN,14)), 1.0, 0.0)",
    150: "(CLOSE+HIGH+LOW)/3*VOL",
    151: "SMA(CLOSE-DL(CLOSE,20), 20, 1)",
    152: "SMA(MEAN(DL(SMA(DL(CLOSE/DL(CLOSE,9),1), 9, 1), 1), 12) "
         "- MEAN(DL(SMA(DL(CLOSE/DL(CLOSE,9),1), 9, 1), 1), 26), 9, 1)",
    153: "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4",
    154: "IFF(VWAP-TSMIN(VWAP,16) < CORR(VWAP, MEAN(VOL,180), 18), 1.0, 0.0)",
    155: "SMA(VOL,13,2) - SMA(VOL,27,2) - SMA(SMA(VOL,13,2)-SMA(VOL,27,2), 10, 2)",
    156: "-MAX(RANK(DECAY(DLT(VWAP,5),3)), "
         "RANK(DECAY(DLT(OPEN*0.15+LOW*0.85,2)/(OPEN*0.15+LOW*0.85), 3)))",
    157: "MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK(-RANK(DLT(CLOSE-1,5)))),2),1)))),5), 1.0) "
         "+ TSRANK(DL(-RET,6), 5)",
    158: "((HIGH-SMA(CLOSE,15,2)) - (LOW-SMA(CLOSE,15,2)))/CLOSE",
    159: "((CLOSE-SUM(MIN(LOW,DC1),6))/SUM(MAX(HIGH,DC1)-MIN(LOW,DC1),6)*12*24 "
         "+ (CLOSE-SUM(MIN(LOW,DC1),12))/SUM(MAX(HIGH,DC1)-MIN(LOW,DC1),12)*6*24 "
         "+ (CLOSE-SUM(MIN(LOW,DC1),24))/SUM(MAX(HIGH,DC1)-MIN(LOW,DC1),24)*24*6) "
         "* 100/(6*12+6*24+12*24)",
    160: "SMA(IFF(CLOSE<=DC1, STD(CLOSE,20), 0.0), 20, 1)",
    161: "MEAN(MAX(MAX(HIGH-LOW, ABS(DC1-HIGH)), ABS(DC1-LOW)), 12)",
    162: "(RSV12 - MIN(RSV12,12))/(MAX(RSV12,12) - MIN(RSV12,12))",
    163: "RANK(((-RET)*MEAN(VOL,20)*VWAP) * (HIGH-CLOSE))",
    164: "SMA((IFF(CLOSE>DC1, 1/(CLOSE-DC1), 1.0) - MIN(IFF(CLOSE>DC1, 1/(CLOSE-DC1), 1.0), 12)) "
         "/(HIGH-LOW)*100, 13, 2)",
    166: "-20*19**1.5 * SUM(RET-MEAN(RET,20),20) / ((19*18) * SUM(RET**2,20)**1.5)",
    167: "SUM(IFF(CLOSE>DC1, CLOSE-DC1, 0.0), 12)",
    168: "-VOL/MEAN(VOL,20)",
    169: "SMA(MEAN(DL(SMA(CLOSE-DC1,9,1),1),12) - MEAN(DL(SMA(CLOSE-DC1,9,1),1),26), 10, 1)",
    170: "RANK(1/CLOSE)*VOL/MEAN(VOL,20) * (HIGH*RANK(HIGH-CLOSE))/(SUM(HIGH,5)/5) "
         "- RANK(VWAP-DL(VWAP,5))",
    171: "(-(LOW-CLOSE)*OPEN**5) / ((CLOSE-HIGH)*CLOSE**5)",
    172: "DMI_X",
    173: "3*SMA(CLOSE,13,2) - 2*SMA(SMA(CLOSE,13,2),13,2) + SMA3LOG",
    174: "SMA(IFF(CLOSE>DC1, STD(CLOSE,20), 0.0), 20, 1)",
    175: "MEAN(MAX(MAX(HIGH-LOW, ABS(DC1-HIGH)), ABS(DC1-LOW)), 6)",
    176: "CORR(RANK((CLOSE-TSMIN(LOW,12))/(TSMAX(HIGH,12)-TSMIN(LOW,12))), RANK(VOL), 6)",
    177: "(20-HIGHDAY(HIGH,20))/20*100",
    178: "(CLOSE-DC1)/DC1*VOL",
    179: "RANK(CORR(VWAP, VOL, 4)) * RANK(CORR(RANK(LOW), RANK(MEAN(VOL,50)), 12))",
    180: "IFF(MEAN(VOL,20) < VOL, -TSRANK(ABS(DLT(CLOSE,7)),60)*SIGN(DLT(CLOSE,7)), -VOL)",
    184: "RANK(CORR(DL(OPEN-CLOSE,1), CLOSE, 200)) + RANK(OPEN-CLOSE)",
    185: "RANK(-((1-OPEN/CLOSE)**2))",
    186: "(DMI_X + DL(DMI_X,6))/2",
    187: "SUM(IFF(OPEN<=DO1, 0.0, MAX(HIGH-OPEN, OPEN-DO1)), 20)",
    188: "((HIGH-LOW - SMA(HIGH-LOW,11,2)) / SMA(HIGH-LOW,11,2))*100",
    189: "MEAN(ABS(CLOSE-MEAN(CLOSE,6)), 6)",
    190: "LOG((COUNT(RET > (CLOSE/DL(CLOSE,19))**(1/20)-1, 20) - 1) "
         "* SUMIF((RET - ((CLOSE/DL(CLOSE,19))**(1/20)-1))**2, 20, RET < (CLOSE/DL(CLOSE,19))**(1/20)-1) "
         "/ ((COUNT(RET < (CLOSE/DL(CLOSE,19))**(1/20)-1, 20)) "
         "* SUMIF((RET - ((CLOSE/DL(CLOSE,19))**(1/20)-1))**2, 20, RET > (CLOSE/DL(CLOSE,19))**(1/20)-1)))",
    191: "CORR(MEAN(VOL,20), LOW, 5) + (HIGH+LOW)/2 - CLOSE",
}

# 明显不该给 .replace 处理的名字（避免误替换），先做一次自检
_LOW_CONF = {4, 9, 56, 78, 101, 127, 146, 148, 154, 157, 159, 162, 172, 186, 190}

EXCLUDED: dict[int, str] = {
    30: "需要 Fama-French 三因子（MKT/SMB/HML）日频数据，本项目未接入",
    54: "公式 `STD(ABS(CLOSE-OPEN))` 缺少窗口参数，无法确定口径",
    75: "仅依赖基准指数，横截面无差异（等于常数 1）",
    101: "聚宽原文公式被截断（以 `\\` 结尾）",
    123: "聚宽原文公式被截断（以 `\\` 结尾）",
    143: "含递归变量 SELF（引用前一日自身值），需逐日递归，且原文未给初值",
    149: "需按基准指数涨跌过滤样本后做 252 日回归，口径不唯一",
    165: "使用 SUMAC（累计求和算子），聚宽标注「尚未实现」",
    181: "基准指数参与幂次运算，原文口径可疑",
    182: "需基准指数的日内涨跌方向，缺基准分钟/日线口径（且横截面仅有 0/0.5/1 三档）",
    183: "使用 SUMAC，聚宽标注「尚未实现」",
}


def available() -> list[int]:
    return sorted(FORMULAS)


def compute_one(k: int, ns: dict) -> np.ndarray:
    """计算单个因子；入参为 :func:`build_inputs` 的命名空间。

    公式表是静态可信内容，但 eval 期间必须屏蔽 numpy 的浮点告警：
    ``eval`` 的 globals 里没有 ``__builtins__`` 时，warnings 模块取
    ``__import__`` 会抛 ``KeyError``，所以这里同时用 ``errstate`` + 忽略告警。
    """
    if k not in FORMULAS:
        raise KeyError(f"alpha_{k:03d} 未实现：{EXCLUDED.get(k, '未知')}")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = eval(FORMULAS[k], {"__builtins__": {}}, ns)  # noqa: S307
    return np.asarray(v, dtype=np.float64)


def compute(md, codes=None, vwap=None, verbose: bool = True) -> tuple[dict, dict]:
    """计算全部可实现的因子。

    返回 ``(panels, errors)``：``panels`` 为 ``{因子号: (T,N) float32}``，
    ``errors`` 为 ``{因子号: 异常信息}``。逐因子独立计算，单个失败不影响其余。
    """
    ns = build_inputs(md, vwap=vwap, verbose=verbose)
    panels, errors = {}, {}
    for k in available():
        try:
            a = compute_one(k, ns)
            panels[k] = a.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            errors[k] = f"{type(exc).__name__}: {exc}"
    return panels, errors


def compute_panel(md, ks, rows=None, vwap=None, verbose: bool = False) -> dict[int, np.ndarray]:
    """按需计算若干因子，取指定行（默认全部），返回 ``{因子号: (len(rows), N)}``。

    用于把 Alpha191 接到模型/评估流程里：只算需要的那几十个，比全量快得多。
    """
    ns = build_inputs(md, vwap=vwap, verbose=verbose)
    out = {}
    for k in ks:
        x = compute_one(k, ns)
        out[k] = x if rows is None else x[np.asarray(rows)]
    return out
