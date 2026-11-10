# -*- coding: utf-8 -*-
"""量价背离因子（ΔCorr(High, Volume, 5) × 波动加权）的**算式单一来源**。

来源
----
量化博主「国金七哥」的思路：

    Alpha = -1 × ΔCorrelation(High, Volume, 5) × Rank(StdDev(Close, 20))

- ``corr(high, volume, 5)``：过去 5 日**价格水平**与**成交量水平**的皮尔逊相关
- ``Δ``：今日值 − 5 交易日前值（负值 = 相关性在减弱 = 背离在增强）
- ``Rank(StdDev(close, 20))``：收盘价 20 日标准差的横截面排名
- 前导负号：因子值越大 = 背离越强 = 看空

为什么单独成模块
----------------
与 ``qf/newf.py`` 同一个理由：能进池的因子必须**历史面板**与**最新一日行**共用同一份算式，
否则回测漂亮、实盘对不上，是最难查的一类 bug。这里只留一份 :func:`panels`。

相对原文的两处工程处理（报告里必须披露）
----------------------------------------
1. 波动权重取 ``1 + w``（w 为 rank − 0.5，∈[−0.5, 0.5]，均值 1）而不是裸 ``w∈[0,1]``。
   裸 rank 会把当天排序靠后的一半股票压到接近 0，等于随机扔掉一半样本，且系数随日期漂。
   消融变体 ``D`` 保留原文严格写法，供对照。
2. 相关用 :func:`qf.ops.roll_corr` 的**成对缺失**口径（停牌日 NaN 自动跳出窗口），
   不做前向填充 —— 停牌价 ffill 后与 0 成交量配对会伪造「完全背离」。

数据口径
--------
- ``high`` 用 ``md.high``（停牌日 ffill，停牌期间价格视为不变）
- ``volume`` 用 ``md.volume_orig`` 且只在 ``alive`` 日为值，停牌日 NaN → 成对剔除
- 横截面排名只在 ``md.eligible`` 内做，域外 NaN
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ops

W_CORR = 5      # 相关窗口（原文 5）
W_DELTA = 5     # 差分滞后（原文「5 天前」）
W_VOL = 20      # 波动窗口（原文 20）

VARIANTS = ("A", "B", "D")
SLOW_VARIANTS = ("E", "F", "G")   # 降频版（第二次预登记），不加权，只动窗口/平滑
W_SMOOTH = 20                     # 降频版的日频平滑窗口


def _dcorr(md, w_corr: int, w_delta: int) -> np.ndarray:
    """``corr(high, volume, w_corr)`` 的 ``w_delta`` 日差分（日频，float64）。

    成对缺失口径：停牌日的 volume 是 NaN，会连同当天的 ffill 价格一起跳出窗口，
    不会伪造「零成交量」的极端相关。
    """
    high = md.high.astype(np.float64)
    vol = np.where(md.alive & np.isfinite(md.volume_orig), md.volume_orig, np.nan).astype(np.float64)
    with np.errstate(all="ignore"):
        c = ops.roll_corr(high, vol, w_corr, minp=max(3, w_corr - 1))
        return c - ops.shift(c, w_delta)


def _xs_rank_pct(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐行横截面百分位排名（pct=True，1/n ~ 1），域外 NaN。"""
    df = pd.DataFrame(np.where(mask & np.isfinite(x), x, np.nan))
    r = df.rank(axis=1, method="average", pct=True)
    return r.to_numpy(dtype=np.float32)


def raw_parts(md) -> dict[str, np.ndarray]:
    """中间件面板（诊断用，便于拆解「到底是相关变化还是波动权重在起作用」）。"""
    T, N = md.T, md.N
    high = md.high.astype(np.float64)
    vol = np.where(md.alive & np.isfinite(md.volume_orig), md.volume_orig, np.nan).astype(np.float64)
    close = md.close.astype(np.float64)

    with np.errstate(all="ignore"):
        c5 = ops.roll_corr(high, vol, W_CORR, minp=max(3, W_CORR - 1))
        d_corr = c5 - ops.shift(c5, W_DELTA)
        s20 = ops.roll_std(close, W_VOL, minp=max(10, int(W_VOL * 0.75)))
        s20_ret = ops.roll_std(ops.pct_change(close, 1), W_VOL, minp=max(10, int(W_VOL * 0.75)))
        rk = _xs_rank_pct(s20, md.eligible)          # 原文口径：对价格水平标准差排名
        rk_ret = _xs_rank_pct(s20_ret, md.eligible)  # 对照：对收益率波动排名（消除价格水平）

    return {
        "corr5": c5.astype(np.float32),
        "d_corr": d_corr.astype(np.float32),
        "std20": s20.astype(np.float32),
        "std20_ret": s20_ret.astype(np.float32),
        "rank_std20": rk,
        "rank_std20_ret": rk_ret,
    }


def panels(md) -> dict[str, np.ndarray]:
    """三个变体的日频面板 ``(T, N) float32``。

    A：原文口径（波动权重 = 1 + rank − 0.5，均值 1）
    B：消融，去掉波动加权
    D：原文严格写法（权重 = rank ∈ (0, 1]）
    """
    p = raw_parts(md)
    d = p["d_corr"]
    w = p["rank_std20"] - 0.5                       # ∈ (−0.5, 0.5]
    with np.errstate(all="ignore"):
        A = -1.0 * d * (1.0 + w)
        B = -1.0 * d
        D = -1.0 * d * p["rank_std20"]
    return {"A": A.astype(np.float32), "B": B.astype(np.float32), "D": D.astype(np.float32)}


def describe() -> dict[str, str]:
    return {
        "A": "-1 × Δcorr5(high, vol) × (1 + rank_xs(σ20) − 0.5)",
        "B": "-1 × Δcorr5(high, vol)",
        "D": "-1 × Δcorr5(high, vol) × rank_xs(σ20)",
        "E": "-1 × Δ20corr20(high, vol)",
        "F": "MA20(-1 × Δ5corr5(high, vol))",
        "G": "MA20(-1 × Δ20corr20(high, vol))",
    }


def slow_panels(md) -> dict[str, np.ndarray]:
    """降频版三变体（第二次预登记 E/F/G）的日频面板 ``(T, N) float32``。

    第一次实验的结论是「波动加权是净损失、死因在频率」，所以这里一律不加权：

    E：只把窗口从 5 拉到 20
    F：保留 5 日快信号，靠 20 日均值平滑
    G：窗口 20 + 平滑 20（最慢）

    平滑用**日频** 20 交易日（≈4 周）简单均值，``min_periods=15``；
    先取负再平滑（线性算子，顺序不影响结果，写成这样是为了和公式逐字对应）。
    """
    d5 = _dcorr(md, 5, 5)
    d20 = _dcorr(md, 20, 20)
    with np.errstate(all="ignore"):
        sm = lambda x: ops.roll_mean(x, W_SMOOTH, minp=15)
        return {
            "E": (-1.0 * d20).astype(np.float32),
            "F": sm(-1.0 * d5).astype(np.float32),
            "G": sm(-1.0 * d20).astype(np.float32),
        }
