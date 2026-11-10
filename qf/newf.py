# -*- coding: utf-8 -*-
"""新因子族（换手 / 流动性 / 量价形态 / 隔夜）的**算式单一来源**。

为什么要单独成模块
------------------
2026-11-03 把换手族（``newf-extra:N001..N004``）接进生产池后，``daily_signal``
直接崩在 ``qf/model.py: build_live_features`` 的 ``cat.loc[k]``：
因子池里有 ``newf-extra:`` 前缀，但基础 catalog 里没有这些键、也没有对应的
实时计算分支 → ``KeyError: 'newf-extra:N001'``，**今日信号完全出不来**。

教训：能进池的因子族必须同时具备
  1. 历史面板（供 walk-forward）
  2. 最新一日行（供 live_scores）
两者若各写一份算式，实盘打分与训练打分的口径会悄悄分叉——这是最难查的一类
bug（回测漂亮、实盘对不上）。所以这里只留一份 ``build_panels``，
``scripts/build_newf_extra.py`` 落盘、``qf.model.build_live_features`` 取末行，
都调它。

数据口径
--------
- 停牌/退市日一律 NaN（不做 ffill），避免 0 收益污染滚动统计
- 换手 = 成交额 / 总市值（``md.amount`` 与 ``md.mcap`` 单位均为元）
- 收益率用后复权价、``pct_change(fill_method=None)``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIB = "newf-extra"
PREFIX = f"{LIB}:"

# slug -> 中文名 / 英文名 / 类别 / 窗口（写 catalog 用）
SPEC: list[tuple[str, str, str, str, str, int]] = [
    ("turn5", "5日换手率", "5D turnover", "换手", "Turnover", 5),
    ("turn20", "20日换手率", "20D turnover", "换手", "Turnover", 20),
    ("turn60", "60日换手率", "60D turnover", "换手", "Turnover", 60),
    ("turn_std20", "换手率不稳定度", "20D turnover CV", "换手", "Turnover", 20),
    ("turn_accel", "换手率加速", "turnover acceleration", "换手", "Turnover", 60),
    ("amihud20", "20日Amihud非流动性", "20D Amihud illiquidity", "流动性", "Liquidity", 20),
    ("amihud_trend", "非流动性变化", "illiquidity trend 20/120", "流动性", "Liquidity", 120),
    ("pv_elastic20", "量价弹性", "volume-price elasticity 20D", "量价", "PriceVolume", 20),
    ("amt_conc20", "成交额集中度", "amount concentration 5/20", "量价", "PriceVolume", 20),
    ("on_minus_intra20", "隔夜减日内收益", "overnight minus intraday 20D", "隔夜", "Overnight", 20),
    ("on_share20", "隔夜收益占比", "overnight share 20D", "隔夜", "Overnight", 20),
    ("intra_mom20", "日内动量", "intraday momentum 20D", "隔夜", "Overnight", 20),
]

# 族 -> slug 列表（--family 切分落盘时用）
FAMILY = {
    "turnover": ["turn5", "turn20", "turn60", "turn_std20", "turn_accel"],
    "liquidity": ["amihud20", "amihud_trend"],
    "pricevolume": ["pv_elastic20", "amt_conc20"],
    "overnight": ["on_minus_intra20", "on_share20", "intra_mom20"],
}

SLUG_TO_ID = {s: f"N{i + 1:03d}" for i, (s, *_r) in enumerate(SPEC)}
SLUG_TO_NAME = {s: cn for s, cn, *_r in SPEC}


def _roll(df: pd.DataFrame, w: int, how: str = "mean", minp_ratio: float = 0.75):
    return getattr(df.rolling(w, min_periods=max(2, int(w * minp_ratio))), how)()


def build_panels(md) -> dict[str, np.ndarray]:
    """全历史面板 ``slug -> (T, N) float32``。停牌/缺失为 NaN。"""
    alive = md.alive
    px = pd.DataFrame(md.close.astype(np.float64))
    opx = pd.DataFrame(md.open.astype(np.float64))
    amt = pd.DataFrame(np.where(alive, md.amount.astype(np.float64), np.nan))
    mcap = pd.DataFrame(np.where(alive & (md.mcap > 0), md.mcap.astype(np.float64), np.nan))
    # 后复权价收益率（停牌日置 NaN，避免 ffill 造成 0 收益污染滚动统计）
    ret = px.pct_change(fill_method=None)
    ret = ret.where(alive & np.isfinite(ret) & (ret > -1.0))

    # ---------------- 换手族：成交额 / 总市值
    turn = (amt / mcap).where(lambda d: d > 0)
    F: dict[str, pd.DataFrame] = {}
    F["turn5"] = _roll(turn, 5)
    F["turn20"] = _roll(turn, 20)
    F["turn60"] = _roll(turn, 60)
    F["turn_std20"] = _roll(turn, 20, "std") / F["turn20"].where(lambda d: d > 0)
    F["turn_accel"] = F["turn5"] / F["turn60"].where(lambda d: d > 0) - 1.0

    # ---------------- 非流动性族：|r| / 成交额
    illiq = (ret.abs() / amt.where(lambda d: d > 0)) * 1e8
    F["amihud20"] = _roll(illiq, 20)
    a120 = _roll(illiq, 120)
    F["amihud_trend"] = F["amihud20"] / a120.where(lambda d: d > 0) - 1.0

    # ---------------- 量价形态族：收益对成交额变化的弹性（20 日 OLS 斜率）
    dlna = np.log(amt.where(lambda d: d > 0)).diff()
    dlna = dlna.where(np.isfinite(dlna))
    cov = ret.rolling(20, min_periods=15).cov(dlna)
    var = dlna.rolling(20, min_periods=15).var()
    F["pv_elastic20"] = cov / var.where(lambda d: d > 1e-8)
    F["amt_conc20"] = amt.rolling(5, min_periods=4).sum() / amt.rolling(20, min_periods=15).sum()

    # ---------------- 隔夜/日内族
    prev = px.shift(1)
    on = (opx / prev.where(lambda d: d > 0) - 1.0).where(alive)
    intra = (px / opx.where(lambda d: d > 0) - 1.0).where(alive)
    on_s = on.rolling(20, min_periods=15).sum()
    intra_s = intra.rolling(20, min_periods=15).sum()
    F["on_minus_intra20"] = on_s - intra_s
    F["on_share20"] = on_s / (on_s.abs() + intra_s.abs()).where(lambda d: d > 0)
    F["intra_mom20"] = intra_s
    return {k: v.to_numpy(dtype=np.float32) for k, v in F.items()}


def live_row(md) -> dict[str, np.ndarray]:
    """最新交易日行 ``factor_key -> (N,) float32``（与 build_panels 同源同口径）。"""
    panels = build_panels(md)
    return {f"{PREFIX}{SLUG_TO_ID[s]}": a[-1] for s, a in panels.items()}
