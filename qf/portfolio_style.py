# -*- coding: utf-8 -*-
"""个股层约束构建：在「分数排序 + 贪心等权」之上加**行业相对偏离**与**风格暴露**上限。

为什么单独一个模块
------------------
``qf/portfolio.py`` 是生产路径，行为已被 `tests/test_rebalance_policy.py` 等逐位锁定。
本模块只在实验里通过猴子补丁替换 ``qf.backtest.build_target``，**生产默认不启用**，
因此不会改变任何现有回测结果。

与生产版的唯一区别
------------------
生产版只有三个上限：单票 8%、单行业**绝对** 25%、双创 50%。这里的「绝对」是关键——
基准里占比 5% 的行业可以持到 25%，等于允许「整个组合压在少数行业上」。

本模块额外提供两类约束（都可单独关闭）：

1. ``ind_dev_cap``：行业权重 **相对基准** 的偏离上限（pp）。基准取
   ``md.eligible[t]`` 等权，与组合等权构建一致。
2. ``style_caps``：组合风格暴露上限（z 单位）。组合暴露定义为

       E_k = Σ_j w_j · z_kj / Σ_j w_j        （仅对 z 有限的持仓求和）

   即「持仓的加权平均 z」。约束是 ``|E_k| ≤ cap_k``。

约束不满足时的行为
------------------
**跳过该标的、继续沿排名向下扫描**（等价于真实优化器「换一只满足约束的股票」），
而不是放宽约束。代价：可能持有排名更靠后的股票，甚至持仓数不足、留更多现金。
这两种代价都会在实验里如实报告（``n_hold`` / ``invested``），不做美化。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from . import config as C
from . import portfolio as qfp

__all__ = ["make_style_builder", "industry_benchmark"]


def industry_benchmark(md, t: int) -> dict[int, float]:
    """基准行业权重：``md.eligible[t]`` 内等权。"""
    ind = np.asarray(md.industry[t])
    elig = np.asarray(md.eligible[t], dtype=bool)
    if not elig.any():
        return {}
    vals, cnts = np.unique(ind[elig], return_counts=True)
    tot = float(cnts.sum())
    return {int(v): float(c) / tot for v, c in zip(vals, cnts)}


def make_style_builder(
    md,
    style_panels: dict[str, np.ndarray] | None,
    rows: np.ndarray,
    style_caps: dict[str, float] | None = None,
    ind_dev_cap: float | None = None,
    *,
    n_holdings: int | None = None,
) -> Callable:
    """构造一个签名与 ``portfolio.build_target`` 兼容的构建函数。

    参数
    ----
    md : MarketData
    style_panels : ``{"SIZE": (len(rows), N), ...}``，来自 ``attribution.build_styles``
    rows : 与 ``style_panels`` 行对应的交易日行号（升序）；用于把回测里的 ``t`` 映射到行
    style_caps : ``{"VOL": 1.0, ...}``；``None`` 或空 = 不加风格约束
    ind_dev_cap : 行业相对偏离上限（如 0.03）；``None`` = 不加行业相对约束
    n_holdings : 固定持仓数（缺省用 cfg.N_HOLDINGS，但允许调用方覆盖）
    """
    rows = np.asarray(rows, dtype=int)
    pos = {int(t): i for i, t in enumerate(rows)}     # t -> 行号
    caps = {str(k): float(v) for k, v in (style_caps or {}).items() if v is not None}
    style_keys = sorted(caps)
    bench_cache: dict[int, dict[int, float]] = {}
    default_n = n_holdings

    def builder(md_, t: int, scores, prev_weights=None, cfg=C,
                n_holdings=None, weight_mode: str = "equal",
                max_new_entries: int | None = None) -> dict[int, float]:
        n = int(n_holdings or default_n or cfg.N_HOLDINGS)
        if n <= 0:
            return {}
        base = 1.0 / float(n)
        if base < float(cfg.MIN_WEIGHT) - 1e-12:
            raise ValueError(f"n_holdings={n} 与 MIN_WEIGHT={cfg.MIN_WEIGHT} 冲突")

        pool = qfp.selection(md_, t, scores, prev_weights, cfg=cfg, n_holdings=n)
        if not pool:
            return {}

        r = pos.get(int(t))
        z_here = None
        if style_keys and r is not None and style_panels is not None:
            z_here = {k: np.asarray(style_panels[k][r], dtype=np.float64) for k in style_keys}

        industry = np.asarray(md_.industry[t])
        growth = qfp._growth_mask(md_)
        bench = None
        if ind_dev_cap is not None:
            bench = bench_cache.get(int(t))
            if bench is None:
                bench = industry_benchmark(md_, t)
                bench_cache[int(t)] = bench

        max_w = float(cfg.MAX_WEIGHT)
        min_w = float(cfg.MIN_WEIGHT)
        ind_max = float(cfg.INDUSTRY_MAX)
        board_max = float(cfg.BOARD_MAX_GROWTH)

        prev_set = {int(j) for j, w in (prev_weights or {}).items() if float(w) > 0}
        limit = None if max_new_entries is None else max(0, int(max_new_entries))
        old_remaining = sum(1 for j in pool if int(j) in prev_set) if limit is not None else 0

        weights: dict[int, float] = {}
        ind_w: dict[int, float] = defaultdict(float)
        board_w = 0.0
        total = 0.0
        new_used = 0
        # 风格暴露的滚动分子/分母（只看 z 有限的标的）
        num = {k: 0.0 for k in style_keys}
        den = {k: 0.0 for k in style_keys}
        rejected_style = 0
        rejected_ind = 0

        for j in pool:
            if len(weights) >= n:
                break
            j = int(j)
            if j in prev_set:
                if limit is not None:
                    old_remaining -= 1
            elif limit is not None and new_used >= limit:
                if old_remaining <= 0:
                    break
                continue

            ind = int(industry[j])
            room = min(max_w,
                       ind_max - ind_w[ind],
                       board_max - board_w if bool(growth[j]) else np.inf,
                       1.0 - total)
            if bench is not None:
                # 相对基准的偏离空间：基准权重 + 容差 − 已配权重
                room = min(room, float(bench.get(ind, 0.0)) + float(ind_dev_cap) - ind_w[ind])
            w = min(base, room)
            if w < min_w - 1e-12:
                rejected_ind += 1
                continue

            ok = True
            for k in style_keys:
                z = z_here[k][j]
                if not np.isfinite(z):
                    continue
                e_new = (num[k] + w * z) / (den[k] + w)
                if abs(e_new) > caps[k] + 1e-12:
                    ok = False
                    break
            if not ok:
                rejected_style += 1
                continue

            weights[j] = float(w)
            ind_w[ind] += float(w)
            if bool(growth[j]):
                board_w += float(w)
            total += float(w)
            for k in style_keys:
                z = z_here[k][j]
                if np.isfinite(z):
                    num[k] += w * z
                    den[k] += w
            if j not in prev_set:
                new_used += 1

        total = float(sum(weights.values()))
        if total > 1.0 + 1e-12:
            weights = {j: w / total for j, w in weights.items()}
        # 诊断量挂到 builder 上，供实验脚本读取（不参与权重）
        builder.last_stats = {
            "n_hold": len(weights),
            "invested": float(sum(weights.values())),
            "rejected_style": rejected_style,
            "rejected_ind": rejected_ind,
            "style_expo": {k: (num[k] / den[k] if den[k] > 0 else float("nan"))
                           for k in style_keys},
            "constrained": bool(style_keys) or (ind_dev_cap is not None),
        }
        return weights

    builder.last_stats = {}
    return builder
