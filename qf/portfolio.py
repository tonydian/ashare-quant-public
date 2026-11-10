# -*- coding: utf-8 -*-
"""组合构建：t 日收盘分数 → 目标权重（选股域 / 缓冲区 / 行业·板块·单票约束）。

流程（docs/INTERFACES.md §3.2）：

1. **候选池**：``md.eligible[t] & isfinite(scores[t])``（数据层已含上市天数、
   价格区间、流动性、市值、ST、停牌过滤）。
2. **排序**：分数降序（稳定排序，同分按面板下标）。
3. **缓冲区**：上一期持仓中"当期排名 < n_holdings × TURNOVER_BUFFER"的股票优先
   保留（最多 n_holdings 只），降低换手；其余名额按排名从高到低补足。
   若某些标的因约束（行业/双创/最小权重）无法分配，名额继续从后续排名补足
   （候选优先队列长度 > n_holdings，按序试探），保证尽量持满 n_holdings 只。
4. **约束（贪心等权）**：按最终顺序逐只分配 ``1/n_holdings`` 的等权权重，
   受以下上限约束：
   * 单票 ≤ ``MAX_WEIGHT``（8%），每只最低 ``MIN_WEIGHT``（2%），达不到下限
     则跳过该票、继续尝试下一名；
   * 单行业累计 ≤ ``INDUSTRY_MAX``（25%），未知行业（-1）单独一组；
   * 双创（创业板 300/301 + 科创板 688/689）累计 ≤ ``BOARD_MAX_GROWTH``（50%）。
5. 返回 ``{j: weight}``，权重和 ≤ 1（剩余为现金）。**不做整手取整**（在 execution 做）。

本函数只使用 t 日及之前的信息：``md.eligible[t]`` 是 t 日收盘口径，
``md.industry[t]`` / ``md.board`` 是静态/前向填充口径，``scores[t]`` 由模型层保证。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import config as C

__all__ = ["build_target", "selection"]

_GROWTH_BOARDS = ("STAR", "CHINEXT")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _growth_mask(md) -> np.ndarray:
    """双创板掩码（缓存到 md 对象上，避免每次重建）。"""
    gm = getattr(md, "_qf_growth_mask", None)
    if gm is None:
        gm = np.isin(np.asarray(md.board, dtype=object), _GROWTH_BOARDS)
        try:
            md._qf_growth_mask = gm
        except Exception:  # pragma: no cover - 只读数据对象的兜底
            pass
    return gm


def _score_row(scores, t: int) -> np.ndarray:
    """scores 既支持 (T, N) 面板也支持单行 (N,)。"""
    a = np.asarray(scores)
    if a.ndim == 2:
        return a[t]
    if a.ndim == 1:
        return a
    raise ValueError(f"scores 维度必须为 1 或 2，得到 {a.ndim}")


def _ranked_candidates(md, t: int, scores):
    """返回 (scores_row, 候选下标按分数降序, 全市场排名数组)。"""
    s = _score_row(scores, t)
    elig = np.asarray(md.eligible[t])
    cand = elig & np.isfinite(s)
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return s, idx, np.full(int(md.N), -1, dtype=np.int64)
    order = idx[np.argsort(-s[idx], kind="stable")]
    rank = np.full(int(md.N), -1, dtype=np.int64)
    rank[order] = np.arange(order.size)
    return s, order, rank


def selection(md, t: int, scores, prev_weights: dict[int, float] | None,
              cfg=C, n_holdings: int | None = None) -> list[int]:
    """候选优先队列（不含权重）：旧持仓缓冲区优先，其后按排名降序。

    返回长度可以 > ``n_holdings``：因为行业/板块/最小权重约束可能在分配权重时
    跳过前面的标的，权重分配会沿该队列继续补位，尽量持满 n_holdings 只。
    """
    n = int(n_holdings or cfg.N_HOLDINGS)
    if n <= 0:
        return []
    _, order, rank = _ranked_candidates(md, t, scores)
    if order.size == 0:
        return []
    n_buffer = max(n, int(round(n * float(cfg.TURNOVER_BUFFER))))
    prev = [int(j) for j, w in (prev_weights or {}).items() if float(w) > 0]
    keep = sorted({j for j in prev if 0 <= rank[j] < n_buffer},
                  key=lambda j: rank[j])
    seen = set(keep)
    pool = keep + [int(j) for j in order if int(j) not in seen]
    return pool


def _assign_weights(md, t: int, pool: list[int], tentative, n: int, cfg,
                    prev_set: set[int] | None = None,
                    max_new_entries: int | None = None) -> dict[int, float]:
    """按 pool 顺序贪心分配权重（受单票/行业/双创/合计上限约束）。

    ``tentative(j, slot)`` 给出该标的的候选权重；若被上限压缩后 < MIN_WEIGHT 则
    跳过并继续下一名（名额由后续候选补足）。

    换手上限（``max_new_entries``）：
    * 若设置，本次构建最多纳入这么多"上一期未持有"的新标的（prev_set 里的不算新）；
    * 新名额用完后，队列继续向后扫描"上一期持有、但因排名跌出缓冲区"的标的
      （pool 本身按 缓冲区旧持仓 → 分数排名 排列，所以等价于补回排名最高的旧持仓）；
    * 补不满就少持/留现金（不回补成"更差的分数"的新标的）。
    ``prev_set=None`` 或 ``max_new_entries=None`` 表示不限制。
    """
    max_w = float(cfg.MAX_WEIGHT)
    min_w = float(cfg.MIN_WEIGHT)
    ind_max = float(cfg.INDUSTRY_MAX)
    board_max = float(cfg.BOARD_MAX_GROWTH)
    industry = np.asarray(md.industry[t])
    growth = _growth_mask(md)

    prev_set = set(prev_set or ())
    limit: int | None = None if max_new_entries is None else max(0, int(max_new_entries))
    old_remaining = sum(1 for j in pool if int(j) in prev_set) if limit is not None else 0

    weights: dict[int, float] = {}
    ind_w: dict[int, float] = defaultdict(float)
    board_w = 0.0
    total = 0.0
    new_used = 0
    for j in pool:
        if len(weights) >= n:
            break
        j = int(j)
        if j in prev_set:
            if limit is not None:
                old_remaining -= 1
        elif limit is not None and new_used >= limit:
            # 新名额已用完：只剩"跌出缓冲区的旧持仓"可作为补位候选
            if old_remaining <= 0:
                break
            continue
        ind = int(industry[j])
        room_ind = ind_max - ind_w[ind]
        room_board = board_max - board_w if bool(growth[j]) else np.inf
        room_total = 1.0 - total
        w = min(tentative(j, len(weights)), max_w, room_ind, room_board, room_total)
        if w < min_w - 1e-12:
            continue
        weights[j] = float(w)
        ind_w[ind] += float(w)
        if bool(growth[j]):
            board_w += float(w)
        total += float(w)
        if j not in prev_set:
            new_used += 1
    return weights


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------
def build_target(md, t: int, scores, prev_weights: dict[int, float] | None = None,
                 cfg=C, n_holdings: int | None = None,
                 weight_mode: str = "equal",
                 max_new_entries: int | None = None) -> dict[int, float]:
    """t 日收盘打分 → 目标权重字典 ``{j: weight}``（权重和 ≤ 1，余为现金）。

    参数
    ----
    md : MarketData
    t : 交易日行号（信号日，收盘时点）
    scores : (T, N) 或 (N,) 分数面板/单行；NaN = 无信号
    prev_weights : 上一期持仓权重 ``{j: w}``，用于缓冲区保留旧持仓（可为 None）
    cfg : 配置对象（默认 qf.config）
    n_holdings : 目标持仓数，缺省 ``C.N_HOLDINGS``
    weight_mode : ``"equal"``（等权，缺省）或 ``"score"``（按排名斜率加权）
    max_new_entries : 单期最多纳入多少只"上一期未持有"的新标的（换手上限）。
        ``None``（缺省）= 不限制，保持向后兼容；``run_backtest`` 默认传 6。
        新名额用完后，空缺名额由"上一期持有、但因排名跌出缓冲区"的标的中
        排名最高者补回（仍要求 eligible + 分数有限），补不满则少持/留现金。

    返回
    ----
    dict[int, float]，键按买入优先级（排名）插入，权重和 ≤ 1。
    """
    n = int(n_holdings or cfg.N_HOLDINGS)
    if n <= 0:
        return {}
    # 约束可行性：单票目标权重 1/n 必须 ≥ MIN_WEIGHT，否则每一只都会被
    # "不足最小权重"跳过 → 返回空字典 → 回测静默空仓（净值一条直线）。
    if 1.0 / float(n) < float(cfg.MIN_WEIGHT) - 1e-12:
        raise ValueError(
            f"n_holdings={n} 与 MIN_WEIGHT={cfg.MIN_WEIGHT:.1%} 冲突："
            f"1/{n}={1.0/n:.2%} < 最小权重，任何标的都无法建仓。"
            f"上限为 n ≤ {int(1.0 / float(cfg.MIN_WEIGHT))}")
    _, order, rank = _ranked_candidates(md, t, scores)
    if order.size == 0:
        return {}
    pool = selection(md, t, scores, prev_weights, cfg=cfg, n_holdings=n)
    if not pool:
        return {}
    prev_set = {int(j) for j, w in (prev_weights or {}).items() if float(w) > 0}

    base = 1.0 / float(n)
    if weight_mode == "score":
        # 先用等权选定持仓名单，再按排名斜率给权重（最好 1.5×base，最差 0.5×base）
        sel = list(_assign_weights(md, t, pool, lambda j, k: base, n, cfg,
                                   prev_set=prev_set,
                                   max_new_entries=max_new_entries).keys())
        if not sel:
            return {}
        rr = rank[sel].astype(float)
        span = max(float(rr.max() - rr.min()), 1.0)
        pct = 1.0 - (rr - rr.min()) / span          # 1=排名最好, 0=最差
        raw = 0.5 + pct                             # [0.5, 1.5]
        raw = raw / raw.sum()
        wmap = {int(j): float(v) for j, v in zip(sel, raw)}
        return _assign_weights(md, t, sel, lambda j, k: wmap[j], n, cfg)

    weights = _assign_weights(md, t, pool, lambda j, k: base, n, cfg,
                              prev_set=prev_set, max_new_entries=max_new_entries)
    # 二次保护：数值误差导致和略大于 1 时按比例压缩
    total = float(sum(weights.values()))
    if total > 1.0 + 1e-12:
        weights = {j: w / total for j, w in weights.items()}
    return weights
