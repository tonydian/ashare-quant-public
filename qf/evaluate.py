# -*- coding: utf-8 -*-
"""因子评估模块（Rank IC / ICIR / 分组收益 / 换手 / 覆盖率 / 分年度稳健性）。

口径严格按 ``docs/INTERFACES.md`` §2.1：

* 样本行 ``rows``：交易日行号（必须是周频行，见 :func:`weekly_rows`）；横截面统计
  只在 rows 上计算；**周频行固定为 2017-01-03 起每 ``C.REBALANCE_DAYS``（5）个交易日**。
* 掩码 ``mask``：当日选股域 ``md.eligible[t]``，且 factor / label 均有限；有效股票数
  < 100 的日子跳过横截面统计。
* 标签 ``label = md.fwd_open_ret(C.FWD_HORIZON)``（t+1 开盘买入 → t+1+h 开盘卖出），
  **只用于评估，不进入因子计算**。
* Rank IC：每日横截面 Spearman（平均秩后 Pearson，等价 pandas ``rank(axis=1)``）。
* ICIR = mean(IC)/std(IC)（ddof=0）；t 值 = mean/std×sqrt(n)；正 IC 占比 = mean(IC>0)。
* 分组：按因子当日横截面排名分 5 组（等权），组收益 = 组内 5 日标签均值；
  多空 = Q5-Q1（逐周，报告均值/IR/t）。
* 换手：Top20% 组相邻周成员变化 ``1 - |S_t ∩ S_{t-1}| / |S_{t-1}|`` 的均值。
* 覆盖率：``mean_t( finite(factor)&eligible / eligible )``；eligible 全为 0 的日子不计。
* 稳健性：分年度 IC；单调性 = 5 组收益均值与组序号（1..5）的 Spearman 相关。

性能辅助：
* :class:`LabelRankCache` —— 预计算 label 的横截面平均秩，可在任意"联合有效子集"上
  O(1) 次向量化得到**精确**平均秩（含 ties），避免每个因子重复 argsort label；
* :class:`TransformCache` / :func:`transform_at_rows` —— 因子变换只在周频行上取数，
  底层直接复用 ``qf.factors._transform_signal`` 官方实现（数值一致）。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from . import factors as qff
from . import ops

__all__ = [
    "MIN_IC_STOCKS", "N_QUANTILES", "TOP_QUANTILE",
    "weekly_rows", "weekly_dates", "dev_week_mask",
    "row_rank", "rank_ic", "quantile_returns", "top_turnover", "turnover_series",
    "coverage", "ic_statistics", "ic_by_year",
    "factor_metrics", "early_screen",
    "LabelRankCache", "TransformCache", "transform_at_rows", "entry_params",
]

MIN_IC_STOCKS = 100          # 横截面有效股票数下限（§2.1：<100 跳过）
N_QUANTILES = 5              # 分组数
TOP_QUANTILE = 0.2           # 换手统计的 Top 组比例


# ================================================================ 周频行
def weekly_rows(md, start_date=None, step=None) -> np.ndarray:
    """周频行号：``np.arange(date_index(start), T, step)``（默认 2017-01-03 起每 5 日）。"""
    start_date = C.DEV_START if start_date is None else start_date
    step = int(C.REBALANCE_DAYS if step is None else step)
    start = int(md.date_index(start_date))
    return np.arange(start, md.T, step, dtype=np.int64)


def weekly_dates(md, rows: np.ndarray) -> pd.DatetimeIndex:
    return md.dates[np.asarray(rows)]


def dev_week_mask(md, rows: np.ndarray) -> np.ndarray:
    """开发期（<= C.DEV_END，2019-12-31）周标记。"""
    rows = np.asarray(rows)
    return md.dates[rows] <= pd.Timestamp(C.DEV_END)


# ================================================================ 行对齐
def _align_rows(a, rows, T: int, name: str = "array") -> np.ndarray:
    """把因子/label/mask 对齐到 rows：支持完整面板 (T,N) 或已对齐 (len(rows),N)。"""
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{name} 必须是二维面板")
    rows = np.asarray(rows)
    if a.shape[0] == len(rows):
        return a
    if a.shape[0] == T:
        return a[rows]
    raise ValueError(f"{name} 行数 {a.shape[0]} 既不等于 len(rows)={len(rows)} 也不等于 T={T}")


# ================================================================ 横截面平均秩
def row_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask 内的逐行横截面平均秩（1..n），mask 外为 NaN。

    与 ``pandas.DataFrame(x).where(mask).rank(axis=1, method="average")`` 数值等价，
    同样处理 ties。
    """
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    R, N = x.shape
    v = np.where(mask, x, np.inf).astype(np.float32)
    order = np.argsort(v, axis=1)                       # 无需 stable：ties 由值相等判定
    vs = np.take_along_axis(v, order, axis=1)
    diff = np.ones((R, N), dtype=bool)
    diff[:, 1:] = vs[:, 1:] != vs[:, :-1]
    idx = np.arange(N, dtype=np.int32)
    start = np.maximum.accumulate(np.where(diff, idx, 0), axis=1)
    nxtb = np.where(diff, idx, N)
    nxt_shift = np.empty_like(nxtb)
    nxt_shift[:, :-1] = nxtb[:, 1:]
    nxt_shift[:, -1] = N
    nxt_shift = np.minimum.accumulate(nxt_shift[:, ::-1], axis=1)[:, ::-1]
    end = np.minimum(nxt_shift - 1, N - 1)
    rk_sorted = (0.5 * (start + end) + 1.0).astype(np.float32)
    out = np.empty((R, N), dtype=np.float32)
    np.put_along_axis(out, order, rk_sorted, axis=1)
    return np.where(mask, out, np.nan).astype(np.float32)


def _ic_from_ranks(rf: np.ndarray, rl: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """由横截面秩计算逐行 Pearson（=Spearman IC），有效数 < MIN_IC_STOCKS 记 NaN。"""
    n = valid.sum(axis=1).astype(np.float64)
    f = np.where(valid, rf, 0.0).astype(np.float64)
    y = np.where(valid, rl, 0.0).astype(np.float64)
    nn = np.maximum(n, 1.0)
    mf = f.sum(axis=1) / nn
    my = y.sum(axis=1) / nn
    cov = np.where(valid, (f - mf[:, None]) * (y - my[:, None]), 0.0).sum(axis=1) / nn
    vf = np.where(valid, (f - mf[:, None]) ** 2, 0.0).sum(axis=1) / nn
    vy = np.where(valid, (y - my[:, None]) ** 2, 0.0).sum(axis=1) / nn
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = cov / np.sqrt(vf * vy)
    return np.where((n >= MIN_IC_STOCKS) & np.isfinite(ic), ic, np.nan).astype(np.float64)


# ================================================================ label 秩缓存
class LabelRankCache:
    """label 横截面平均秩的预计算缓存（支持任意联合有效子集上的精确平均秩）。

    对每周把 label 升序排序一次，记录每个位置的"同值组"起止（处理 ties）；
    给定任意子集 ``valid ⊆ base_valid`` 时，用前缀计数一次向量化算出子集内平均秩。
    对 475 周 × 5281 股，单因子取得 label 秩约 30-50 ms（比重新 argsort 快 ~4 倍）。
    """

    def __init__(self, label_rows: np.ndarray, mask_rows: np.ndarray):
        base_valid = np.asarray(mask_rows, dtype=bool) & np.isfinite(label_rows)
        self.base_valid = base_valid
        R, N = base_valid.shape
        v = np.where(base_valid, np.asarray(label_rows, dtype=np.float32), np.inf).astype(np.float32)
        order = np.argsort(v, axis=1)
        vs = np.take_along_axis(v, order, axis=1)
        diff = np.ones((R, N), dtype=bool)
        diff[:, 1:] = vs[:, 1:] != vs[:, :-1]
        idx = np.arange(N, dtype=np.int32)
        start = np.maximum.accumulate(np.where(diff, idx, 0), axis=1)
        nxtb = np.where(diff, idx, N)
        nxt_shift = np.empty_like(nxtb)
        nxt_shift[:, :-1] = nxtb[:, 1:]
        nxt_shift[:, -1] = N
        nxt_shift = np.minimum.accumulate(nxt_shift[:, ::-1], axis=1)[:, ::-1]
        end = np.minimum(nxt_shift - 1, N - 1)
        self.order = order
        self.start = start
        self.end = end

    def ranks(self, valid: np.ndarray) -> np.ndarray:
        """返回子集 ``valid``（须 ⊆ base_valid）上的精确平均秩，其余 NaN。"""
        valid = np.asarray(valid, dtype=bool)
        vs = np.take_along_axis(valid, self.order, axis=1)
        cs = np.cumsum(vs, axis=1, dtype=np.int32)
        cs_shift = np.empty_like(cs)
        cs_shift[:, 0] = 0
        cs_shift[:, 1:] = cs[:, :-1]
        cs_prev = np.take_along_axis(cs_shift, self.start, axis=1)
        cs_end = np.take_along_axis(cs, self.end, axis=1)
        rk = 0.5 * (cs_prev.astype(np.float32) + 1.0 + cs_end.astype(np.float32))
        rk = np.where(vs, rk, np.nan)
        out = np.full(cs.shape, np.nan, dtype=np.float32)
        np.put_along_axis(out, self.order, rk.astype(np.float32), axis=1)
        return out


# ================================================================ 核心统计
def rank_ic(factor, label, mask=None, rows=None, label_cache: LabelRankCache | None = None) -> np.ndarray:
    """逐期 Rank IC（Spearman）。

    factor / label / mask 为完整面板 (T,N) 时需给 rows；已按 rows 对齐时 rows 可为 None。
    返回 (len(rows),)，无效周为 NaN。
    """
    if rows is None:
        x = np.asarray(factor)
        n = x.shape[0]
        T = n
        rows_arr = np.arange(n)
    else:
        rows_arr = np.asarray(rows)
        x = _align_rows(factor, rows_arr, None, "factor")
        T = None
    y = _align_rows(label, rows_arr, T, "label")
    if mask is None:
        m = np.ones(x.shape, dtype=bool)
    else:
        m = _align_rows(mask, rows_arr, T, "mask").astype(bool)
    valid = m & np.isfinite(x) & np.isfinite(y)
    rf = row_rank(x, valid)
    if label_cache is not None:
        rl = label_cache.ranks(valid)
    else:
        rl = row_rank(y, valid)
    return _ic_from_ranks(rf, rl, valid)


def _group_index(rf: np.ndarray, valid: np.ndarray, n_q: int) -> np.ndarray:
    """由平均秩把每行分 n_q 组（0..n_q-1；无效 -1）。组大小 ≈ n/n_q。"""
    R, N = rf.shape
    n = valid.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.floor((rf - 1.0) * n_q / np.maximum(n, 1.0)[:, None])
    g = np.clip(g, 0, n_q - 1)
    g = np.where(valid, g, -1)
    return g.astype(np.int32)


def _group_means(grp: np.ndarray, y: np.ndarray, valid: np.ndarray,
                 n_q: int, ok_week: np.ndarray) -> np.ndarray:
    R, N = grp.shape
    row_i = np.arange(R, dtype=np.int64)[:, None]
    flat = (row_i * n_q + grp)[valid]
    w = np.where(valid, y, 0.0).astype(np.float64)[valid]
    sums = np.bincount(flat, weights=w, minlength=R * n_q)
    cnts = np.bincount(flat, minlength=R * n_q)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = sums / np.maximum(cnts, 1)
    q = q.reshape(R, n_q)
    q[cnts.reshape(R, n_q) == 0] = np.nan
    q[~ok_week] = np.nan
    return q


def _quantile_core(x, y, mask, n_q):
    """(x,y,mask) 已对齐 → (qret (R,n_q), rf, valid, ok_week)。"""
    valid = mask & np.isfinite(x) & np.isfinite(y)
    ok_week = valid.sum(axis=1) >= MIN_IC_STOCKS
    rf = row_rank(x, valid)
    grp = _group_index(rf, valid, n_q)
    return _group_means(grp, y, valid, n_q, ok_week), rf, grp, valid, ok_week


def quantile_returns(factor, label, mask=None, rows=None, n_q: int = N_QUANTILES) -> np.ndarray:
    """逐周 n_q 分组等权收益（5 日标签均值）→ (len(rows), n_q)。"""
    if rows is None:
        x = np.asarray(factor)
        rr = np.arange(x.shape[0])
        T = None
    else:
        rr = np.asarray(rows)
        x = _align_rows(factor, rr, None, "factor")
        T = None
    y = _align_rows(label, rr, T, "label")
    m = np.ones(x.shape, dtype=bool) if mask is None else _align_rows(mask, rr, T, "mask").astype(bool)
    q, _, _, _, _ = _quantile_core(x, y, m, n_q)
    return q


def _turnover_from_top(top: np.ndarray, ok_week: np.ndarray) -> tuple[float, np.ndarray]:
    R = top.shape[0]
    tv = np.full(R, np.nan, dtype=np.float64)
    prev = None
    for i in range(R):
        if not ok_week[i]:
            prev = None
            continue
        cur = top[i]
        if prev is not None:
            size = int(prev.sum())
            if size > 0:
                tv[i] = 1.0 - np.count_nonzero(cur & prev) / size
        prev = cur
    vals = tv[np.isfinite(tv)]
    return (float(vals.mean()) if len(vals) else float("nan")), tv


def turnover_series(factor, mask=None, rows=None, top: float = TOP_QUANTILE) -> np.ndarray:
    """Top20% 组逐周换手率序列 (len(rows),)，首个有效周/断档后首周为 NaN。"""
    if rows is None:
        x = np.asarray(factor)
        rr = np.arange(x.shape[0])
        T = None
    else:
        rr = np.asarray(rows)
        x = _align_rows(factor, rr, None, "factor")
        T = None
    m = np.ones(x.shape, dtype=bool) if mask is None else _align_rows(mask, rr, T, "mask").astype(bool)
    _, _, grp, valid, ok_week = _quantile_core(x, m & np.isfinite(x), m, N_QUANTILES)
    top_mask = valid & (grp == N_QUANTILES - 1)
    _, tv = _turnover_from_top(top_mask, ok_week)
    return tv


def top_turnover(factor, mask=None, rows=None, top: float = TOP_QUANTILE) -> float:
    """Top20% 组平均换手。"""
    tv = turnover_series(factor, mask=mask, rows=rows, top=top)
    v = tv[np.isfinite(tv)]
    return float(v.mean()) if len(v) else float("nan")


def coverage(factor, mask=None, rows=None) -> float:
    """覆盖率：mean_t( finite(factor) & eligible / eligible )；eligible=0 的日子不计。"""
    if rows is None:
        x = np.asarray(factor)
        rr = np.arange(x.shape[0])
        T = None
    else:
        rr = np.asarray(rows)
        x = _align_rows(factor, rr, None, "factor")
        T = None
    m = np.ones(x.shape, dtype=bool) if mask is None else _align_rows(mask, rr, T, "mask").astype(bool)
    denom = m.sum(axis=1)
    num = (np.isfinite(x) & m).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = num / np.maximum(denom, 1)
    cov = np.where(denom > 0, cov, np.nan)
    return float(np.nanmean(cov)) if np.isfinite(cov).any() else float("nan")


# ================================================================ IC 统计
def ic_statistics(ic: np.ndarray) -> dict:
    """IC 序列 → {ic_mean, ic_std, icir, tstat, pos_ratio, n}（ICIR/t 用 ddof=0）。"""
    v = np.asarray(ic, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(len(v))
    if n == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan"),
                "tstat": float("nan"), "pos_ratio": float("nan"), "n": 0}
    m = float(v.mean())
    s = float(v.std())
    if s > 1e-12:
        icir = m / s
        tstat = m / s * np.sqrt(n)
    else:
        icir = float("nan")
        tstat = float("nan")
    return {"ic_mean": m, "ic_std": s, "icir": icir, "tstat": float(tstat),
            "pos_ratio": float(np.mean(v > 0)), "n": n}


def ic_by_year(ic: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """逐年 IC 均值 → {"2017": x, ...}（无有效周的年不出现）。"""
    ic = np.asarray(ic, dtype=np.float64)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    out = {}
    for yr in np.unique(years):
        v = ic[(years == yr) & np.isfinite(ic)]
        if len(v):
            out[str(int(yr))] = float(v.mean())
    return out


def _spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy(dtype=np.float64, copy=True)
    rb = pd.Series(b[ok]).rank().to_numpy(dtype=np.float64, copy=True)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else float("nan")


# ================================================================ 单因子完整指标
def _factor_metrics_aligned(md, x: np.ndarray, y: np.ndarray, mask: np.ndarray,
                            rows: np.ndarray, name: str | None = None,
                            n_q: int = N_QUANTILES,
                            label_cache: LabelRankCache | None = None) -> dict:
    """x/y/mask 已按 rows 对齐（(R,N)），计算 §2.1 全部指标（全样本 + DEV 子集）。"""
    rows = np.asarray(rows)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(x) & np.isfinite(y)
    n_valid = valid.sum(axis=1)
    ok_week = n_valid >= MIN_IC_STOCKS

    rf = row_rank(x, valid)
    rl = label_cache.ranks(valid) if label_cache is not None else row_rank(y, valid)
    ic = _ic_from_ranks(rf, rl, valid)

    grp = _group_index(rf, valid, n_q)
    qret = _group_means(grp, y, valid, n_q, ok_week)             # (R,n_q)
    q_mean = np.nanmean(qret, axis=0) if np.isfinite(qret).any() else np.full(n_q, np.nan)
    ls = qret[:, -1] - qret[:, 0]
    ls_ok = ls[np.isfinite(ls)]

    top_mask = valid & (grp == n_q - 1)
    turn, _ = _turnover_from_top(top_mask, ok_week)

    # 覆盖率（全部 rows，eligible>0 的日才计入）
    mask_n = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov_rows = (np.isfinite(x) & mask).sum(axis=1) / np.maximum(mask_n, 1)
    cov_rows = np.where(mask_n > 0, cov_rows, np.nan)
    cov = float(np.nanmean(cov_rows)) if np.isfinite(cov_rows).any() else float("nan")

    dates_w = md.dates[rows]
    dev = dates_w <= pd.Timestamp(C.DEV_END)

    st = ic_statistics(ic)
    st_dev = ic_statistics(ic[dev])
    ls_dev = ls[dev]
    ls_dev = ls_dev[np.isfinite(ls_dev)]
    grp_dev = grp[dev]
    valid_dev = valid[dev]
    ok_dev = ok_week[dev]
    top_dev = valid_dev & (grp_dev == n_q - 1)
    turn_dev, _ = _turnover_from_top(top_dev, ok_dev)
    cov_dev = float(np.nanmean(cov_rows[dev])) if np.isfinite(cov_rows[dev]).any() else float("nan")
    qret_dev = qret[dev]
    qmean_dev = np.nanmean(qret_dev, axis=0) if np.isfinite(qret_dev).any() else np.full(n_q, np.nan)
    ls_dev_mean = float(ls_dev.mean()) if len(ls_dev) else float("nan")
    ls_dev_ir = (float(ls_dev.mean() / ls_dev.std())
                 if len(ls_dev) > 1 and ls_dev.std() > 1e-12 else float("nan"))
    mono_dev = _spearman_1d(qmean_dev, np.arange(1, n_q + 1))

    out = {
        "name": name,
        "n_weeks": int(len(rows)),
        "n_ic": st["n"],
        "ic_5d": st["ic_mean"],
        "ic_std_5d": st["ic_std"],
        "icir_5d": st["icir"],
        "tstat": st["tstat"],
        "ic_pos_ratio": st["pos_ratio"],
        "coverage": cov,
        "ls_mean_5d": float(ls_ok.mean()) if len(ls_ok) else float("nan"),
        "ls_std_5d": float(ls_ok.std()) if len(ls_ok) else float("nan"),
        "ls_ir_5d": (float(ls_ok.mean() / ls_ok.std())
                     if len(ls_ok) > 1 and ls_ok.std() > 1e-12 else float("nan")),
        "turnover_top20": turn,
        "monotonicity": _spearman_1d(q_mean, np.arange(1, n_q + 1)),
        "ic_by_year": ic_by_year(ic, dates_w),
        "dev_ic_5d": st_dev["ic_mean"],
        "dev_icir_5d": st_dev["icir"],
        "dev_tstat": st_dev["tstat"],
        "dev_ic_pos_ratio": st_dev["pos_ratio"],
        "dev_ic_n": st_dev["n"],
        "dev_coverage": cov_dev,
        "dev_ls_mean_5d": ls_dev_mean,
        "dev_ls_ir_5d": ls_dev_ir,
        "dev_turnover_top20": turn_dev,
        "dev_monotonicity": mono_dev,
    }
    for i in range(n_q):
        out[f"q{i + 1}_5d"] = float(q_mean[i])
        out[f"dev_q{i + 1}_5d"] = float(qmean_dev[i])
    return out


def factor_metrics(md, factor, label, rows, name: str | None = None,
                   mask=None, n_q: int = N_QUANTILES,
                   label_cache: LabelRankCache | None = None) -> dict:
    """单因子完整指标。factor/label (T,N) 或已对齐 (len(rows),N)；mask 缺省 md.eligible。"""
    rows = np.asarray(rows)
    x = _align_rows(factor, rows, md.T, "factor")
    y = _align_rows(label, rows, md.T, "label")
    m = md.eligible[rows] if mask is None else _align_rows(mask, rows, md.T, "mask").astype(bool)
    if label_cache is None:
        label_cache = LabelRankCache(y, m)
    return _factor_metrics_aligned(md, x, y, m, rows, name=name, n_q=n_q,
                                   label_cache=label_cache)


# ================================================================ 周频行变换（复用官方实现）
class TransformCache:
    """缓存 vol_scaled / confirm_volume 所需的滚动面板（全库共享，按 norm 维度）。"""

    def __init__(self, inp):
        self.inp = inp
        self._ret_std: dict[int, np.ndarray] = {}
        self._vol_mean: dict[int, np.ndarray] = {}

    def ret_std(self, w: int) -> np.ndarray:
        if w not in self._ret_std:
            self._ret_std[w] = ops.roll_std(self.inp.ret, w)
        return self._ret_std[w]

    def volume_mean(self, w: int) -> np.ndarray:
        if w not in self._vol_mean:
            self._vol_mean[w] = ops.roll_mean(self.inp.volume, w)
        return self._vol_mean[w]


def entry_params(entry, window=None) -> dict:
    """catalog 一行 → qf.factors 计算参数（window / norm 用列值覆盖）。"""
    p = dict(entry["params"])
    p["window"] = int(entry["window"] if window is None else window)
    p["norm"] = int(entry["norm"])
    return p


def transform_at_rows(sig: np.ndarray, inp, transform: str, params: dict,
                      rows: np.ndarray, cache: TransformCache | None = None) -> np.ndarray:
    """计算因子变换面板并只取周频行（(len(rows), N) float32）。

    直接复用 ``qf.factors._transform_signal`` 的官方语义（含 std/mean=0 → NaN 口径）；
    仅对 vol_scaled / confirm_volume 复用全库共享的滚动面板缓存（公式逐字一致）。
    """
    rows = np.asarray(rows).astype(np.int64)
    norm = int(params.get("norm", 20))
    with np.errstate(all="ignore"):
        if transform == "vol_scaled" and cache is not None:
            vol = cache.ret_std(norm)
            out = np.where(vol > 0, sig / np.where(vol > 0, vol, np.nan), np.nan)
            return out[rows].astype(np.float32)
        if transform == "confirm_volume" and cache is not None:
            vm = cache.volume_mean(norm)
            out = np.where(vm > 0, sig * (inp.volume / np.where(vm > 0, vm, np.nan)), np.nan)
            return out[rows].astype(np.float32)
        full = qff._transform_signal(sig, inp, transform, params)
        return full[rows].astype(np.float32)


# ================================================================ 批量初筛
def early_screen(md, cat, inp, rows_screen, label, verbose: bool = True,
                 progress_path=None, progress_every: int = 10,
                 done_keys=None) -> pd.DataFrame:
    """Stage-1 初筛：遍历 ``qf.factors.iter_groups`` 的 (base,window) 组，按组复用
    base 信号，逐 transform 在 rows 上计算因子面板并立即评估（算完即丢）。

    返回 800 行 DataFrame（列 = INTERFACES §2.5 因子卡字段 + DEV 期字段）。
    ``progress_path`` 给定则每 ``progress_every`` 组落盘一次（支持断点续跑），
    ``done_keys`` 为已完成的 factor_key 集合。
    """
    rows = np.asarray(rows_screen).astype(np.int64)
    label_rows = _align_rows(label, rows, md.T, "label")
    mask_w = md.eligible[rows]
    label_cache = LabelRankCache(label_rows, mask_w)
    cache = TransformCache(inp)
    done = set(done_keys or ())
    out: list[dict] = []
    groups = list(qff.iter_groups(cat))
    n_groups = len(groups)
    t_start = time.time()
    t_group_sum = 0.0
    n_done_groups = 0
    meta_cols = ["lib", "factor_id", "name_cn", "name_en", "category_cn",
                 "category_en", "base", "transform", "window", "norm"]

    def _flush():
        if progress_path is not None and out:
            pd.DataFrame(out).to_parquet(progress_path)

    for gi, (base, w, g) in enumerate(groups):
        g_new = g[~g["factor_key"].isin(done)]
        if len(g_new) == 0:
            continue
        t0 = time.time()
        p0 = entry_params(g.iloc[0], window=w)
        sig = qff._base_signal(inp, base, p0)
        for _, r in g_new.iterrows():
            p = entry_params(r, window=w)
            x = transform_at_rows(sig, inp, r["transform"], p, rows, cache)
            met = _factor_metrics_aligned(md, x, label_rows, mask_w, rows,
                                          name=r["factor_key"], label_cache=label_cache)
            for c in meta_cols:
                met[c] = r[c]
            met["factor_key"] = r["factor_key"]
            out.append(met)
            del x
        del sig
        dt = time.time() - t0
        t_group_sum += dt
        n_done_groups += 1
        if verbose:
            avg = t_group_sum / n_done_groups
            remain = n_groups - gi - 1
            eta = avg * remain / 60.0
            print(f"[stage1] {gi + 1:3d}/{n_groups} {base:16s} w={int(w):2d} "
                  f"{len(g_new)}f {dt:5.1f}s | elapsed {(time.time()-t_start)/60:.1f}min "
                  f"ETA {eta:.1f}min | {len(out)} factors", flush=True)
        if progress_path is not None and (gi + 1) % progress_every == 0:
            _flush()
    _flush()
    df = pd.DataFrame(out)
    front = ["factor_key"] + meta_cols
    cols = front + [c for c in df.columns if c not in front]
    return df[cols]
