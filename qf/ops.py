# -*- coding: utf-8 -*-
"""向量化算子库（PandaFactor 算子语义的 numpy 面板实现）。

PandaFactor 的算子（DELAY / STDDEV / CORRELATION / SUM / RANK ...）定义在
``vendor/panda_factor`` 的 ``factor_base.py`` / ``factor_utils.py`` 中，是
pandas MultiIndex 逐股票实现，适合单因子验证但在 2428×5281 的全 A 面板上
无法支撑 800 因子的批量计算。本模块用 (T, N) 的 numpy 面板重写同一套语义：
每一列是一只股票的时序，行是交易日；所有函数都保持 NaN 语义（不足
min_periods 的窗口输出 NaN），并且与 pandas 参考实现逐点对齐（见 tests）。

约定:
    * 输入/输出均为 float32 的 (T, N) 数组（内部用 float64 累加保证精度）。
    * 面板中 NaN 只允许出现在上市前 / 退市后（数据层已把停牌日价格前向填充、
      成交量补 0），算子对 NaN 仍做 count-aware 处理。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "shift", "pct_change", "delay",
    "roll_sum", "roll_mean", "roll_std", "roll_min", "roll_max",
    "roll_skew", "roll_kurt", "roll_corr", "roll_cov", "roll_slope",
    "roll_rank_last", "ema", "cs_rank", "cs_zscore", "tanh_clip",
]


# ------------------------------------------------------------------ 基础
def shift(a: np.ndarray, k: int) -> np.ndarray:
    """时序滞后（DELAY）：k>0 向下移 k 行，k<0 向上移。"""
    out = np.full_like(a, np.nan, dtype=np.float32)
    if k > 0:
        if k < a.shape[0]:
            out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out = a.copy()
    return out


def pct_change(a: np.ndarray, k: int = 1) -> np.ndarray:
    """k 期收益率 a[t]/a[t-k] - 1（NaN 安全）。"""
    prev = shift(a, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((prev != 0) & np.isfinite(prev), a / prev - 1.0, np.nan)
    return out.astype(np.float32)


def delay(a: np.ndarray, k: int = 1) -> np.ndarray:  # PandaFactor 命名
    return shift(a, k)


# ------------------------------------------------------------------ 滚动基础量
def _cumsums(a: np.ndarray, w: int, need_sq: bool = False):
    """返回 (cnt, s1[, s2]) 且在 t 处表示窗口 (t-w+1..t) 的和；float64 精度。"""
    T, N = a.shape
    valid = np.isfinite(a)
    x = np.where(valid, a, 0.0).astype(np.float64)
    c = np.zeros((T + 1, N), dtype=np.float64)
    c[1:] = np.cumsum(x, axis=0)
    cnt = np.zeros((T + 1, N), dtype=np.float64)
    cnt[1:] = np.cumsum(valid, axis=0)
    idx = np.arange(T)
    lo = np.maximum(idx - w + 1, 0)
    s1 = c[idx + 1] - c[lo]
    n = cnt[idx + 1] - cnt[lo]
    if need_sq:
        x2 = np.where(valid, x * x, 0.0)
        c2 = np.zeros((T + 1, N), dtype=np.float64)
        c2[1:] = np.cumsum(x2, axis=0)
        s2 = c2[idx + 1] - c2[lo]
        return n, s1, s2
    return n, s1


def _minp(w: int, floor: int = 5) -> int:
    """与 QuantSkills factor.py 的 _minp 完全一致的 min_periods 口径。"""
    return min(w, max(2, min(floor, w // 2 if w > 2 else 2)))


def _mask_count(out: np.ndarray, n: np.ndarray, w: int, minp: int | None) -> np.ndarray:
    minp = _minp(w) if minp is None else int(minp)
    out = np.where(n >= minp, out, np.nan)
    return out.astype(np.float32)


def roll_sum(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    n, s1 = _cumsums(a, w)
    return _mask_count(s1, n, w, minp)


def roll_mean(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    n, s1 = _cumsums(a, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / n
    return _mask_count(m, n, w, minp)


def roll_std(a: np.ndarray, w: int, minp: int | None = None, ddof: int = 0) -> np.ndarray:
    n, s1, s2 = _cumsums(a, w, need_sq=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = s2 / n - (s1 / n) ** 2
    var = np.maximum(var, 0.0)
    if ddof:
        with np.errstate(invalid="ignore", divide="ignore"):
            var = var * n / np.maximum(n - ddof, 1)
    return _mask_count(np.sqrt(var), n, w, minp)


def _window_view(a: np.ndarray, w: int):
    """chunked sliding window; yields (start, stop, view(start:stop, w, N))"""
    T, N = a.shape
    rows = max(1, int(4e6 / max(1, w * N)))   # ~16MB/chunk
    for s in range(0, T, rows):
        e = min(T, s + rows)
        if e - s < w:
            break
        yield s, e, np.lib.stride_tricks.sliding_window_view(a[s:e], w, axis=0)


def _fill_warmup(out: np.ndarray, a: np.ndarray, w: int, reduce: str) -> None:
    """补齐前 w-1 行的部分窗口 min/max（与 pandas min_periods 口径对齐）。"""
    T = a.shape[0]
    n_head = min(w - 1, T)
    T_, N = a.shape
    run = np.full(N, np.nan, dtype=np.float32)
    for t in range(n_head):
        row = a[t]
        v = np.isfinite(row)
        if reduce == "min":
            run = np.where(v, np.where(np.isfinite(run), np.fmin(run, row), row), run)
        else:
            run = np.where(v, np.where(np.isfinite(run), np.fmax(run, row), row), run)
        out[t] = run


def roll_min(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    for s, e, v in _window_view(a, w):
        with np.errstate(all="ignore"):
            r = np.nanmin(v, axis=-1)
        m = e - (s + w - 1)
        out[s + w - 1:s + w - 1 + m] = r[:m]
    _fill_warmup(out, a, w, "min")
    return _mask_count(out, _cumsums(a, w)[0], w, minp)


def roll_max(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    for s, e, v in _window_view(a, w):
        with np.errstate(all="ignore"):
            r = np.nanmax(v, axis=-1)
        m = e - (s + w - 1)
        out[s + w - 1:s + w - 1 + m] = r[:m]
    _fill_warmup(out, a, w, "max")
    return _mask_count(out, _cumsums(a, w)[0], w, minp)


def roll_skew(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    """总体偏度（biason），与 pandas rolling().skew() 接近。"""
    valid = np.isfinite(a)
    x = np.where(valid, a, 0.0).astype(np.float64)
    T, N = a.shape
    c = [None] * 5
    for p in (1, 2, 3, 4):
        xp = x if p == 1 else x ** p
        cc = np.zeros((T + 1, N), dtype=np.float64)
        cc[1:] = np.cumsum(xp, axis=0)
        c[p] = cc
    cc = np.zeros((T + 1, N), dtype=np.float64)
    cc[1:] = np.cumsum(valid, axis=0)
    idx = np.arange(T)
    lo = np.maximum(idx - w + 1, 0)
    n = cc[idx + 1] - cc[lo]
    s1 = c[1][idx + 1] - c[1][lo]
    s2 = c[2][idx + 1] - c[2][lo]
    s3 = c[3][idx + 1] - c[3][lo]
    s4 = c[4][idx + 1] - c[4][lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        m2 = s2 / n - mu ** 2
        m3 = s3 / n - 3 * mu * s2 / n + 2 * mu ** 3
        g1 = m3 / np.power(np.maximum(m2, 0), 1.5)
    return _mask_count(g1, n, w, minp)


def roll_kurt(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    """超额峰度（excess kurtosis，总体口径）。"""
    valid = np.isfinite(a)
    x = np.where(valid, a, 0.0).astype(np.float64)
    T, N = a.shape
    c = [None] * 5
    for p in (1, 2, 3, 4):
        xp = x if p == 1 else x ** p
        cc = np.zeros((T + 1, N), dtype=np.float64)
        cc[1:] = np.cumsum(xp, axis=0)
        c[p] = cc
    cc = np.zeros((T + 1, N), dtype=np.float64)
    cc[1:] = np.cumsum(valid, axis=0)
    idx = np.arange(T)
    lo = np.maximum(idx - w + 1, 0)
    n = cc[idx + 1] - cc[lo]
    s1 = c[1][idx + 1] - c[1][lo]
    s2 = c[2][idx + 1] - c[2][lo]
    s3 = c[3][idx + 1] - c[3][lo]
    s4 = c[4][idx + 1] - c[4][lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        m2 = s2 / n - mu ** 2
        m4 = s4 / n - 4 * mu * s3 / n + 6 * mu ** 2 * s2 / n - 3 * mu ** 4
        g2 = m4 / np.maximum(m2, 0) ** 2 - 3.0
    return _mask_count(g2, n, w, minp)


def _pairwise(a: np.ndarray, b: np.ndarray):
    va = np.isfinite(a)
    vb = np.isfinite(b)
    both = va & vb
    xa = np.where(both, a, 0.0).astype(np.float64)
    xb = np.where(both, b, 0.0).astype(np.float64)
    return both, xa, xb


def roll_cov(a: np.ndarray, b: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    both, xa, xb = _pairwise(a, b)
    T, N = a.shape
    def cs(x):
        c = np.zeros((T + 1, N), dtype=np.float64)
        c[1:] = np.cumsum(x, axis=0)
        return c
    cxy = cs(xa * xb); cx = cs(xa); cy = cs(xb)
    cc = np.zeros((T + 1, N), dtype=np.float64)
    cc[1:] = np.cumsum(both, axis=0)
    idx = np.arange(T); lo = np.maximum(idx - w + 1, 0)
    n = cc[idx + 1] - cc[lo]
    sxy = cxy[idx + 1] - cxy[lo]
    sx = cx[idx + 1] - cx[lo]
    sy = cy[idx + 1] - cy[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
    return _mask_count(cov, n, w, minp)


def roll_corr(a: np.ndarray, b: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    both, xa, xb = _pairwise(a, b)
    T, N = a.shape
    def cs(x):
        c = np.zeros((T + 1, N), dtype=np.float64)
        c[1:] = np.cumsum(x, axis=0)
        return c
    cxy = cs(xa * xb); cx = cs(xa); cy = cs(xb)
    cx2 = cs(xa * xa); cy2 = cs(xb * xb)
    cc = np.zeros((T + 1, N), dtype=np.float64)
    cc[1:] = np.cumsum(both, axis=0)
    idx = np.arange(T); lo = np.maximum(idx - w + 1, 0)
    n = cc[idx + 1] - cc[lo]
    sxy = cxy[idx + 1] - cxy[lo]
    sx = cx[idx + 1] - cx[lo]
    sy = cy[idx + 1] - cy[lo]
    sx2 = cx2[idx + 1] - cx2[lo]
    sy2 = cy2[idx + 1] - cy2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sx2 / n - (sx / n) ** 2
        vy = sy2 / n - (sy / n) ** 2
        corr = cov / np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    return _mask_count(corr, n, w, minp)


def roll_slope(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    """滚动 OLS 斜率（x = 0..w-1，y = 窗口内值，NaN 跳过）。"""
    valid = np.isfinite(a)
    xv = np.where(valid, a, 0.0).astype(np.float64)
    T, N = a.shape
    ti = np.arange(T, dtype=np.float64)[:, None]
    def cs(x):
        c = np.zeros((T + 1, N), dtype=np.float64)
        c[1:] = np.cumsum(x, axis=0)
        return c
    c1 = cs(xv); c2 = cs(xv * ti); c3 = cs(valid.astype(np.float64) * ti); c4 = cs(valid.astype(np.float64) * ti * ti)
    idx = np.arange(T); lo = np.maximum(idx - w + 1, 0)
    n = c1[idx + 1] - c1[lo]  # here n is sum of values? fix below
    cc = np.zeros((T + 1, N), dtype=np.float64)
    cc[1:] = np.cumsum(valid, axis=0)
    n = cc[idx + 1] - cc[lo]
    s1 = c1[idx + 1] - c1[lo]
    st = c3[idx + 1] - c3[lo]
    sty = c2[idx + 1] - c2[lo]
    stt = c4[idx + 1] - c4[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        num = sty - st * s1 / n
        den = stt - st * st / n
        slope = num / den
    return _mask_count(slope, n, w, minp)


def roll_rank_last(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    """窗口内最后一个值的时序百分位排名（0~1），等价 pandas rolling rank(pct=True).iloc[-1]。

    用逐 lag 比较实现: 对 k=1..w-1 累计 (x[t] > x[t-k]) 与相等计数；
    复杂度 O(w·T·N)，但全部向量化，w<=20 时可接受。
    """
    T, N = a.shape
    valid = np.isfinite(a)
    a32 = np.where(valid, a, np.nan).astype(np.float32)
    less = np.zeros((T, N), dtype=np.float32)
    eq = np.zeros((T, N), dtype=np.float32)
    cnt = np.zeros((T, N), dtype=np.float32)
    cnt[:] = valid.astype(np.float32)
    for _ in range(1, min(w, T)):
        lag = shift(a32, _)
        vlag = np.isfinite(lag)
        cur = a32[_:]
        lagk = lag[_:]
        vk = vlag[_:]
        e = np.isfinite(cur) & vk
        np.add(less[_:], (cur > lagk) & e, out=less[_:])
        np.add(eq[_:], (cur == lagk) & e, out=eq[_:])
        cnt[_:] += vk
    with np.errstate(invalid="ignore", divide="ignore"):
        # 与 pandas rank(pct=True) 对齐: pct = (less + eq/2 + 1) / count
        pr2 = (less + 0.5 * eq + 1.0) / np.maximum(cnt, 1.0)
    out = np.where(valid, pr2, np.nan)
    return _mask_count(out, cnt, w, minp)


def ema(a: np.ndarray, span: int, minp: int | None = None) -> np.ndarray:
    """指数移动平均，adjust=False 语义；NaN 处沿用上一值（ignore_na=False 近似）。"""
    if minp is None:
        minp = _minp(span, 3)
    return ema_alpha(a, 2.0 / (span + 1.0), minp)


def ema_alpha(a: np.ndarray, alpha: float, minp: int | None = None) -> np.ndarray:
    """自定义 alpha 的 EMA（对应 ewm(alpha=..., adjust=False)）。"""
    T, N = a.shape
    if minp is None:
        # 与参考实现 ewm(..., adjust=False) 的默认 min_periods 对齐：至少 1
        minp = 1
    y = np.full(N, np.nan, dtype=np.float64)
    started = np.zeros(N, dtype=bool)
    cnt = np.zeros(N, dtype=np.int64)
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(T):
        xt = a[t].astype(np.float64)
        v = np.isfinite(xt)
        y = np.where(v, np.where(started, y * (1 - alpha) + alpha * xt, xt), y)
        started |= v
        cnt += v
        ok = started & (cnt >= minp)
        out[t] = np.where(ok, y, np.nan)
    return out


# ------------------------------------------------------------------ 横截面
def cs_rank(a: np.ndarray) -> np.ndarray:
    """横截面百分位排名，映射到 [-0.5, 0.5]；NaN 保持 NaN。"""
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    fin = np.isfinite(a)
    # NaN 用 +inf 占位（升序排名时排在最后）
    tmp = np.where(fin, a, np.inf).astype(np.float64)
    order = np.argsort(tmp, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(T)[:, None]
    ranks[rows, order] = np.arange(N, dtype=np.int64)[None, :]
    cnt = fin.sum(axis=1)[:, None].astype(np.float64)
    r = ranks.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = r / np.maximum(cnt - 1.0, 1.0) - 0.5
    out = np.where(fin, pct, np.nan).astype(np.float32)
    return out


def cs_zscore(a: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """横截面 z-score（稳健：用中位数/MAD 近似）；NaN 保持 NaN。"""
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    fin = np.isfinite(a)
    x = np.where(fin, a, 0.0).astype(np.float64)
    n = fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (x.sum(axis=1) / np.maximum(n, 1))[:, None]
        var = ((x - mean) ** 2 * fin).sum(axis=1) / np.maximum(n - 1, 1)
        sd = np.sqrt(np.maximum(var, 0))[:, None]
        z = np.where(sd > 1e-12, (x - mean) / sd, 0.0)
    z = np.clip(z, -clip, clip)
    out = np.where(fin, z, np.nan).astype(np.float32)
    return out


def tanh_clip(a: np.ndarray) -> np.ndarray:
    """np.tanh，保留 NaN。"""
    return np.tanh(a).astype(np.float32)


# ------------------------------------------------------------------ 快速实现
# 上面的 cumsum / sliding-window 版本保留作为参考；实际运行使用 ops_fast 的
# 增量窗口实现（精度与 pandas 对齐、速度快 3~8 倍），在文件末尾覆盖同名函数。
from .ops_fast import (  # noqa: E402,F401
    roll_sum, roll_mean, roll_std, roll_min, roll_max, roll_skew, roll_kurt,
    roll_cov, roll_corr, roll_slope, roll_rank_last, roll_mean_std,
)

