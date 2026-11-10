# -*- coding: utf-8 -*-
"""Alpha191 需要的时序/横截面算子（在 (T, N) 面板上向量化）。

设计约定
--------
* 面板统一为 ``float64`` 的 ``(T, N)``，行=交易日，列=股票；``NaN`` 表示"无数据"。
* 窗口算子采用 **满窗口径**：窗口内只要有一个 NaN，输出就是 NaN（等价
  ``pandas.rolling(w, min_periods=w)``）。这与 JQ Alpha191 的取数口径一致
  （上市前 350 个交易日不足 → 因子值为空）。
* 所有算子都是纯函数，不修改入参。

性能
----
全部基于 ``cumsum`` / 逐 lag 原地 ``np.maximum`` / ``lfilter``，
单次调用量级 O(T·N) 或 O(w·T·N)（w 为窗口），对 2440×5281 的面板
单算子约 0.1~2 秒。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

__all__ = [
    "shift", "delta", "roll_sum", "roll_mean", "roll_std", "roll_count",
    "roll_corr", "roll_cov", "ts_max", "ts_min", "ts_rank", "ts_argmax",
    "ts_argmin", "decay", "wma", "sma", "sequence_beta", "prod",
    "rank_pct", "sum_where", "sign", "count_true",
]

NAN = np.nan


# ------------------------------------------------------------------ 基础
def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """向后平移 k 行（k=1 表示昨天），前 k 行填 NaN；保留入参 dtype。"""
    if k <= 0:
        return a
    out = np.full_like(a, NAN)
    if k < a.shape[0]:
        out[k:] = a[:-k]
    return out


def shift(a, n=1):
    return _shift(np.asarray(a), int(n))


def delta(a, n=1):
    """DELTA(A, n) = A_t - A_{t-n}。"""
    a = np.asarray(a, dtype=np.float64)
    return a - _shift(a, int(n))


def sign(a):
    return np.sign(a)


# ------------------------------------------------------------ 滚动求和/计数
def _cum(a: np.ndarray):
    ok = np.isfinite(a)
    s = np.cumsum(np.where(ok, a, 0.0), axis=0, dtype=np.float64)
    c = np.cumsum(ok, axis=0, dtype=np.int32)
    return s, c


def _win(s: np.ndarray, w: int) -> np.ndarray:
    """窗口差分：输出第 i 行 = 输入第 i+w-1 行减第 i-1 行。"""
    pad = np.zeros((1, s.shape[1]), dtype=s.dtype)
    sp = np.concatenate([pad, s], axis=0)
    return sp[w:] - sp[:-w]


def _roll(a: np.ndarray, w: int, fn) -> np.ndarray:
    w = int(w)
    out = np.full(a.shape, NAN, dtype=np.float64)
    if a.shape[0] < w:
        return out
    s, c = _cum(a)
    ss, cc = _win(s, w), _win(c, w)
    valid = cc >= w
    out[w - 1:] = np.where(valid, fn(ss, w), NAN)
    return out


def roll_sum(a, w):
    return _roll(np.asarray(a, dtype=np.float64), w, lambda s, w: s)


def roll_mean(a, w):
    return _roll(np.asarray(a, dtype=np.float64), w, lambda s, w: s / w)


def roll_count(a, w):
    """窗口内"有限值"的个数（入参可为布尔/0-1 数组）。"""
    a = np.asarray(a)
    return _roll(np.where(np.isfinite(a), 1.0, 0.0), w, lambda s, w: s)


def roll_std(a, w):
    a = np.asarray(a, dtype=np.float64)
    m = roll_mean(a, w)
    m2 = roll_mean(a * a, w)
    v = m2 - m * m
    np.maximum(v, 0.0, out=v)
    return np.sqrt(v, out=v)


def _col_blocks(n_cols: int, step: int = 1000):
    for j0 in range(0, n_cols, step):
        yield slice(j0, min(j0 + step, n_cols))


def _rolling_second(x, y, w):
    """返回 (Sx, Sy, Sxy, Sxx, Syy) 的窗口和；x/y 需已做 NaN→0 填充。"""
    sx = _win(np.cumsum(x, axis=0), w)
    sy = _win(np.cumsum(y, axis=0), w)
    sxy = _win(np.cumsum(x * y, axis=0), w)
    sxx = _win(np.cumsum(x * x, axis=0), w)
    syy = _win(np.cumsum(y * y, axis=0), w)
    return sx, sy, sxy, sxx, syy


def roll_cov(x, y, w):
    """滚动协方差（总体口径，除以 w）。按列分块，控制峰值内存。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = int(w)
    out = np.full(x.shape, NAN)
    if x.shape[0] < w:
        return out
    for sl in _col_blocks(x.shape[1]):
        xb, yb = x[:, sl], y[:, sl]
        ok = np.isfinite(xb) & np.isfinite(yb)
        x0, y0 = np.where(ok, xb, 0.0), np.where(ok, yb, 0.0)
        sx, sy, sxy, _, _ = _rolling_second(x0, y0, w)
        n = _win(np.cumsum(ok, axis=0, dtype=np.int32), w)
        v = sxy / w - (sx / w) * (sy / w)
        out[w - 1:, sl] = np.where(n >= w, v, NAN)
    return out


def roll_corr(x, y, w):
    """滚动 Pearson 相关（满窗口径，方差为 0 → NaN）。按列分块，控制峰值内存。

    5 个累加量（Σx, Σy, Σxy, Σx², Σy²）本来需要 5 份 (T,N) 的 float64，
    2440×5281 时峰值超过 1GB；分块后每块只有 ~1/5 的内存占用。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = int(w)
    out = np.full(x.shape, NAN)
    if x.shape[0] < w:
        return out
    for sl in _col_blocks(x.shape[1]):
        xb, yb = x[:, sl], y[:, sl]
        ok = np.isfinite(xb) & np.isfinite(yb)
        x0, y0 = np.where(ok, xb, 0.0), np.where(ok, yb, 0.0)
        sx, sy, sxy, sxx, syy = _rolling_second(x0, y0, w)
        n = _win(np.cumsum(ok, axis=0, dtype=np.int32), w)
        mx, my = sx / w, sy / w
        vx = sxx / w - mx * mx
        vy = syy / w - my * my
        cov = sxy / w - mx * my
        scale = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cov / np.where(scale > 1e-12, scale, NAN)
        r[(n < w) | ~np.isfinite(r)] = NAN
        out[w - 1:, sl] = r
    return out


# ---------------------------------------------------------------- 滚动极值
def _ts_extreme(a, w, fill, ufunc):
    a = np.asarray(a, dtype=np.float64)
    w = int(w)
    ok = np.isfinite(a)
    z = np.where(ok, a, fill)
    m = z.copy()
    for k in range(1, w):
        ufunc(m[k:], z[:-k], out=m[k:])
    cnt = roll_count(a, w)
    out = np.where(np.isfinite(cnt) & (cnt >= w), m, NAN)
    return out


def ts_max(a, w):
    return _ts_extreme(a, w, -np.inf, np.maximum)


def ts_min(a, w):
    return _ts_extreme(a, w, np.inf, np.minimum)


def _ts_pos(a, w, better):
    """窗口内最优值的位置（0=当天，n-1=窗口最早），用于 TSARGMAX/TSARGMIN。"""
    a = np.asarray(a, dtype=np.float64)
    w = int(w)
    T, N = a.shape
    best = np.full((T, N), np.nan)
    pos = np.full((T, N), np.nan)
    for k in range(w):
        sh = _shift(a, k)
        v = sh
        if k == 0:
            best = np.where(np.isfinite(v), v, NAN)
            pos = np.where(np.isfinite(v), float(k), NAN)
            continue
        upd = np.isfinite(v) & (~np.isfinite(best) | better(v, best))
        best = np.where(upd, v, best)
        pos = np.where(upd, float(k), pos)
    cnt = roll_count(a, w)
    return np.where(np.isfinite(cnt) & (cnt >= w), pos, NAN)


def ts_argmax(a, w):
    return _ts_pos(a, w, np.greater)


def ts_argmin(a, w):
    return _ts_pos(a, w, np.less)


# ---------------------------------------------------------------- 时序排名
def ts_rank(a, w):
    """TSRANK(A, n)：末位值在过去 n 个值中的排位（0~1，越大越高）。"""
    a = np.asarray(a, dtype=np.float64)
    w = int(w)
    cnt = np.zeros(a.shape)
    for k in range(w):
        sh = _shift(a, k)
        cnt += np.where(np.isfinite(sh) & (a > sh), 1.0, 0.0)
    valid = roll_count(a, w)
    out = cnt / float(w)
    return np.where(np.isfinite(valid) & (valid >= w), out, NAN)


# ---------------------------------------------------------------- 加权均线
def decay(a, w):
    """DECAYLINEAR(A, n)：权重 n, n-1, ..., 1 归一化。"""
    a = np.asarray(a, dtype=np.float64)
    w = int(w)
    acc = np.zeros(a.shape)
    for k in range(w):
        acc += (w - k) * _shift(a, k)
    acc /= (w * (w + 1) / 2.0)
    valid = roll_count(a, w)
    return np.where(np.isfinite(valid) & (valid >= w), acc, NAN)


def wma(a, w):
    """WMA(A, n)：GTJA 研报口径，权重 0.9^i（i=0 为当日），归一化。"""
    a = np.asarray(a, dtype=np.float64)
    w = int(w)
    wt = 0.9 ** np.arange(w)
    acc = np.zeros(a.shape)
    for k in range(w):
        acc += wt[k] * _shift(a, k)
    acc /= wt.sum()
    valid = roll_count(a, w)
    return np.where(np.isfinite(valid) & (valid >= w), acc, NAN)


def _ffill(a: np.ndarray) -> np.ndarray:
    """按列向前填充（不改动行首仍为 NaN 的位置）。"""
    T, N = a.shape
    idx = np.where(np.isfinite(a), np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(N)[None, :]]


def _bfill(a: np.ndarray) -> np.ndarray:
    return _ffill(a[::-1])[::-1]


def sma(a, n, m=1):
    """SMA(A, n, m)：``y_t = (m/n)·A_t + (1 - m/n)·y_{t-1}``（等价 ewm(alpha=m/n)）。

    递推用 ``scipy.signal.lfilter`` 实现。NaN 先做「前向 + 后向填充」，
    并把 ``y_{-1}`` 的初值设为每列的首个有效值（``zi``）——否则递推从 0 起步，
    会在上市后前 ``~3n`` 个交易日产生 ``(1-m/n)^t`` 的预热偏差。
    输出在原 NaN 位置仍为 NaN（等价 pandas ``ewm(adjust=False)`` 的取值）。
    """
    a = np.asarray(a, dtype=np.float64)
    al = float(m) / float(n)
    if al >= 1.0:
        return a.copy()
    z = _ffill(_bfill(a))
    z = np.where(np.isfinite(z), z, 0.0)
    zi = ((1.0 - al) * z[0])[None, :]
    y, _ = lfilter([al], [1.0, -(1.0 - al)], z, axis=0, zi=zi)
    return np.where(np.isfinite(a), y, NAN)


# ------------------------------------------------------------------ 其他
def sequence_beta(y, n):
    """REGBETA(y, SEQUENCE(n))：y 对时间序号 0..n-1 的滚动 OLS 斜率。"""
    y = np.asarray(y, dtype=np.float64)
    n = int(n)
    i = np.arange(n, dtype=np.float64)
    s_i = i.sum()
    s_ii = (i * i).sum()
    den = n * s_ii - s_i * s_i
    acc = np.zeros(y.shape)
    for k in range(n):
        acc += i[n - 1 - k] * _shift(y, k)
    mean = roll_mean(y, n)
    num = n * acc - s_i * n * mean
    out = num / den
    valid = roll_count(y, n)
    return np.where(np.isfinite(valid) & (valid >= n), out, NAN)


def prod(a, n):
    """PROD(A, n)：窗口内连乘（对正数用对数求和实现）。"""
    a = np.asarray(a, dtype=np.float64)
    n = int(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(np.where(a > 0, a, NAN))
    return np.exp(roll_sum(lg, n))


def rank_pct(a):
    """RANK(A)：横截面升序百分位（居中到 [-0.5, 0.5]，NaN 保持 NaN）。"""
    a = np.asarray(a, dtype=np.float64)
    T, N = a.shape
    ok = np.isfinite(a)
    if N == 0:
        return np.full_like(a, NAN)
    z = np.where(ok, a, np.inf)
    order = np.argsort(z, axis=1, kind="stable")
    r = np.empty((T, N), dtype=np.float64)
    np.put_along_axis(r, order, np.broadcast_to(np.arange(N, dtype=np.float64), (T, N)), axis=1)
    cnt = ok.sum(axis=1, keepdims=True).astype(np.float64)
    den = np.maximum(cnt - 1.0, 1.0)
    out = (r - (cnt - 1.0) / 2.0) / den
    out = np.where(ok, out, NAN)
    return out


def sum_where(x, n, cond):
    """SUMIF(x, n, cond)：窗口内 cond 为真处累加 x。"""
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(cond, dtype=bool)
    return roll_sum(np.where(c, x, 0.0), n)


def count_true(cond, n):
    """COUNT(cond, n)：窗口内条件为真的天数。

    注意：入参是布尔矩阵（比较结果为 False 表示"不成立"，也可能来自 NaN 比较），
    因此**上市前**的区间会算成 0 而不是 NaN；因子评估按 ``eligible`` 掩码取数，
    这部分不会进入横截面统计。
    """
    c = np.asarray(cond)
    return roll_sum(np.where(c.astype(bool), 1.0, 0.0), n)
