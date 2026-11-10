# -*- coding: utf-8 -*-
"""ops_patch：用增量窗口和替代 cumsum 版本（更快），保持与 ops 相同的 API。

实现方式：对每一行 t 做 O(1) 的窗口更新（加新值/减旧值），只维护长度 N 的累加器，
避免反复分配 (T,N) float64 数组；精度用 float64 累加器，输出 float32。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["roll_sum", "roll_mean", "roll_std", "roll_min", "roll_max",
           "roll_skew", "roll_kurt", "roll_cov", "roll_corr", "roll_slope",
           "roll_rank_last", "roll_mean_std"]


def _minp(w: int, floor: int = 5) -> int:
    return min(w, max(2, min(floor, w // 2 if w > 2 else 2)))


def _check(a):
    return np.asarray(a, dtype=np.float32)


def _win_loop(a: np.ndarray, w: int):
    """逐行更新窗口，yield (t, add, remove, n_add) ; add/remove 为 (N,) 向量。"""
    T, N = a.shape
    x = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    valid = np.isfinite(a)
    for t in range(T):
        add = x[t]
        vadd = valid[t]
        if t >= w:
            rem = x[t - w]
            vrem = valid[t - w]
        else:
            rem = None
            vrem = None
        yield t, add, rem, vadd, vrem


# ------------------------------------------------------------------ sum / mean
def roll_sum(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    a = _check(a)
    T, N = a.shape
    minp = _minp(w) if minp is None else int(minp)
    out = np.full((T, N), np.nan, dtype=np.float32)
    s = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.float64)
    x = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    valid = np.isfinite(a)
    for t in range(T):
        s += x[t]
        cnt += valid[t]
        if t >= w:
            s -= x[t - w]
            cnt -= valid[t - w]
        ok = cnt >= minp
        r = out[t]
        np.divide(s, 1.0, out=r, casting="unsafe")
        r[~ok] = np.nan
    return out


def roll_mean(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    a = _check(a)
    T, N = a.shape
    minp = _minp(w) if minp is None else int(minp)
    out = np.full((T, N), np.nan, dtype=np.float32)
    s = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.float64)
    x = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    valid = np.isfinite(a)
    for t in range(T):
        s += x[t]
        cnt += valid[t]
        if t >= w:
            s -= x[t - w]
            cnt -= valid[t - w]
        r = out[t]
        np.divide(s, np.maximum(cnt, 1.0), out=r)
        r[cnt < minp] = np.nan
    return out


def roll_mean_std(a: np.ndarray, w: int, minp: int | None = None, ddof: int = 0):
    """一次遍历同时返回 mean 和 std（z-score 变换用）。"""
    a = _check(a)
    T, N = a.shape
    minp = _minp(w) if minp is None else int(minp)
    mean = np.full((T, N), np.nan, dtype=np.float32)
    std = np.full((T, N), np.nan, dtype=np.float32)
    s1 = np.zeros(N, dtype=np.float64)
    s2 = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.float64)
    x = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    x2 = x * x
    valid = np.isfinite(a)
    for t in range(T):
        s1 += x[t]
        s2 += x2[t]
        cnt += valid[t]
        if t >= w:
            s1 -= x[t - w]
            s2 -= x2[t - w]
            cnt -= valid[t - w]
        n = np.maximum(cnt, 1.0)
        m = s1 / n
        var = s2 / n - m * m
        var = np.maximum(var, 0.0)
        if ddof:
            var = var * cnt / np.maximum(cnt - ddof, 1.0)
        r_m = mean[t]
        r_s = std[t]
        r_m[:] = m
        np.sqrt(var, out=r_s)
        bad = cnt < minp
        if bad.any():
            r_m[bad] = np.nan
            r_s[bad] = np.nan
    return mean, std


def roll_std(a: np.ndarray, w: int, minp: int | None = None, ddof: int = 0) -> np.ndarray:
    _, sd = roll_mean_std(a, w, minp=minp, ddof=ddof)
    return sd


# ------------------------------------------------------------------ cov / corr
def _roll_pair(a, b, w, minp, kind: str):
    a = _check(a)
    b = _check(b)
    T, N = a.shape
    minp = _minp(w) if minp is None else int(minp)
    out = np.full((T, N), np.nan, dtype=np.float32)
    both = np.isfinite(a) & np.isfinite(b)
    xa = np.where(both, a, 0.0).astype(np.float64)
    xb = np.where(both, b, 0.0).astype(np.float64)
    sxy = np.zeros(N, dtype=np.float64)
    sx = np.zeros(N, dtype=np.float64)
    sy = np.zeros(N, dtype=np.float64)
    sx2 = np.zeros(N, dtype=np.float64)
    sy2 = np.zeros(N, dtype=np.float64)
    cnt = np.zeros(N, dtype=np.float64)
    for t in range(T):
        sxy += xa[t] * xb[t]
        sx += xa[t]
        sy += xb[t]
        sx2 += xa[t] * xa[t]
        sy2 += xb[t] * xb[t]
        cnt += both[t]
        if t >= w:
            sxy -= xa[t - w] * xb[t - w]
            sx -= xa[t - w]
            sy -= xb[t - w]
            sx2 -= xa[t - w] * xa[t - w]
            sy2 -= xb[t - w] * xb[t - w]
            cnt -= both[t - w]
        n = np.maximum(cnt, 1.0)
        mx = sx / n
        my = sy / n
        cov = sxy / n - mx * my
        if kind == "cov":
            val = cov
        else:
            vx = np.maximum(sx2 / n - mx * mx, 0.0)
            vy = np.maximum(sy2 / n - my * my, 0.0)
            den = np.sqrt(vx * vy)
            val = np.where(den > 1e-15, cov / np.maximum(den, 1e-15), np.nan)
        r = out[t]
        r[:] = val
        r[cnt < minp] = np.nan
    return out


def roll_cov(a: np.ndarray, b: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    return _roll_pair(a, b, w, minp, "cov")


def roll_corr(a: np.ndarray, b: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    return _roll_pair(a, b, w, minp, "corr")


# ------------------------------------------------------------------ skew / kurt
def _roll_moment(a, w, minp, kind: str) -> np.ndarray:
    a = _check(a)
    T, N = a.shape
    minp = _minp(w) if minp is None else int(minp)
    out = np.full((T, N), np.nan, dtype=np.float32)
    x = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    valid = np.isfinite(a)
    s1 = np.zeros(N); s2 = np.zeros(N); s3 = np.zeros(N); s4 = np.zeros(N)
    cnt = np.zeros(N)
    for t in range(T):
        v = x[t]
        v2 = v * v
        s1 += v
        s2 += v2
        if kind in ("skew", "kurt"):
            s3 += v2 * v
            s4 += v2 * v2
        cnt += valid[t]
        if t >= w:
            u = x[t - w]
            u2 = u * u
            s1 -= u
            s2 -= u2
            s3 -= u2 * u
            s4 -= u2 * u2
            cnt -= valid[t - w]
        n = np.maximum(cnt, 1.0)
        mu = s1 / n
        m2 = np.maximum(s2 / n - mu * mu, 0.0)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            ok_var = m2 > 1e-20
            if kind == "skew":
                m3 = s3 / n - 3 * mu * s2 / n + 2 * mu ** 3
                g1 = m3 / np.power(np.maximum(m2, 1e-30), 1.5)
                # pandas 无偏修正: sqrt(n(n-1))/(n-2) * G1
                val = g1 * np.sqrt(n * (n - 1.0)) / np.maximum(n - 2.0, 1.0)
                ok = ok_var & (n >= 3)
            else:
                m4 = s4 / n - 4 * mu * s3 / n + 6 * mu * mu * s2 / n - 3 * mu ** 4
                g2 = m4 / np.maximum(m2, 1e-30) ** 2 - 3.0
                # pandas 无偏修正
                val = ((n - 1.0) * ((n + 1.0) * (g2 + 3.0) - 3.0 * (n - 1.0))) / \
                      np.maximum((n - 2.0) * (n - 3.0), 1.0)
                ok = ok_var & (n >= 4)
            val = np.where(ok, np.clip(val, -1e8, 1e8), np.nan)
        r = out[t]
        r[:] = val.astype(np.float32)
        r[cnt < minp] = np.nan
    return out


def roll_skew(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    return _roll_moment(a, w, minp, "skew")


def roll_kurt(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    return _roll_moment(a, w, minp, "kurt")


# ------------------------------------------------------------------ slope
def roll_slope(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    """滚动 OLS 斜率（x=0..w-1，NaN 跳过；默认要求全窗口）。"""
    a = _check(a)
    T, N = a.shape
    if minp is None:
        minp = w
    out = np.full((T, N), np.nan, dtype=np.float32)
    x = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    valid = np.isfinite(a)
    sy = np.zeros(N); sty = np.zeros(N); cnt = np.zeros(N)
    # Σ(i*y_i)，i 为绝对行号；窗口内相对斜率换算：Σ(k*y) = Σ(ty) - t0*Σy
    for t in range(T):
        sy += x[t]
        sty += t * x[t]
        cnt += valid[t]
        if t >= w:
            sy -= x[t - w]
            sty -= (t - w) * x[t - w]
            cnt -= valid[t - w]
        t0 = t - w + 1
        # Σ(k - kbar)(y - ybar) = Σk y - (Σk)(Σy)/w, k=0..w-1
        num = sty - t0 * sy - (w - 1) / 2.0 * sy
        st = w * (w - 1) / 2.0
        stt = (w - 1) * w * (2 * w - 1) / 6.0
        den = stt - st * st / w
        r = out[t]
        r[:] = num / den
        r[cnt < minp] = np.nan
    return out


# ------------------------------------------------------------------ min / max
def _roll_mm(a: np.ndarray, w: int, minp, kind: str) -> np.ndarray:
    a = _check(a)
    T, N = a.shape
    minp = _minp(w) if minp is None else int(minp)
    out = np.full((T, N), np.nan, dtype=np.float32)
    # 窗口有效计数
    cnt = np.zeros((T, N), dtype=np.float32)
    run = np.zeros(N, dtype=np.float64)
    valid = np.isfinite(a)
    vv = valid.astype(np.float64)
    for t in range(T):
        run += vv[t]
        if t >= w:
            run -= vv[t - w]
        cnt[t] = run
    # 真正的 min/max 用 chunked sliding window（向量化 min/max 求值性价比高）
    # 注意：块之间必须重叠 w-1 行，否则会在块边界丢行（verifier F-01）。
    # 大窗口（w>40）改用 pandas rolling 按列分块：sliding_window_view 的中间张量
    # 会随 w 线性膨胀，且 rows<w 时原实现会直接 break（w=55/250 曾全 NaN）。
    if 1 < w <= 40:
        rows = max(w + 1, int(8e6 / max(1, w * N)))
        step = max(1, rows - (w - 1))
        for s in range(0, T, step):
            e = min(T, s + rows)
            if e - s < w:
                break
            v = np.lib.stride_tricks.sliding_window_view(a[s:e], w, axis=0)
            with np.errstate(all="ignore"):
                r = np.nanmin(v, axis=-1) if kind == "min" else np.nanmax(v, axis=-1)
            m = e - (s + w - 1)
            out[s + w - 1:s + w - 1 + m] = r[:m]
    elif w > 40:
        # 列分块 pandas rolling（C 实现，内存有界）
        block = max(256, int(8_000_000 / max(T, 1)))
        for c0 in range(0, N, block):
            c1 = min(N, c0 + block)
            df = pd.DataFrame(a[:, c0:c1])
            rdf = (df.rolling(w, min_periods=minp).min() if kind == "min"
                   else df.rolling(w, min_periods=minp).max())
            out[:, c0:c1] = rdf.to_numpy(dtype=np.float32)
    # 前 w-1 行部分窗口
    n_head = min(w - 1, T)
    runmin = np.full(N, np.nan, dtype=np.float32)
    for t in range(n_head):
        row = a[t]
        v = np.isfinite(row)
        if kind == "min":
            runmin = np.where(v, np.where(np.isfinite(runmin), np.fmin(runmin, row), row), runmin)
        else:
            runmin = np.where(v, np.where(np.isfinite(runmin), np.fmax(runmin, row), row), runmin)
        out[t] = runmin
    out[cnt < minp] = np.nan
    return out


def roll_min(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    return _roll_mm(a, w, minp, "min")


def roll_max(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    return _roll_mm(a, w, minp, "max")


# ------------------------------------------------------------------ ts rank
def roll_rank_last(a: np.ndarray, w: int, minp: int | None = None) -> np.ndarray:
    """窗口内最后一个值的时序百分位排名（与 pandas rank(pct=True).iloc[-1] 对齐）。"""
    a = _check(a)
    T, N = a.shape
    minp = _minp(w) if minp is None else int(minp)
    a = np.where(np.isfinite(a), a, np.nan).astype(np.float32)
    less = np.zeros((T, N), dtype=np.float32)
    eq = np.zeros((T, N), dtype=np.float32)
    for k in range(1, min(w, T)):
        np.add(less[k:], (a[k:] > a[:-k]), out=less[k:])
        np.add(eq[k:], (a[k:] == a[:-k]), out=eq[k:])
    # 窗口内有效值计数（含自身）
    valid = np.isfinite(a)
    cnt = np.zeros((T, N), dtype=np.float32)
    run = np.zeros(N, dtype=np.float64)
    vv = valid.astype(np.float64)
    for t in range(T):
        run += vv[t]
        if t >= w:
            run -= vv[t - w]
        cnt[t] = run
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (less + 0.5 * eq + 1.0) / np.maximum(cnt, 1.0)
    out = np.where(valid & (cnt >= minp), out, np.nan)
    return out.astype(np.float32)
