# -*- coding: utf-8 -*-
"""统计显著性校正工具（backtest-guard 体检中 1/中 2 项修复）。

解决的问题
----------
1) **标签重叠**：周频步长 5 个交易日、h10 标签跨 11 个交易日 → 相邻 IC 样本重叠 54.5%，
   朴素 t 统计量被高估（IC naive t=14.2 vs 有效样本 t≈9.6）。提供：
   - ``newey_west_tstat``：Newey-West (Bartlett) HAC 方差下的 t；
   - ``effective_n``：按 step/horizon 折算有效样本数的保守近似；
   - ``circular_block_bootstrap_ci``：循环块 bootstrap 置信区间（同时适用 IC 与 Sharpe）。
2) **多重筛选**：800 因子→34 入池没有多重检验校正。提供：
   - ``bh_fdr``：Benjamini-Hochberg FDR 控制，返回拒绝集与阈值；
   - ``p_value_two_sided``：正态近似双侧 p 值（用于把 t 统计量转 p）。
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "p_value_two_sided", "newey_west_var", "newey_west_tstat",
    "effective_n", "circular_block_bootstrap_ci", "sharpe_ratio", "bh_fdr",
]


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_value_two_sided(t: float) -> float:
    """正态近似双侧 p 值。"""
    t = float(t)
    if not np.isfinite(t):
        return float("nan")
    return float(2.0 * (1.0 - _norm_cdf(abs(t))))


def newey_west_var(x: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett kernel) 长期方差估计（对均值）。

    返回 ``Var(mean)``，即已除以 n 的长期方差，可直接用于 t = mean / sqrt(Var(mean))。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    lag = int(max(0, min(lag, n - 2)))
    e = x - x.mean()
    gamma0 = float(np.dot(e, e) / n)
    var = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        gl = float(np.dot(e[l:], e[:-l]) / n)
        var += 2.0 * w * gl
    return float(max(var, 0.0) / n)


def newey_west_tstat(x: np.ndarray, lag: int | None = None) -> dict:
    """均值显著性的 Newey-West t 统计量。

    lag 缺省用 ``round(1.5 * n^(1/3))``（周频 IC 序列通常得到 5~8）；
    对"标签跨 h、步长 s"的 IC 序列，建议显式传 ``lag=ceil(h/s)``。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "t": float("nan"), "lag": 0, "n": n}
    if lag is None:
        lag = int(round(1.5 * n ** (1.0 / 3.0)))
    var_mean = newey_west_var(x, lag)
    mean = float(x.mean())
    t = mean / math.sqrt(var_mean) if (np.isfinite(var_mean) and var_mean > 0) else float("nan")
    naive = mean / (x.std(ddof=1) / math.sqrt(n)) if x.std(ddof=1) > 0 else float("nan")
    return {"mean": mean, "t": float(t), "naive_t": float(naive),
            "lag": int(lag), "n": int(n), "se": float(math.sqrt(var_mean))}


def effective_n(n: int, step: int, horizon: int) -> float:
    """重叠标签折算后的有效样本数：n * step / horizon（保守近似）。"""
    if horizon <= 0:
        return float(n)
    return float(n) * float(step) / float(horizon)


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    excess = r - float(rf) / periods
    sd = excess.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(periods))


def circular_block_bootstrap_ci(x: np.ndarray, stat_fn, n_boot: int = 2000,
                                block_len: int = 20, alpha: float = 0.05,
                                seed: int = 20261023) -> dict:
    """循环块 bootstrap 置信区间（保留序列的局部相关性，适合重叠样本/自相关收益）。

    ``stat_fn(sample) -> float``；返回点估计与 ``alpha/2, 1-alpha/2`` 分位。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10 or block_len < 1:
        return {"point": float(stat_fn(x)) if n else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offs = np.arange(block_len)
    idx = (starts[:, :, None] + offs[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    samples = x[idx]
    stats = np.array([stat_fn(s) for s in samples], dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    if len(stats) < 50:
        return {"point": float(stat_fn(x)), "lo": float("nan"), "hi": float("nan"),
                "n_boot": int(len(stats))}
    return {"point": float(stat_fn(x)),
            "lo": float(np.quantile(stats, alpha / 2)),
            "hi": float(np.quantile(stats, 1 - alpha / 2)),
            "n_boot": int(len(stats))}


def bh_fdr(p_values: np.ndarray, q: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR 控制（独立/正相关假设下控制 FDR ≤ q）。

    返回：``reject``（bool 掩码，与输入同序）、``k``（拒绝个数）、``threshold``
    （第 k 个 p 的 BH 临界值）、``p_sorted``/``crit``（供报告展示曲线）。
    """
    p = np.asarray(p_values, dtype=np.float64)
    ok = np.isfinite(p)
    n = int(ok.sum())
    reject = np.zeros(len(p), dtype=bool)
    if n == 0:
        return {"reject": reject, "k": 0, "threshold": float("nan"), "n_tests": 0}
    idx = np.flatnonzero(ok)
    order = np.argsort(p[idx], kind="stable")
    p_sorted = p[idx][order]
    crit = q * (np.arange(1, n + 1) / n)
    passed = p_sorted <= crit
    if passed.any():
        k = int(np.flatnonzero(passed)[-1]) + 1
        reject[idx[order[:k]]] = True
        threshold = float(crit[k - 1])
    else:
        k, threshold = 0, float(crit[0])
    return {"reject": reject, "k": int(k), "threshold": threshold, "n_tests": int(n),
            "p_sorted": p_sorted, "crit": crit}
