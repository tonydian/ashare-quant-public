# -*- coding: utf-8 -*-
"""多重检验折扣：PSR / DSR（Deflated Sharpe Ratio）/ 有效独立试验数。

背景
----
我们过去靠"8 相位带 + bootstrap CI"判断一个方案能不能上线，但**从没对"试了多少个方案"
做折扣**。试 30 个格子挑最好的那个，和只试 1 个就选中，可信度完全不同。这个模块提供
Bailey & López de Prado (2014) 的标准工具：

* ``probabilistic_sharpe_ratio``（PSR）：给定 SR、样本长度、偏度峰度，SR > 基准的概率；
* ``expected_max_sharpe``（SR\\*）：在 N 个**独立**试验里，纯运气能达到的最大 SR 期望；
* ``deflated_sharpe_ratio``（DSR）：把 SR 减去 SR\\* 之后再算概率，即"扣掉选择偏差后的可信度"；
* ``effective_n_trials``：试验高度相关时的有效独立试验数（N_eff < N）。

口径（**必须遵守**）
------------------
所有函数的 ``sr`` 都是**单期（非年化）**夏普，即 ``mean(r)/std(r)``。
年化夏普 ``SR_ann = sr * sqrt(ppY)``。用年化 SR 直接套公式会把 SR\\* 放大 sqrt(252) 倍，
结论完全错。用 ``sharpe_stat`` 取口径正确的值。

公式
----
``SR* = sqrt(Var(SR_trials)) · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]``，
γ = 0.5772156649（Euler–Mascheroni），e 为自然常数。

``DSR = Φ( (SR - SR*) / sqrt(Var_null) )``，其中
``Var_null = (1 - γ3·SR + (γ4-1)/4·SR²) / (T-1)``（γ3 偏度、γ4 **非超额**峰度，正态=3）。

当拿不到"N 个试验的 SR 分布"时（通常拿不到），``sr_std`` 用 ``Var_null`` 的平方根代替——
这是 Bailey 原文给的做法，也是**偏保守**的一侧（假设试验之间的 SR 离散度等于单次估计的
抽样误差）。能拿到横截面 SR 分布时，直接传 ``sr_std`` 更准。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


# ---------------------------------------------------------------- 基础统计
def sharpe_stat(returns, periods_per_year: int = TRADING_DAYS) -> dict:
    """把日收益序列压成 DSR 需要的四个量（SR 为**单期**值）。"""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = int(r.size)
    if t < 3:
        raise ValueError(f"样本不足：T={t}，至少需要 3 个观测")
    sd = float(np.std(r, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("收益序列标准差为 0，无法计算夏普")
    mean = float(np.mean(r))
    sr = mean / sd
    z = (r - mean) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))          # 非超额峰度：正态 = 3
    return {
        "sr": sr,
        "sr_ann": sr * math.sqrt(periods_per_year),
        "T": t,
        "skew": skew,
        "kurt": kurt,
        "ann_vol": sd * math.sqrt(periods_per_year),
        "periods_per_year": int(periods_per_year),
    }


def sr_variance_null(sr: float, t: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """单次 SR 估计在零假设下的方差（Bailey 式 (9)）。"""
    if t <= 1:
        raise ValueError("T 必须 > 1")
    v = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (t - 1.0)
    return float(max(v, 1e-18))


# ---------------------------------------------------------------- 概率换算
def probabilistic_sharpe_ratio(sr: float, t: int, skew: float = 0.0, kurt: float = 3.0,
                               benchmark: float = 0.0) -> float:
    """PSR：真实 SR 大于 ``benchmark`` 的概率（单期 SR）。"""
    var = sr_variance_null(sr, t, skew, kurt)
    z = (sr - benchmark) / math.sqrt(var)
    return float(norm.cdf(z))


def expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """N 个独立试验下、纯运气能达到的最大 SR 的期望（单期 SR 口径）。"""
    n = int(n_trials)
    if n <= 1:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n)
    b = norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(sr_std * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(sr: float, t: int, skew: float = 0.0, kurt: float = 3.0,
                          n_trials: int = 1, sr_std: float | None = None) -> float:
    """DSR：扣掉"试了 N 个方案"的选择偏差后，SR > 0 的概率。"""
    var = sr_variance_null(sr, t, skew, kurt)
    std = math.sqrt(var) if sr_std is None else float(sr_std)
    sr_star = expected_max_sharpe(std, n_trials)
    return float(norm.cdf((sr - sr_star) / math.sqrt(var)))


def effective_n_trials(n_trials: int, rho_bar: float) -> float:
    """试验两两相关性为 ``rho_bar`` 时的有效独立试验数。

    ``N_eff = N / (1 + (N-1)·ρ̄)``：ρ̄=0 → N；ρ̄=1 → 1。
    """
    n = float(n_trials)
    if n <= 1:
        return 1.0
    rho = float(np.clip(rho_bar, 0.0, 0.999999))
    return float(n / (1.0 + (n - 1.0) * rho))


def min_sr_for_confidence(target_dsr: float = 0.95, t: int = 252, skew: float = 0.0,
                          kurt: float = 3.0, n_trials: int = 1,
                          sr_std: float | None = None,
                          hi: float = 1.0) -> float:
    """反解：要拿到 ``target_dsr`` 的置信度，单期 SR 至少要多大。

    返回值可直接 ``× sqrt(252)`` 成年化，用来回答"以后新方案得做到多少才算数"。
    """
    lo = -hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if deflated_sharpe_ratio(mid, t, skew, kurt, n_trials, sr_std) < target_dsr:
            lo = mid
        else:
            hi = mid
    return float(hi)
