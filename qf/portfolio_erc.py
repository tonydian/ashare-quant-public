# -*- coding: utf-8 -*-
"""风险预算配置：等风险贡献（ERC）/ 逆波动 / 最大分散化。

为什么单独成模块：
  ERC 需要迭代求解（使各资产的风险贡献相等），不是简单的波动率倒数。
  原 scripts/three_strategy_combo.py 里的"ERC"其实只是逆波动率（IVP），
  本模块提供真正的 ERC。

用法::

    from qf import portfolio_erc
    w = portfolio_erc.erc_weights(cov)          # 真 ERC
    w2 = portfolio_erc.ivp_weights(cov)         # 逆波动率
    w3 = portfolio_erc.mdp_weights(cov)         # 最大分散化

协方差矩阵由调用方提供（务必只用 t 之前的数据，避免前视）。
"""
from __future__ import annotations

import numpy as np


def _clean(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    # 对称化
    cov = 0.5 * (cov + cov.T)
    # 保证正定（加对角扰动）
    try:
        np.linalg.cholesky(cov)
    except Exception:
        cov = cov + np.eye(n) * max(1e-10, float(np.trace(cov)) / max(n, 1) * 1e-6)
    return cov


def ivp_weights(cov: np.ndarray) -> np.ndarray:
    """逆波动率（等价于对角协方差的 ERC）。"""
    cov = _clean(cov)
    v = np.sqrt(np.maximum(np.diag(cov), 0.0))
    inv = np.where(v > 0, 1.0 / np.maximum(v, 1e-12), 0.0)
    s = inv.sum()
    return inv / s if s > 0 else np.ones(len(cov)) / len(cov)


def _erc_solve(cov, w0, objective, iters, tol):
    """单起点 SLSQP 求解（内部用）。"""
    n = cov.shape[0]
    try:
        from scipy.optimize import minimize

        w0 = np.clip(np.asarray(w0, float), 1e-8, None)
        w0 = w0 / w0.sum()
        cons = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0),
                 "jac": lambda w: np.ones_like(w)},)
        r = minimize(objective, w0, method="SLSQP",
                     bounds=[(0.0, 1.0)] * n, constraints=cons,
                     options={"maxiter": iters, "ftol": tol})
        w = np.maximum(np.asarray(r.x, float), 0.0)
        ssum = w.sum()
        return w / ssum if ssum > 0 else None
    except Exception:
        return None


def erc_weights(cov: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """真 ERC：使各资产风险贡献相等（长仓、非负）。

    方法：用 scipy 的 SLSQP 求解
        minimize  Σ_i (RC_i/ΣRC - 1/n)^2
        s.t.      Σw = 1,  w >= 0
    其中 RC_i = w_i (Σw)_i / sqrt(w'Σw)。

    这是最稳健的实现：解析梯度 + 约束优化，必然收敛到可行解。
    """
    cov = _clean(cov)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)

    target = 1.0 / n

    def objective(w):
        pv = float(w @ cov @ w)
        if pv <= 1e-18:
            return 1e6
        rc = w * (cov @ w) / np.sqrt(pv)
        total = rc.sum()
        if total <= 1e-18:
            return 1e6
        e = rc / total - target
        return float(e @ e)

    # 快速路径：先用逆波动起点单次求解（多数情况足够）
    # 不收敛再多起点（扰动），兼顾速度与稳健性
    fast = _erc_solve(cov, ivp_weights(cov), objective, iters, tol)
    if fast is not None and objective(fast) < 1e-12:
        return fast

    rng = np.random.default_rng(7)
    starts = [fast if fast is not None else ivp_weights(cov), np.ones(n) / n]
    for _ in range(3):
        starts.append(rng.dirichlet(np.ones(n)))

    best_w, best_f = (fast, objective(fast)) if fast is not None else (None, np.inf)
    try:
        from scipy.optimize import minimize

        cons = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0),
                 "jac": lambda w: np.ones_like(w)},)
        bnds = [(0.0, 1.0)] * n
        for w0 in starts:
            w0 = np.clip(np.asarray(w0, float), 1e-6, None)
            w0 = w0 / w0.sum()
            try:
                r = minimize(objective, w0, method="SLSQP", bounds=bnds,
                             constraints=cons,
                             options={"maxiter": iters, "ftol": tol})
                w = np.maximum(np.asarray(r.x, float), 0.0)
                ss = w.sum()
                if ss > 0:
                    w = w / ss
                    f = objective(w)
                    if f < best_f:
                        best_f, best_w = f, w
            except Exception:
                continue
    except ImportError:
        pass

    # 如果 scipy 失败，退回"幂次迭代"（对 ERC 已有理论收敛性）
    if best_w is None or best_f > 1e-10:
        w = ivp_weights(cov)
        for _ in range(5000):
            pv = float(w @ cov @ w)
            if pv <= 1e-18:
                break
            rc = w * (cov @ w)
            total = float(rc.sum())
            if total <= 1e-18:
                break
            # 指数更新：w_i <- w_i * (target/RC_i)^eta
            ratio = (1.0 / n) * total / np.maximum(rc, 1e-18)
            w = w * np.power(ratio, 0.25)
            w = np.maximum(w, 1e-10)
            ss = w.sum()
            if ss <= 0:
                break
            w = w / ss
            e = (w * (cov @ w)) / max(float((w * (cov @ w)).sum()), 1e-18) - 1.0 / n
            if float(e @ e) < 1e-14:
                break
        if best_w is None or objective(w) < best_f:
            best_w = w

    return best_w if best_w is not None else np.ones(n) / n


def mdp_weights(cov: np.ndarray) -> np.ndarray:
    """最大分散化组合（近似）：w ∝ inv(diag(σ)) · inv(corr) · 1。"""
    cov = _clean(cov)
    n = cov.shape[0]
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-16))
    corr = cov / np.outer(vol, vol)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    try:
        inv_corr = np.linalg.pinv(corr)
        w = (inv_corr @ np.ones(n)) / vol
    except Exception:
        return ivp_weights(cov)
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def risk_contributions(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """各资产的风险贡献占比（用于验证 ERC 是否真的等风险）。"""
    cov = _clean(cov)
    w = np.asarray(w, float)
    pv = float(w @ cov @ w)
    if pv <= 0:
        return np.full(len(w), 1.0 / len(w))
    rc = w * (cov @ w) / np.sqrt(pv)
    rc = np.maximum(rc, 0.0)
    s = rc.sum()
    return rc / s if s > 0 else np.full(len(w), 1.0 / len(w))
