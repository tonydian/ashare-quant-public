# -*- coding: utf-8 -*-
"""风格归因模块测试：恒等式闭合 + 无未来函数。"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_EXT = ROOT.parent / "data_extract" / "was_stock_data_ext"
if _EXT.exists():          # 迁移环境只有 was_stock_data 时不要污染全局 QF_DATA_RAW
    os.environ.setdefault("QF_DATA_RAW", str(_EXT))

from qf import attribution as qfa  # noqa: E402


def test_row_z():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan] + [5.0] * 30)
    m = np.ones_like(x, dtype=bool)
    z = qfa._row_z(x, m)
    finite = np.isfinite(z)
    assert abs(np.nanmean(z[finite])) < 1e-9
    assert abs(np.nanstd(z[finite]) - 1.0) < 1e-9
    assert np.isnan(z[4])          # NaN 保持
    assert np.isnan(qfa._row_z(x, np.zeros_like(m))).all()


def test_wls_identity_closed():
    """横截面 WLS 的逐股分解必须恒等：Σw·y = w·c + Σ coef·E + Σ w·u。"""
    rng = np.random.default_rng(7)
    n, k = 800, 4
    X = rng.normal(size=(n, k))
    w = np.abs(rng.normal(size=n)) + 0.1
    true = np.array([0.01, 0.02, -0.03, 0.005, 0.001])
    y = true[0] + X @ true[1:] + rng.normal(scale=0.01, size=n)

    coef, mu, sd, u, ok = qfa.wls_factor_returns(X, y, w)
    assert coef is not None and ok.all()

    Z = (X - mu) / sd
    E = (w[:, None] * Z).sum(axis=0) / w.sum()
    lhs = float((w * y).sum() / w.sum())
    rhs = coef[0] + float((E * coef[1:]).sum()) + float((w * u).sum() / w.sum())
    assert abs(lhs - rhs) < 1e-9
    # 系数应接近真值（同量级容差）
    assert np.allclose(coef[1:], true[1:], atol=0.05)


def test_styles_no_lookahead():
    """把 t 之后的所有行情/市值/成交额改掉，t 日的风格暴露必须逐位不变。"""
    from qf import data as qfdata

    md = qfdata.load_market(verbose=False)
    t = int(md.date_index("2024-06-28"))

    base = qfa.build_styles(md, [t])

    price = np.asarray(md.price).copy()
    mcap = np.asarray(md.mcap).copy()
    amount = np.asarray(md.amount).copy()
    rng = np.random.default_rng(11)
    with np.errstate(invalid="ignore"):
        price[t + 1:] = price[t + 1:] * (1.0 + rng.normal(scale=0.3, size=price[t + 1:].shape))
        mcap[t + 1:] = mcap[t + 1:] * (1.0 + rng.normal(scale=0.5, size=mcap[t + 1:].shape))
        amount[t + 1:] = amount[t + 1:] * (1.0 + rng.normal(scale=0.5, size=amount[t + 1:].shape))
    md.price, md.mcap, md.amount = price, mcap, amount
    try:
        pert = qfa.build_styles(md, [t])
    finally:
        pass

    for k in qfa.STYLES:
        a, b = base[k][0], pert[k][0]
        assert np.allclose(a, b, equal_nan=True), f"{k} 用了未来数据"


def test_attribute_accounting_on_real_run():
    """真实数据上跑 6 个调仓期：账目闭合 + R_actual 与 R_est 同量级。"""
    from qf import attribution as qfa2
    from qf import data as qfdata

    md = qfdata.load_market(verbose=False)
    run = ROOT / "build" / "runs" / "main"
    if not (run / "holdings.parquet").exists():
        import pytest
        pytest.skip("缺少 build/runs/main 回测产物")

    hold = pd.read_parquet(run / "holdings.parquet")
    nav = pd.read_parquet(run / "nav.parquet")["nav"]
    hold["date"] = pd.to_datetime(hold["date"])
    nav.index = pd.to_datetime(nav.index)

    dates = np.array(sorted(hold["date"].unique()))[:7]
    rows = np.array([md.date_index(d) for d in dates], dtype=int)
    styles = qfa2.build_styles(md, rows[:-1])
    _, con, _ = qfa2.attribute_periods(md, rows, hold, nav, styles,
                                       cash_daily_ret=None, ret_mode="close")
    assert not con.empty
    gap = con["R_est"] - con[[c for c in qfa2.COMPONENTS] + ["SPECIFIC"]].sum(axis=1)
    assert gap.abs().max() < 0.02, f"恒等式未闭合：{gap.abs().max():.4f}"
    # 分解总和应与可实现收益接近（差为现金/成本/时点）
    assert (con["R_actual"] - con["R_est"]).abs().max() < 0.15
