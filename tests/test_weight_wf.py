# -*- coding: utf-8 -*-
"""``build/exp_weight_wf.py`` 的三条不变量。

权重实验最容易在三个地方悄悄出错，任一发生则整份对比作废：

1. **约束投影不生效** —— 转债权重超过用户设的 10% 上限，实验组与基线不可比。
2. **权重和不为 1** —— 隐式加杠杆或留现金，收益会被系统性放大/缩小。
3. **滚动窗口漏用未来数据** —— 若用 ``A[t-win:t]`` 之外的切片，样本外就变成样本内。

另附一条回归：FIX 基线的样本外 Sharpe 必须稳定在记录值上，否则说明数据或成本口径被改过。
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("w_wf", ROOT / "build" / "exp_weight_wf.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


def test_project_respects_caps_and_sum():
    """投影后必须满足 Σw=1、0≤w≤caps。"""
    caps = W.CAPS
    for raw in ([9, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0]):
        w = W.project(np.array(raw, dtype=float))
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= -1e-12).all()
        assert (w <= caps + 1e-9).all()


def test_project_never_exceeds_cb_cap():
    """转债 10% 是用户硬约束，任何输入都不能突破。"""
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = W.project(rng.random(4) * 10)
        assert w[3] <= W.CB_CAP + 1e-9, w


def test_inv_vol_rule_obeys_constraints():
    rng = np.random.default_rng(1)
    H = rng.normal(0, 0.01, size=(300, 4)) * np.array([1.0, 1.2, 0.5, 1.1])
    w = W.w_inv_vol(H)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w <= W.CAPS + 1e-9).all()
    assert w[3] <= W.CB_CAP + 1e-9
    # 波动最小的腿（ETF）应拿到最大权重
    assert int(np.argmax(w)) == 2


def test_optimizers_return_feasible_weights():
    rng = np.random.default_rng(2)
    H = rng.normal(0, 0.01, size=(500, 4)) * np.array([1.0, 1.1, 0.6, 1.0])
    for kind in ("minvar", "erc", "maxsr"):
        w = W._solve(H, kind)
        assert abs(w.sum() - 1.0) < 1e-8, (kind, w)
        assert (w >= -1e-12).all(), (kind, w)
        assert (w <= W.CAPS + 1e-8).all(), (kind, w)
        assert w[3] <= W.CB_CAP + 1e-8, (kind, w)


@pytest.fixture(scope="module")
def rets():
    import eval_combo_frontier_cb as F
    return F.common_returns(F.load_legs())


def test_fix_baseline_regression(rets):
    """FIX 基线的样本外 Sharpe 钉在 0.926（2026-11-06 记录值）。

    偏离超过 0.02 说明腿数据、成本口径或窗口被改过 —— 那本轮所有结论都要重跑。
    """
    res = W.walk_forward(rets, "fix")
    p = W.perf(res["nav"])
    assert abs(p["sharpe"] - 0.926) < 0.02, p
    assert abs(p["cagr"] - 0.0810) < 0.01, p
    # 固定权重下换手只来自漂移修复，应远小于优化方案
    assert res["turnover"] < 1.0
