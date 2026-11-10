# -*- coding: utf-8 -*-
"""PSR / DSR / 有效试验数 的单元测试。

不变式（都对公式成立，与具体数据无关）：
 1. N=1 时 SR\\*=0，DSR 必须等于 PSR；
 2. DSR 随 N 单调下降（试得越多越难通过）；
 3. DSR 随 T 单调上升（样本越长越可信）；
 4. ``effective_n_trials`` 边界：ρ=0→N、ρ=1→1、N=1→1；
 5. ``min_sr_for_confidence`` 反解出的 SR 代回 DSR 必须命中目标置信度；
 6. 与手算的 PSR 一致（正态、SR=0 时 PSR 恰为 0.5）。
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import dsr  # noqa: E402


def test_sharpe_stat_matches_manual():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 500)
    s = dsr.sharpe_stat(r)
    assert s["T"] == 500
    assert s["sr"] == pytest.approx(r.mean() / r.std(ddof=1))
    assert s["sr_ann"] == pytest.approx(s["sr"] * math.sqrt(252))
    assert s["kurt"] > 0


def test_n1_equals_psr():
    sr, t, skew, kurt = 0.06, 1600, -0.3, 4.2
    assert dsr.expected_max_sharpe(0.01, 1) == 0.0
    a = dsr.deflated_sharpe_ratio(sr, t, skew, kurt, n_trials=1)
    b = dsr.probabilistic_sharpe_ratio(sr, t, skew, kurt, benchmark=0.0)
    assert a == pytest.approx(b)


def test_dsr_decreases_with_trials():
    vals = [dsr.deflated_sharpe_ratio(0.05, 1600, 0.0, 3.0, n_trials=n)
            for n in (1, 5, 20, 100, 400)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_dsr_increases_with_T():
    vals = [dsr.deflated_sharpe_ratio(0.05, t, 0.0, 3.0, n_trials=30)
            for t in (250, 500, 1000, 2000)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def test_psr_zero_sr_is_half():
    assert dsr.probabilistic_sharpe_ratio(0.0, 1000) == pytest.approx(0.5)


def test_effective_n_trials_bounds():
    assert dsr.effective_n_trials(1, 0.5) == 1.0
    assert dsr.effective_n_trials(30, 0.0) == pytest.approx(30.0)
    assert dsr.effective_n_trials(30, 1.0) == pytest.approx(1.0, abs=1e-4)
    mid = dsr.effective_n_trials(30, 0.5)
    assert 1.0 < mid < 30.0


def test_expected_max_sharpe_monotone_and_positive():
    v = [dsr.expected_max_sharpe(0.02, n) for n in (2, 10, 100, 1000)]
    assert all(x > 0 for x in v)
    assert all(v[i] < v[i + 1] for i in range(len(v) - 1))
    # 与闭式解的一致性：N=2 时 Φ⁻¹(0.5)=0，只剩 γ 项
    g = dsr.EULER_GAMMA
    manual = 0.02 * ((1 - g) * norm.ppf(0.5) + g * norm.ppf(1 - 1 / (2 * math.e)))
    assert dsr.expected_max_sharpe(0.02, 2) == pytest.approx(manual)


def test_min_sr_round_trip():
    t, n, target = 1600, 30, 0.95
    sr = dsr.min_sr_for_confidence(target_dsr=target, t=t, n_trials=n)
    assert dsr.deflated_sharpe_ratio(sr, t, n_trials=n) == pytest.approx(target, abs=1e-4)
