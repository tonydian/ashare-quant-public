# -*- coding: utf-8 -*-
"""统计校正工具的单元测试（数值正确性 + 边界）。"""
from __future__ import annotations

import numpy as np
import pytest

from qf import stats_adjust as qsa


def test_p_value_two_sided_known_values():
    assert qsa.p_value_two_sided(0.0) == pytest.approx(1.0, abs=1e-12)
    assert qsa.p_value_two_sided(1.96) == pytest.approx(0.05, abs=5e-3)
    assert qsa.p_value_two_sided(-1.96) == pytest.approx(0.05, abs=5e-3)
    assert qsa.p_value_two_sided(10.0) < 1e-20


def test_newey_west_equals_naive_for_iid_large_n():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 1.0, size=20000)
    nw = qsa.newey_west_tstat(x, lag=0)
    # lag=0 时 NW 退化为普通标准误（NW 用 1/n 分母，naive 用 ddof=1，大 n 下一致）
    assert nw["t"] == pytest.approx(nw["naive_t"], rel=5e-3)


def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(1)
    n = 2000
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + e[i]          # AR(1)，正自相关
    nw = qsa.newey_west_tstat(x, lag=10)
    assert abs(nw["t"]) < abs(nw["naive_t"]), "正自相关应让 t 变小"


def test_effective_n_formula():
    assert qsa.effective_n(326, 5, 10) == pytest.approx(163.0)
    assert qsa.effective_n(100, 5, 5) == pytest.approx(100.0)


def test_bootstrap_ci_contains_point_and_reasonable_width():
    rng = np.random.default_rng(2)
    x = rng.normal(0.1, 1.0, size=600)
    ci = qsa.circular_block_bootstrap_ci(x, lambda s: float(np.mean(s)),
                                         n_boot=500, block_len=10)
    assert ci["lo"] < ci["point"] < ci["hi"]
    assert ci["hi"] - ci["lo"] > 0


def test_sharpe_ratio_positive_for_positive_drift():
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, size=1000)
    assert qsa.sharpe_ratio(r) > 0


def test_bh_fdr_known_example():
    # 全不显著 → 0 个
    r = qsa.bh_fdr(np.full(100, 0.9), q=0.05)
    assert r["k"] == 0 and not r["reject"].any()
    # 全极显著 → 全部
    r = qsa.bh_fdr(np.full(100, 1e-10), q=0.05)
    assert r["k"] == 100 and r["reject"].all()
    # 加入 NaN 不报错
    p = np.array([0.001, np.nan, 0.5, 0.9])
    r = qsa.bh_fdr(p, q=0.05)
    assert r["n_tests"] == 3 and len(r["reject"]) == 4


def test_bh_fdr_monotone_in_q():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, size=200)
    k05 = qsa.bh_fdr(p, q=0.05)["k"]
    k10 = qsa.bh_fdr(p, q=0.10)["k"]
    assert k10 >= k05


def test_bh_fdr_matches_manual_threshold():
    # BH: 取最大的 k 使 p_(k) <= q*k/n（n=8, q=0.05 → 临界 0.00625*k）
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    r = qsa.bh_fdr(p, q=0.05)
    # k=1: .001<=.00625 OK; k=2: .008<=.0125 OK; k=3: .039<=.01875 NO
    assert r["k"] == 2
    # 手算：k=1: .001<=.00625 ✓; k=2: .008<=.0125 ✓; k=3: .039<=.01875 ✗ → k=2
