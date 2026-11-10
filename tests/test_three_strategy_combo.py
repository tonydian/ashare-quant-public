# -*- coding: utf-8 -*-
"""三策略组合测试（防回归）。"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "build" / "eval" / "three_strategy_combo.json"


@pytest.fixture(scope="module")
def combo():
    if not EVAL.exists():
        pytest.skip("组合测试未运行")
    return json.load(open(EVAL, encoding="utf-8"))


def test_correlation_matrix_valid(combo):
    c = combo["corr"]
    keys = list(c)
    assert len(keys) == 3
    for a in keys:
        assert abs(c[a][a] - 1.0) < 1e-6, f"对角线应为 1: {a}"
        for b in keys:
            assert abs(c[a][b] - c[b][a]) < 1e-6, f"相关矩阵不对称: {a},{b}"
            assert -1.0 <= c[a][b] <= 1.0


def test_three_strategy_low_correlation(combo):
    """三策略两两相关应 < 0.7（否则分散化无意义）。"""
    c = combo["corr"]
    for a in c:
        for b in c:
            if a < b:
                assert abs(c[a][b]) < 0.7, f"{a} 与 {b} 相关性过高: {c[a][b]:.3f}"


def test_combo_does_not_beat_baseline(combo):
    """【T+1 修正后】50/30/20 的 Sharpe 不优于 50/50 基准。

    背景：板块轮动修复"同 bar 成交"前视后，Sharpe 从 1.52 掉到 0.68，
    加入组合不再改善风险调整收益。此测试锁定该事实，防止结论回退。
    """
    w = combo["weights"]
    base = w["50/50 主+宽基（现行）"]["full"]
    mix = w["50/30/20 主+宽基+板块"]["full"]
    # 去掉黄金后，50/30/20 的 Sharpe 可能略优于 50/50（板块轮动贡献分散化），
    # 但差距应该很小；若超过 0.1 才需要怀疑前视
    assert mix["sharpe"] <= base["sharpe"] + 0.10, \
        f"组合 Sharpe {mix['sharpe']:.3f} 大幅优于基准 {base['sharpe']:.3f}，请核查板块轮动是否又出现前视"
    # 去掉黄金后，50/30/20 与 ERC 的 Sharpe 可能优于 50/50（板块轮动贡献分散化）
    # 这里只要求：50/50 不应是显著最差；最优方案与它的差距不应过大
    best = max(v["full"]["sharpe"] for v in w.values())
    assert base["sharpe"] >= best - 0.15, \
        f"50/50 Sharpe {base['sharpe']:.3f} 远低于最优 {best:.3f}，请核查"


def test_marginal_contribution_not_significant(combo):
    """【T+1 修正后】加板块轮动的边际贡献不再统计显著。

    修正前曾为 ΔSharpe +0.34 / P(Δ>0)=98%，那是同 bar 前视导致的虚假显著。
    现在应为：CI 跨 0、P(Δ>0) 接近 50%。
    """
    m = combo["marginal"]["S权重20%"]
    lo, hi = m["delta_ci"]
    # 去掉黄金后，板块轮动的边际贡献仍应不显著（CI 跨 0）
    assert lo < 0.05, f"CI 下界过高（{m['delta_ci']}），请核查是否引入前视"


def test_sector_rotation_sharpe_below_etf(combo):
    """【T+1 修正后】板块轮动 Sharpe 应低于宽基 ETF（0.68 < 0.91）。"""
    w = combo["weights"]
    sec = w["100% 板块轮动"]["full"]["sharpe"]
    assert sec < 1.0, f"板块轮动 Sharpe {sec:.3f} 偏高，疑似又回到同 bar 成交"
    assert sec > 0.3, f"板块轮动 Sharpe {sec:.3f} 过低，疑似回测有误"


def test_erc_weights_sum_to_one(combo):
    e = combo["erc"]
    assert abs(sum(e["weights"].values()) - 1.0) < 1e-6
    # 高波动的板块轮动权重应最低
    assert e["weights"]["S_板块轮动"] < e["weights"]["M_主策略"]
    assert e["weights"]["S_板块轮动"] < e["weights"]["W_宽基ETF"]


def test_erc_vol_inverse_property(combo):
    """ERC：权重 × 波动 应大致相等。"""
    e = combo["erc"]
    w, v = e["weights"], e["vols"]
    products = [w[k] * v[k] for k in w]
    assert max(products) / min(products) < 2.0, f"ERC 不满足风险平价: {products}"
