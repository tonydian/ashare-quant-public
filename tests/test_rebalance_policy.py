# -*- coding: utf-8 -*-
"""再平衡政策（漂移带 / 换手预算）单元测试。

三个不变式：
 1. 关闭时与历史口径**逐位一致**（NAV 完全相同）——新参数不得改变任何旧结果；
 2. 漂移带只在"要动的量太小"时跳过，**空仓首次建仓永不跳过**；
 3. 换手预算只收缩偏离，不改变"该卖还是要卖"（风险出口不被拦）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import data as qfdata                        # noqa: E402
from qf import ops                                   # noqa: E402
from qf.backtest import run_backtest                 # noqa: E402

START = "2021-01-04"
END = "2022-12-30"


@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def scores(md):
    r20 = ops.pct_change(md.price, 20)               # 只依赖 <=t 的价格
    return ops.cs_rank(-r20)


def _run(md, scores, **kw):
    return run_backtest(md, scores, md.date_index(START), md.date_index(END),
                        rebalance_days=40, trade_band_rel=0.30,
                        trade_band_abs=6000.0, max_new_entries=6,
                        verbose=False, **kw)


def test_disabled_matches_legacy_bitwise(md, scores):
    """关闭（None）与"零强度"参数必须得到完全相同的净值。"""
    base = _run(md, scores)
    zero = _run(md, scores, drift_band=0.0, turnover_budget=10.0)
    a = np.asarray(base.nav, dtype=float)
    b = np.asarray(zero.nav, dtype=float)
    assert np.array_equal(a, b), "关闭状态下净值必须逐位一致（新参数不得影响旧结果）"
    assert base.metrics["n_drift_skipped"] == 0
    assert base.metrics["n_budget_capped"] == 0
    assert zero.metrics["n_drift_skipped"] == 0


def test_drift_band_skips_and_cuts_turnover(md, scores):
    """漂移带足够大 → 出现跳过，且换手下降、但仍会真正调仓。"""
    base = _run(md, scores)
    hi = _run(md, scores, drift_band=0.50)
    assert hi.metrics["n_drift_skipped"] > 0, "漂移带未生效"
    assert hi.metrics["turnover_ann"] < base.metrics["turnover_ann"]
    assert hi.metrics["n_rebalances"] > 0, "不能把调仓全跳过"


def test_first_build_never_skipped(md, scores):
    """空仓首次建仓必须照常买入（否则整段回测都是空仓吃现金）。"""
    r = _run(md, scores, drift_band=10.0)
    assert r.metrics["n_buy_trades"] > 0, "首次建仓被漂移带跳过"
    assert r.metrics["avg_holdings"] > 0


def test_turnover_budget_caps_but_still_sells(md, scores):
    """换手预算只收缩偏离：换手下降、出现收缩计数，但卖出/清仓仍会发生。"""
    base = _run(md, scores)
    tight = _run(md, scores, turnover_budget=0.03)
    assert tight.metrics["n_budget_capped"] > 0, "换手预算未触发"
    assert tight.metrics["turnover_ann"] < base.metrics["turnover_ann"] * 0.9
    assert tight.metrics["n_sell_trades"] > 0, "风险出口不应被预算拦住"
    assert tight.metrics["max_dd"] > -0.60, "回撤异常，预算可能破坏了风控"
