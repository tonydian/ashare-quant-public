# -*- coding: utf-8 -*-
"""``n_holdings`` 透传参数的单元测试。

三个不变式：
 1. 不传 ``n_holdings``（生产路径）→ 与显式传 ``cfg.N_HOLDINGS`` **逐位一致**；
 2. 传小 N → 实际平均持仓数确实变小（参数真的生效，不是被忽略）；
 3. ``config`` 里记录的是**生效值**，不是全局默认值。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C                            # noqa: E402
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
    r20 = ops.pct_change(md.price, 20)
    return ops.cs_rank(-r20)


def _run(md, scores, **kw):
    return run_backtest(md, scores, md.date_index(START), md.date_index(END),
                        rebalance_days=40, trade_band_rel=0.30,
                        trade_band_abs=6000.0, max_new_entries=6,
                        verbose=False, **kw)


def test_default_matches_explicit_cfg_bitwise(md, scores):
    """缺省 None 必须等价于显式传 cfg.N_HOLDINGS（生产路径零改动）。"""
    base = _run(md, scores)
    same = _run(md, scores, n_holdings=int(C.N_HOLDINGS))
    a = np.asarray(base.nav, dtype=float)
    b = np.asarray(same.nav, dtype=float)
    assert np.array_equal(a, b), "缺省 n_holdings 必须与 cfg.N_HOLDINGS 逐位一致"


def test_smaller_n_actually_holds_fewer(md, scores):
    """N=20 的平均持仓数必须明显小于 N=40（参数真的生效，不是被忽略）。"""
    big = _run(md, scores, n_holdings=40)
    small = _run(md, scores, n_holdings=20)
    # 测试窗内可用标的不足 40 只，avg_holdings 会低于目标 N，故按相对幅度断言
    assert small.metrics["avg_holdings"] < big.metrics["avg_holdings"] * 0.90
    assert small.config["n_holdings"] == 20
    assert big.config["n_holdings"] == 40


def test_infeasible_n_raises(md, scores):
    """1/N < MIN_WEIGHT 时必须报错，而不是静默空仓。"""
    with pytest.raises(ValueError):
        _run(md, scores, n_holdings=60)
