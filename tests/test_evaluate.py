# -*- coding: utf-8 -*-
"""qf/evaluate.py + 因子构建链路自测（factor-lab owner）。

覆盖：
1. 周频行契约（2017-01-03 起每 5 个交易日）；
2. row_rank / LabelRankCache 与 pandas rank 逐点等价（含 ties / NaN）；
3. Rank IC 方向、有效股票数 <100 跳过、分组单调性、换手极值、覆盖率；
4. factor_metrics 内部一致性（ICIR=mean/std、分年度、DEV 子集、单调性）；
5. 仅周频行变换与 qf.factors 官方实现逐点一致；
6. 无未来函数：扰动 t 之后的数据不改变 t 及以前的因子值。

运行：``python -m pytest tests/test_evaluate.py -q``（真实数据部分 ~25s）。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C          # noqa: E402
from qf import data as qfdata       # noqa: E402
from qf import evaluate as qfe      # noqa: E402
from qf import factors as qff       # noqa: E402
from qf import ops                  # noqa: E402


# ================================================================ fixtures
@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def inp(md):
    return qff.FactorInputs(md)


@pytest.fixture(scope="module")
def rows_weekly(md):
    return qfe.weekly_rows(md)


def _fake_md(dates, n_stocks, eligible=None):
    """最小 md 桩：factor_metrics 只用到 dates / eligible / T / N。"""
    T = len(dates)
    elig = np.ones((T, n_stocks), dtype=bool) if eligible is None else eligible
    return types.SimpleNamespace(dates=pd.DatetimeIndex(dates), eligible=elig, T=T, N=n_stocks)


# ================================================================ 1) 周频行契约
def test_weekly_rows_contract(md, rows_weekly):
    first = md.date_index("2017-01-03")
    assert rows_weekly[0] == first
    assert str(md.dates[rows_weekly[0]].date()) == "2017-01-03"
    assert np.all(np.diff(rows_weekly) == C.REBALANCE_DAYS) == True  # noqa: E712
    expected = np.arange(first, md.T, C.REBALANCE_DAYS)
    assert np.array_equal(rows_weekly, expected)
    # 最后一行必须在数据末尾 REBALANCE_DAYS 之内
    assert md.T - 1 - rows_weekly[-1] < C.REBALANCE_DAYS


# ================================================================ 2) 秩
def test_row_rank_matches_pandas():
    rng = np.random.default_rng(C.RANDOM_SEED)
    R, N = 40, 400
    x = rng.normal(size=(R, N)).astype(np.float32)
    x[rng.random((R, N)) < 0.15] = np.nan
    x[:, :80] = np.round(x[:, :80] * 2)          # ties
    x[:, 80:110] = np.repeat(x[:, 80:81], 30, axis=1)  # 全等
    mask = np.isfinite(x) & (rng.random((R, N)) < 0.9)
    ref = pd.DataFrame(np.where(mask, x, np.nan)).rank(
        axis=1, method="average").to_numpy()
    mine = qfe.row_rank(x, mask)
    assert np.array_equal(np.isfinite(ref), np.isfinite(mine))
    m = np.isfinite(ref)
    assert np.nanmax(np.abs(ref[m] - mine[m])) == 0.0


def test_label_rank_cache_matches_row_rank():
    rng = np.random.default_rng(C.RANDOM_SEED + 1)
    R, N = 30, 300
    x = np.round(rng.normal(size=(R, N)).astype(np.float32) * 3)  # 大量 ties
    base_mask = rng.random((R, N)) < 0.95
    cache = qfe.LabelRankCache(np.where(base_mask, x, np.nan), base_mask)
    valid = base_mask & (rng.random((R, N)) < 0.7)
    mine = cache.ranks(valid)
    ref = qfe.row_rank(np.where(base_mask, x, np.nan), valid)
    assert np.array_equal(np.isfinite(ref), np.isfinite(mine))
    m = np.isfinite(ref)
    assert np.nanmax(np.abs(ref[m] - mine[m])) == 0.0


# ================================================================ 3) IC / 分组 / 换手 / 覆盖率
def test_rank_ic_perfect_inverse_and_skip():
    rng = np.random.default_rng(7)
    R, N = 20, 300
    lab = rng.normal(size=(R, N)).astype(np.float32)
    rows = np.arange(R)
    ic_pos = qfe.rank_ic(lab, lab, None, rows=rows)
    ic_neg = qfe.rank_ic(-lab, lab, None, rows=rows)
    assert np.allclose(ic_pos, 1.0)
    assert np.allclose(ic_neg, -1.0)
    # 有效股票数 < MIN_IC_STOCKS → NaN
    small = np.zeros((R, N), bool)
    small[:, :50] = True
    assert np.all(~np.isfinite(qfe.rank_ic(lab, lab, small, rows=rows)))
    # 50% NaN 的因子仍可计算（有效数 150 >= 100）
    half = np.ones((R, N), bool)
    half[:, N // 2:] = False
    ic_half = qfe.rank_ic(lab, lab, half, rows=rows)
    assert np.allclose(ic_half, 1.0)


def test_quantile_returns_monotone():
    rng = np.random.default_rng(8)
    R, N = 10, 500
    lab = rng.normal(size=(R, N)).astype(np.float32)
    q = qfe.quantile_returns(lab, lab, None, rows=np.arange(R))
    assert q.shape == (R, C.REBALANCE_DAYS // C.REBALANCE_DAYS + 4)  # (R,5)
    means = np.nanmean(q, axis=0)
    assert np.all(np.diff(means) > 0), means
    q_neg = qfe.quantile_returns(-lab, lab, None, rows=np.arange(R))
    assert np.all(np.diff(np.nanmean(q_neg, axis=0)) < 0)


def test_turnover_extremes_and_coverage():
    R, N = 12, 500
    const = np.tile(np.arange(N, dtype=np.float32), (R, 1))
    assert qfe.top_turnover(const, None, rows=np.arange(R)) == pytest.approx(0.0)
    alt = np.where(np.arange(R)[:, None] % 2 == 0,
                   np.arange(N, dtype=np.float32)[None, :],
                   -np.arange(N, dtype=np.float32)[None, :]).astype(np.float32)
    assert qfe.top_turnover(alt, None, rows=np.arange(R)) == pytest.approx(1.0)
    # 覆盖率：一半 NaN → 0.5
    x = np.ones((R, N), dtype=np.float32)
    x[:, N // 2:] = np.nan
    assert qfe.coverage(x, None, rows=np.arange(R)) == pytest.approx(0.5)


# ================================================================ 4) factor_metrics 一致性
def test_factor_metrics_consistency():
    rng = np.random.default_rng(9)
    R, N = 60, 400
    rows = np.arange(R)
    dates = pd.date_range("2018-01-05", periods=R, freq="7D")   # 2018-01 ~ 2019-04
    md = _fake_md(dates, N)
    lab = rng.normal(size=(R, N)).astype(np.float32)
    fac = (lab + 0.5 * rng.normal(size=(R, N))).astype(np.float32)
    met = qfe.factor_metrics(md, fac, lab, rows, name="t")
    # ICIR 定义
    assert met["icir_5d"] == pytest.approx(met["ic_5d"] / met["ic_std_5d"], rel=1e-6)
    assert met["ic_5d"] > 0.5
    assert 0.0 <= met["ic_pos_ratio"] <= 1.0
    assert met["coverage"] == pytest.approx(1.0)
    # 分组单调性
    qs = [met[f"q{i}_5d"] for i in range(1, 6)]
    assert np.all(np.diff(qs) > 0)
    assert met["monotonicity"] == pytest.approx(1.0)
    assert 0.0 <= met["turnover_top20"] <= 1.0
    # 分年度：只有 2018/2019；DEV 子集 <= 2019-12-31 全部 60 周
    assert set(met["ic_by_year"].keys()) == {"2018", "2019"}
    assert met["dev_ic_n"] == R
    assert met["dev_ic_5d"] == pytest.approx(met["ic_5d"])
    # 未来数据不进入因子/标签口径：全 NaN 的周被跳过
    fac2 = fac.copy()
    fac2[3] = np.nan
    met2 = qfe.factor_metrics(md, fac2, lab, rows, name="t2")
    assert met2["n_ic"] == met["n_ic"] - 1


def test_rank_ic_scipy_equivalence():
    """与 scipy.stats.spearmanr 逐周对比（含 ties）。"""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(10)
    R, N = 12, 600
    x = np.round(rng.normal(size=(R, N)), 1).astype(np.float32)  # 有 ties
    y = rng.normal(size=(R, N)).astype(np.float32)
    mask = np.isfinite(x) & np.isfinite(y)
    rows = np.arange(R)
    ic = qfe.rank_ic(x, y, mask, rows=rows)
    for i in range(R):
        m = mask[i]
        rho = spearmanr(x[i][m], y[i][m]).statistic
        assert ic[i] == pytest.approx(rho, rel=0, abs=1e-6)


# ================================================================ 5) 变换等价（真实数据）
def test_transform_at_rows_matches_official(md, inp, rows_weekly):
    cat = qff.load_catalog()
    cache = qfe.TransformCache(inp)
    checked = 0
    for base, w, g in qff.iter_groups(cat):
        if base not in ("volume_ratio", "drawdown"):
            continue
        p0 = qfe.entry_params(g.iloc[0], window=w)
        sig = qff._base_signal(inp, base, p0)
        for _, r in g.iterrows():
            p = qfe.entry_params(r, window=w)
            mine = qfe.transform_at_rows(sig, inp, r["transform"], p, rows_weekly, cache)
            ref = qff._transform_signal(sig, inp, r["transform"], p)[rows_weekly]
            assert mine.dtype == np.float32
            assert mine.shape == ref.shape
            assert np.array_equal(np.isfinite(ref), np.isfinite(mine))
            m = np.isfinite(ref)
            assert np.max(np.abs(ref[m] - mine[m])) == 0.0, (base, r["transform"])
            checked += 1
    assert checked >= 16


# ================================================================ 6) 无未来函数
def _perturbed_inputs(inp, split_row: int, seed: int = 0):
    """复制 FactorInputs，把 split_row 之后的价格/成交量改成随机数。"""
    rng = np.random.default_rng(seed)
    T, N = inp.close.shape
    out = {}
    for name in ("close", "open", "high", "low", "volume", "amount"):
        a = getattr(inp, name).copy()
        tail = a[split_row + 1:]
        finite = np.isfinite(tail)
        scale = float(np.nanstd(a)) if np.isfinite(a).any() else 1.0
        noise = (scale * 0.3 * rng.normal(size=tail.shape)).astype(np.float32)
        tail[finite] = np.abs(tail[finite] + noise[finite])
        a[split_row + 1:] = tail
        out[name] = a
    out["ret"] = ops.pct_change(out["close"], 1)
    out["valid_close"] = np.isfinite(out["close"])
    return types.SimpleNamespace(**out)


def test_no_future_function_in_factor_values(md, inp, rows_weekly):
    split = rows_weekly[len(rows_weekly) // 2]
    inp2 = _perturbed_inputs(inp, split)
    cat = qff.load_catalog()
    # 取 3 个不同 base 的因子（覆盖 rolling / ema / arithmetics）
    picked = []
    for base, w, g in qff.iter_groups(cat):
        if base in ("sma_gap", "ret_kurt", "obv_slope") and len(picked) < 3:
            picked.append((base, w, g))
    cache1 = qfe.TransformCache(inp)
    cache2 = qfe.TransformCache(inp2)
    checked = 0
    for base, w, g in picked:
        p0 = qfe.entry_params(g.iloc[0], window=w)
        sig1 = qff._base_signal(inp, base, p0)
        sig2 = qff._base_signal(inp2, base, p0)
        r = g.iloc[0]
        p = qfe.entry_params(r, window=w)
        f1 = qfe.transform_at_rows(sig1, inp, r["transform"], p, rows_weekly, cache1)
        f2 = qfe.transform_at_rows(sig2, inp2, r["transform"], p, rows_weekly, cache2)
        past = rows_weekly <= split
        a = f1[past]
        b = f2[past]
        assert np.array_equal(np.isfinite(a), np.isfinite(b))
        m = np.isfinite(a)
        assert np.max(np.abs(a[m] - b[m])) == 0.0, base
        checked += 1
    assert checked == 3


def test_label_forward_return_uses_open_prices(md):
    """标签口径：fwd_open_ret(5)[t] = open(t+6)/open(t+1) - 1（后复权→缩放价）。"""
    h = C.FWD_HORIZON
    lab = md.fwd_open_ret(h)
    t = int(md.date_index("2020-06-01"))
    j = int(np.flatnonzero(np.isfinite(md.price_open[t + 1]) & np.isfinite(md.price_open[t + 1 + h]))[0])
    ref = float(md.price_open[t + 1 + h, j]) / float(md.price_open[t + 1, j]) - 1.0
    assert lab[t, j] == pytest.approx(ref, rel=1e-5)
    # 最后 h+1 行无标签
    assert np.all(~np.isfinite(lab[-(h + 1):]))
