# -*- coding: utf-8 -*-
"""独立审计：数据层（qf/data.py）口径检查。

由 verifier 编写，独立于业务代码（不 import 被测模块的内部函数做同义反复，
而是用原始 npz/csv + numpy 重新推导，再与 md 字段比对）。

覆盖：
* 缩放后复权价：最后有效日 price == 实际（不复权）收盘价；收益与后复权 close 一致；
* alive / 停牌 / 退市区间；
* 选股域 eligible / buyable / sellable 的逐元素独立重算（检查是否用了 t 日收盘价等前视）；
* 涨跌停价规则（主板 10%、双创 20%、创业板 2020-08-24 前 10%）；
* fwd_ret / fwd_open_ret 的对齐（t 收盘 → t+h；t+1 开盘 → t+1+h 开盘）；
* 基准对齐（md.bench 与 indices.csv、all_eq 等权基准）；
* 板块 / 静态 ST / 退市股 alive 口径。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C                      # noqa: E402
from qf import data as qfdata                   # noqa: E402

# 原始数据目录（不经过 qf.data 的二次加工）
RAW = C.DATA_RAW


@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def raw_arrays():
    z = np.load(RAW / "panel_ohlcv.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# 1. 缩放后复权价
# ---------------------------------------------------------------------------
def test_price_scale_equals_actual_price_on_last_valid(md):
    """price = close × 常数 scale，使最后一个有效交易日的 price == 真实（不复权）收盘。"""
    lv = md.last_valid
    has = (lv >= 0) & (lv < md.T)
    jj = np.flatnonzero(has)
    t = lv[jj]
    p = md.price[t, jj].astype(np.float64)
    r = md.raw_close[t, jj].astype(np.float64)
    ok = np.isfinite(p) & np.isfinite(r) & (r > 0)
    assert ok.sum() > 4000, f"可核对的股票太少：{ok.sum()}"
    rel = np.abs(p[ok] / r[ok] - 1.0)
    assert rel.max() < 1e-5, f"最后有效日 price 与不复权价偏差过大：max rel={rel.max():.2e}"


def test_price_preserves_adjusted_returns(md):
    """常数缩放不改变收益：price 日收益 == 后复权 close 日收益。"""
    ret_p = md.price[1:].astype(np.float64) / md.price[:-1].astype(np.float64) - 1.0
    ok = np.isfinite(ret_p) & np.isfinite(md.ret_daily[1:])
    assert ok.sum() > 1e6
    diff = np.abs(ret_p[ok] - md.ret_daily[1:][ok].astype(np.float64))
    assert diff.max() < 1e-5


def test_price_finite_where_alive(md):
    """alive 区间内 price/price_open 均应有值（停牌已前向填充），否则整手/估值会失真。

    另外审计原始数据缺陷：后复权 close 存在 <=0 的格子（分红/复权因子导致的
    负后复权价，如 2019-06 的 600595），data.py 未清洗。要求：这些格子必须
    落在选股域之外（价格过滤 MIN_PRICE 兜底），不能被买入/估值。
    """
    assert np.isfinite(md.price[md.alive]).all()
    assert np.isfinite(md.price_open[md.alive]).all()
    neg = md.alive & (md.price <= 0)
    n_neg = int(neg.sum())
    print(f"[audit] 后复权价<=0 的格子：{n_neg} 个；涉及股票 "
          f"{sorted(set(md.codes[np.nonzero(neg)[1]]))[:10]}")
    assert not (neg & md.eligible).any(), "负/零价格格子进入了选股域"
    assert not (neg & md.buyable).any(), "负/零价格格子可买入"
    assert n_neg < 1000, "负后复权价格子异常多，需进一步排查原始数据"


# ---------------------------------------------------------------------------
# 2. alive / 停牌 / 退市
# ---------------------------------------------------------------------------
def test_alive_bounds(raw_arrays, md):
    close = raw_arrays["close"].astype(np.float64)
    fin = np.isfinite(close) & (close > 0)
    first = np.where(fin.any(axis=0), fin.argmax(axis=0), md.T)
    rev = fin[::-1]
    has = rev.any(axis=0)
    last = np.where(has, md.T - 1 - rev.argmax(axis=0), -1)
    assert np.array_equal(md.first_valid, first)
    assert np.array_equal(md.last_valid, last)
    idx = np.arange(md.T)[:, None]
    expected_alive = (idx >= first[None, :]) & (idx <= last[None, :])
    assert np.array_equal(md.alive, expected_alive)
    # 停牌 = alive 但当日无有效 bar
    assert np.array_equal(md.is_suspended, md.alive & ~fin)
    # 上市前/退市后价格必须 NaN
    assert np.isnan(md.close[~md.alive]).all()


def test_delisted_codes_stay_alive_until_last_bar(md):
    delist = pd.read_csv(RAW / "delisted_universe.csv", dtype={"code": str})
    delist["code"] = delist["code"].str.zfill(6)
    found = 0
    for _, row in delist.head(60).iterrows():
        j = md.code_index(row["code"])
        if j < 0:
            continue
        found += 1
        lv = md.last_valid[j]
        assert lv >= 0
        assert lv < md.T - 1, f"{row['code']} 退市股最后有效 bar 落在样本末尾"
        assert md.alive[lv, j] and not md.alive[lv + 1, j]
        assert md.is_delisted[j] if np.isfinite(md.is_delisted[j]) else True
    assert found >= 30, f"退市股样本命中过少：{found}"


def test_board_mapping(md):
    for j, code in enumerate(md.codes):
        if code.startswith(("688", "689")):
            expected = "STAR"
        elif code.startswith(("300", "301")):
            expected = "CHINEXT"
        else:
            expected = "MAIN"
        assert md.board[j] == expected
    assert set(np.unique(md.board)) <= {"MAIN", "STAR", "CHINEXT"}


# ---------------------------------------------------------------------------
# 3. 涨跌停
# ---------------------------------------------------------------------------
def test_limit_prices_match_board_rules(md):
    prev = md.prev_raw_close
    pct = md.limit_pct
    ok = np.isfinite(prev) & (prev > 0)
    exp_up = np.floor(prev * (1.0 + pct) * 100 + 0.5) / 100.0
    exp_dn = np.floor(prev * (1.0 - pct) * 100 + 0.5) / 100.0
    both = ok & np.isfinite(md.limit_up) & np.isfinite(md.limit_down)
    assert np.allclose(md.limit_up[both], exp_up[both], atol=1e-4)
    assert np.allclose(md.limit_down[both], exp_dn[both], atol=1e-4)

    # 板块规则
    chi = np.array([b == "CHINEXT" for b in md.board])
    star = np.array([b == "STAR" for b in md.board])
    pre = md.dates < pd.Timestamp(C.CHINEXT_20PCT_DATE)
    main = ~(chi | star)
    assert np.allclose(pct[:, main], 0.10, atol=1e-6)
    assert np.allclose(pct[:, star], 0.20, atol=1e-6)
    assert np.allclose(pct[np.ix_(pre, chi)], 0.10, atol=1e-6)
    assert np.allclose(pct[np.ix_(~pre, chi)], 0.20, atol=1e-6)


def test_limit_flags_consistent_with_open(md):
    """开盘涨停标记必须与开盘价 >= 涨停价（带极小容差）一致。"""
    op = md.raw_open
    m = md.limit_up_at_open
    assert m.sum() > 1000
    eps = np.maximum(np.abs(md.prev_raw_close) * 0.002, 0.005)
    assert (op[m] >= (md.limit_up - eps)[m] - 1e-6).all()
    # 未标记涨停时，开盘价不应显著高于涨停价
    not_m = np.isfinite(op) & np.isfinite(md.limit_up) & ~m
    assert (op[not_m] < (md.limit_up - eps)[not_m] + 1e-6).all()


# ---------------------------------------------------------------------------
# 4. 选股域 / 可交易性 —— 逐元素独立重算（发现前视的第一道防线）
# ---------------------------------------------------------------------------
def test_eligible_independent_recompute(md):
    """按 docs/DATA_NOTES 口径独立重算 eligible，必须与 md.eligible 完全一致。

    关键点：价格 / 市值过滤用 **t-1** 日值（否则是未来函数），流动性用
    t 日及之前的 20 日均额，上市天数用 list_days。
    价格带用 **PIT 实际价 md.price_exec**（2026-10-23 backtest-guard 修复后口径）。
    """
    amount = np.where(md.alive, md.amount, np.nan)
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy(dtype=np.float64)
    liq_ok = np.isfinite(amt20) & (amt20 >= C.MIN_AMOUNT_20D)

    # 价格带过滤用 PIT 实际价（md.price_exec）：旧口径用末日常数缩放价，含未来复权信息，
    # 已被 backtest-guard 判为高危前视并修复（2026-10-23）。
    price_prev = pd.DataFrame(np.where(md.alive, md.price_exec, np.nan)).shift(1).to_numpy(dtype=np.float64)
    price_ok = (np.isfinite(price_prev)
                & (price_prev >= C.MIN_PRICE) & (price_prev <= C.MAX_PRICE))

    mcap_prev = pd.DataFrame(np.where(md.alive, md.mcap, np.nan)).shift(1).to_numpy(dtype=np.float64)
    mcap_ok = (np.isfinite(mcap_prev)
               & (mcap_prev >= C.MIN_MCAP) & (mcap_prev <= C.MAX_MCAP))

    list_ok = md.list_days >= C.MIN_LIST_DAYS
    st_ok = (~np.tile(md.name_st[None, :], (md.T, 1))) if C.EXCLUDE_ST else np.ones((md.T, md.N), bool)
    expected = (md.alive & liq_ok & price_ok & mcap_ok & list_ok & st_ok & ~md.is_suspended)
    n_diff = int((expected ^ md.eligible).sum())
    assert n_diff == 0, f"eligible 独立重算不一致：{n_diff} 个格子"


def test_buyable_sellable_independent_recompute(md):
    op_ok = np.isfinite(md.raw_open) & (md.raw_open > 0)
    exp_buy = md.eligible & op_ok & ~md.limit_up_at_open
    exp_sell = md.alive & ~md.is_suspended & op_ok & ~md.limit_dn_at_open
    assert np.array_equal(md.buyable, exp_buy), "buyable 与独立重算不一致"
    assert np.array_equal(md.sellable, exp_sell), "sellable 与独立重算不一致"
    assert not (md.buyable & ~md.eligible).any()
    assert not (md.sellable & ~md.alive).any()


def test_eligible_does_not_use_current_close_price(md):
    """构造"t-1 价格越界、t 价格回到区间"的格子：若用了当日价格，会被错误纳入。"""
    price = np.where(md.alive, md.price_exec, np.nan)
    prev = np.vstack([np.full((1, md.N), np.nan), price[:-1]])
    hi = np.isfinite(prev) & (prev > C.MAX_PRICE) & np.isfinite(price) & (price <= C.MAX_PRICE)
    lo = np.isfinite(prev) & (prev < C.MIN_PRICE) & np.isfinite(price) & (price >= C.MIN_PRICE)
    sel = hi | lo
    n = int(sel.sum())
    assert n > 0, "样本中不存在价格穿越过滤线的格子，无法验证"
    assert not (md.eligible & sel).any(), "选股域疑似使用了 t 日收盘价（未来函数）"


# ---------------------------------------------------------------------------
# 5. 前向收益对齐
# ---------------------------------------------------------------------------
def test_fwd_ret_alignment(md):
    h = C.FWD_HORIZON
    exp = np.full((md.T, md.N), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        exp[:-h] = (md.close[h:].astype(np.float64) / md.close[:-h].astype(np.float64) - 1.0)
    both = np.isfinite(exp) & np.isfinite(md.fwd_ret(h))
    assert np.allclose(md.fwd_ret(h)[both], exp[both], rtol=1e-5, atol=1e-6)
    assert np.isnan(md.fwd_ret(h)[-h:]).all(), "fwd_ret 末尾应有 h 行为 NaN"
    # 标签只用于评估：抽查 fwd_ret 不能与任何 t 时点已实现收益相同（结构性检查略）


def test_fwd_open_ret_alignment(md):
    """fwd_open_ret(h)[t] = price_open[t+1+h] / price_open[t+1] - 1；末尾 h+1 行 NaN。"""
    h = C.FWD_HORIZON
    o = md.price_open.astype(np.float64)
    exp = np.full((md.T, md.N), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        exp[: md.T - h - 1] = o[1 + h:] / o[1: md.T - h] - 1.0
    got = md.fwd_open_ret(h)
    both = np.isfinite(exp) & np.isfinite(got)
    assert both.sum() > 1e6
    assert np.allclose(got[both], exp[both], rtol=1e-5, atol=1e-6)
    assert np.isnan(got[md.T - h - 1:]).all(), "fwd_open_ret 末尾应有 h+1 行为 NaN"


def test_fwd_open_ret_label_uses_ffilled_open_on_suspended_days(md):
    """审计记录：t+1 停牌的格子，标签仍用前向填充的开盘价（并非真实可成交价）。

    这不是未来函数，但会污染研究/训练标签（不可买入的停牌期收益被计入）。
    本测试记录问题规模，供报告量化；不改变业务代码。
    """
    susp_next = np.zeros((md.T, md.N), dtype=bool)
    susp_next[:-1] = md.is_suspended[1:]
    label = md.fwd_open_ret(C.FWD_HORIZON)
    bad = susp_next & np.isfinite(label)
    frac = bad.sum() / max(np.isfinite(label).sum(), 1)
    # 记录到审计产物（failure 时也能在报告里看到）
    print(f"[audit] fwd_open_ret 在 t+1 停牌格子上有值：{int(bad.sum())} "
          f"({frac:.2%} of finite labels)")
    assert np.isfinite(label).sum() > 0


# ---------------------------------------------------------------------------
# 6. 基准对齐
# ---------------------------------------------------------------------------
def test_bench_alignment(md):
    idx = pd.read_csv(RAW / "indices.csv", index_col=0, parse_dates=True)
    assert md.bench.index.equals(md.dates), "基准索引必须与行情日期完全对齐"
    assert list(md.bench.columns[:4]) == C.BENCHMARKS, md.bench.columns.tolist()
    src = idx.reindex(md.dates).ffill()
    for name in C.BENCHMARKS:
        got = md.bench[name].to_numpy(dtype=np.float64)
        exp = src[name].pct_change().fillna(0.0).to_numpy(dtype=np.float64)
        assert np.allclose(got, exp, atol=1e-12, equal_nan=True), f"{name} 基准收益对齐错误"
    assert np.isfinite(md.bench.to_numpy(dtype=np.float64)).all()
    assert md.bench.iloc[0]["hs300"] == 0.0


def test_all_eq_benchmark_independent_recompute(md):
    """all_eq = 当日全部 alive 股票的等权日收益（含退市股，无生存偏差）。"""
    price = md.price.astype(np.float64)
    r = np.full((md.T, md.N), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = price[1:] / price[:-1] - 1.0
    valid = np.isfinite(r) & md.alive
    n = valid.sum(axis=1)
    s = np.where(valid, r, 0.0).sum(axis=1)
    exp = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    assert np.allclose(md.bench["all_eq"].to_numpy(dtype=np.float64), exp, atol=1e-10)
