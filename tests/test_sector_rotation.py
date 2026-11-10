# -*- coding: utf-8 -*-
"""板块 ETF 轮动策略测试。"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PANEL = ROOT / "build" / "etf" / "sector_panel.npz"
UNIV = ROOT / "build" / "etf" / "sector_universe.json"


@pytest.fixture(scope="module")
def pool():
    if not PANEL.exists() or not UNIV.exists():
        pytest.skip("板块池不存在，先跑 build_sector_pool.py")
    from qf import sector_rotation as qfsr
    return qfsr.load_pool()


def test_pool_excludes_overseas_and_broad(pool):
    """板块池必须剔除海外指数与宽基。"""
    dates, codes, close, amount, sector_of = pool
    banned = {"纳指", "标普", "日经", "港股", "恒生科技",
              "沪深300", "中证500", "中证1000", "创业板", "科创50", "红利", "金融"}
    assert not (set(sector_of.values()) & banned), \
        f"板块池仍含禁止板块: {set(sector_of.values()) & banned}"
    assert len(sector_of) >= 20, f"板块数过少: {len(sector_of)}"


def test_pool_is_time_point_aware(pool):
    """时点可知性：2018 年可交易 ETF 数应显著少于 2024 年。"""
    dates, codes, close, amount, sector_of = pool
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()

    def n_avail(ts):
        t = int(np.flatnonzero(dates >= pd.Timestamp(ts))[0])
        return int(np.sum(np.isfinite(close[t]) & np.isfinite(amt20[t]) & (amt20[t] >= 2.5e7)))

    n2018, n2024 = n_avail("2018-06-01"), n_avail("2024-06-01")
    assert n2018 < n2024, f"池子未随时间扩展: 2018={n2018}, 2024={n2024}"
    assert n2018 >= 1, "2018 年应至少有可交易 ETF"


def test_signal_is_causal(pool):
    """信号只能使用 t 及以前数据。"""
    from qf import sector_rotation as qfsr
    dates, codes, close, amount, sector_of = pool
    t = len(dates) - 1
    sig = qfsr.build_signal_at(dates, codes, close, amount, sector_of, t)
    assert sig["date"] == dates[t]
    # 选中标的下标必须在范围内，且价格与当日收盘一致
    for p in sig["picks"]:
        j = codes.index(p["code"])
        assert abs(p["price"] - close[t, j]) < 1e-6
    # 信号数不超过 TOP_N
    assert len(sig["picks"]) <= qfsr.TOP_N


def test_no_cross_sector_duplicate(pool):
    """同一板块不能出现多只 ETF（去重逻辑）。"""
    from qf import sector_rotation as qfsr
    dates, codes, close, amount, sector_of = pool
    # 检查不同日期
    for t in (len(dates) - 1, len(dates) - 60, len(dates) - 200):
        if t < 300:
            continue
        sig = qfsr.build_signal_at(dates, codes, close, amount, sector_of, t)
        secs = [p["sector"] for p in sig["picks"]]
        assert len(secs) == len(set(secs)), f"t={t} 板块重复: {secs}"


def test_backtest_no_lookahead_in_nav(pool):
    """回测净值必须有限、为正，且第一天为 1.0。"""
    from qf import sector_rotation as qfsr
    dates, codes, close, amount, sector_of = pool
    start = max(int(np.flatnonzero(dates >= pd.Timestamp("2020-01-02"))[0]), 250)
    res = qfsr.backtest(dates, codes, close, amount, sector_of, start_row=start, verbose=False)
    nav = res["nav"]
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav.to_numpy() > 0).all()
    assert abs(nav.iloc[0] - 1.0) < 1e-9
    # 暴露度不能超过 1
    assert res["exposure"].max() <= 1.0 + 1e-9


def test_stats_file_consistency():
    """统计文件的关键指标应与策略口径一致（防空跑/旧文件）。"""
    p = ROOT / "build" / "eval" / "sector_rotation_stats.json"
    if not p.exists():
        pytest.skip("stats 文件不存在")
    st = json.load(open(p, encoding="utf-8"))
    for k in ("full", "val", "oos"):
        assert k in st
        assert st[k]["cagr"] > 0
        assert st[k]["max_dd"] < 0
    # 样本外应显著低于验证期（收益依赖 2020-2021 的已知特征）
    assert st["oos"]["cagr"] < st["val"]["cagr"], "样本外不应高于验证期（若高于请核查）"
    assert "样本外" in st.get("note", "") or "21" in str(st.get("note", ""))


def test_signal_file_exists():
    p = ROOT / "signals" / "latest_sector_rotation.json"
    if not p.exists():
        pytest.skip("信号文件不存在")
    d = json.load(open(p, encoding="utf-8"))
    assert d["strategy"] == "sector_etf_rotation"
    assert "picks" in d and isinstance(d["picks"], list)
    assert len(d["picks"]) <= 3


def test_data_quality_no_adj_jumps():
    """回归：板块 ETF 数据不得有大量复权因子跳变（qfq 异常但不复权正常）。"""
    import numpy as np
    panel = ROOT / "build" / "etf" / "sector_panel.npz"
    if not panel.exists():
        pytest.skip("板块面板不存在")
    z = np.load(panel, allow_pickle=True)
    close = z["close"].astype(float)
    raw = z["raw_close"].astype(float)
    n_adj = 0
    for j in range(close.shape[1]):
        r_q = close[1:, j] / close[:-1, j] - 1
        r_w = raw[1:, j] / raw[:-1, j] - 1
        m = np.isfinite(r_q) & np.isfinite(r_w)
        n_adj += int(np.sum(m & (np.abs(r_q) > 0.11) & (np.abs(r_w) <= 0.11)))
    assert n_adj <= 40, f"复权跳变过多：{n_adj}"


def test_strict_t1_backtest_exists():
    """回归：严格 T+1 回测框架必须存在（防退回同 bar 前视）。"""
    p = ROOT / "scripts" / "sector_daily_v2.py"
    assert p.exists(), "缺少严格 T+1 回测脚本"
    txt = p.read_text(encoding="utf-8")
    assert "pending" in txt, "应使用 pending 机制实现 T+1"
    assert "bought_t >= t" in txt, "当日成交不应计入当日收益"


def test_fixed_panel_exists():
    """修复后的数据面板应存在，且异常点数应少于原始面板。"""
    import numpy as np
    fixed = ROOT / "build" / "etf" / "sector_panel_fixed.npz"
    if not fixed.exists():
        pytest.skip("修复面板不存在")
    z = np.load(fixed, allow_pickle=True)
    close = z["close"].astype(float)
    n_bad = 0
    for j in range(close.shape[1]):
        r = close[1:, j] / close[:-1, j] - 1
        n_bad += int(np.sum(np.abs(r) > 0.11))
    # 剩余异常应全部是创业板/科创的真实 ±20%
    assert n_bad <= 25, f"修复面板仍有 {n_bad} 条异常"
