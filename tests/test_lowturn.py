# -*- coding: utf-8 -*-
"""低换手月度策略（qf/lowturn.py + scripts/lowturn_signal.py + Web 卡片）测试。

注意：``TestBacktest`` 会加载全量行情（约 15s、2GB 内存），建议单独运行::

    pytest tests/test_lowturn.py -q
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
STATIC = ROOT / "web" / "static"


# ---------------------------------------------------------------- 纯函数
def test_month_bounds_cover_every_day():
    from qf import lowturn as lt

    dates = pd.bdate_range("2024-01-01", "2024-04-30")
    mb = lt.month_bounds(pd.DatetimeIndex(dates))
    assert len(mb) == 4
    seen = set()
    for (f, l) in mb.values():
        assert f <= l
        seen |= set(range(f, l + 1))
    assert seen == set(range(len(dates))), "月度区间必须无缝覆盖每个交易日"


def test_month_bounds_handle_partial_month():
    """数据在月中截断时，最后一个月只能覆盖到已有交易日。"""
    from qf import lowturn as lt

    dates = pd.bdate_range("2024-01-01", "2024-03-13")
    mb = lt.month_bounds(pd.DatetimeIndex(dates))
    last = sorted(mb)[-1]
    f, l = mb[last]
    assert l == len(dates) - 1
    assert dates[l].month == 3


def test_month_role_month_end_uses_next_month():
    """本月最后一个交易日 → 用当天数据算下月名单。"""
    from lowturn_signal import _month_role

    dates = pd.bdate_range("2024-01-01", "2024-03-29")
    idx = pd.DatetimeIndex(dates)
    # 2 月最后一个交易日
    feb_last = max(i for i, d in enumerate(idx) if d.month == 2)
    t_sig, tgt, role = _month_role(idx, feb_last)
    assert role == "month_end"
    assert t_sig == feb_last
    assert tgt == "2024-03"


def test_month_role_mid_month_uses_prev_month_end():
    """月中 → 用上月最后交易日算本月名单（已在持有）。"""
    from lowturn_signal import _month_role

    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-03-29"))
    mid = max(i for i, d in enumerate(idx) if d.month == 3 and d.day <= 15)
    t_sig, tgt, role = _month_role(idx, mid)
    assert role == "mid_month"
    assert tgt == "2024-03"
    assert idx[t_sig].month == 2


def test_month_role_last_row_of_data_still_month_end():
    """数据最后一行恰是月末时，仍要给出下月名单（这是实盘最常用的情形）。"""
    from lowturn_signal import _month_role

    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", "2024-02-29"))
    t_sig, tgt, role = _month_role(idx, len(idx) - 1)
    assert role == "month_end"
    assert tgt == "2024-03"


# ---------------------------------------------------------------- 静态注册
def test_web_kind_registered():
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert '"lowturn"' in svc
    assert "def _bt_lowturn(" in svc
    assert "def _lowturn_bt(" in svc


def test_frontend_card_and_fields():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'data-kind="lowturn"' in html, "回测页缺少低换手策略卡"
    assert "lowturn: [" in js, "缺少参数配置"
    assert "BT_KIND === 'lowturn'" in js, "缺少请求组装分支"
    for fid in ("btkLookback", "btkMinList", "btkMinAdv", "btkRebW"):
        assert fid in js, f"缺少参数 {fid}"


def test_signal_script_exists():
    p = ROOT / "scripts" / "lowturn_signal.py"
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "lowturn_target.csv" in txt
    assert "lowturn_orders.csv" in txt


# ---------------------------------------------------------------- 需要行情
@pytest.fixture(scope="module")
def md():
    from qf import data as qfdata

    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def bt(md):
    from qf import lowturn as lt

    return lt.backtest(md, start="2020-01-02")


class TestBacktest:
    def test_three_books_ordered(self, bt):
        """毛 ≥ 净 ≥ 严格（严格每月全卖全买，成本最重）。"""
        from qf import lowturn as lt

        g = lt.performance(bt["nav_gross"])["cagr"]
        n = lt.performance(bt["nav"])["cagr"]
        s = lt.performance(bt["nav_strict"])["cagr"]
        assert g > n > s, f"口径排序异常 gross={g:.4f} net={n:.4f} strict={s:.4f}"
        assert g - n < 0.10, "成本拖累过大，检查是否重复扣费"
        assert n - s > 0.02, "严格口径应显著更差（约 11pp）"

    def test_cost_monotonic(self, md):
        """成本越高年化越低，且近似线性（防止漏扣/重复扣）。"""
        from qf import lowturn as lt

        a = lt.performance(lt.backtest(md, cost_bp=0.0)["nav_strict"])["cagr"]
        b = lt.performance(lt.backtest(md, cost_bp=49.3)["nav_strict"])["cagr"]
        c = lt.performance(lt.backtest(md, cost_bp=98.6)["nav_strict"])["cagr"]
        assert a > b > c
        d1, d2 = a - b, b - c
        assert abs(d1 - d2) / max(d1, 1e-9) < 0.35, f"成本非线性：{d1:.4f} vs {d2:.4f}"

    def test_picks_are_cheapest(self, md):
        """每期选出的必须是候选里换手率最低的那批。"""
        from qf import lowturn as lt

        turn = lt.turnover_panel(md, 20)
        mb = lt.month_bounds(md.dates)
        months = sorted(mb)
        for m in months[-6:-1]:                    # 近半年（早期月份数据覆盖不足）
            t_sig = mb[months[months.index(m) - 1]][1]
            picks = lt.select(md, turn, t_sig, 10)
            assert len(picks) == 10
            mask = lt.candidate_mask(md, turn, t_sig)
            others = np.setdiff1d(np.where(mask)[0], picks)
            assert turn[t_sig, picks].max() <= turn[t_sig, others].min() + 1e-12

    def test_picks_exclude_st_and_new(self, md):
        from qf import lowturn as lt

        turn = lt.turnover_panel(md, 20)
        mb = lt.month_bounds(md.dates)
        months = sorted(mb)
        t_sig = mb[months[-3]][1]
        picks = lt.select(md, turn, t_sig, 10)
        assert not md.name_st[picks].any(), "选中了 ST 股"
        assert md.list_days[t_sig, picks].min() >= lt.MIN_LIST_DAYS
        assert md.is_suspended[t_sig, picks].sum() == 0
        assert md.alive[t_sig, picks].all()

    def test_min_adv_filter_effective(self, md):
        """加流动性下限后，持仓的 20 日均额必须都达标。"""
        from qf import lowturn as lt

        turn = lt.turnover_panel(md, 20)
        mb = lt.month_bounds(md.dates)
        months = sorted(mb)
        t_sig = mb[months[-3]][1]
        picks = lt.select(md, turn, t_sig, 10, min_adv20=0.5e8)
        assert len(picks) == 10
        assert (md.amount_20d[t_sig, picks] >= 0.5e8).all()

    def test_no_lookahead_in_signal(self, md):
        """把信号日之后的价格全部抹掉，名单必须不变。"""
        from qf import lowturn as lt

        turn = lt.turnover_panel(md, 20)
        mb = lt.month_bounds(md.dates)
        months = sorted(mb)
        t_sig = mb[months[-3]][1]
        base = lt.select(md, turn, t_sig, 10)

        turn2 = turn.copy()
        turn2[t_sig + 1:] = np.nan
        again = lt.select(md, turn2, t_sig, 10)
        assert base == again, "后视数据影响了选股"

    def test_turnover_annual_range(self, bt):
        assert 2.0 < bt["turnover_ann"] < 15.0, "单边年换手异常"
        assert 0.4 < bt["keep_mean"] < 0.95, "名单重合度异常"

    def test_nav_index_and_values(self, bt):
        nav = bt["nav"]
        assert isinstance(nav.index, pd.DatetimeIndex)
        assert nav.index.is_monotonic_increasing
        assert np.isfinite(nav.to_numpy()).all()
        assert (nav > 0).all()
        assert nav.index[0] >= pd.Timestamp("2020-01-02")
