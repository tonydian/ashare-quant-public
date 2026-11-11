"""实时行情层（qf/realtime.py）的离线测试。

不联网：所有 HTTP 入口都用 monkeypatch 替换。
重点锁住三件容易错的事：
  1. 涨跌幅口径必须是小数（0.0046 = +0.46%），前端 fmtPct 会 ×100
  2. 二分定位涨跌分界的边界（全涨 / 全跌 / 末页不满 / 报告总数偏小）
  3. 大面积取数失败必须返回 None（不进缓存），部分缺失是正常的（停牌）
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qf import realtime as rt

PAGE = rt.PAGE_SIZE


# ----------------------------------------------------------------- 工具

def _rows(chgs):
    """把涨跌幅列表转成 _sina_page 的返回结构。"""
    return [{"chg": c, "code": f"{i:06d}", "name": f"股票{i}", "amount": 1e8}
            for i, c in enumerate(chgs)]


def _patch_market(monkeypatch, chgs, reported_total=None):
    """按降序 chgs 构造一个假市场，替换分页接口。"""
    total = len(chgs)

    def fake_page(page, num=PAGE):
        lo = (page - 1) * num
        return _rows(chgs[lo:lo + num]) or None

    monkeypatch.setattr(rt, "_sina_page", fake_page)
    monkeypatch.setattr(rt, "_sina_total", lambda: reported_total or total)
    monkeypatch.setattr(rt, "fetch_limits", lambda: None)
    monkeypatch.setattr(rt, "_CACHE", {})
    return total


# ----------------------------------------------------------------- 基础

def test_sina_code_prefix():
    assert rt.sina_code("600519") == "sh600519"
    assert rt.sina_code("688981") == "sh688981"
    assert rt.sina_code("000001") == "sz000001"
    assert rt.sina_code("300750") == "sz300750"
    # 转债：沪 110/111/113，深 123/127/128
    assert rt.sina_code("110081") == "sh110081"
    assert rt.sina_code("113050") == "sh113050"
    assert rt.sina_code("123456") == "sz123456"
    assert rt.sina_code("127001") == "sz127001"
    assert rt.sina_code("128002") == "sz128002"


def test_sina_code_pads_short_input():
    assert rt.sina_code("1") == "sz000001"        # 补零后 000001 是深市
    assert rt.sina_code("518880") == "sh518880"


def test_is_trading_time_boundaries():
    from datetime import datetime
    t = lambda h, m, wd=0: rt.is_trading_time(datetime(2026, 11, 9 + wd, h, m))
    assert t(9, 14) is False
    assert t(9, 15) is True
    assert t(11, 30) is True
    assert t(11, 31) is False
    assert t(12, 59) is False
    assert t(13, 0) is True
    assert t(15, 0) is True
    assert t(15, 1) is False
    assert rt.is_trading_time(datetime(2026, 11, 14, 10, 0)) is False   # 周六


# ----------------------------------------------------------------- 解析

_SAMPLE = (
    'var hq_str_sh600519="贵州茅台,1445.000,1439.000,1402.300,1445.620,1402.000,'
    '1402.300,1402.370,3984791,5633865699.000,187,1402.300,100,1402.230,100,'
    '1402.220,100,1402.130,100,1402.040,200,1402.370,100,1402.500,1900,'
    '1402.570,2100,1402.600,200,1402.700,2026-11-11,15:34:59,00,D|2100|2944830.00";\n'
)


def test_parse_sina_fields():
    d = rt._parse_sina(_SAMPLE)
    q = d["sh600519"]
    assert q["name"] == "贵州茅台"
    assert q["last"] == pytest.approx(1402.30)
    assert q["prev_close"] == pytest.approx(1439.00)
    assert q["open"] == pytest.approx(1445.00)
    assert q["amount"] == pytest.approx(5633865699.0)
    assert q["date"] == "2026-11-11"
    assert q["time"] == "15:34:59"


def test_parse_sina_chg_pct_is_decimal():
    """必须是小数口径：前端 fmtPct 内部 ×100，给百分数会放大一百倍。"""
    q = rt._parse_sina(_SAMPLE)["sh600519"]
    assert q["chg_pct"] == pytest.approx(1402.30 / 1439.00 - 1.0, abs=1e-9)
    assert -0.03 < q["chg_pct"] < -0.02          # 实际约 -2.55%
    assert abs(q["chg_pct"]) < 1                 # 是小数，不是百分数


def test_parse_sina_skips_suspended():
    """现价或昨收为 0（停牌/未开盘）应被丢弃。"""
    line = _SAMPLE.replace("1402.300,1445.620", "0.000,1445.620")
    assert rt._parse_sina(line) == {}


# ----------------------------------------------------------------- 二分

def test_breadth_normal_split(monkeypatch):
    """250 只：100 涨 / 50 平 / 100 跌。"""
    _patch_market(monkeypatch, [0.02] * 100 + [0.0] * 50 + [-0.02] * 100)
    b = rt.fetch_breadth()
    assert b is not None
    assert (b["up"], b["flat"], b["down"]) == (100, 50, 100)
    assert b["total"] == 250


def test_breadth_all_up(monkeypatch):
    """全市场都在涨：不能落进「最后一页是分界」的分支。"""
    _patch_market(monkeypatch, [0.01] * 250)
    b = rt.fetch_breadth()
    assert b is not None
    assert (b["up"], b["flat"], b["down"]) == (250, 0, 0)


def test_breadth_all_down(monkeypatch):
    _patch_market(monkeypatch, [-0.01] * 250)
    b = rt.fetch_breadth()
    assert b is not None
    assert (b["up"], b["flat"], b["down"]) == (0, 0, 250)


def test_breadth_reported_total_smaller_than_real(monkeypatch):
    """新浪报告的总数会偏小（实测报 5341、实际 5350）。

    下跌家数若用「报告总数 - 位置」反算就会少算，必须按实际页数取。
    """
    _patch_market(monkeypatch, [0.02] * 100 + [0.0] * 50 + [-0.02] * 240,
                  reported_total=340)       # 实际 390，报告只给 340
    b = rt.fetch_breadth()
    assert b is not None
    assert (b["up"], b["flat"], b["down"]) == (100, 50, 240)
    assert b["total"] == 390
    assert b["up"] + b["down"] + b["flat"] == b["total"]


def test_breadth_zero_on_page_boundary(monkeypatch):
    """分界恰好落在页边界（第 100 条是最后一个上涨）。"""
    _patch_market(monkeypatch, [0.02] * 100 + [-0.02] * 100)
    b = rt.fetch_breadth()
    assert (b["up"], b["flat"], b["down"]) == (100, 0, 100)


def test_breadth_returns_none_on_fetch_failure(monkeypatch):
    monkeypatch.setattr(rt, "_sina_page", lambda p, num=PAGE: None)
    monkeypatch.setattr(rt, "_sina_total", lambda: 100)
    monkeypatch.setattr(rt, "fetch_limits", lambda: None)
    assert rt.fetch_breadth() is None


# ----------------------------------------------------------------- 个股

def test_fetch_quotes_majority_missing_returns_none(monkeypatch):
    """大面积取不到 → 当故障处理，返回 None 且不进缓存。

    历史事故：只拿到 9/59 只却因为有 1 批成功就当成正常结果、
    被缓存 5 分钟（非交易时段 TTL），页面上 50 只票凭空消失。
    """
    monkeypatch.setattr(rt, "_get", lambda *a, **k: None)
    assert rt.fetch_quotes(["600519"] * 40) is None


def test_fetch_quotes_partial_ok(monkeypatch):
    """少量缺失（停牌）属于正常，应照常返回。

    阈值是「拿到数 < 请求数的一半」才判失败，所以 2 只里回 1 只（=50%）要放行。
    """
    quote = ('var hq_str_sh600519="贵州茅台,1445.000,1439.000,1402.300,1445.620,'
             '1402.000,1402.300,1402.370,3984791,5633865699.000,187,1402.300,100,'
             '1402.230,100,1402.220,100,1402.130,100,1402.040,200,1402.370,100,'
             '1402.500,1900,1402.570,2100,1402.600,200,1402.700,2026-11-11,15:34:59,00";\n')
    monkeypatch.setattr(rt, "_get", lambda *a, **k: quote.encode("gbk"))
    got = rt.fetch_quotes(["600519", "000001"])
    assert got is not None
    assert list(got) == ["600519"]     # 另一只是缺失（如停牌），不影响整体返回


# ----------------------------------------------------------------- 板块

def test_fetch_sectors_parses_and_converts(monkeypatch):
    """新浪板块字段是百分数，必须转成小数。"""
    payload = (
        'var S_Finance_bankuai_sinaindustry = {'
        '"new_blhy":"new_blhy,玻璃行业,19,14.479,0.480,3.4325889164599,'
        '627178811,9756763507,sz002623,9.995,23.330,2.120,亚玛顿",'
        '"new_cbzz":"new_cbzz,船舶制造,8,13.404,-0.042,-0.3187,'
        '145340950,2418073187,sh600072,0.577,8.720,0.050,中船科技"};'
    )
    monkeypatch.setattr(rt, "_get", lambda *a, **k: payload.encode("gbk"))
    s = rt.fetch_sectors(5)
    assert s is not None
    assert len(s) == 2
    assert s[0]["name"] == "玻璃行业"          # 按涨幅降序
    assert s[0]["chg_pct"] == pytest.approx(0.034326, abs=1e-5)
    assert s[0]["leader"] == "亚玛顿"
    assert s[0]["leader_chg"] == pytest.approx(0.09995, abs=1e-5)
    assert s[1]["chg_pct"] < 0
