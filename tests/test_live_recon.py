# -*- coding: utf-8 -*-
"""``scripts/live_recon.py`` 的不变量测试。

这个脚本的价值全在**口径**上，所以测试盯的也是口径，不是数字好不好看：

1. **进场价 = 开盘价**。执行偏差只能用 (成交价 / 开盘价 − 1)，
   混进信号日收盘价就会把隔夜跳空算成"你的滑点"。
2. **三段可加**：`filled/ref_close − 1 ≈ gap + exec`，拆开后必须能拼回去。
3. **费用按腿分开**：个股有 5 元最低佣金和卖出印花，转债两样都没有。
4. **现金账要对得上**：买 1000 元股票，现金减少的必须是成交额 + 费用，一分不差。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("live_recon_mod", ROOT / "scripts" / "live_recon.py")
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)
from qf import config as C  # noqa: E402


# ---------------------------------------------------------------------------
# 一、口径：进场价必须是开盘价
# ---------------------------------------------------------------------------
def test_exec_bp_买入贵了为正():
    assert L.exec_bp(10.05, 10.00, "买入") == pytest.approx(50.0, abs=1e-9)
    assert L.exec_bp(10.00, 10.00, "买入") == pytest.approx(0.0, abs=1e-9)


def test_exec_bp_卖出贱了为正():
    """卖出方向要翻号：10 元的票卖成 9.95 是吃亏 50bp，不是赚。"""
    assert L.exec_bp(9.95, 10.00, "卖出") == pytest.approx(50.0, abs=1e-9)
    assert L.exec_bp(10.05, 10.00, "卖出") == pytest.approx(-50.0, abs=1e-9)


def test_exec_bp_缺价返回nan():
    for f, o in ((np.nan, 10.0), (10.0, np.nan), (0.0, 10.0), (10.0, -1.0)):
        assert np.isnan(L.exec_bp(f, o, "买入"))


def test_三段拆解可拼回总偏差():
    """gap + exec 必须能还原 filled/ref_close，否则拆解口径是错的。"""
    ref, op, filled = 10.00, 10.20, 10.26
    g = L.gap_bp(op, ref)
    e = L.exec_bp(filled, op, "买入")
    total = (filled / ref - 1.0) * 1e4
    assert (1 + g / 1e4) * (1 + e / 1e4) - 1.0 == pytest.approx(total / 1e4, rel=1e-12)
    # 数量级也符合直觉：跳空 200bp、执行 59bp，总 260bp
    assert g == pytest.approx(200.0, abs=1e-9)
    assert e == pytest.approx(58.8235, abs=1e-3)
    assert total == pytest.approx(260.0, abs=1e-9)


def test_gap_bp_方向():
    assert L.gap_bp(10.5, 10.0) == pytest.approx(500.0, abs=1e-9)   # 高开
    assert L.gap_bp(9.5, 10.0) == pytest.approx(-500.0, abs=1e-9)   # 低开


# ---------------------------------------------------------------------------
# 二、回测假设按腿取值
# ---------------------------------------------------------------------------
def test_各腿让价不同():
    assert L.leg_price_bp("main") == 40.0
    assert L.leg_price_bp("lowturn") == 49.3
    assert L.leg_price_bp("etf") == 5.0
    assert L.leg_price_bp("cb") == 5.0
    # 未知腿退回全局滑点，不能静默变成 0
    assert L.leg_price_bp("whoknows") == float(C.SLIPPAGE_BP)


def test_model_fill_买卖方向():
    assert L.model_fill(10.0, "main", "买入") == pytest.approx(10.04, abs=1e-12)
    assert L.model_fill(10.0, "main", "卖出") == pytest.approx(9.96, abs=1e-12)


# ---------------------------------------------------------------------------
# 三、费用：个股有最低佣金和印花，转债没有
# ---------------------------------------------------------------------------
def test_个股买入_最低佣金主导():
    fee = L.stock_fee(1000.0, "买入")
    assert fee == pytest.approx(C.MIN_COMMISSION + 1000.0 * C.TRANSFER_FEE, abs=1e-9)
    # 1348 元的小单，佣金一项就是 37bp —— 这正是当前主策略腿的真实处境
    assert C.MIN_COMMISSION / 1348.0 * 1e4 == pytest.approx(37.09, abs=0.01)


def test_个股卖出_加印花税():
    amt = 1e6
    buy = L.stock_fee(amt, "买入")
    sell = L.stock_fee(amt, "卖出")
    assert sell - buy == pytest.approx(C.STAMP_TAX_SELL * amt, abs=1e-6)


def test_个股大额时费率制生效():
    amt = 1e6
    comm = max(C.MIN_COMMISSION, C.COMMISSION_RATE * amt)
    assert comm == pytest.approx(C.COMMISSION_RATE * amt)      # 100 元 > 5 元
    assert L.stock_fee(amt, "买入") == pytest.approx(comm + C.TRANSFER_FEE * amt, abs=1e-9)


def test_转债无最低佣金无印花():
    assert L.trade_fee("cb", 1000.0, "买入") == pytest.approx(0.5, abs=1e-12)
    assert L.trade_fee("cb", 1000.0, "卖出") == pytest.approx(0.5, abs=1e-12)
    # 同样 1000 元，转债比股票便宜 10 倍以上
    assert L.trade_fee("cb", 1000.0, "买入") * 10 < L.trade_fee("stock", 1000.0, "买入")


# ---------------------------------------------------------------------------
# 四、成交状态分类
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("plan,filled,want", [
    (100, 100, "足额"),
    (100, 0, "未成交"),
    (100, np.nan, "未成交"),
    (100, 60, "部分成交"),
    (100, 120, "超量"),
])
def test_classify(plan, filled, want):
    assert L.classify(plan, filled) == want


def test_wavg_按金额加权且容忍空值():
    x = pd.Series([100.0, 0.0, np.nan])
    w = pd.Series([1.0, 9.0, 5.0])
    assert L.wavg(x, w) == pytest.approx(10.0)      # 大单 0bp 主导，不被 100bp 小单带跑
    assert np.isnan(L.wavg(pd.Series([np.nan]), pd.Series([1.0])))
    # 权重全零时退回等权，不能除零
    assert L.wavg(pd.Series([3.0, 5.0]), pd.Series([0.0, 0.0])) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 五、净值对账：现金必须一分不差
# ---------------------------------------------------------------------------
class _FakePrices:
    """固定价格，避免测试去加载 2400 天行情。"""

    def __init__(self, open_px, close_px, dates):
        self.o, self.c = open_px, close_px
        self.dates = pd.DatetimeIndex(dates)
        self.last = self.dates[-1]

    def available(self, d):
        return pd.Timestamp(d) <= self.last

    def open_close(self, code, kind, d):
        if not self.available(d):
            return np.nan, np.nan
        return self.o.get(code, np.nan), self.c.get(code, np.nan)

    def close(self, code, kind, d):
        return self.open_close(code, kind, d)[1]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """把 fills / combo / 输出目录都指到临时目录。"""
    fills = tmp_path / "fills"
    fills.mkdir()
    recon = tmp_path / "recon"
    combo = tmp_path / "combo.json"
    combo.write_text(json.dumps({"signal_date": "2026-11-04", "capital": 10000.0}),
                     encoding="utf-8")
    monkeypatch.setattr(L, "FILL_DIR", fills)
    monkeypatch.setattr(L, "RECON_DIR", recon)
    monkeypatch.setattr(L, "COMBO", combo)
    return fills, recon


def _write_fills(fills_dir, rows):
    cols = ["signal_date", "trade_date", "code", "name", "kind", "side",
            "plan_shares", "unit", "ref_close", "open_px", "filled_px", "filled_shares", "note"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(fills_dir / "fills_2026-11-05.csv", index=False, encoding="utf-8-sig")


def test_nav_影子组合现金账(sandbox, monkeypatch, capsys):
    fills, recon = sandbox
    _write_fills(fills, [["2026-11-04", "2026-11-05", "000001", "平安银行", "stock", "买入",
                          100, "股", 10.0, 10.0, "", "", ""]])
    monkeypatch.setattr(L, "Prices", lambda *a, **k: _FakePrices(
        {"000001": 10.0}, {"000001": 11.0}, ["2026-11-05"]))

    L.cmd_nav(argparse.Namespace(date=None, capital=10000.0, holdings=None))

    out = pd.read_csv(recon / "nav_recon.csv", index_col=0)
    fee = C.MIN_COMMISSION + 1000.0 * C.TRANSFER_FEE
    assert out["cash_shadow"].iloc[-1] == pytest.approx(10000.0 - 1000.0 - fee, abs=1e-9)
    assert out["mv_shadow"].iloc[-1] == pytest.approx(1100.0, abs=1e-9)
    assert out["nav_shadow"].iloc[-1] == pytest.approx(10000.0 - fee + 100.0, abs=1e-9)
    # 没有成交记录时不该冒出实盘列
    assert "nav_live" not in out.columns


def test_nav_实盘比影子贵的钱等于差价(sandbox, monkeypatch, capsys):
    fills, recon = sandbox
    # 计划 100 股 @ 开盘 10 元；实际成交 100 股 @ 10.05（买贵了 50bp）
    _write_fills(fills, [["2026-11-04", "2026-11-05", "000001", "平安银行", "stock", "买入",
                          100, "股", 10.0, "", 10.05, 100, ""]])
    monkeypatch.setattr(L, "Prices", lambda *a, **k: _FakePrices(
        {"000001": 10.0}, {"000001": 11.0}, ["2026-11-05"]))

    L.cmd_nav(argparse.Namespace(date=None, capital=10000.0, holdings=None))

    out = pd.read_csv(recon / "nav_recon.csv", index_col=0)
    shadow, live = out["nav_shadow"].iloc[-1], out["nav_live"].iloc[-1]
    # 多付的价差 0.05 元/股 × 100 股 = 5 元；佣金两边都被最低 5 元托住不变，
    # 但过户费是按金额的，多成交的 5 元也要多付 0.1bp —— 所以是 5 元再多一丝。
    expect = 5.0 + C.TRANSFER_FEE * 5.0
    assert shadow - live == pytest.approx(expect, abs=1e-9)
    assert out["diff_bp"].iloc[-1] == pytest.approx(-expect / shadow * 1e4, abs=1e-9)


def test_nav_持仓穿越行情末日时不硬算(sandbox, monkeypatch, capsys):
    """成交日在行情末日之后：必须明确报"无法估值"，不能拿别的价格顶上。"""
    fills, recon = sandbox
    _write_fills(fills, [["2026-11-05", "2026-11-06", "000001", "平安银行", "stock", "买入",
                          100, "股", 10.0, "", "", "", ""]])
    monkeypatch.setattr(L, "Prices", lambda *a, **k: _FakePrices(
        {"000001": 10.0}, {"000001": 11.0}, ["2026-11-05"]))
    L.cmd_nav(argparse.Namespace(date=None, capital=10000.0, holdings=None))
    txt = capsys.readouterr().out
    assert "无法估值" in txt


def test_recon_无成交时只报数量(sandbox, monkeypatch, capsys):
    fills, _ = sandbox
    _write_fills(fills, [["2026-11-04", "2026-11-05", "000001", "平安银行", "stock", "买入",
                          100, "股", 10.0, "", "", "", ""]])
    monkeypatch.setattr(L, "Prices", lambda *a, **k: _FakePrices(
        {"000001": 10.0}, {"000001": 11.0}, ["2026-11-05"]))
    L.cmd_recon(argparse.Namespace(date=None, turnover=None))
    txt = capsys.readouterr().out
    assert "成交率 0.0%" in txt
    assert "价格类指标无法计算" in txt
