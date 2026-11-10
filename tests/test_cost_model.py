# -*- coding: utf-8 -*-
"""成本口径（cost_model）测试：gross/net 的判定与扣减行为。

不加载行情数据，只测纯逻辑函数 `DataService._cost_apply`。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _svc():
    """绕过 __init__（避免 20s 数据加载），只要类属性与方法。"""
    from web.services import DataService

    return DataService.__new__(DataService)


def _nav(n=600, drift=0.0004):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(np.cumprod(np.full(n, 1.0 + drift)), index=idx)


@pytest.mark.parametrize("kind", ["etf", "sector", "cb"])
def test_net_applies_for_undercharged_strategies(kind):
    """ETF/板块/转债只内嵌 5bp 滑点，net 模式应补扣差额并降低净值。"""
    s = _svc()
    nav = _nav()
    out, info = s._cost_apply(nav, kind, {"cost_model": "net"})

    assert info["model"] == "net"
    assert info["applied"] is True
    assert info["embedded_bp"] == pytest.approx(5.0)
    assert info["delta_bp"] == pytest.approx(info["target_bp"] - 5.0)
    assert info["ann_cost"] > 0
    assert out.iloc[-1] < nav.iloc[-1]
    # 扣费是纯乘法衰减，不应改变净值形状（首日以后逐日比值恒定）
    ratio = (out / nav).to_numpy()
    assert np.all(np.diff(ratio) <= 0)


def test_stock_already_fully_loaded():
    """个股回测已按 40bp 全成本扣费，实测 50.7bp > 目标 49.3bp → 不应再扣。"""
    s = _svc()
    nav = _nav()
    out, info = s._cost_apply(nav, "stock", {"cost_model": "net"},
                              turnover_ann=0.7, embedded_bp=50.7)
    assert info["delta_bp"] == 0.0
    assert info["applied"] is False
    assert out.equals(nav)


def test_gross_is_default_and_untouched():
    """不传 cost_model 时必须是 gross，且净值原样返回（向后兼容）。"""
    s = _svc()
    nav = _nav()
    for p in ({}, {"cost_model": "gross"}):
        out, info = s._cost_apply(nav, "cb", p)
        assert info["model"] == "gross"
        assert info["applied"] is False
        assert out is nav


def test_ann_cost_formula():
    """年成本 = 换手 × 2 × 补扣bp / 1e4。"""
    s = _svc()
    _, info = s._cost_apply(_nav(), "cb", {"cost_model": "net"},
                            turnover_ann=7.2, embedded_bp=5.0)
    expect = 7.2 * 2 * (20.0 - 5.0) / 1e4
    assert info["ann_cost"] == pytest.approx(expect, rel=1e-6)


def test_bad_model_falls_back_to_gross():
    s = _svc()
    nav = _nav()
    out, info = s._cost_apply(nav, "etf", {"cost_model": "whatever"})
    assert info["model"] == "gross"
    assert out is nav


def test_frontend_and_api_wired():
    """前端开关与后端参数必须同时存在。"""
    html = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")

    assert 'id="btkCostNet"' in html
    assert "cost_model: checkedById('btkCostNet'" in js
    assert "renderCostNote" in js
    assert '"cost_model"' in svc
    for kind in ("stock", "etf", "sector", "cb"):
        assert f'"{kind}"' in svc
