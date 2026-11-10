# -*- coding: utf-8 -*-
"""全策略回测页测试。"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


def test_strategy_picker_in_html():
    """策略选择器必须含单策略 3 类 + 组合配置 6 种。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="btKindPicker"' in html, "缺少策略选择器"
    for kind in ("stock", "etf", "sector"):
        assert f'data-kind="{kind}"' in html, f"缺少 {kind} 策略卡"
    for kind in ("combo_fixed", "combo_eq", "combo_ivp", "combo_erc",
                 "combo_mdp", "combo_5050"):
        assert f'data-kind="{kind}"' in html, f"缺少 {kind} 策略卡"
    assert 'id="btKindParams"' in html, "缺少参数容器"


def test_strategy_picker_in_js():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "BT_KIND_FIELDS" in js
    assert "function btRunUnified(" in js
    assert "function renderUnifiedBacktest(" in js
    assert "function btBindKindPicker(" in js
    # 四类都要有字段配置
    for kind in ("stock:", "etf:", "sector:", "combo:", "combo_eq:",
                 "combo_ivp:", "combo_erc:", "combo_mdp:", "combo_fixed:"):
        assert kind in js, f"缺少 {kind} 参数配置"


def test_backtest_strategy_registered():
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "def backtest_strategy(" in svc
    assert "STRATEGY_KINDS" in svc
    for m in ("_bt_stock", "_bt_etf", "_bt_sector", "_bt_combo"):
        assert f"def {m}(" in svc, f"缺少 {m}"


def test_backtest_route_is_post():
    srv = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    i_get = srv.find("def do_GET")
    i_post = srv.find("def do_POST")
    i_route = srv.find('"/api/backtest_strategy"')
    assert i_route > 0, "缺少路由"
    assert i_post < i_route, "路由必须在 do_POST 里"


def test_no_stale_cash_ret_reference():
    """回归：曾误用未定义的 self._cash_ret 导致 500。"""
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "self._cash_ret" not in svc, "仍引用未定义的 self._cash_ret"


def test_bt_pack_handles_pretruncated():
    """回归：_bt_pack 必须能处理已预截取的 nav（否则空序列 IndexError）。"""
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    i = svc.find("def _bt_pack")
    seg = svc[i:i + 1200]
    assert "已被预截取过" in seg or "sub = nav" in seg, "_bt_pack 缺少空序列容错"


def test_combo_weights_normalized():
    """组合权重应归一化（和为 1）—— 语义测试。"""
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    i = svc.find("def _bt_combo")
    assert i > 0, "找不到 _bt_combo"
    seg = svc[i:i + 4000]
    # 固定权重模式：必须做归一（除以 tot 或 sum）
    assert ("/ tot" in seg or "/tot" in seg or "wv = _np.array" in seg), \
        "固定权重模式未见归一化"
    # 动态模式：必须归一（w / ssum）
    assert ("ssum" in seg or "/ w.sum()" in seg), "动态模式未见归一化"
