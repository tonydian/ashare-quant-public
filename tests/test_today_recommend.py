# -*- coding: utf-8 -*-
"""今日推荐接口与页面测试。"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
COMBO = ROOT / "signals" / "latest_combo.json"
STATIC = ROOT / "web" / "static"


@pytest.fixture(scope="module")
def combo():
    if not COMBO.exists():
        pytest.skip("latest_combo.json 不存在")
    return json.load(open(COMBO, encoding="utf-8"))


def test_today_page_exists():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'data-page="today"' in html, "缺少 今日推荐 页签"
    assert 'id="page-today"' in html, "缺少 page-today section"
    assert "今日操作清单" in html
    assert "荐股" in html and "荐基" in html


def test_today_is_default_page():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "switchPage(PAGE_TITLES[initial] ? initial : 'today')" in js, "默认页应为 today"
    assert "today: '今日推荐'" in js


def test_today_renderer_registered():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function loadToday(" in js
    assert "function renderToday(" in js
    assert "case 'today': loadToday(force); break;" in js


def test_combo_has_both_kind_fields(combo):
    """组合信号应能区分个股与基金。"""
    m = combo.get("main") or {}
    e = combo.get("etf") or {}
    s = combo.get("sector") or {}
    assert "targets" in m
    assert "leg" in e
    if combo.get("mode") == "three":
        assert "legs" in s


def test_stock_targets_are_a_shares(combo):
    """荐股必须是 A 股 6 位代码，且在合理价格区间。"""
    for t in (combo.get("main") or {}).get("targets", []):
        code = str(t["code"])
        assert len(code) == 6 and code.isdigit(), f"非法代码 {code}"
        assert 1.0 < float(t["price"]) < 500, f"价格异常 {code}: {t['price']}"
        assert int(t["shares"]) % 100 == 0, f"{code} 非整手"
        assert int(t["shares"]) > 0


def test_etf_legs_are_etf_codes(combo):
    """荐基代码必须是 ETF 代码段（51xxxx/159xxx/56xxxx/58xxxx）。"""
    legs = []
    if (combo.get("etf") or {}).get("leg"):
        legs.append(combo["etf"]["leg"])
    legs += (combo.get("sector") or {}).get("legs", []) or []
    for leg in legs:
        code = str(leg["code"])
        assert code[:2] in ("51", "15", "56", "58"), f"非 ETF 代码: {code}"
