# -*- coding: utf-8 -*-
"""个股页与区间表优化后的回归测试。"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


def test_stock_page_hero_structure():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function renderStock(" in js
    for token in ("stk-hero", "stk-box", "stk-perf", "stk-score-big",
                  "stk-score-bar", "stk-warn", "多周期涨跌"):
        assert token in js, f"个股页缺少 {token}"


def test_stock_page_explains_exclusion():
    """域外股票必须说明原因，不能只显示空白。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "为什么没有复合分" in js
    assert "exclusion" in js
    assert "选股域内" in js and "选股域外" in js


def test_stock_page_factor_empty_explained():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "该股不在选股域内，模型不为其计算因子" in js


def test_fundamentals_grouped():
    """基本面按"估值/盈利/成长/质量"分组。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for g in ("估值", "盈利", "成长", "质量"):
        assert "'" + g + "'" in js or g in js
    assert "fund-group" in js
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert ".fund-group" in css and ".fund-cell" in css


def test_candidates_field_fixed():
    """候选榜必须读 candidates/holdings 字段（曾只读 rows 导致空）。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "d.candidates" in js or "d.holdings" in js


def test_bt_range_ui_has_years_and_verdict():
    """分区间 UI 已从"表格+说明"改为"三卡对比+衰减判定"。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "年数" in js, "应显示区间年数"
    assert "function renderPeriodCards(" in js, "应使用三卡展示"
    assert "严重衰减" in js or "decay" in js.lower(), "应给出衰减判定"
    assert "2022 至今 · 真实水平" in js, "样本外应标注真实水平"
    assert "2020-2021 · 牛市" in js, "验证期应标注牛市属性"


def test_period_cards_css_exists():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for cls in (".pd-wrap", ".pd-grid", ".pd-card", ".pd-verdict"):
        assert cls in css, f"CSS 缺少 {cls}"
    assert ".bt-note" not in css, "旧样式应已移除"


def test_periods_include_n_days():
    """后端 periods 必须带 n_days（区间表显示年数需要）。"""
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    i = svc.find("def _bt_periods")
    seg = svc[i:i + 900]
    assert 'out[k]["n_days"]' in seg, "periods 缺少 n_days"


def test_no_pending_raw_score_row():
    """raw_score_row 方案已撤销（因子层面对域外股票全 NaN，无法给分）。"""
    model = (ROOT / "qf" / "model.py").read_text(encoding="utf-8")
    assert "def raw_score_row" not in model, "已撤销的方案不应残留"
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "raw_score_row" not in svc
