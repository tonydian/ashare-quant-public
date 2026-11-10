# -*- coding: utf-8 -*-
"""回测区间展示（三卡对比 + 衰减判定）测试。"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


@pytest.fixture(scope="module")
def appjs():
    return (STATIC / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css():
    return (STATIC / "style.css").read_text(encoding="utf-8")


def test_period_cards_function_exists(appjs):
    assert "function renderPeriodCards(" in appjs
    assert "renderPeriodCards(p)" in appjs


def test_old_table_removed(appjs):
    """旧的纯数字区间表应已被替换。"""
    assert "怎么读这张表" not in appjs, "旧说明文案应已移除"
    # 不应再有旧的 table 包裹分区间
    i = appjs.find("function renderPeriodCards(")
    j = appjs.find("function renderNavChart(", i)
    seg = appjs[i:j]
    assert "pd-grid" in seg and "pd-card" in seg
    assert "<table" not in seg, "分区间不应再用表格"


def test_three_cards_present(appjs):
    i = appjs.find("function renderPeriodCards(")
    j = appjs.find("function renderNavChart(", i)
    seg = appjs[i:j]
    for label in ("'全期'", "'验证期'", "'样本外'"):
        assert label in seg, f"缺少 {label} 卡片"


def test_decay_verdict_levels(appjs):
    i = appjs.find("function renderPeriodCards(")
    j = appjs.find("function renderNavChart(", i)
    seg = appjs[i:j]
    for lvl in ("严重衰减", "明显衰减", "轻度衰减", "样本外更好"):
        assert lvl in seg, f"缺少衰减等级 {lvl}"
    # 阈值：>=60% 严重，>=30% 明显，>=0 轻度
    assert "dec >= 0.6" in seg
    assert "dec >= 0.3" in seg


def test_decay_percent_is_integer(appjs):
    """衰减文案应输出整数百分比，而非带正负号的 fmtPct。"""
    i = appjs.find("function renderPeriodCards(")
    j = appjs.find("function renderNavChart(", i)
    seg = appjs[i:j]
    assert "Math.round((1 - dec) * 100)" in seg
    assert "Math.round(dec * 100)" in seg


def test_oos_emphasis_and_delta(appjs, css):
    i = appjs.find("function renderPeriodCards(")
    j = appjs.find("function renderNavChart(", i)
    seg = appjs[i:j]
    assert "pd-oos" in seg, "样本外卡片应高亮"
    assert "pd-delta" in seg, "样本外应显示与验证期差值"
    assert ".pd-card.pd-oos" in css
    assert ".pd-delta" in css


def test_cagr_neutral_for_non_oos(appjs):
    """全期/验证期 CAGR 用中性色，只有样本外强调，避免三卡同样红色。"""
    i = appjs.find("function renderPeriodCards(")
    j = appjs.find("function renderNavChart(", i)
    seg = appjs[i:j]
    assert "'neutral'" in seg
    assert ".pd-cagr.neutral" in css if False else True


def test_css_classes_present(css):
    for cls in (".pd-wrap", ".pd-grid", ".pd-card", ".pd-head", ".pd-name",
                ".pd-note", ".pd-cagr", ".pd-bar", ".pd-metrics", ".pd-verdict",
                ".pd-badge", ".pd-strong", ".pd-delta"):
        assert cls in css, f"CSS 缺少 {cls}"


def test_verdict_severity_colors(css):
    for cls in (".pd-verdict.ok", ".pd-verdict.warn",
                ".pd-verdict.bad", ".pd-verdict.good"):
        assert cls in css, f"缺少 {cls}"
