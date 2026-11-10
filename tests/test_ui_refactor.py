# -*- coding: utf-8 -*-
"""UI 重构后的回归测试：页面合并、参数生效、图表元素。"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


def test_recommend_page_removed():
    """「推荐」页已合并进「今日推荐」。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="page-recommend"' not in html, "旧推荐页应已删除"
    assert 'data-page="recommend"' not in html, "旧推荐导航应已删除"
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "recommend: '推荐'" not in js
    assert "case 'recommend'" not in js


def test_today_has_candidates():
    """今日推荐页包含候选榜与信号说明（原推荐页核心内容）。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="todayCandidates"' in html
    assert 'id="todaySignalNote"' in html
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function loadTodayCandidates(" in js
    assert "function renderTodayCandidates(" in js


def test_no_orphan_high_params_card():
    """旧的「回测参数」卡片（字段不生效）应已删除。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "高级参数（仅「个股模型」生效）" not in html
    # btModel 等旧 id 不应再出现（会被新面板替代）
    for old_id in ('id="btModel"', 'id="btNHoldings"', 'id="btCashBond"',
                   'id="btBandRel"', 'id="btOverlay"'):
        assert old_id not in html, f"旧参数字段仍在: {old_id}"


def test_run_button_present():
    """运行回测按钮必须存在（曾因删卡片被误删）。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="btnRunBacktest"' in html


def test_all_kind_params_have_ids():
    """四类策略的参数 id 必须在 BT_KIND_FIELDS 中定义且能读取。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "BT_KIND_FIELDS" in js
    # stock 专有
    for k in ("btkModel", "btkNHold", "btkRebal", "btkMaxNew", "btkCashBond",
              "btkCashGold", "btkOverlay", "btkStopLoss"):
        assert k in js, f"缺少 stock 参数 {k}"
    # etf / sector 共用
    for k in ("btkEntry", "btkExit", "btkMinAmt", "btkMom", "btkTop"):
        assert k in js, f"缺少 ETF/板块参数 {k}"
    # combo
    for k in ("btkWMain", "btkWETF", "btkWSector"):
        assert k in js, f"缺少组合参数 {k}"


def test_bt_run_reads_all_params():
    """btRunUnified 必须实际读取每个参数（而非写死）。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    i = js.find("function btRunUnified()")
    j = js.find("function checkedById(")
    seg = js[i:j]
    for k in ("btkModel", "btkNHold", "btkRebal", "btkMaxNew", "btkCashBond",
              "btkCashGold", "btkOverlay", "btkStopLoss",
              "btkEntry", "btkExit", "btkMinAmt", "btkMom", "btkTop",
              "btkWMain", "btkWETF", "btkWSector"):
        assert k in seg, f"btRunUnified 未读取 {k}"


def test_chart_has_axis_and_drawdown():
    """收益图应包含坐标轴刻度、网格、回撤阴影与悬停元素。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function renderNavChart(" in js
    assert "function btBindChartHover(" in js
    for token in ("gridline", "btbase", "bthover", "btHoverDot", "最大回撤",
                  "btNavGrad", "btTip"):
        assert token in js, f"图表缺少 {token}"
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for cls in (".bt-chart", ".bt-chart .gridline", ".bt-chart .bttip"):
        assert cls in css, f"CSS 缺少 {cls}"


def test_stock_params_match_official_semantics():
    """max_new 默认必须为 6（与 run_backtest 默认一致），否则结果不可比。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "numById('btkMaxNew', 6)" in js, "max_new 默认应为 6"
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "max_new = 6" in svc or "max_new = 6\n" in svc
    # trade_band 必须显式传给 run_backtest（正式回测默认 0.3/6000）
    assert "trade_band_rel=0.30" in svc


def test_today_export_button():
    """今日推荐页必须有导出 CSV 按钮 + 对应路由。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    srv = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    assert 'id="btnTodayExport"' in html
    assert "bindTodayExport" in js
    assert "/api/today_export" in srv


def test_holdings_input_panel():
    """必须能录入当前持仓并计算买卖差异。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for k in ('id="holdingsPanel"', 'id="holdingsText"',
              'id="btnHoldingsApply"', 'id="btnHoldingsInput"'):
        assert k in html, f"缺少 {k}"
    assert "parseHoldings" in js
    assert "pill-down" in js


def test_stale_data_warning():
    """数据过期必须有警示条。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="dataStaleBar"' in html
    assert 'id="dataStaleMsg"' in html
    assert "checkDataFreshness" in js
    assert "stale-bad" in js and "stale-warn" in js


def test_loading_stage_hint():
    """冷启动应有分阶段提示。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "LOAD_STAGES" in js
    assert "initLoadingStages" in js


def test_stock_detail_fast_path():
    """域外个股必须走快速路径（不加载全市场打分）。"""
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "_stock_detail_impl" in svc
    assert "fast_path" in svc
    assert "in_domain" in svc


def test_live_scores_prewarmed():
    """live_scores 必须预热（否则首次搜股要等 20s+）。"""
    srv = (ROOT / "scripts" / "web_server.py").read_text(encoding="utf-8")
    assert "_warm_live_scores" in srv, "缺少 live_scores 预热"
    assert "_get_live_features" in srv, "缺少 live features 预热"
    svc = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "_get_live_scores" in svc, "缺少 live_scores 缓存读取"
