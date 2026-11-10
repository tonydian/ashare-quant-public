# -*- coding: utf-8 -*-
"""研究页 UI 的回归测试。"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


def test_research_page_exists():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="page-research"' in html, "缺 research 页面"
    assert 'data-page="research"' in html, "缺 research 导航"


def test_research_page_structure():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    i = html.find('id="page-research"')
    j = html.find("</section>", i)
    seg = html[i:j]
    for cid in ("rsKpis", "rsCost", "rsRebal", "rsEtfFlex", "rsPool",
                "btnResearchRefresh"):
        assert f'id="{cid}"' in seg, f"缺 {cid}"


def test_research_js_functions():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for fn in ("loadResearch", "renderRsKpis", "renderRsCost",
               "renderRsRebal", "renderRsEtfFlex", "renderRsPool"):
        assert f"function {fn}" in js, f"缺 {fn}"
    assert "'/api/research/overview'" in js
    assert "case 'research'" in js, "未挂到 switchPage"


def test_page_titles_has_research():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    i = js.find("var PAGE_TITLES")
    assert i > 0
    seg = js[i:i + 400]
    assert "research:" in seg, "PAGE_TITLES 缺 research"


def test_research_css():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for cls in (".rs-kpis", ".rs-kpi", ".rs-bar-row", ".rs-bar-track",
                ".rs-bar-fill", ".rs-note"):
        assert cls in css, f"缺样式 {cls}"


def test_bar_tags_balanced():
    """条形图必须用 div（span 在 flex 里不可靠）。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '<span class="rs-bar-track">' not in js, "仍有 span 版 bar-track"
    assert '<span class="rs-bar-fill"' not in js, "仍有 span 版 bar-fill"
    assert '<div class="rs-bar-track">' in js


def test_js_syntax():
    import os
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node 不可用")
    # 清掉 NODE_OPTIONS：IDE/CI 注入的 --require 若失效会让 node 直接报错，与 app.js 本身无关
    env = {k: v for k, v in os.environ.items() if k != "NODE_OPTIONS"}
    r = subprocess.run([node, "--check", str(STATIC / "app.js")],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, f"语法错误: {(r.stderr or '')[:300]}"


def test_backend_methods():
    s = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    for fn in ("cost_sensitivity", "rebalance_study", "etf_pool_health",
               "research_overview"):
        assert f"def {fn}" in s, f"缺 {fn}"


def test_server_routes():
    s = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    for ep in ("/api/research/overview", "/api/research/cost",
               "/api/research/rebalance", "/api/research/pool"):
        assert f'"{ep}"' in s, f"缺路由 {ep}"


def test_research_api():
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8765/api/research/overview", timeout=180).read())
    except Exception:
        pytest.skip("web 服务未运行")
    assert "cost" in d and "rebalance" in d and "pool" in d
    assert d["cost"]["has_data"], "成本数据缺失"
    assert len(d["cost"]["sweep"]) >= 5, "滑点档位太少"


def test_kpi_uses_current_figures():
    """KPI 必须用现行口径（40日/40bp），不能是旧数据。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "主策略 CAGR（现行口径）" in js
    assert "已采用调仓周期" in js
    # 不应再出现"推荐调仓周期"这种与已采用值冲突的表述
    assert "推荐调仓周期'" not in js
