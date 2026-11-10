# -*- coding: utf-8 -*-
"""一键更新管道 + UI 的回归测试。"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


def test_pipeline_script_exists():
    p = ROOT / "scripts" / "run_pipeline.py"
    assert p.exists(), "缺 run_pipeline.py"
    s = p.read_text(encoding="utf-8")
    # 必须包含全部关键阶段
    for key in ("data", "factors", "model", "backtest", "etf", "sector",
                "combo", "three", "erc", "tracking", "signal"):
        assert f'"{key}"' in s, f"管道缺阶段 {key}"
    # 必须写状态文件
    assert "status.json" in s
    # 必须支持跳过
    assert "--skip-data" in s and "--skip-factors" in s


def test_pipeline_script_syntax():
    import ast
    p = ROOT / "scripts" / "run_pipeline.py"
    ast.parse(p.read_text(encoding="utf-8"))


def test_backend_has_pipeline_methods():
    s = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "def pipeline_status" in s
    assert "def pipeline_run" in s
    assert "run_pipeline.py" in s


def test_server_routes_registered():
    s = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    assert '"/api/pipeline/status"' in s
    assert '"/api/pipeline/run"' in s


def test_ui_card_exists():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="pipelineCard"' in html, "缺更新卡片"
    # 重设计后：只保留 2 个更新按钮 + 1 个检测按钮
    for bid in ("btnPipeRun", "btnPipeFast", "pipeStatusText",
                "pipelineStages", "btnConsCheck"):
        assert f'id="{bid}"' in html, f"缺 {bid}"
    # 旧的冗余按钮已移除
    for old_id in ("btnPipeRefresh", "btnDataUpdate", "btnReloadPage"):
        assert f'id="{old_id}"' not in html, f"旧按钮 {old_id} 应已移除"


def test_ui_js_functions():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for fn in ("loadPipelineStatus", "bindPipeline", "renderPipeline"):
        assert f"function {fn}" in js, f"缺 {fn}"
    assert "'/api/pipeline/run'" in js
    assert "skip_factors: true" in js


def test_css_styles():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for cls in (".pl-actions", ".pl-stage", ".pl-dot", ".pl-status", ".pl-log"):
        assert cls in css, f"缺样式 {cls}"


def test_api_status_works():
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8765/api/pipeline/status", timeout=120).read())
    except Exception:
        pytest.skip("web 服务未运行")
    assert "running" in d
    assert "stages" in d
