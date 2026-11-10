# -*- coding: utf-8 -*-
"""数据页重设计的回归测试（6 按钮 → 3 按钮）。"""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


def _data_section() -> str:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    i = html.find('id="page-data"')
    assert i > 0, "找不到数据页"
    j = html.find("</section>", i)
    return html[i:j]


def test_only_three_buttons():
    """数据页只应有 3 个按钮（全部更新 / 只重算策略 / 重新检测）。"""
    seg = _data_section()
    ids = re.findall(r'<button[^>]*id="(\w+)"', seg)
    assert len(ids) == 3, f"按钮数 {len(ids)} != 3: {ids}"
    assert set(ids) == {"btnPipeRun", "btnPipeFast", "btnConsCheck"}


def test_removed_buttons():
    """旧的冗余按钮必须移除。"""
    seg = _data_section()
    for old in ("btnDataUpdate", "btnReloadPage", "btnPipeRefresh"):
        assert old not in seg, f"旧按钮 {old} 仍在"


def test_main_buttons_have_description():
    """主按钮需带副标题说明。"""
    seg = _data_section()
    assert seg.count('class="up-txt"') >= 2, "主按钮缺说明文字"
    assert "重建因子" in seg
    assert re.search(r"约\s*\d+\s*分钟", seg), "主按钮缺耗时说明"


def test_pipeline_stages_container():
    seg = _data_section()
    for cid in ("pipelineStages", "pipelineTail", "pipeStatusText",
                "pipeLastRun", "pipeEta"):
        assert f'id="{cid}"' in seg, f"缺 {cid}"


def test_consistency_container():
    seg = _data_section()
    for cid in ("consistencyBox", "consSummary", "btnConsCheck"):
        assert f'id="{cid}"' in seg, f"缺 {cid}"


def test_old_data_job_functions_removed():
    """旧的 dataJob 体系函数应已删除。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for fn in ("startDataUpdate", "pollDataJob", "refreshDataPanel",
               "attachDataJob", "dataJobDone", "dataJobFail", "bindDataPage"):
        assert f"function {fn}(" not in js, f"{fn} 未删除"


def test_no_dangling_calls():
    """不应有对已删除函数的调用。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for fn in ("bindDataPage", "refreshDataPanel", "loadDataStatus"):
        # 排除定义行
        calls = re.findall(r"(?<!function )" + fn + r"\(", js)
        assert len(calls) == 0, f"{fn} 仍有 {len(calls)} 处调用"


def test_load_data_info_exists():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function loadDataInfo" in js
    assert "loadDataInfo()" in js, "loadDataInfo 未被调用"


def test_data_page_css():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for cls in (".up-row", ".btn-lg", ".up-ico", ".up-txt", ".up-meta"):
        assert cls in css, f"缺样式 {cls}"


def test_js_syntax_ok():
    """用 node 检查语法（若可用）。"""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node 不可用")
    # 清掉 NODE_OPTIONS：CI/IDE 注入的 --require 预加载如果路径失效，会让 node 直接报错，
    # 与本项目 app.js 的语法无关（本地曾因此误报）。
    import os
    env = {k: v for k, v in os.environ.items() if k != "NODE_OPTIONS"}
    r = subprocess.run([node, "--check", str(STATIC / "app.js")],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, f"app.js 语法错误: {(r.stderr or '')[:300]}"


def test_api_available():
    """数据页依赖的 API 都应可用。"""
    import json
    import urllib.request
    for ep in ("data/status", "pipeline/status", "consistency"):
        try:
            d = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:8765/api/{ep}", timeout=120).read())
        except Exception:
            pytest.skip("web 服务未运行")
        assert isinstance(d, dict), f"{ep} 返回异常"
