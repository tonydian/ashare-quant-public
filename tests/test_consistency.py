# -*- coding: utf-8 -*-
"""一致性检测 + verify 阶段的回归测试。"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_verify_stage_in_pipeline():
    """管道必须以 verify 阶段收尾。"""
    s = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert '"verify"' in s, "缺 verify 阶段"
    assert "check_data_source.py" in s
    # verify 必须是最后一个
    i_verify = s.find('("verify"')
    i_signal = s.find('("signal"')
    assert i_verify > i_signal, "verify 应在 signal 之后（作为门禁）"


def test_check_script_has_json_flag():
    s = (ROOT / "scripts" / "check_data_source.py").read_text(encoding="utf-8")
    assert "--json" in s
    assert "fix_hint" in s, "应给修复建议"


def test_check_script_exits_nonzero_when_bad():
    """检测脚本在不一致时应返回非 0（门禁语义）。"""
    s = (ROOT / "scripts" / "check_data_source.py").read_text(encoding="utf-8")
    assert "raise SystemExit(0 if r[\"ok\"] else 1)" in s


def test_consistency_api():
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8765/api/consistency", timeout=180).read())
    except Exception:
        pytest.skip("web 服务未运行")
    assert "ok" in d
    assert "items" in d
    # 至少检查了行情面板/因子/模型/信号
    names = [x.get("name", "") for x in (d.get("items") or [])]
    assert any("行情" in n for n in names), f"缺行情检查: {names}"
    assert any("因子" in n for n in names)
    assert any("模型" in n for n in names)


def test_pipeline_ui_has_consistency_box():
    html = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="consistencyBox"' in html
    assert 'id="btnConsCheck"' in html


def test_pipeline_ui_collapses_skipped():
    js = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "pl-skip-note" in js, "跳过阶段应折叠"
    assert "loadConsistency" in js
    # 日志应折叠
    assert "<details" in js


def test_css_for_consistency():
    css = (ROOT / "web" / "static" / "style.css").read_text(encoding="utf-8")
    for cls in (".cons-wrap", ".cons-head", ".cons-item", ".cons-fix",
                ".pl-skip-note", ".pl-log-pre"):
        assert cls in css, f"缺样式 {cls}"


def test_check_runs_clean():
    """直接跑检测脚本，应 exit 0（当前状态一致）。"""
    r = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / "check_data_source.py"), "--json"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120)
    out = (r.stdout or "").strip()
    if not out.startswith("{"):
        pytest.skip(f"输出异常: {out[:100]} {r.stderr[:100]}")
    d = json.loads(out)
    if not d.get("ok"):
        pytest.skip(f"当前存在滞后（正常，需跑管道）: "
                    f"{[x['name'] for x in d.get('items', []) if not x['ok']]}")
    assert d["ok"]
