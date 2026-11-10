# -*- coding: utf-8 -*-
"""实盘跟踪台账的回归测试。"""
import json
import urllib.error
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"
TRACK = ROOT / "build" / "tracking"
SCRIPT = ROOT / "scripts" / "track_record.py"


def test_script_exists():
    assert SCRIPT.exists(), "缺 track_record.py"


def test_script_commands():
    """脚本必须支持 5 个子命令。"""
    s = SCRIPT.read_text(encoding="utf-8")
    for cmd in ("signal", "fill", "nav", "report", "reference"):
        assert f'"{cmd}"' in s or f"'{cmd}'" in s, f"缺 {cmd}"


def test_tracking_dir():
    assert TRACK.exists(), "缺 build/tracking 目录"


def test_state_file():
    p = TRACK / "tracking_state.json"
    assert p.exists(), "缺 tracking_state.json"
    st = json.load(open(p, encoding="utf-8"))
    for k in ("capital", "holdings", "cash"):
        assert k in st, f"state 缺 {k}"


def test_backtest_reference():
    p = TRACK / "backtest_reference.json"
    assert p.exists(), "缺回测参考"
    r = json.load(open(p, encoding="utf-8"))
    assert r.get("cagr") is not None
    assert r.get("rebalance_days") == 40


def test_script_runs_reference():
    """reference 命令应能运行成功。"""
    r = subprocess.run([sys.executable, str(SCRIPT), "reference"],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, f"运行失败: {(r.stderr or '')[:300]}"


def test_script_runs_report():
    """report 命令应能运行（无数据也应友好提示）。"""
    r = subprocess.run([sys.executable, str(SCRIPT), "report"],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, f"运行失败: {(r.stderr or '')[:300]}"
    assert "实盘跟踪报告" in (r.stdout or "")


# ---------------------------------------------------------------- 后端 / UI
def test_backend_method():
    s = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "def tracking_data" in s


def test_server_route():
    s = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    assert '"/api/tracking"' in s


def test_ui_page():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="page-tracking"' in html
    assert 'data-page="tracking"' in html
    for cid in ("tkKpis", "tkNav", "tkHold", "tkTrades", "tkConsistency",
                "btnTrackRefresh"):
        assert f'id="{cid}"' in html, f"缺 {cid}"


def test_ui_js():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function loadTracking" in js
    assert "function renderTracking" in js
    assert "case 'tracking'" in js, "未挂 switchPage"
    assert "tracking: '跟踪'" in js, "PAGE_TITLES 缺 tracking"


def test_ui_css():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for cls in (".tk-chart", ".tk-bar", ".tk-axis"):
        assert cls in css, f"缺 {cls}"


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


def test_api():
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8765/api/tracking", timeout=180).read())
    except Exception:
        pytest.skip("web 服务未运行")
    for k in ("state", "navs", "trades", "signals", "reference"):
        assert k in d, f"缺 {k}"
    assert d["reference"], "回测参考缺失"

# ---------------------------------------------------------------- 操作 API
def _post(path, payload=None):
    import urllib.request
    req = urllib.request.Request(
        "http://127.0.0.1:8765" + path,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=180).read()), 200
    except urllib.error.HTTPError as e:
        return None, e.code
    except Exception:
        return None, 0


def test_fill_api():
    d, code = _post("/api/tracking/fill", {
        "code": "000001", "side": "buy", "price": 10.0,
        "shares": 100, "fee": 5, "date": "2026-10-28",
        "reason": "pytest"})
    if code == 0:
        pytest.skip("web 服务未运行")
    assert code == 200 and d and d.get("ok"), f"fill 失败: code={code} d={d}"
    assert d["n_holdings"] >= 1
    _post("/api/tracking/delete",
          {"date": "2026-10-28", "code": "000001", "side": "buy"})


def test_nav_api():
    d, code = _post("/api/tracking/nav", {})
    if code == 0:
        pytest.skip("web 服务未运行")
    assert code == 200 and d and d.get("ok"), f"nav 失败: {d}"
    assert d.get("date")


def test_fill_validation():
    """非法代码应被拒绝（400）。"""
    d, code = _post("/api/tracking/fill",
                    {"code": "ABC", "side": "buy", "price": 10, "shares": 100})
    if code == 0:
        pytest.skip("web 服务未运行")
    assert code == 400, f"非法代码应返回 400，实得 {code}"


def test_ui_ops_elements():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for bid in ("tkOps", "tkBtnSignal", "tkBtnNav", "tkBtnFill", "tkBtnReset",
                "tkDate", "tkCode", "tkSide", "tkPrice", "tkShares",
                "tkFee", "tkReason", "tkOpHint"):
        assert f'id="{bid}"' in html, f"缺 {bid}"


def test_ui_ops_js():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function bindTrackingOps" in js
    assert "function tkPost" in js
    assert "tk-del" in js, "缺删除按钮逻辑"
    for ep in ("/api/tracking/fill", "/api/tracking/nav",
               "/api/tracking/signal", "/api/tracking/delete",
               "/api/tracking/reset"):
        assert ep in js, f"前端未调用 {ep}"


def test_ui_ops_css():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for cls in (".tk-op-row", ".tk-form", ".tk-f", ".tk-hint"):
        assert cls in css, f"缺 {cls}"
