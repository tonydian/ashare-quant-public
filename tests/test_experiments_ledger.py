# -*- coding: utf-8 -*-
"""实验台账的回归测试。"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"
LEDGER = ROOT / "build" / "eval" / "experiments_ledger.json"


def test_ledger_file_exists():
    assert LEDGER.exists(), "缺实验台账文件"


def test_ledger_structure():
    d = json.load(open(LEDGER, encoding="utf-8"))
    assert "experiments" in d and "kept" in d
    assert len(d["experiments"]) >= 3, "弃用记录太少"
    assert len(d["kept"]) >= 3, "保留记录太少"
    for x in d["experiments"]:
        assert x.get("name"), "缺 name"
        assert x.get("verdict") == "弃用"
        assert len(x.get("reasons") or []) >= 2, f"{x.get('name')} 原因太少"


def test_ledger_has_fund_rsi():
    """基本面 RSI 的否决必须记录（防止重复尝试）。"""
    d = json.load(open(LEDGER, encoding="utf-8"))
    names = [x["name"] for x in d["experiments"]]
    assert any("RSI" in n for n in names), f"未记录 RSI 实验: {names}"


def test_backend_method():
    s = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "def experiments_ledger" in s
    assert '"ledger": self.experiments_ledger()' in s


def test_ui_elements():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="rsLedger"' in html
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function renderRsLedger" in js
    assert "renderRsLedger(d)" in js


def test_css():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    for cls in (".rs-dead", ".rs-dead-h", ".rs-dead-tag", ".rs-dead-list"):
        assert cls in css, f"缺 {cls}"


def test_api_returns_ledger():
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8765/api/research/overview", timeout=180).read())
    except Exception:
        pytest.skip("web 服务未运行")
    lg = d.get("ledger") or {}
    assert lg.get("experiments"), "API 未返回台账"
    assert lg.get("kept"), "API 未返回保留列表"
