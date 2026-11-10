# -*- coding: utf-8 -*-
"""数据源一致性 + 新鲜度诊断的回归测试。"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_config_auto_picks_latest_data():
    """config 应自动选行数更多的数据目录（ext 优先）。"""
    from qf import config as C
    import numpy as np

    assert hasattr(C, "DATA_SOURCE"), "缺 DATA_SOURCE 标记"
    assert C.DATA_SOURCE in ("base", "ext")
    z = np.load(C.DATA_RAW / "panel_ohlcv.npz")
    T = int(z["close"].shape[0])
    # 与 ext/base 中较大者一致
    REPO = ROOT.parent
    base = REPO / "data_extract" / "was_stock_data" / "panel_ohlcv.npz"
    ext = REPO / "data_extract" / "was_stock_data_ext" / "panel_ohlcv.npz"
    sizes = []
    for p in (base, ext):
        if p.exists():
            sizes.append(int(np.load(p)["close"].shape[0]))
    assert sizes, "两个数据目录都不存在"
    assert T == max(sizes), f"未选最新数据目录（{T} vs max {max(sizes)}）"


def test_check_data_source_script():
    p = ROOT / "scripts" / "check_data_source.py"
    assert p.exists()
    s = p.read_text(encoding="utf-8")
    for k in ("weekly_meta", "composite_score", "signal_date"):
        assert k in s, f"检测脚本缺 {k}"


def test_pipeline_passes_data_env():
    """管道必须给子进程传 QF_DATA_RAW（避免混用新旧数据）。"""
    s = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "QF_DATA_RAW" in s, "管道未传数据目录"
    assert "_data_env" in s
    assert "env=_data_env()" in s


def test_stale_diag_exists():
    s = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "def _stale_diag" in s
    # 必须区分两种类型
    assert "model_stale" in s
    assert "data_stale" in s
    # 必须有动作建议
    assert '"retrain"' in s


def test_stale_api_distinguishes():
    """API 返回的 stale 应含 kind/action 字段。"""
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8765/api/today_recommend", timeout=180).read())
    except Exception:
        pytest.skip("web 服务未运行")
    st = d.get("stale") or {}
    assert st, "缺 stale 字段"
    for k in ("level", "kind", "action", "data_last"):
        assert k in st, f"stale 缺 {k}"
    assert st["level"] in ("ok", "warn", "bad")
    assert st["kind"] in ("none", "data_stale", "model_stale")


def test_signal_not_stale_after_pipeline():
    """跑完管道后，信号日期应与数据最后日期一致。"""
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8765/api/today_recommend", timeout=180).read())
    except Exception:
        pytest.skip("web 服务未运行")
    st = d.get("stale") or {}
    if st.get("kind") == "model_stale":
        pytest.skip(f"模型滞后 {st.get('gap_days')} 日（需跑管道）")
    assert st.get("level") != "bad", f"数据状态异常: {st.get('message')}"


def test_js_freshness_uses_backend():
    """前端应优先用后端 stale 诊断。"""
    js = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
    i = js.find("function checkDataFreshness")
    assert i > 0
    seg = js[i:i + 1200]
    assert "d.stale" in seg or "stale" in seg, "前端未用后端诊断"
    assert "action" in seg, "前端未区分动作"
