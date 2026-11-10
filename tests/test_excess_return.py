# -*- coding: utf-8 -*-
"""超额收益指标的回归测试。"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


def test_backend_has_excess_stats():
    s = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    assert "_excess_stats" in s
    assert '"excess": self._excess_stats(sub)' in s
    # 必须覆盖四个基准
    for k in ("csi_all", "hs300", "csi500", "csi1000"):
        assert k in s


def test_frontend_renders_excess_cards():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "d.excess" in js
    assert "超额 vs " in js
    assert "信息比率" in js
    for k in ("csi_all", "hs300", "csi500", "csi1000"):
        assert k in js


def test_excess_api_returns_metrics():
    """实际调 API 验证超额字段存在且合理。"""
    import json
    import urllib.request

    body = {"kind": "combo", "start_date": "2020-01-02",
            "weights": {"main": 0.5, "etf": 0.5, "sector": 0}}
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8765/api/backtest_strategy",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    except Exception:
        import pytest
        pytest.skip("web 服务未运行")
    ex = r.get("excess") or {}
    assert ex, "缺少 excess 字段"
    assert "csi_all" in ex
    e = ex["csi_all"]
    for k in ("excess_ann", "bench_ann", "info_ratio"):
        assert k in e
    # 合理性：超额 + 基准 ≈ 策略年化
    total = e["excess_ann"] + e["bench_ann"]
    assert abs(total - r["metrics"]["cagr"]) < 0.02, \
        f"超额+基准={total:.4f} 与策略 CAGR={r['metrics']['cagr']:.4f} 不符"
