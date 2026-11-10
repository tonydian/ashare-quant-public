# -*- coding: utf-8 -*-
"""Web UI 后端轻量测试：不启动 HTTP 服务、不加载全量行情，只验证纯逻辑/图表/静态产物接口。

（完整 API 冒烟见 scripts/web_smoke.py：需先启动 services/web_server.py）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from web import charts
from web.server import _sanitize
from web.services import DataService, _rolling_mean_blocked


def test_sanitize_json_safe():
    out = _sanitize({"a": np.float32(1.5), "b": float("nan"), "c": np.int64(3),
                     "d": [np.float64("inf")], "e": pd.Timestamp("2026-01-01")})
    assert out["a"] == 1.5 and out["c"] == 3
    assert out["b"] is None and out["d"] == [None]
    assert out["e"].startswith("2026-01-01")
    import json
    json.dumps(out, allow_nan=False)


def test_rolling_mean_blocked():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 30)).astype(np.float32)
    out = _rolling_mean_blocked(a, 5)
    assert out.shape == a.shape
    ref = pd.DataFrame(a).rolling(5, min_periods=5).mean().to_numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_chart_index_line_png():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2025-01-01", periods=250)
    close = 5000 + np.cumsum(rng.normal(size=250))
    png = charts.index_line(dates, close, "中证全指", days=120)
    assert png[:4] == b"\x89PNG" and len(png) > 10_000


def test_chart_kline_png():
    rng = np.random.default_rng(2)
    n = 180
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(size=n))
    op = close + rng.normal(0, 0.5, n)
    hi = np.maximum(op, close) + 1
    lo = np.minimum(op, close) - 1
    vol = np.abs(rng.normal(1e6, 2e5, n))
    png = charts.stock_kline(dates, op, hi, lo, close, vol, "测试股份", "000001", days=120)
    assert png[:4] == b"\x89PNG"


def test_chart_industry_and_nav_png():
    rows = [{"industry": f"行业{i}", "ret_20d": v} for i, v in enumerate(np.linspace(-0.1, 0.15, 12))]
    assert charts.industry_bar(rows, days=20)[:4] == b"\x89PNG"
    dates = pd.bdate_range("2024-01-01", periods=120)
    nav = 300000 * np.cumprod(1 + np.random.default_rng(3).normal(0.0004, 0.01, 120))
    dd = nav / np.maximum.accumulate(nav) - 1
    assert charts.nav_chart(dates, nav, dd)[:4] == b"\x89PNG"


def test_factors_summary_static():
    svc = DataService()
    s = svc.factors_summary()
    assert s["counts"]["total"] >= 823
    assert s["counts"]["quant"] == 800 and s["counts"]["momentum"] == 8
    assert s["stage2"] > 0 and s["pool"] > 0
    assert len(s["pool_detail"]) == s["pool"]
    # 基本面族只允许 bp 一个进池（2026-11「换维度方案 A」：价值维度叠加，不顶替既有槽）
    _fund = [x["factor_key"] for x in s["pool_detail"] if x["factor_key"].startswith("fundamental-extra:")]
    assert _fund in ([], ["fundamental-extra:F002"]), _fund
    assert s["composite_ic"].get("mean") is not None
    # 报告列表
    r = svc.reports()
    assert any(f["name"].endswith(".md") for f in r["files"])


def test_family_share_monitor():
    """族占比监控：只展示、不干预权重。

    背景（2026-11-03）：族上限 cap 0.30/0.25/0.20 三档全否，损伤随干预强度单调
    递增，23 期最大族始终是 risk-pattern → 结论「集中度是结果不是病因」，
    所以这里只做展示层监控，超 FAMILY_SHARE_WARN 标黄。
    """
    from web.services import FAMILY_SHARE_WARN

    svc = DataService()
    s = svc.factors_summary()
    fw = s["family_weights"]
    assert fw, "族占比为空（pool_detail 的族名解析失败？）"
    assert abs(sum(x["share"] for x in fw) - 1.0) < 1e-9
    assert all(0 < x["share"] <= 1 for x in fw)
    assert sum(x["n"] for x in fw) == s["pool"]
    # 阈值判定必须与常量一致（>33% 才标黄）
    assert all(x["over"] == (x["share"] > FAMILY_SHARE_WARN) for x in fw)
    # 族名口径 = factor_key 前缀，与族预算/族上限实验一致
    assert all(":" not in x["family"] and x["family"] for x in fw)
    share_of = {x["family"]: x["share"] for x in fw}
    for r in s["pool_detail"]:
        assert r["family"] in share_of, r["factor_key"]
        assert abs(r["family_share"] - share_of[r["family"]]) < 1e-12
    # 降序排列，前端第一行就是最大族
    assert [x["share"] for x in fw] == sorted((x["share"] for x in fw), reverse=True)


def test_model_registry_is_whitelist():
    """界面可选模型只来自白名单：实验残留目录不得自动冒进下拉框。

    历史问题：模型列表靠扫描 build/models/* 生成，跑一次实验就多几项，
    且字母序下主模型 wf_h10 排在第 23 位。
    """
    from web.services import MODEL_NAME, MODEL_SHOW

    svc = DataService()
    names = [m["name"] for m in svc.registry]
    assert names, "白名单模型一个都没装配上"
    assert set(names) <= set(MODEL_SHOW), f"越权暴露：{set(names) - set(MODEL_SHOW)}"
    assert MODEL_NAME in names
    # 主模型必须排第一（默认选中项在前）
    assert names[0] == MODEL_NAME

    # 磁盘上存在、但不在白名单里的目录一律不可见
    import pathlib
    base = pathlib.Path(__file__).resolve().parent.parent / "build" / "models"
    if base.exists():
        stray = {d.name for d in base.iterdir()
                 if (d / "composite_score.npz").exists() and d.name not in MODEL_SHOW}
        assert stray, "预期存在未被展示的实验模型目录"
        assert not (stray & set(names))

    for m in svc.registry:
        assert m["label"] and m["desc"], f"{m['name']} 缺 label/desc"
        assert m["kind"] in ("prod", "research")
        assert m["pool"] >= 0 and m["horizon"] >= 1
