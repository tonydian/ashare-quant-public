# -*- coding: utf-8 -*-
"""ERC 风险平价 + 配置对比页的回归测试。"""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


# ---------------------------------------------------------------- ERC 数学
def test_erc_equal_risk_contribution():
    """ERC 的核心性质：各资产风险贡献必须相等。"""
    from qf import portfolio_erc as pe

    rng = np.random.default_rng(42)
    for _ in range(5):
        A = rng.normal(size=(3, 3))
        cov = A @ A.T + np.eye(3) * 0.3
        w = pe.erc_weights(cov)
        assert abs(w.sum() - 1.0) < 1e-8, "权重和应为 1"
        assert (w >= 0).all(), "不允许负权重"
        rc = pe.risk_contributions(cov, w)
        assert rc.max() - rc.min() < 1e-4, \
            f"风险贡献应相等，实际极差 {rc.max()-rc.min():.2e}"


def test_erc_vs_ivp_different():
    """有相关性时，ERC 与 IVP 必须不同（否则 ERC 没意义）。"""
    from qf import portfolio_erc as pe

    cov = np.array([[0.010, 0.006, 0.003],
                    [0.006, 0.020, 0.004],
                    [0.003, 0.004, 0.008]])
    w_erc = pe.erc_weights(cov)
    w_ivp = pe.ivp_weights(cov)
    assert np.max(np.abs(w_erc - w_ivp)) > 1e-3, \
        "存在相关性时 ERC 应不同于 IVP"


def test_ivp_single_asset_limits():
    """单资产时权重应为 1。"""
    from qf import portfolio_erc as pe

    w = pe.ivp_weights(np.array([[0.01]]))
    assert abs(w[0] - 1.0) < 1e-9


def test_mdp_weights_valid():
    from qf import portfolio_erc as pe

    cov = np.array([[0.01, 0.002, 0.001],
                    [0.002, 0.015, 0.003],
                    [0.001, 0.003, 0.02]])
    w = pe.mdp_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()


# ---------------------------------------------------------------- 结果文件
def test_erc_result_exists():
    p = ROOT / "build" / "eval" / "erc_deep.json"
    if not p.exists():
        pytest.skip("未跑 erc_deep.py")
    d = json.load(open(p, encoding="utf-8"))
    assert "methods" in d
    assert len(d["methods"]) >= 6
    for name, v in d["methods"].items():
        for seg in ("full", "val", "oos"):
            assert seg in v, f"{name} 缺 {seg}"
        assert "avg_weights" in v
        assert abs(sum(v["avg_weights"]) - 1.0) < 1e-6


def test_erc_is_best_sharpe():
    """ERC 的 Sharpe 应为六种方法中最高的。"""
    p = ROOT / "build" / "eval" / "erc_deep.json"
    if not p.exists():
        pytest.skip("未跑 erc_deep.py")
    d = json.load(open(p, encoding="utf-8"))
    ms = d["methods"]
    best = max(ms.items(), key=lambda kv: kv[1]["full"]["sharpe"])
    assert "ERC" in best[0], f"最高 Sharpe 应为 ERC，实际是 {best[0]}"


def test_candidate_tracking_exists():
    p = ROOT / "build" / "eval" / "candidate_rolling_oos.json"
    if not p.exists():
        pytest.skip("未跑 candidate_tracking.py")
    d = json.load(open(p, encoding="utf-8"))
    assert d.get("rolling_start"), "缺 rolling_start"
    assert d.get("starts"), "缺 starts"
    assert d.get("latest"), "缺 latest"


# ---------------------------------------------------------------- 前端
def test_portfolio_page_exists():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'data-page="portfolio"' in html, "缺导航项"
    assert 'id="page-portfolio"' in html, "缺页面容器"
    # 必须有 data-page 属性（否则 switchPage 找不到）
    assert 'id="page-portfolio" data-page="portfolio"' in html, \
        "section 缺 data-page 属性，页面不会显示"


def test_portfolio_js_renderers():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for fn in ("renderPfMethods", "renderPfWeights", "renderPortfolioCorr",
               "renderPfGrid", "renderCandidates", "loadPortfolio"):
        assert f"function {fn}" in js, f"缺 {fn}"
    # PAGE_TITLES 必须含 portfolio（否则 switchPage 会重置为 market）
    assert "portfolio: '配置对比'" in js, "PAGE_TITLES 缺 portfolio"
    # switch case 必须正确
    assert "case 'portfolio': loadPortfolio(); break;" in js


def test_portfolio_api_methods():
    """实际调 API 验证。"""
    import urllib.request
    try:
        d = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8765/api/combo_methods", timeout=120).read())
    except Exception:
        pytest.skip("web 服务未运行")
    assert "methods" in d and len(d["methods"]) >= 6
    assert "corr" in d
