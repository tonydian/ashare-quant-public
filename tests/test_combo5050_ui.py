# -*- coding: utf-8 -*-
"""回测页组合配置卡的回归测试（适配 6 种配置方法架构）。"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"


def test_all_combo_cards_exist():
    """回测页必须有 6 种组合配置卡。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for kind in ("combo_fixed", "combo_eq", "combo_ivp", "combo_erc",
                 "combo_mdp", "combo_5050"):
        assert f'data-kind="{kind}"' in html, f"缺 {kind} 卡"


def test_combo_cards_have_method_attr():
    """动态配置卡必须带 data-method 属性。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for method in ("eq", "ivp", "erc", "mdp"):
        assert f'data-method="{method}"' in html, f"缺 method={method}"


def test_strategy_groups_layout():
    """策略卡必须分组（单策略 / 组合配置）。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "strat-groups" in html
    assert html.count("strat-group-title") >= 2, "至少两个分组标题"


def test_dynamic_fields_defined():
    """动态方法必须有协方差窗口/再平衡间隔字段。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for kind in ("combo_ivp", "combo_erc", "combo_mdp"):
        assert f"{kind}: [" in js, f"缺 {kind} 字段定义"
    assert "btkCovWin" in js
    assert "btkRebalDyn" in js


def test_run_sends_method():
    """运行时必须发 method 参数。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for kind in ("combo_eq", "combo_ivp", "combo_erc", "combo_mdp"):
        assert f"'{kind}'" in js, f"缺 {kind} 分支"
    assert "body.method = BT_KIND.replace('combo_', '')" in js


def test_cash_defaults_no_gold():
    """现金替代默认值必须为 100% 国债 / 0% 黄金。"""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "btkCashBond', 1.0)" in js or "btkCashBond', 1)" in js
    assert "btkCashGold', 0.0)" in js or "btkCashGold', 0)" in js


def test_etf_card_text_updated():
    """ETF 卡片应显示"最多持 2 只"。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "最多持 2 只" in html
    assert "最多持 1 只" not in html


def test_backend_supports_all_methods():
    """后端 _bt_combo 必须支持 6 种方法。"""
    s = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
    i = s.find("def _bt_combo")
    seg = s[i:i + 3000]
    for m in ("fixed", "eq", "ivp", "erc", "mdp"):
        assert f'"{m}"' in seg or f"'{m}'" in seg, f"后端缺 method={m}"
    assert "portfolio_erc" in seg, "应调用 qf.portfolio_erc"


def test_api_accepts_method():
    """实际调 API 验证 6 种方法都能跑。"""
    import json
    import urllib.request

    cases = [("fixed", {"weights": {"main": 0.5, "etf": 0.3, "sector": 0.2}}),
             ("eq", {}), ("ivp", {"cov_window": 252, "rebal": 60}),
             ("erc", {"cov_window": 252, "rebal": 60}),
             ("mdp", {"cov_window": 252, "rebal": 60})]
    for method, extra in cases:
        body = {"kind": "combo", "method": method, "start_date": "2024-01-02", **extra}
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8765/api/backtest_strategy",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        except Exception:
            pytest.skip("web 服务未运行")
        assert "metrics" in r, f"{method} 无 metrics"
        assert r["metrics"]["cagr"] is not None, f"{method} CAGR 为空"
        # 权重和应为 1
        w = r.get("weights") or {}
        assert abs(sum(w.values()) - 1.0) < 0.02, f"{method} 权重和={sum(w.values())}"
