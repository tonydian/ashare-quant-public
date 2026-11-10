# -*- coding: utf-8 -*-
"""本月推荐（低换手月度）页面与接口测试：不启动 HTTP、不加载全量行情。"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.services import DataService  # noqa: E402

STATIC = ROOT / "web" / "static"
SIG = ROOT / "signals" / "lowturn_signal.json"
COMBO = ROOT / "signals" / "latest_combo.json"
PICKS = ROOT / "build" / "runs" / "lowturn" / "picks.json"


def _svc():
    """不触发行情加载的 DataService 骨架（这些接口只读已落盘产物）。"""
    s = object.__new__(DataService)
    s.md = None
    return s


@pytest.fixture(scope="module")
def monthly():
    if not SIG.exists():
        pytest.skip("lowturn_signal.json 不存在")
    return _svc().monthly_recommend()


# ---------------------------------------------------------------- 页面/前端
def test_monthly_page_exists():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'data-page="monthly"' in html
    assert 'id="page-monthly"' in html
    # 导航项要在「决策」组里
    nav = html.split('<div class="nav-group-title">决策</div>')[1].split("</nav>")[0]
    assert 'data-page="monthly"' in nav, "本月推荐应放在决策组"


def test_monthly_renderer_registered():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "monthly: '本月推荐'" in js
    assert "function loadMonthly(" in js
    assert "function renderMonthly(" in js
    assert "case 'monthly': loadMonthly(force); break;" in js


def test_today_page_has_strategy_board_and_filter():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="todayStrategies"' in html
    assert 'id="todayFilter"' in html
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "function renderTodayStrategies(" in js
    assert "data-st-filter" in js and "setTodayFilter" in js
    # 低换手卡片要能跳到本月推荐
    assert "data-jump=\"monthly\"" in js or "data-jump" in js


# ---------------------------------------------------------------- 接口契约
def test_monthly_payload_shape(monthly):
    for k in ("target_month", "asof", "status_cn", "next_action", "rows",
              "rotation", "sector_dist", "perf", "yearly", "bench",
              "monthly_returns", "nav_series", "budget", "invested", "cash"):
        assert k in monthly, f"缺少字段 {k}"
    assert monthly["strategy"] == "lowturn_monthly"
    assert monthly["n_pick"] == len(monthly["rows"]) == 10


def test_monthly_budget_follows_combo_weight(monthly):
    """预算必须按组合权重切分，否则股数与今日清单对不上。"""
    if not COMBO.exists():
        pytest.skip("latest_combo.json 不存在")
    combo = json.load(open(COMBO, encoding="utf-8"))
    w = float((combo.get("weights") or {}).get("lowturn") or 0)
    cap = float(combo.get("capital") or 0)
    if not w or combo.get("lowturn") is None:
        pytest.skip("组合信号里没有低换手侧")
    assert monthly["weight"] == pytest.approx(w)
    assert monthly["budget"] == pytest.approx(cap * w)
    assert monthly["sized_by"] == "combo"
    # 股数之和 == 已投金额（同一份 legs，不能各算一套）
    assert monthly["invested"] == pytest.approx(
        sum(r["amount"] or 0 for r in monthly["rows"]))
    assert monthly["cash"] == pytest.approx(max(0.0, monthly["budget"] - monthly["invested"]))
    assert 0.85 <= (monthly["invested"] / monthly["budget"]) <= 1.02


def test_monthly_rows_match_combo_legs(monthly):
    if not COMBO.exists():
        pytest.skip("latest_combo.json 不存在")
    combo = json.load(open(COMBO, encoding="utf-8"))
    legs = (combo.get("lowturn") or {}).get("legs") or []
    if not legs:
        pytest.skip("组合信号里没有低换手 legs")
    assert [r["code"] for r in monthly["rows"]] == [str(x.get("code")) for x in legs]
    for r in monthly["rows"]:
        assert r["shares"] > 0 and r["shares"] % 100 == 0, "必须是整手"
        assert r["turn20_pct"] is not None and r["turn20_pct"] < 0.01


def test_monthly_rotation_consistent_with_picks(monthly):
    if not PICKS.exists():
        pytest.skip("picks.json 不存在")
    picks = json.load(open(PICKS, encoding="utf-8"))
    prev = picks[-1]
    rot = monthly["rotation"]
    if rot.get("prev_month") == prev.get("month"):
        prev_codes = set(prev.get("codes") or [])
        cur = {r["code"] for r in monthly["rows"]}
        assert {x["code"] for x in rot["entered"]} == cur - prev_codes
        assert {x["code"] for x in rot["exited"]} == prev_codes - cur
        assert rot["kept"] == len(cur & prev_codes)
        # 换入换出都要给名字，前端直接展示
        for x in rot["entered"] + rot["exited"]:
            assert x["name"], f"{x['code']} 缺名称"


def test_monthly_perf_and_history(monthly):
    p = monthly["perf"]
    assert 0.05 < p["cagr"] < 0.20
    # 只换变动 > 每月全清全建（否则说明账本口径又坏了）
    assert p["cagr"] > p["strict_cagr"]
    assert p["gross_cagr"] > p["cagr"]
    assert p["cost_drag_pp"] == pytest.approx((p["gross_cagr"] - p["cagr"]) * 100, rel=1e-6)
    assert p["cost_bp"] == pytest.approx(49.27, abs=0.01)
    assert 0.5 < p["avg_overlap"] < 1.0
    assert len(monthly["monthly_returns"]) == 12
    assert len(monthly["nav_series"]) > 100
    assert monthly["nav_series"][0]["nav"] == pytest.approx(1.0, abs=1e-9)


def test_monthly_sector_dist_sums_to_one(monthly):
    dist = monthly["sector_dist"]
    assert dist, "行业分布不能为空"
    assert sum(x["n"] for x in dist) == monthly["n_pick"]
    assert sum(x["pct"] for x in dist) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- 今日推荐分解
def test_today_strategies_decomposition():
    if not COMBO.exists():
        pytest.skip("latest_combo.json 不存在")
    combo = json.load(open(COMBO, encoding="utf-8"))
    d = _svc().today_recommend()
    ss = d.get("strategies") or []
    keys = [s["key"] for s in ss]

    # 腿数不写死：实盘走 50/50 两腿还是带低换手/板块/转债，由 latest_combo.json 决定。
    # 界面分解必须与信号文件逐腿对齐——多一条或少一条都说明分解逻辑坏了。
    # 2026-11-05：改成直接读 weights 的 key（原来硬编码四腿名单，加 cb 后误报）。
    expected = [k for k in (combo.get("weights") or {})
                if combo.get(k) is not None]
    assert keys == expected, f"分解腿 {keys} 与信号文件 {expected} 不一致"
    assert "main" in keys

    total_w = sum((s.get("weight") or 0) for s in ss)
    assert total_w == pytest.approx(1.0, abs=1e-6)
    for s in ss:
        assert s["jump"], f"{s['key']} 缺跳转目标"
        assert s["budget"] > 0
    if "lowturn" in keys:
        low = next(s for s in ss if s["key"] == "lowturn")
        assert low["jump"] == "monthly", "低换手侧应跳到本月推荐页"
    # 主策略指标走 qf.metrics 口径，不要自算。注意 build/runs/main 的 nav 是
    # 「生产相位」那一档（2026-11 价值因子上线后 8.38%），不是 8 相位均值 7.01%。
    main = next(s for s in ss if s["key"] == "main")
    assert 0.080 <= main["cagr"] <= 0.087
    assert 0.63 <= main["sharpe"] <= 0.73
    # 单点 CAGR 会被误读成策略能力，所以相位带必须一起给
    pb = main.get("phase_band") or {}
    assert pb, "主策略缺 phase_band：只给生产相位单点会被当成策略能力"
    assert 0.065 <= pb["cagr_mean"] <= 0.075
    assert pb["n_phases"] == 8
