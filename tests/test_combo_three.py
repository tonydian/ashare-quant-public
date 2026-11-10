# -*- coding: utf-8 -*-
"""三策略组合实盘信号测试。"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
COMBO = ROOT / "signals" / "latest_combo.json"


@pytest.fixture(scope="module")
def combo():
    if not COMBO.exists():
        pytest.skip("latest_combo.json 不存在（先跑 daily_signal.py --three）")
    return json.load(open(COMBO, encoding="utf-8"))


def test_three_mode_structure(combo):
    if combo.get("mode") != "three":
        pytest.skip("当前不是三策略模式（可能是 50/50）")
    for k in ("main", "etf", "sector", "cash", "weights", "parts"):
        assert k in combo, f"缺少字段 {k}"
    assert combo["weights"] == {"main": 0.5, "etf": 0.3, "sector": 0.2}
    assert "50%" in combo["parts"] and "30%" in combo["parts"] and "20%" in combo["parts"]


def test_money_conserved_three(combo):
    """各部分金额之和 == 本金（±2 元，整手取整）；四策略模式要算上 lowturn。"""
    if not combo.get("sector"):
        pytest.skip("当前是两策略模式，无 sector 部分")
    total = (combo["main"]["allocated"] + combo["etf"]["allocated"]
             + combo["sector"]["allocated"] + combo["cash"]["value"]
             + ((combo.get("lowturn") or {}).get("allocated") or 0))
    assert abs(total - combo["capital"]) < 2.0, f"资金不守恒: {total} vs {combo['capital']}"


def test_budget_respected(combo):
    """各部分实际金额不得超过其预算（整手可能略低，但不得超）。"""
    if not combo.get("sector"):
        pytest.skip("当前是两策略模式")
    cap = combo["capital"]
    for k in ("main", "etf", "sector"):
        seg = combo.get(k) or {}
        if not seg:
            continue
        assert seg["allocated"] <= cap * seg["budget_pct"] + 1, \
            f"{k} 侧 {seg['allocated']:.0f} 超过预算 {cap * seg['budget_pct']:.0f}"
    low = combo.get("lowturn") or {}
    if low:
        assert low["allocated"] <= cap * low["budget_pct"] + 1, \
            f"lowturn 侧 {low['allocated']:.0f} 超过预算 {cap * low['budget_pct']:.0f}"


def test_sector_legs_valid(combo):
    """板块轮动侧：有 leg 时必须整手、金额自洽、板块不重复。"""
    sec = combo.get("sector") or {}
    legs = sec.get("legs") or []
    assert len(legs) <= 3, "板块轮动最多持 3 个板块"
    sectors = [x["sector"] for x in legs]
    assert len(sectors) == len(set(sectors)), f"板块重复: {sectors}"
    for x in legs:
        assert x["shares"] % 100 == 0, "必须整手"
        assert abs(x["shares"] * x["price"] - x["target_value"]) < 1.0


def test_etf_leg_valid(combo):
    etf = combo.get("etf") or {}
    leg = etf.get("leg")
    if leg:
        assert leg["shares"] % 100 == 0
        assert abs(leg["shares"] * leg["price"] - leg["target_value"]) < 1.0
        assert etf["allocated"] > 0
    else:
        assert etf["allocated"] == 0


def test_report_exists_and_matches(combo):
    d = combo["signal_date"]
    md = ROOT / "reports" / f"实盘组合_{d}.md"
    assert md.exists(), f"缺少报告 {md.name}"
    text = md.read_text(encoding="utf-8")
    if combo.get("mode") == "three":
        assert "三策略" in text
        assert "板块 ETF 轮动" in text
    assert "主策略" in text
