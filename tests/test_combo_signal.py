# -*- coding: utf-8 -*-
"""组合信号一致性测试（50/50 / 三策略 / 四策略，按 latest_combo.json 的 mode 适配）。"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SIG = ROOT / "signals"
COMBO = SIG / "latest_combo.json"
MAIN = SIG / "latest_signal.json"


@pytest.fixture(scope="module")
def combo():
    if not COMBO.exists():
        pytest.skip("latest_combo.json 不存在（先跑 daily_signal.py --combo）")
    return json.load(open(COMBO, encoding="utf-8"))


# mode → 主策略权重（2026-11-02 起 default 跑 --four）
MAIN_W = {"two": 0.5, "three": 0.5, "four": 0.4}


def test_combo_structure(combo):
    for k in ("signal_date", "capital", "weights", "main", "etf", "cash"):
        assert k in combo, f"缺少字段 {k}"
    w = combo["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-9, f"权重和不为 1: {w}"
    mode = combo.get("mode") or "two"
    assert mode in MAIN_W, f"未知 mode {mode}"
    assert abs(w["main"] - MAIN_W[mode]) < 1e-9, f"{mode} 模式下主策略权重应为 {MAIN_W[mode]:.0%}"
    if mode == "four":
        assert abs(w["lowturn"] - 0.30) < 1e-9, "四策略模式低换手权重应为 30%"
        assert "lowturn" in combo and "sector" in combo
    assert abs(combo["capital"] - 300000) < 1


def test_combo_money_conserved(combo):
    """各部分金额之和 == 本金（±2 元容差，整手取整）。

    2026-11-05：第四腿从 sector 换成 cb，这里改成遍历「权重里出现的所有腿」，
    以后再加腿不会再因为写死 key 而漏算（上次漏的是 29,320 元，差 9.8%）。
    """
    total = combo["cash"]["value"]
    for k in (combo.get("weights") or {}):
        seg = combo.get(k)
        if isinstance(seg, dict):
            total += seg.get("allocated") or 0.0
    assert abs(total - combo["capital"]) < 2.0, f"资金不守恒：{total} vs {combo['capital']}"


def test_combo_main_within_budget(combo):
    """主策略侧金额不得超过其预算，且应等于「预算 × 自身投入比例」。

    2026-11-05 修：此前 scale = main_alloc / base 把腿拉满预算，等于把 regime
    系数（破位 0.45）归一化掉了 —— 实盘 50% 股票 vs 回测 22.5%。
    """
    cap = combo["capital"]
    budget = cap * combo["main"]["budget_pct"]
    alloc = combo["main"]["allocated"]
    target = budget * combo["main"].get("invested_ratio", 1.0)
    assert alloc <= budget + 1, \
        f"主策略侧 {alloc:.0f} 超过预算 {budget:.0f}"
    assert alloc <= target + 1, \
        f"主策略侧 {alloc:.0f} 超过「预算×投入比例」{target:.0f}（regime 被抵消）"
    assert alloc > target * 0.6, \
        f"主策略侧低于目标的 60%（{alloc:.0f} vs {target:.0f}）"


def test_combo_main_keeps_regime(combo):
    """回归：主策略腿必须保留 regime 系数，不能被预算归一化掉。"""
    if not MAIN.exists():
        pytest.skip("latest_signal.json 不存在")
    sig = json.load(open(MAIN, encoding="utf-8"))
    inv = float(sig.get("invested_pct", 1.0))
    assert abs(combo["main"]["invested_ratio"] - inv) < 1e-9, \
        "combo.main.invested_ratio 与主策略信号自身的 invested_pct 不一致"
    got = combo["main"]["allocated"] / combo["capital"] / combo["weights"]["main"]
    assert abs(got - inv) < 0.02, \
        f"组合里主策略腿的投入比例 {got:.1%} ≠ 主策略自身 {inv:.1%}（regime 被抹掉）"


def test_combo_etf_leg_consistent(combo):
    """ETF 侧：有 leg 时金额>0，无 leg 时金额==0。"""
    leg = combo["etf"].get("leg")
    if leg:
        assert combo["etf"]["allocated"] > 0
        assert leg["shares"] % 100 == 0, "ETF 必须整手"
        assert abs(leg["shares"] * leg["price"] - leg["target_value"]) < 1.0
    else:
        assert combo["etf"]["allocated"] == 0


def test_combo_reports_exist(combo):
    d = combo["signal_date"]
    md = ROOT / "reports" / f"实盘组合_{d}.md"
    assert md.exists(), f"缺少组合报告 {md.name}"
    text = md.read_text(encoding="utf-8")
    assert "主策略" in text and "ETF" in text
    mode = combo.get("mode") or "two"
    if mode == "four":
        assert "低换手" in text and "40%" in text
    elif mode == "three":
        assert "三策略" in text and "50%" in text


def test_pipeline_runs_four_leg():
    """管道必须默认跑四腿（2026-11-05：此前 --combo 两腿，实盘与回测口径脱节）。"""
    src = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert '"scripts/daily_signal.py", "--four"' in src, \
        "run_pipeline 的 signal 阶段应传 --four"


def test_etf_leg_uses_half_budget_each():
    """ETF 腿每只固定占 1/2 预算：2 只突破各拿一半，只 1 只突破则半仓（与回测 equal 同口径）。"""
    sys.path.insert(0, str(ROOT))
    from scripts.daily_signal import _size_etf_leg

    picks = [{"code": "510300", "price": 4.0}, {"code": "510500", "price": 6.0}]
    legs, used = _size_etf_leg(picks, 60000)
    assert len(legs) == 2, "两只突破应各建一只"
    for x in legs:
        assert x["target_value"] <= 30000 + 1, "单只不得超过腿预算的一半"
    assert used > 59000, f"两只合计应接近腿预算，实得 {used:.0f}"

    legs1, used1 = _size_etf_leg(picks[:1], 60000)
    assert len(legs1) == 1
    assert used1 <= 30000 + 1, f"只有 1 只突破时应自动半仓，实得 {used1:.0f}"


def test_etf_leg_max_two():
    """最多同时持 2 只（第三只突破也不加仓）。"""
    sys.path.insert(0, str(ROOT))
    from scripts.daily_signal import _size_etf_leg

    picks = [{"code": c, "price": 4.0} for c in ("510300", "510500", "512100")]
    legs, _ = _size_etf_leg(picks, 60000)
    assert len(legs) == 2


def test_cb_leg_uses_10_sheet_lot():
    """转债腿整手必须是 10 张（不是 100 股），且不能因整手把腿投成半仓。"""
    sys.path.insert(0, str(ROOT))
    from scripts.daily_signal import _size_cb_leg

    picks = [{"code": f"11000{i}", "price": 110.0} for i in range(10)]
    legs, used = _size_cb_leg(picks, 30000)
    assert len(legs) == 10
    for x in legs:
        assert x["shares"] % 10 == 0, "张数必须是 10 的倍数"
        assert x["lots"] == x["shares"] // 10
    assert used <= 30000 + 1, f"转债腿不得超预算，实得 {used:.0f}"
    # 10% 腿 × 30 万 = 3 万，一手约 1,100 元 → 至少能投到 95%
    assert used > 30000 * 0.95, f"整手损耗过大，只投 {used / 30000:.1%}"


def test_cb_leg_present_when_weight_set(combo):
    """权重含 cb 时，latest_combo.json 必须有 cb 段且金额与预算匹配。"""
    if (combo.get("weights") or {}).get("cb", 0) <= 0:
        pytest.skip("当前信号不是转债腿口径")
    cb = combo.get("cb") or {}
    assert cb, "weights 含 cb 但组合里没有 cb 段"
    assert cb["allocated"] <= combo["capital"] * cb["budget_pct"] + 1


def test_pipeline_cb_signal_runs_before_signal():
    """转债信号必须在 signal 之前：signal 要读 latest_cb_signal.json 建腿。"""
    src = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    i_cb = src.index('"cb_signal"')
    i_sig = src.index('"scripts/daily_signal.py"')
    assert i_cb < i_sig, "cb_signal 阶段必须排在 signal 之前，否则一直用前一天的转债榜单"
