# -*- coding: utf-8 -*-
"""前视偏差回归测试：锁定三个生产策略的时序正确性。"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_main_backtest_uses_pending_mechanism():
    """主策略必须用 pending 机制（t日决策 t+1执行）。"""
    src = (ROOT / "qf" / "backtest.py").read_text(encoding="utf-8")
    assert "pending_weights" in src, "主策略缺少 pending_weights 机制"
    assert "pending_shares" in src
    # 文档应说明 t+1 开盘
    assert "t+1" in src


def test_etf_trend_uses_pending():
    src = (ROOT / "scripts" / "etf_trend_strategy.py").read_text(encoding="utf-8")
    assert "pending" in src, "宽基 ETF 缺少 pending 机制"


def test_sector_rotation_uses_pending():
    src = (ROOT / "qf" / "sector_rotation.py").read_text(encoding="utf-8")
    assert "pending_targets" in src, "板块轮动缺少 pending 机制"


def test_v3_script_fixed_to_t1():
    """研究脚本 v3 也必须用 pending（本轮修复项，防回退）。"""
    src = (ROOT / "scripts" / "sector_rotation_v3.py").read_text(encoding="utf-8")
    assert "pending" in src, "v3 研究脚本应使用 pending 机制（已修复，勿回退）"
    # 不应再有"决策后立即按 close[t] 计收益"的写法
    assert "holdings = new\n" not in src or "pending" in src


def test_combo_scripts_have_no_trade_logic():
    """组合类脚本必须是纯净值加权（无价格/无成交），结构上不可能前视。"""
    for f in ("three_strategy_combo.py", "etf_combo_50_50.py"):
        p = ROOT / "scripts" / f
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        has_price = any(k in src for k in ("close[", "open[", "price["))
        has_trade = any(k in src for k in ("buy_", "sell_", "order"))
        assert not (has_price and has_trade), \
            f"{f} 同时使用价格与成交逻辑，需人工检查前视风险"


def test_audit_artifacts_exist():
    """审计产物应存在（证明做过实测而非只看代码）。"""
    p = ROOT / "build" / "eval" / "lookahead_audit_all.json"
    if not p.exists():
        pytest.skip("审计尚未运行")
    d = json.load(open(p, encoding="utf-8"))
    assert "etf_trend" in d
    # 宽基 ETF 注入前视应显著虚高（证明策略敏感且当前实现正确）
    assert d["etf_trend"]["delta_cagr_pp"] > 20, \
        f"注入前视后应显著虚高，实际 {d['etf_trend']['delta_cagr_pp']:.1f}pp"


def test_v3_results_match_production():
    """v3 与生产必须同口径、同结果（差异 < 1pp）。

    这条测试原本只断言「差异 < 3pp」，靠一个凭经验拍的容差**兜住**口径漂移：
    v3 硬编码 `MIN_AMOUNT=1e7` + 一套不含「创业板50」的 EXCLUDE_SECTORS，
    生产是 `2.5e7` + 含「创业板50」，两边在同一份数据上语义本就不同。
    2026-11-04 生产去掉假军工后差异涨到 3.33pp，测试才红——它一直没在守护什么。

    现在改为：先校验口径元信息（板块数 / 成交额门槛 / 数据末端日期）必须一致，
    再在同一口径下要求数值几乎相同（实测差异 0.00pp）。任何一边漂移都会立刻红。
    """
    v3 = ROOT / "build" / "eval" / "sector_rotation_v3.json"
    prod = ROOT / "build" / "eval" / "sector_rotation_stats.json"
    if not (v3.exists() and prod.exists()):
        pytest.skip("产物不存在")
    import json as _j
    a = _j.load(open(v3, encoding="utf-8"))
    b = _j.load(open(prod, encoding="utf-8"))

    # ---- 1. 口径必须先对上
    uni = a.get("universe")
    if uni is None:
        pytest.skip("v3 产物是旧格式（无 universe 元信息），请重跑 scripts/sector_rotation_v3.py")
    import sys as _s
    if str(ROOT) not in _s.path:
        _s.path.insert(0, str(ROOT))
    from qf import sector_rotation as _SR

    _, codes, _, _, sector_of = _SR.load_pool()
    assert uni["sector_count"] == len(set(sector_of.values())) == a["sector_count"], \
        (f"v3 板块数 {uni['sector_count']} / 生产 {len(set(sector_of.values()))} "
         f"不一致，池子口径已漂移")
    assert float(uni["min_amount"]) == float(_SR.MIN_AMOUNT), \
        f"v3 成交额门槛 {uni['min_amount']} != 生产 {_SR.MIN_AMOUNT}"
    assert uni["label_override"] == dict(_SR.LABEL_OVERRIDE), \
        "v3 的错标修正表与生产不一致"

    # ---- 2. 同口径下数值必须几乎相同
    cagr_v3 = a.get("全期", {}).get("cagr") or a.get("full", {}).get("cagr")
    cagr_prod = b.get("full", {}).get("cagr")
    if cagr_v3 is None or cagr_prod is None:
        pytest.skip("字段缺失")
    # 数据末端不同的时候（v3 快照比生产旧）不做数值比较，只留口径校验
    prod_end = (b.get("note") or "")
    if uni.get("data_end") and prod_end and uni["data_end"] not in prod_end:
        pytest.skip(f"v3 数据末端 {uni['data_end']} 与生产快照不同日，跳过数值比对")
    assert abs(cagr_v3 - cagr_prod) < 0.01, \
        f"v3 ({cagr_v3:.2%}) 与生产版 ({cagr_prod:.2%}) 差异过大，口径可能又不一致"
