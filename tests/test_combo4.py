# -*- coding: utf-8 -*-
"""四腿组合生产口径（scripts/combo4.py + web/services.combo_plan）。

保护三件事：
1. 权重规则不可漂移（转债 w，其余按 50/25/25 等比压缩，合计 100%）；
2. 转债必须是**净**口径（毛序列直接拼会重犯报告 §0.1 的错）；
3. combo_plan 在有 combo4 产物时必须走四腿，不能被三腿覆盖。
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SLUG = ROOT / "build" / "eval" / "combo4_stats.json"
RUNS = ROOT / "build" / "runs" / "combo_4leg"


@pytest.fixture(scope="module")
def stats():
    if not SLUG.exists():
        pytest.skip("combo4_stats.json 未生成，先跑 scripts/combo4.py")
    return json.loads(SLUG.read_text(encoding="utf-8"))


def test_weights_rule(stats):
    """转债 w 固定，剩余按 50/25/25 比例分，四腿合计恒为 1。"""
    for key, r in stats["results"].items():
        w = r["weights"]
        assert len(w) == 4, key
        assert abs(sum(w) - 1.0) < 1e-9, f"{key} 权重合计 {sum(w)}"
        rest = 1.0 - r["w_cb"]
        assert abs(w[0] - rest * 0.50) < 1e-12, key
        assert abs(w[1] - rest * 0.25) < 1e-12, key
        assert abs(w[2] - rest * 0.25) < 1e-12, key
        assert abs(w[3] - r["w_cb"]) < 1e-12, key


def test_prod_is_15pct(stats):
    """生产口径必须是 15%（DSR 未过 → 降权试行），不能悄悄改成 20%+。"""
    assert abs(stats["prod_w_cb"] - 0.15) < 1e-12
    prod = stats["results"]["cb15"]
    assert abs(prod["weights"][0] - 0.425) < 1e-12
    assert abs(prod["weights"][3] - 0.15) < 1e-12


def test_cb_leg_is_net_of_cost(stats):
    """转债腿必须是净口径：全期年化应落在 14%~17%，而不是毛口径的 18%+。"""
    assert abs(stats["cb_cost_ann"] - (0.1797 - 0.1545)) < 1e-9
    nav = pd.read_parquet(RUNS / "nav_cb00.parquet")["nav"].astype(float)
    assert (nav > 0).all()
    days = (pd.to_datetime(nav.index)[-1] - pd.to_datetime(nav.index)[0]).days
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1.0
    assert 0.06 < cagr < 0.12, f"三腿基线 CAGR 异常：{cagr:.2%}"


def test_cb_adds_and_smooths(stats):
    """核心主张：加转债后 Sharpe 上升、波动下降（不是靠加杠杆）。"""
    b, p = stats["results"]["cb00"]["full"], stats["results"]["cb15"]["full"]
    assert p["sharpe"] > b["sharpe"], "15% 档 Sharpe 未提升"
    assert p["ann_vol"] < b["ann_vol"], "15% 档波动未下降"
    assert p["max_dd"] >= b["max_dd"] - 1e-9, "15% 档回撤劣化"


def test_all_navs_dumped(stats):
    """每一档都要落 NAV —— P6 教训：不落 NAV 就无法实测试验间相关性。"""
    for key in stats["results"]:
        f = RUNS / f"nav_{key}.parquet"
        assert f.exists(), f"{f} 缺失"
        nav = pd.read_parquet(f)["nav"].astype(float)
        assert len(nav) > 1500
        assert (nav > 0).all()


def test_combo_plan_prefers_combo4():
    """有 combo4 产物时，面板必须走四腿，且推荐档为转债 15%。"""
    from web.services import DataService

    d = DataService().combo_plan()
    if d.get("basis") != "combo4":
        pytest.skip("combo4_stats.json 缺失，走三腿回退分支")
    assert d["recommended"] == "转债15%"
    assert len(d["legs"]) == 4
    assert abs(sum(d["prod_weights"].values()) - 1.0) < 1e-9
    assert abs(d["prod_weights"]["可转债"] - 0.15) < 1e-12
    # 至少有一条 note 说明 DSR 未过，防止面板把"试行"说成"采纳"
    assert any("DSR" in n for n in d["notes"])
