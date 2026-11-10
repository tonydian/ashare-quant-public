# -*- coding: utf-8 -*-
"""相位带（phase_band）+ 换手族接入生产 + 绩效披露口径的回归测试。

背景（2026-11-03）：主策略 build/runs/main 的 nav 由
``rebalance_offset = (phase - start_row) % rb`` 决定，等价于「拿到分数就立刻建仓」。
实测这一档恰好是 8 个对齐相位里 CAGR 最高的一个（7.21% vs 中位 4.54%）。
因此绩效披露必须附相位分布，不能让单点数字冒充策略能力。
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BAND = ROOT / "build" / "eval" / "phase_band.json"
FINAL = ROOT / "build" / "eval" / "final_numbers.json"
PY = sys.executable


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(ROOT / "scripts" / "phase_band.py"), *args],
                          cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)


# ---------------------------------------------------------------- 相位约束
def test_phase_band_rejects_non_multiple_of_5_rb():
    """rb 不是 5 的倍数 → 调仓网格会逐渐偏离周频打分行，必须直接拒绝。"""
    r = _run("--rebalance-days", "42")
    assert r.returncode != 0
    assert "5 的倍数" in (r.stdout + r.stderr)


def test_phase_band_rejects_step_not_aligned():
    """step 不是 5 的倍数 → offset%5!=0 的相位会退化成全现金且不报错，必须拒绝。"""
    r = _run("--step", "3")
    assert r.returncode != 0
    assert "全现金" in (r.stdout + r.stderr)


def test_summary_stats():
    from importlib import import_module

    sys.path.insert(0, str(ROOT / "scripts"))
    mod = import_module("phase_band")
    s = mod._summary(np.array([0.03, 0.05, 0.07, np.nan]))
    assert s["n"] == 3, "NaN 必须被剔除"
    assert s["min"] == pytest.approx(0.03)
    assert s["max"] == pytest.approx(0.07)
    assert s["median"] == pytest.approx(0.05)
    assert mod._summary(np.array([]))["n"] == 0


# ---------------------------------------------------------------- 产出 schema
@pytest.mark.skipif(not BAND.exists(), reason="尚未生成 phase_band.json")
def test_phase_band_schema_and_invariants():
    b = json.load(open(BAND, encoding="utf-8"))
    for k in ("model", "start", "end", "rebalance_days", "prod_offset",
              "n_phases", "cagr", "sharpe", "max_dd", "prod", "rows"):
        assert k in b, f"缺字段 {k}"

    rb = b["rebalance_days"]
    assert rb % 5 == 0
    offsets = [r["offset"] for r in b["rows"]]
    assert len(offsets) == b["n_phases"]
    assert all(o % 5 == 0 for o in offsets), "所有相位都必须与周频行对齐"
    assert b["prod_offset"] in offsets, "生产相位必须在跑过的相位集合里"
    assert len(set(offsets)) == len(offsets), "相位不能重复"

    c = b["cagr"]
    assert c["min"] <= c["median"] <= c["max"]
    assert c["min"] <= c["mean"] <= c["max"]
    # 生产那一档必须和明细里对得上（防止两处口径漂移）
    row = next(r for r in b["rows"] if r["offset"] == b["prod_offset"])
    assert b["prod"]["cagr"] == pytest.approx(row["cagr"], abs=1e-12)
    # pct_better 的定义：比生产更好的相位占比
    better = sum(1 for r in b["rows"] if r["cagr"] > b["prod"]["cagr"] + 1e-12)
    assert b["prod"]["n_better"] == better
    assert b["prod"]["pct_better"] == pytest.approx(better / max(b["n_phases"] - 1, 1))


# ---------------------------------------------------------------- 披露链路
@pytest.mark.skipif(not (BAND.exists() and FINAL.exists()), reason="尚未刷新基准数字")
def test_final_numbers_carries_phase_band():
    fn = json.load(open(FINAL, encoding="utf-8"))
    pb = (fn.get("main") or {}).get("phase_band")
    assert pb, "final_numbers.json 的 main 必须带 phase_band（否则报告只能引用单点）"
    for k in ("cagr_mean", "cagr_median", "cagr_min", "cagr_max",
              "n_phases", "prod_offset", "n_better"):
        assert k in pb, f"phase_band 缺 {k}"
    assert pb["cagr_min"] <= pb["cagr_median"] <= pb["cagr_max"]
    # 单点披露（main.cagr）不应超出相位区间
    one = (fn.get("main") or {}).get("cagr")
    if one is not None:
        assert pb["cagr_min"] - 0.01 <= one <= pb["cagr_max"] + 0.01


def test_final_numbers_config_rebalance_days_is_40():
    """config.rebalance_days 曾写成 10（与 run_backtest 默认 40 不符），会误导仓位推算。"""
    if not FINAL.exists():
        pytest.skip("无 final_numbers.json")
    fn = json.load(open(FINAL, encoding="utf-8"))
    assert (fn.get("config") or {}).get("rebalance_days") == 40


# ---------------------------------------------------------------- 管道接线
def test_pipeline_model_stage_includes_turnover_family():
    """换手族必须真的进生产候选池；只写在实验脚本里等于没接。"""
    src = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "newf_turnover_extra.npz" in src, "model 阶段没有 --extra 换手族"
    # 阶段必须带 --extra 参数对
    i = src.index("newf_turnover_extra.npz")
    seg = src[max(0, i - 400):i]
    assert '"--extra"' in seg, "--extra 参数名缺失"


def test_pipeline_has_phase_band_stage():
    from importlib import import_module

    sys.path.insert(0, str(ROOT / "scripts"))
    mod = import_module("run_pipeline")
    keys = [s[0] for s in mod.STAGES]
    assert "phase_band" in keys, "管道缺相位带阶段 → 绩效披露无法自动更新"
    assert keys.index("phase_band") > keys.index("backtest")
    assert "phase_band" in mod.STAGE_TIMEOUT


def test_research_extra_is_repeatable():
    """--extra 必须支持重复：单次只能传一个时，换手族会把 momentum 族挤掉。"""
    src = (ROOT / "scripts" / "run_research.py").read_text(encoding="utf-8")
    i = src.index('"--extra"')
    seg = src[i:i + 200]
    assert 'action="append"' in seg, "--extra 必须 action=append"


@pytest.mark.skipif(not BAND.exists(), reason="尚未生成 phase_band.json")
def test_backtest_report_s1_discloses_phase_band():
    """回测报告 §1 是主报告的门面，必须带相位区间，否则单点会继续被当能力。"""
    rp = ROOT / "reports" / "回测报告.md"
    if not rp.exists():
        pytest.skip("无回测报告")
    txt = rp.read_text(encoding="utf-8")
    i = txt.find("## 1. 摘要")
    j = txt.find("## 2. 策略逻辑")
    assert i >= 0 and j > i
    sec = txt[i:j]
    assert "调仓相位敏感度" in sec, "§1 缺相位带说明（请重跑 refresh_report_s1.py）"
    assert "中位" in sec and "铺到" in sec
