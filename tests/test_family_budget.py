# -*- coding: utf-8 -*-
"""族预算（family budget）的约束测试。

背景：方案 C =「按族分配权重预算」，族间等分 1/F、族内保留原权重比例。
动机是防止一族因子靠数量优势吃掉大半权重（risk-pattern 在 21 池里占 35%）。

这个功能默认关闭，开启后必须满足三条不变量：
  1. 每个非空族的 Σ|w| 严格等于 1/F（族预算的核心承诺）
  2. 族内各因子的相对比例（含正负号）保持不变
  3. 权重全被地板清零的族要跳过，剩余族重新归一到 Σ|w|=1
     （否则 Σ|w| < 1，复合分的量纲会跟着变）
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import model as qfm  # noqa: E402


def test_family_shares_equal():
    """三个族、槽数不等（1/2/3 个因子），族间必须等预算。"""
    w = np.array([0.50, 0.20, 0.10, 0.08, 0.06, 0.04, 0.02])
    fam = ["fund"] + ["rp"] * 2 + ["mom"] * 4
    out = qfm.apply_family_budget(w, fam)
    assert abs(np.abs(out).sum() - 1.0) < 1e-9
    for f in ("fund", "rp", "mom"):
        idx = [i for i, x in enumerate(fam) if x == f]
        assert abs(np.abs(out[idx]).sum() - 1.0 / 3) < 1e-9, f


def test_within_family_ratio_kept():
    """族内比例与符号不变：只有族间预算被改写。"""
    w = np.array([0.30, -0.20, 0.10, 0.25, 0.15])
    fam = ["a", "a", "a", "b", "b"]
    out = qfm.apply_family_budget(w, fam)
    # a 族内 30:-20:10 的比例必须保留，b 族内 25:15 同理
    assert abs(out[0] / out[1] + 1.5) < 1e-9
    assert abs(out[0] / out[2] - 3.0) < 1e-9
    assert abs(out[3] / out[4] - 25 / 15) < 1e-9
    assert out[1] < 0 and out[2] > 0


def test_zero_family_skipped_and_renormalized():
    """某族权重全零时必须跳过，剩余族重新归一（不能留下 Σ|w|<1）。"""
    w = np.array([0.0, 0.0, 0.40, 0.35, 0.25])
    fam = ["dead", "dead", "a", "b", "b"]
    out = qfm.apply_family_budget(w, fam)
    assert np.abs(out[:2]).sum() == 0.0
    assert abs(np.abs(out).sum() - 1.0) < 1e-9, "跳过空族后必须重新归一到 1"
    assert abs(np.abs(out[2:]).sum() - 1.0) < 1e-9


def test_strength_zero_is_noop():
    """强度 0 = 不干预，必须逐元素相等。"""
    w = np.array([0.5, 0.3, 0.2])
    fam = ["a", "b", "b"]
    out = qfm.apply_family_budget(w, fam, strength=0.0)
    assert np.allclose(out, w)


def test_single_family_is_noop():
    """只有一族时族预算无意义，不能把权重打断。"""
    w = np.array([0.5, 0.3, 0.2])
    out = qfm.apply_family_budget(w, ["only", "only", "only"])
    assert np.allclose(out, w)


def test_config_default_off():
    """默认必须关闭：生产行为不能被新功能改变。"""
    cfg = qfm.ModelConfig()
    assert cfg.family_budget is False
    assert cfg.family_strength == 1.0


@pytest.mark.skipif(not (ROOT / "build" / "models" / "wf_dimC" / "model_weights.parquet").exists(),
                    reason="方案 C 实验产物不存在")
def test_experiment_artifact_obeys_budget():
    """验产物：方案 C 模型每期每族 |w| 必须等于 1/6（21 池共 6 族）。"""
    import pandas as pd

    w = pd.read_parquet(ROOT / "build" / "models" / "wf_dimC" / "model_weights.parquet")
    w["fam"] = w["factor_key"].str.split(":").str[0]
    per = w.groupby(["refit_date", "fam"])["weight"].apply(lambda s: s.abs().sum()).unstack()
    n_fam = per.shape[1]
    assert n_fam == 6, f"21 池应有 6 族，实际 {n_fam}"
    assert np.allclose(per.fillna(0).to_numpy(), 1.0 / n_fam, atol=1e-6), per.head()

    meta = json.loads((ROOT / "build" / "models" / "wf_dimC" / "model_meta.json").read_text(encoding="utf-8"))
    assert len(meta["pool"]) == 21
