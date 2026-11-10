# -*- coding: utf-8 -*-
"""族上限（apply_family_cap）的行为测试。

设计要点（区别于族预算 apply_family_budget）：
- 未越界 → 一根手指都不动（这是与族预算最重要的行为差异）
- 越界 → 只削越界族，让出的预算按未越界族原占比分配，族内相对比例保留
- cap 不可行（cap×有效族数 < 1）→ 不干预，绝不能 silently 违反上限
- cap<=0 → 关闭
"""
import numpy as np
import pytest

from qf.model import apply_family_cap


def fam_sums(w, fams):
    return {
        f: float(np.abs(w[[i for i, x in enumerate(fams) if x == f]]).sum())
        for f in sorted(set(fams))
    }


def test_no_exceed_returns_input_unchanged():
    w = np.array([0.30, 0.10, 0.10, 0.20, 0.20, 0.10])
    fams = ["a", "a", "a", "b", "b", "b"]
    out = apply_family_cap(w, fams, cap=0.60)  # 最大族 0.50 < 0.60
    assert np.allclose(out, w)


def test_exceeding_family_capped_others_scale_up():
    w = np.array([0.50, 0.30, 0.20])            # 族 a=0.50 b=0.30 c=0.20
    fams = ["a", "b", "c"]
    out = apply_family_cap(w, fams, cap=0.40)
    fs = fam_sums(out, fams)
    assert fs["a"] == pytest.approx(0.40, abs=1e-9)
    assert abs(out).sum() == pytest.approx(1.0, abs=1e-9)
    # 让出的 0.10 按 b:c = 3:2 分配
    assert fs["b"] == pytest.approx(0.36, abs=1e-9)
    assert fs["c"] == pytest.approx(0.24, abs=1e-9)
    # 族内相对比例保留（每族只有 1 个因子，检查族间即可）


def test_within_family_proportions_preserved():
    w = np.array([0.30, 0.10, 0.10, 0.18, 0.18, 0.09, 0.05])  # a=0.50(6:2:2) b=0.45 c=0.05
    fams = ["a", "a", "a", "b", "b", "b", "c"]
    out = apply_family_cap(w, fams, cap=0.45)
    ra = out[:3] / out[:3].sum()
    rb = out[3:6] / out[3:6].sum()
    assert np.allclose(ra, [0.6, 0.2, 0.2], atol=1e-9)
    assert np.allclose(rb, [0.4, 0.4, 0.2], atol=1e-9)
    fs = fam_sums(out, fams)
    assert fs["a"] == pytest.approx(0.45, abs=1e-9)


def test_water_filling_caps_second_family():
    # 削 a 后按原占比分配会把 b 顶过线，需要第二次迭代把 b 也钉住
    w = np.array([0.60, 0.35, 0.05])   # a=0.60 b=0.35 c=0.05，cap=0.40
    fams = ["a", "b", "c"]
    out = apply_family_cap(w, fams, cap=0.40)
    fs = fam_sums(out, fams)
    # rem=0.60 按 b:c=7:1 -> b=0.525 越界 -> 钉 0.40，c 拿剩余 0.20
    assert fs["a"] == pytest.approx(0.40, abs=1e-9)
    assert fs["b"] == pytest.approx(0.40, abs=1e-9)
    assert fs["c"] == pytest.approx(0.20, abs=1e-9)
    assert abs(out).sum() == pytest.approx(1.0, abs=1e-9)


def test_infeasible_cap_does_nothing():
    # 2 个族 + cap=0.40：0.4*2 < 1.0，任何分配都违反上限 -> 不干预
    # （回归防线：曾误用族等分兜底，会给出各 0.5 反而破上限）
    w = np.array([0.5, 0.5])
    out = apply_family_cap(w, ["a", "b"], cap=0.40)
    assert np.allclose(out, w)


def test_infeasible_with_zero_family_counts_active_only():
    w = np.array([0.5, 0.0, 0.5])
    out = apply_family_cap(w, ["a", "b", "c"], cap=0.40)  # 有效族=2
    assert np.allclose(out, w)


def test_cap_zero_or_single_family_disabled():
    w = np.array([0.6, 0.4])
    assert np.allclose(apply_family_cap(w, ["a", "b"], cap=0.0), w)
    assert np.allclose(apply_family_cap(w, ["a", "a"], cap=0.25), w)


def test_negative_weights_use_absolute_mass():
    # 空头权重按 |w| 计入族质量
    w = np.array([0.60, -0.20, 0.20])   # a=0.60(按|w|) b=0.20 c=0.20
    fams = ["a", "a", "b"]
    out = apply_family_cap(w, fams, cap=0.50)
    assert fam_sums(out, fams)["a"] == pytest.approx(0.50, abs=1e-9)
    assert abs(out).sum() == pytest.approx(1.0, abs=1e-9)
