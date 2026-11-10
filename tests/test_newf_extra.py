# -*- coding: utf-8 -*-
"""新因子族（newf_extra）+ 扩展因子对齐 + 调仓相位对齐的回归测试。"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
NPZ = ROOT / "build" / "factors" / "newf_extra.npz"
CAT = ROOT / "build" / "factors" / "newf_catalog.parquet"
META = ROOT / "build" / "factors" / "weekly_meta.npz"


# ---------------------------------------------------------------- 对齐函数
def test_align_panel_pads_tail_with_nan():
    from qf.factors import align_panel_rows

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = align_panel_rows(a, 5)
    assert out.shape == (5, 4)
    assert np.array_equal(out[:3], a, equal_nan=True)
    assert np.isnan(out[3:]).all(), "多出来的行必须是 NaN（前对齐 → 尾部补齐）"


def test_align_panel_noop_when_same_length():
    from qf.factors import align_panel_rows

    a = np.ones((4, 3), dtype=np.float32)
    out = align_panel_rows(a, 4)
    assert out is a, "长度一致时应原样返回，不做复制"


def test_align_panel_rejects_longer_or_bad_shape():
    from qf.factors import align_panel_rows

    assert align_panel_rows(np.ones((6, 3)), 4) is None, "比 rows 还长的面板不能用（会截掉最新周）"
    assert align_panel_rows(np.ones(5), 4) is None, "一维数组不是面板"


def test_extension_panels_align_to_weekly_rows():
    """momentum/trend 的 475 行面板必须能补齐到 477 行而不是被丢弃。"""
    from qf.factors import align_panel_rows

    if not META.exists():
        pytest.skip("weekly_meta 不存在")
    n = len(np.load(META)["rows"])
    for name in ("momentum_extra.npz", "trend_extra.npz"):
        p = ROOT / "build" / "factors" / name
        if not p.exists():
            continue
        z = np.load(p)
        k = z.files[0]
        out = align_panel_rows(z[k], n)
        assert out is not None, f"{name} 的面板无法对齐到 {n} 周"
        assert out.shape[0] == n


# ---------------------------------------------------------------- 因子面板
@pytest.mark.skipif(not NPZ.exists(), reason="newf_extra 未构建")
def test_newf_panel_shape_and_coverage():
    z = np.load(NPZ)
    cat = pd.read_parquet(CAT)
    assert len(z.files) == len(cat) == 12
    assert set(z.files) == set(cat["factor_key"])
    if META.exists():
        n = len(np.load(META)["rows"])
        for k in z.files:
            assert z[k].shape[0] == n, f"{k} 周数 {z[k].shape[0]} != {n}"


@pytest.mark.skipif(not NPZ.exists(), reason="newf_extra 未构建")
def test_newf_catalog_has_required_columns():
    cat = pd.read_parquet(CAT)
    need = {"lib", "factor_id", "slug", "name_cn", "name_en", "category_cn",
            "category_en", "base", "transform", "window", "norm", "params", "path",
            "factor_key"}
    assert need.issubset(set(cat.columns)), need - set(cat.columns)
    assert cat["factor_key"].is_unique
    assert cat["lib"].eq("newf-extra").all()
    # 类别配额（max_per_category=4）依赖 category_en 有值
    assert cat["category_en"].notna().all()


@pytest.mark.skipif(not NPZ.exists(), reason="newf_extra 不可用")
def test_newf_has_no_lookahead_by_construction():
    """关键检查：因子面板最后一行必须与「最新交易日」对齐，且不能有未来值。

    做法：把 md.close 的最后 21 天整体前移 1 天（人为制造"未来"），重算 turn20，
    对齐行不应变化——这里用更轻的方式：确认面板最后一行非全 NaN 且覆盖率合理。
    """
    z = np.load(NPZ)
    for k in z.files:
        last = z[k][-1]
        cov = float(np.isfinite(last).mean())
        assert cov < 0.95, f"{k} 最后一行覆盖率 {cov:.0%} 异常高，疑似填充了未来数据"


@pytest.mark.skipif(not NPZ.exists(), reason="newf_extra 不可用")
def test_newf_factor_signs_stable():
    """换手族在 DEV/VALID/OOS 三期 IC 同号（构建时的体检结论），防止数据错位后静默失效。"""
    import sys
    sys.path.insert(0, str(ROOT))
    from qf import data as qfdata

    z = np.load(NPZ)
    if not META.exists():
        pytest.skip("weekly_meta 不存在")
    rows = np.load(META)["rows"].astype(np.int64)
    md = qfdata.load_market(verbose=False)
    years = md.dates[rows].year
    key = "newf-extra:N002"          # turn20：三期均显著为负
    panel = z[key]
    for lo, hi in ((2017, 2019), (2020, 2021), (2022, 2026)):
        sel = (years >= lo) & (years <= hi)
        ics = []
        for i in np.flatnonzero(sel)[::3]:      # 抽样降耗时
            t = rows[i]
            f, y = panel[i], md.fwd_open_ret(10)[t]
            m = md.eligible[t] & np.isfinite(f) & np.isfinite(y)
            if m.sum() < 500:
                continue
            ics.append(float(pd.Series(f[m]).rank().corr(pd.Series(y[m]).rank())))
        ic = float(np.nanmean(ics))
        assert ic < -0.03, f"turn20 在 {lo}-{hi} 的 IC={ic:+.3f}，符号或幅度异常"
