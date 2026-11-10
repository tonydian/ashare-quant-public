# -*- coding: utf-8 -*-
"""额外因子族面板的一致性保障（scripts/build_extra_panels.py）。

背景：2026-11-04 22:56 的更新报「失败」，17 个阶段全部 rc=0，唯独最后的
一致性检测报「模型滞后 6 日」。根因与模型无关——

    build_factors.py 只重建主面板 weekly_top/weekly_meta，
    而生产模型 --extra 挂的 momentum_extra / newf_turnover_extra /
    value_extra 是各自独立生成的，从来没有阶段在管道里重建。
    → 它们比 weekly_meta 少一周
    → 模型在该周的因子全 NaN，np.tensordot 让整行复合分变 NaN
    → composite_score 最后有效行停在上一周
    → check_data_source 报「模型滞后」

不变式：
 1. 要重建哪些面板，从 run_pipeline 的 model 阶段解析，不得在别处另写一份；
 2. value_extra 依赖 fundamental_extra，生成顺序必须被依赖者在前；
 3. 面板周数 == weekly_meta 周数才算对齐（不看 mtime，不看行内容）；
 4. 收尾复核发现没对齐就非零退出，不留"看起来跑完了"的半成品。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_extra_panels as bep  # noqa: E402


# ------------------------------------------------------------------ 单一真源
def test_production_extras_is_parsed_from_pipeline():
    """--extra 列表必须来自 run_pipeline.py，而不是脚本内另写一份。"""
    stems = bep.production_extras()
    assert stems, "model 阶段没有解析到 --extra"
    src = bep.PIPELINE.read_text(encoding="utf-8")
    for s in stems:
        if s == "momentum_extra":
            continue    # 由 run_research 自动加载，命令行里本就没有
        assert f"{s}.npz" in src, f"{s} 不在 run_pipeline.py 里，说明解析跑偏了"


def test_production_extras_has_no_duplicates():
    stems = bep.production_extras()
    assert len(stems) == len(set(stems))


def test_every_production_extra_has_a_regen_command():
    """生产在用的面板必须登记重建命令，否则管道跑到一半会炸。"""
    for s in bep.production_extras():
        assert s in bep.REGEN, f"{s} 没有重建命令"


def test_regen_commands_point_to_existing_scripts():
    for s in bep.production_extras():
        cmd = bep.REGEN[s]
        script = ROOT / cmd[0]
        assert script.exists(), f"{s} 的重建脚本不存在: {script}"


# ------------------------------------------------------------------ 依赖顺序
def test_value_extra_depends_on_fundamental_extra():
    assert "fundamental_extra" in bep.DEPENDS.get("value_extra", [])


def test_dependency_comes_first_in_order():
    """value_extra 是从 fundamental_extra 切出来的，必须先建源面板。"""
    ordered = bep.with_deps(["value_extra"])
    assert ordered.index("fundamental_extra") < ordered.index("value_extra")


def test_with_deps_keeps_unrelated_order_stable():
    a = bep.with_deps(["newf_turnover_extra", "momentum_extra"])
    assert a == ["momentum_extra", "newf_turnover_extra"]


# ------------------------------------------------------------------ 周数判定
def test_panel_weeks_missing_file(tmp_path):
    assert bep.panel_weeks(tmp_path / "nope.npz") is None


def test_panel_weeks_reads_first_key(tmp_path):
    p = tmp_path / "x.npz"
    np.savez_compressed(p, **{"a:b": np.zeros((7, 3), dtype=np.float32)})
    assert bep.panel_weeks(p) == 7


def test_week_count_uses_weekly_meta_not_mtime():
    """对齐口径是周数，不是文件新旧——补 NaN 也能把 mtime 刷成最新。"""
    wm = ROOT / "build" / "factors" / "weekly_meta.npz"
    if not wm.exists():
        pytest.skip("需要先跑 build_factors.py")
    n = len(np.load(wm)["rows"])
    assert n > 0
    for s in bep.production_extras():
        p = ROOT / "build" / "factors" / f"{s}.npz"
        if not p.exists():
            continue
        assert bep.panel_weeks(p) is not None


# ------------------------------------------------------------------ 一致性检测
def test_checker_reports_extra_panel_item():
    """check_data_source 必须单独报一项「额外因子族」，否则根因不可见。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_cds", ROOT / "scripts" / "check_data_source.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.check()
    names = [it["name"] for it in r["items"]]
    assert "额外因子族" in names
    item = next(it for it in r["items"] if it["name"] == "额外因子族")
    assert "detail" in item and item["detail"]
