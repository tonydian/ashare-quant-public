# -*- coding: utf-8 -*-
"""额外因子族合并进 catalog 时必须去重（2026-11-03）。

背景：newf-extra 的 N001~N005 与 newf-turnover 是同一批换手因子的两份副本，
factor_key 完全相同。panels 是字典、天然去重，但 catalog 直接 concat 会留下
重复行 —— 一旦同时传两个 --extra，按类配额计数与池内去重都会算错。
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAC = ROOT / "build" / "factors"


def test_merge_guards_against_duplicate_factor_key():
    src = (ROOT / "scripts" / "run_research.py").read_text(encoding="utf-8")
    i = src.index("add = extra_cat[")
    seg = src[i:i + 900]
    assert 'drop_duplicates(subset="factor_key"' in seg, \
        "合并前必须对 extra catalog 自身去重"
    assert 'add = add[~add["factor_key"].isin(set(cat["factor_key"]))]' in seg, \
        "合并前必须剔除已存在于 catalog 的 factor_key"


def test_duplicate_keys_exist_on_disk():
    """锁定现实：这两份产物确实有 5 个重复 key，所以防护不是空的。"""
    a = pd.read_parquet(FAC / "newf_catalog.parquet")
    b = pd.read_parquet(FAC / "newf_turnover_catalog.parquet")
    dup = set(a["factor_key"]) & set(b["factor_key"])
    assert len(dup) == 5, f"预期 5 个重复 key，实际 {len(dup)}"
    assert dup == {f"newf-extra:N00{i}" for i in range(1, 6)}

    # 合并后若不去重，catalog 会出现重复行；去重后必须唯一
    merged = pd.concat([a, b], ignore_index=True)
    assert not merged["factor_key"].is_unique
    fixed = merged.drop_duplicates(subset="factor_key", keep="first")
    assert fixed["factor_key"].is_unique
    assert len(fixed) == len(set(a["factor_key"]) | set(b["factor_key"]))


def test_max_per_category_is_exposed():
    """取消/收紧族配额必须能从命令行调，否则「配额是否拖后腿」无法验证。"""
    src = (ROOT / "scripts" / "run_research.py").read_text(encoding="utf-8")
    assert '"--max-per-category"' in src
    assert '_kw["max_per_category"] = int(args.max_per_category)' in src
    from qf import model as qfm
    assert qfm.ModelConfig().max_per_category == 4, "默认值必须保持 4，避免静默改生产行为"


def test_pool_slim_products_consistent():
    """精简实验产物必须自洽：基准方案 = 34 个，族数 = 8，有效独立数 < 槽位数。"""
    import json
    p = ROOT / "build" / "eval" / "pool_slim.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n_pool"] == 34
    assert len(d["families"]) == 8
    assert sum(f["n"] for f in d["families"]) == 34, "族成员必须覆盖全部槽位"
    assert d["n_effective"] < d["n_pool"], "有效独立数必须小于槽位数，否则去冗余无意义"
    assert d["n_pairs_rho70"] > 0
