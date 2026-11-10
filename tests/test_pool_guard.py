# -*- coding: utf-8 -*-
"""因子池缓冲机制的回归测试。"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import pool_guard as pg


# ---------------------------------------------------------------- 基本行为
def test_initial_pool_takes_top_k():
    ranked = [f"F{i:03d}" for i in range(50)]
    pool, state, info = pg.apply_guard(None, ranked, k=10, verbose=False)
    assert pool == ranked[:10]
    assert info["mode"] == "initial"
    assert len(pool) == 10


def test_pool_size_always_k():
    """无论输入如何，池大小必须等于 k。"""
    ranked = [f"F{i:03d}" for i in range(60)]
    pool, _ = pg.apply_guard(None, ranked, k=12, verbose=False)[:2]
    assert len(pool) == 12
    # 再跑几次
    state = {}
    cur = pool
    for _ in range(5):
        cur, state, _ = pg.apply_guard(cur, ranked[::-1], k=12, state=state,
                                       verbose=False)
        assert len(cur) == 12, f"池大小 {len(cur)} != 12"


def test_stable_when_ranking_unchanged():
    """排名不变时，池不应变化（换手 0）。"""
    ranked = [f"F{i:03d}" for i in range(50)]
    pool, state, _ = pg.apply_guard(None, ranked, k=10, verbose=False)
    for _ in range(5):
        new_pool, state, info = pg.apply_guard(pool, ranked, k=10, state=state,
                                               verbose=False)
        assert info["turnover"] == 0, f"排名不变却换手 {info['turnover']}"
        assert new_pool == pool


def test_small_noise_causes_little_turnover():
    """小幅噪声（尾部抖动）不应导致频繁换人。"""
    rng = np.random.default_rng(42)
    N, K = 60, 16
    quality = np.sort(rng.normal(0.5, 0.3, N))[::-1]
    base = [f"F{i:03d}" for i in np.argsort(-quality)]
    pool, state, _ = pg.apply_guard(None, base, k=K, verbose=False)

    total_turn = 0
    for _ in range(10):
        q = quality + rng.normal(0, 0.03, N)
        order = [f"F{i:03d}" for i in np.argsort(-q)]
        pool, state, info = pg.apply_guard(pool, order, k=K, state=state,
                                           verbose=False)
        total_turn += info["turnover"]
    # 无缓冲时 10 次换手约 20~30；有缓冲应明显更少
    assert total_turn < 15, f"缓冲后换手仍偏高: {total_turn}"


def test_strong_factor_enters_immediately():
    """真正变强的因子（排名前 50%）必须能进池，不能被缓冲僵化。"""
    rng = np.random.default_rng(7)
    N, K = 60, 16
    quality = np.sort(rng.normal(0.5, 0.3, N))[::-1]
    base = [f"F{i:03d}" for i in np.argsort(-quality)]
    pool, state, _ = pg.apply_guard(None, base, k=K, verbose=False)

    # 让一个池外因子变最强
    weak = [x for x in base if x not in pool][0]
    idx = int(weak[1:])
    q2 = quality.copy()
    q2[idx] = 5.0
    order2 = [f"F{i:03d}" for i in np.argsort(-q2)]

    pool2, state, info = pg.apply_guard(pool, order2, k=K, state=state,
                                        need_streak=2, verbose=False)
    assert weak in pool2, f"最强因子 {weak} 未能进池（缓冲僵化）"


def test_removal_requires_streak():
    """移除需要连续确认（单次跌出不应立即踢）。"""
    ranked_a = [f"F{i:03d}" for i in range(40)]
    pool, state, _ = pg.apply_guard(None, ranked_a, k=10, verbose=False)

    # 第一次：让原池某因子跌到很后面
    ranked_b = [x for x in ranked_a if x != "F005"] + ["F005"]
    pool1, state, info1 = pg.apply_guard(pool, ranked_b, k=10, state=state,
                                        need_streak=2, verbose=False)
    # F005 应还在（缓冲期）
    assert "F005" in pool1, "单次跌出就踢了（缓冲失效）"

    # 第二次：继续跌
    pool2, state, info2 = pg.apply_guard(pool1, ranked_b, k=10, state=state,
                                        need_streak=2, verbose=False)
    assert "F005" not in pool2, "连续两次跌出仍未踢"


def test_state_persistence(tmp_path):
    state = {"F001": {"in_streak": 2, "out_streak": 0}}
    pg.save_state(tmp_path, state, meta={"note": "test"})
    loaded = pg.load_state(tmp_path)
    assert loaded.get("F001", {}).get("in_streak") == 2


def test_prev_pool_from_meta(tmp_path):
    import json
    d = tmp_path / "wf_test"
    d.mkdir()
    (d / "model_meta.json").write_text(
        json.dumps({"pool": ["A", "B"]}), encoding="utf-8")
    assert pg.prev_pool_from_meta(d) == ["A", "B"]
    assert pg.prev_pool_from_meta(tmp_path / "nonexist") is None


# ---------------------------------------------------------------- 接入
def test_model_config_has_guard_fields():
    s = (ROOT / "qf" / "model.py").read_text(encoding="utf-8")
    for f in ("pool_guard", "pool_prev", "pool_need_streak",
              "pool_entry_rank", "pool_exit_rank"):
        assert f in s, f"ModelConfig 缺 {f}"


def test_select_pool_wired():
    s = (ROOT / "qf" / "model.py").read_text(encoding="utf-8")
    assert "pool_guard" in s
    assert "apply_guard" in s


def test_pipeline_uses_guard():
    s = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "--pool-guard" in s, "管道未启用缓冲"


def test_run_research_has_cli():
    s = (ROOT / "scripts" / "run_research.py").read_text(encoding="utf-8")
    for arg in ("--pool-guard", "--pool-streak", "--pool-entry-rank",
                "--pool-exit-rank"):
        assert arg in s, f"缺参数 {arg}"

def test_category_quota_respected_after_guard():
    """缓冲的挤占不能破坏类别配额（回归测试）。"""
    import json
    import pandas as pd
    from qf import config as C

    md_dir = C.BUILD_MODELS / "wf_h10"
    meta_p = md_dir / "model_meta.json"
    pm_p = md_dir / "pool_meta.parquet"
    if not (meta_p.exists() and pm_p.exists()):
        pytest.skip("模型不存在")
    meta = json.load(open(meta_p, encoding="utf-8"))
    if not meta.get("cfg", {}).get("pool_guard"):
        pytest.skip("未启用缓冲")
    pm = pd.read_parquet(pm_p)
    if "category" not in pm.columns:
        pytest.skip("pool_meta 无 category")
    max_per = int(meta["cfg"].get("max_per_category", 4))
    actual = pm["category"].value_counts().max()
    assert actual <= max_per, (
        f"类别配额被破坏：{actual} > {max_per}\n"
        f"{pm['category'].value_counts()}"
    )


def test_pool_size_matches_meta():
    """pool_meta 行数应等于 model_meta 里的 pool 长度。"""
    import json
    import pandas as pd
    from qf import config as C

    md_dir = C.BUILD_MODELS / "wf_h10"
    if not (md_dir / "model_meta.json").exists():
        pytest.skip("模型不存在")
    meta = json.load(open(md_dir / "model_meta.json", encoding="utf-8"))
    pm = pd.read_parquet(md_dir / "pool_meta.parquet")
    assert len(pm) == len(meta["pool"]), (
        f"pool_meta {len(pm)} 行 vs pool {len(meta['pool'])} 个"
    )


def test_guard_does_not_break_fdr():
    """启用缓冲后，池内因子仍应通过 FDR（pool_meta 的 n 不超过 DEV 周数）。"""
    import json
    import numpy as np
    import pandas as pd
    from qf import config as C

    md_dir = C.BUILD_MODELS / "wf_h10"
    meta_p = md_dir / "model_meta.json"
    pm_p = md_dir / "pool_meta.parquet"
    if not (meta_p.exists() and pm_p.exists()):
        pytest.skip("模型不存在")
    meta = json.load(open(meta_p, encoding="utf-8"))
    pm = pd.read_parquet(pm_p)
    # n = DEV 期有效周数，不应超过 DEV 期总周数
    from qf import data as qfdata
    md = qfdata.load_market(verbose=False)
    zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
    rows = zm["rows"].astype(int)
    dev_end = pd.Timestamp(C.DEV_END)
    dev_weeks = int((pd.to_datetime(md.dates[rows]) <= dev_end).sum())
    if "n" in pm.columns:
        assert (pm["n"] <= dev_weeks).all(), "池内因子用了 DEV 期以外的数据"
