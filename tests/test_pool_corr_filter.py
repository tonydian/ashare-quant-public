# -*- coding: utf-8 -*-
"""相关性剪枝的数值口径（2026-11-03）。

回归的 bug：
    `select_pool` 里的相关矩阵是 `G / (mean_n_i * mean_n_j)`，少了「除以周数 T」，
    于是 |ρ|=1 的镜像对只算出 0.008 量级，`corr_max=0.70` 永远不可能触发。
    实测生产池：R737/R738 逐周 ρ=-1.000，却被算成 -0.079；池内 60 对 |ρ|>0.7
    全部漏过 → 34 个槽里塞进 5.5 个独立信号。

用合成数据做端到端断言，不依赖真实行情。
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qf import model as qfm  # noqa: E402


def _fake_market(T: int = 120, N: int = 420, n_ind: int = 6, seed: int = 7):
    rng = np.random.default_rng(seed)
    industry = np.tile(np.arange(N) % n_ind, (T, 1)).astype(np.int32)
    logmcap = rng.normal(10.0, 0.8, size=(T, N)).astype(np.float32)
    eligible = np.ones((T, N), dtype=bool)
    return SimpleNamespace(industry=industry, logmcap=logmcap, eligible=eligible,
                           N=N, T=T)


def _catalog(keys: list[str], cat: str = "Pattern") -> pd.DataFrame:
    return pd.DataFrame({"factor_key": keys, "category_en": [cat] * len(keys),
                         "lib": ["synthetic"] * len(keys),
                         "base": ["x"] * len(keys)})


def _run(panels: dict, keys: list[str], *, max_factors: int, corr_max: float,
         max_per_category: int = 8):
    md = _fake_market()
    T, N = md.T, md.N
    rows = np.arange(T)
    mask = md.eligible.copy()
    label = panels[keys[0]].copy()             # 让首个因子有强 IC
    label = label + np.random.default_rng(0).normal(0, 0.35, size=label.shape)
    cfg = qfm.ModelConfig(max_factors=max_factors, corr_max=corr_max,
                          max_per_category=max_per_category, pool_guard=False,
                          min_ic_obs=40)
    pool, meta = qfm.select_pool(md, rows, panels, label, mask, cfg,
                                 _catalog(keys), verbose=False)
    return pool


def _base_panels(T: int = 120, N: int = 420) -> dict:
    rng = np.random.default_rng(11)
    a = rng.normal(size=(T, N)).astype(np.float32)
    return {"f_a": a}


def test_mirror_pair_is_pruned():
    """ρ=-1 的镜像对只能进一个：这正是 R473/R474、R737/R738 的情形。"""
    p = _base_panels()
    p["f_mirror"] = -p["f_a"]
    pool = _run(p, ["f_a", "f_mirror"], max_factors=2, corr_max=0.70)
    assert pool == ["f_a"], f"镜像对被同时选入（相关剪枝失效）：{pool}"


def test_duplicate_series_is_pruned():
    """完全复制（ρ=+1）同样只能进一个。"""
    p = _base_panels()
    p["f_copy"] = p["f_a"].copy()
    pool = _run(p, ["f_a", "f_copy"], max_factors=2, corr_max=0.70)
    assert len(pool) == 1


def test_independent_factors_are_kept():
    """近似独立的因子必须能同时进池——避免「剪枝过头」把池子清空。"""
    rng = np.random.default_rng(23)
    T, N = 120, 420
    p = {"f_a": rng.normal(size=(T, N)).astype(np.float32)}
    for i in range(3):
        p[f"f_r{i}"] = rng.normal(size=(T, N)).astype(np.float32)
    keys = list(p)
    pool = _run(p, keys, max_factors=4, corr_max=0.70)
    assert len(pool) == 4, f"独立因子被误剪：{pool}"


def test_partial_corr_respects_threshold():
    """0.85ρ 的因子对：阈值 0.70 时只能进一个，阈值 0.95 时都能进。"""
    T, N = 120, 420
    rng = np.random.default_rng(31)
    a = rng.normal(size=(T, N)).astype(np.float32)
    b = 0.85 * a + np.sqrt(1 - 0.85 ** 2) * rng.normal(size=(T, N)).astype(np.float32)
    p = {"f_a": a, "f_b": b.astype(np.float32)}
    assert len(_run(p, ["f_a", "f_b"], max_factors=2, corr_max=0.70)) == 1
    assert len(_run(p, ["f_a", "f_b"], max_factors=2, corr_max=0.95)) == 2


def test_guard_path_also_enforces_corr():
    """缓冲池的「保留上次池」路径也必须过相关剪枝：否则镜像对会永远留在池里。"""
    md = _fake_market()
    T, N = md.T, md.N
    rng = np.random.default_rng(41)
    a = rng.normal(size=(T, N)).astype(np.float32)
    p = {"f_a": a, "f_mirror": -a}
    for i in range(4):
        p[f"f_r{i}"] = rng.normal(size=(T, N)).astype(np.float32)
    keys = list(p)
    label = a + rng.normal(0, 0.35, size=(T, N)).astype(np.float32)
    # 缓冲的目标池大小 = 上次池大小（2 个），剪掉镜像对后必须从 ranked 里补回 1 个
    cfg = qfm.ModelConfig(max_factors=3, corr_max=0.70, max_per_category=8,
                          pool_guard=True, pool_prev=["f_a", "f_mirror"],
                          pool_state={}, pool_need_streak=2, min_ic_obs=40)
    pool, _ = qfm.select_pool(md, np.arange(T), p, label, md.eligible.copy(),
                             cfg, _catalog(keys), verbose=False)
    assert "f_mirror" not in pool, f"缓冲路径把镜像对留在了池内：{pool}"
    assert len(pool) == 2, f"剪掉相关对后应补回上次池规模（2 个）：{pool}"
    assert "f_a" in pool


def test_corr_normalization_divides_by_week_count():
    """源码级防线：归一化必须除以「有效交集总数」，不能再是 mean_n_i*mean_n_j。"""
    src = (ROOT / "qf" / "model.py").read_text(encoding="utf-8")
    assert "G / np.maximum(J, 1.0)" in src
    assert "cnt[:, None] * cnt[None, :]" not in src, "旧的错误归一化又回来了"


def test_production_pool_respects_corr_max():
    """生产池的不变量：池内两两 |ρ| 必须都在 corr_max 内。

    这条是给 2026-11-03 那个 bug 上锁的——当时 34 个槽里有 49~60 对 |ρ|>0.7，
    而剪枝阈值 0.70 因为归一化写错从未触发。
    """
    import json
    from qf import config as C
    from qf import data as qfdata

    meta = ROOT / "build" / "models" / "wf_h10" / "model_meta.json"
    if not meta.exists():
        pytest.skip("生产模型不存在")
    pool = json.loads(meta.read_text(encoding="utf-8"))["pool"]
    rows = np.load(ROOT / "build" / "factors" / "weekly_meta.npz")["rows"].astype(np.int64)
    zs = [np.load(ROOT / "build" / "factors" / "weekly_top.npz")]
    zs += [np.load(p) for p in sorted((ROOT / "build" / "factors").glob("*_extra.npz"))]
    pan = {}
    for k in pool:
        for z in zs:
            if k in z.files:
                a = z[k]
                if a.shape[0] < len(rows):
                    pad = np.full((len(rows) - a.shape[0], a.shape[1]), np.nan, np.float32)
                    a = np.concatenate([a, pad], axis=0)
                pan[k] = a
                break
        else:
            pytest.skip(f"因子 {k} 面板缺失")
    md = qfdata.load_market(verbose=False)
    dev = md.dates[rows] <= pd.Timestamp(C.DEV_END)
    rd = rows[dev]
    mk = md.eligible[rd]
    A = np.stack([qfm.build_feature_block(md, rd, {k: pan[k][dev]}, k, mk) for k in pool], 0)
    V = np.isfinite(A) & mk[None]
    A[~V] = 0.0
    G = np.einsum("kwn,lwn->kl", A.astype(np.float64), A.astype(np.float64), optimize=True)
    J = np.einsum("kwn,lwn->kl", V.astype(np.float32), V.astype(np.float32), optimize=True)
    c = np.clip(G / np.maximum(J, 1.0), -1, 1)
    iu = np.triu_indices(len(pool), 1)
    worst = float(np.abs(c[iu]).max())
    assert worst <= 0.72, f"生产池里仍有 |ρ|={worst:.3f} 的冗余对（阈值 0.70）"
