# -*- coding: utf-8 -*-
"""独立审计：模型层（qf/model.py）无未来函数 + walk-forward 复现。

检查项：
1. **refit 训练窗口审计**：monkeypatch ``qf.model.fit_weights``，截获每次 refit 实际
   传给权重估计的标签矩阵；用独立实现的 weekly label 对比，证明训练窗口 =
   ``[r-train_weeks, r-2)``，即只用 refit 时点已实现（且留 2 周缓冲）的标签；
2. **训练标签时间边界**：对每个 refit r，训练窗口最后一个标签的卖出日
   ``rows[r-3]+h+1`` 必须 <= refit 日 ``rows[r]``（标签已实现，才允许用于训练）；
3. **复现性**：用 weekly_top.npz + 保存的 pool/cfg 重跑 walk_forward，
   scores 必须与 build/models/wf_v1 保存的完全一致；
4. **关键扰动（任务要求）**：把**最后一个 refit 之后**的因子面板替换为随机数重跑，
   refit 日之前的 scores 必须逐位不变；refit 之后的 scores 必须变化；
5. 候选池只用 DEV 期（pool_meta 的 n ≤ DEV 周数）；权重归一化/refit 节奏检查。
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C                      # noqa: E402
from qf import data as qfdata                   # noqa: E402
from qf import model as qfm                     # noqa: E402

MODEL_DIR = C.BUILD_MODELS / os.environ.get("QF_AUDIT_MODEL", "wf_h10")
FACTOR_NPZ = C.BUILD_FACTORS / "weekly_top.npz"
META_NPZ = C.BUILD_FACTORS / "weekly_meta.npz"
EXTRA_NPZ = C.BUILD_FACTORS / "momentum_extra.npz"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def saved():
    if not (MODEL_DIR / "composite_score.npz").exists():
        pytest.skip(f"{MODEL_DIR}/composite_score.npz 尚未生成")
    return qfm.load_result(MODEL_DIR)


@pytest.fixture(scope="module")
def inputs(saved):
    if not (FACTOR_NPZ.exists() and META_NPZ.exists()):
        pytest.skip(f"{FACTOR_NPZ.name} 尚未生成")
    zm = np.load(META_NPZ)
    rows = zm["rows"].astype(np.int64)
    zf = np.load(FACTOR_NPZ)
    panels = {}
    for k in saved.pool:                       # 只加载池内因子，控制内存
        if k in zf.files:
            panels[k] = zf[k]
    if EXTRA_NPZ.exists():                     # 额外因子族（momentum-extra 等）
        ze = np.load(EXTRA_NPZ)
        for k in saved.pool:
            if k not in panels and k in ze.files:
                panels[k] = ze[k]
    missing = [k for k in saved.pool if k not in panels]
    if missing:
        pytest.skip(f"因子面板缺失池内因子：{missing[:5]}")
    n_weeks = len(rows)
    for k, v in panels.items():
        assert v.shape[0] == n_weeks, f"{k} 周数 {v.shape[0]} != meta {n_weeks}"
    return {"rows": rows, "panels": panels, "n_weeks": n_weeks}


def _independent_weekly_label(md, rows: np.ndarray, mask: np.ndarray, h: int) -> np.ndarray:
    """独立实现 training label：fwd_open_ret(h) 逐周横截面 rank 到 [-0.5,0.5]。

    与 qf.model.weekly_label 的现行语义对齐：只保留 t+1 开盘**可买**（md.buyable[t+1]）
    的样本（Lead 已按 verifier P2#2 修复）。这里用 md.buyable 作为"可执行性"数据源
    （其口径已在 tests/test_audit_data.py 中独立重算验证），重点验证训练窗口边界。
    """
    lab = md.fwd_open_ret(h)[rows].astype(np.float64)
    buy = np.zeros_like(mask, dtype=bool)
    okrow = rows + 1 < md.T
    buy[okrow] = md.buyable[rows[okrow] + 1]
    mask = mask & buy
    out = np.full(lab.shape, np.nan, dtype=np.float32)
    for i in range(lab.shape[0]):
        v = mask[i] & np.isfinite(lab[i])
        idx = np.flatnonzero(v)
        cnt = len(idx)
        if cnt < 10:
            continue
        order = np.argsort(lab[i, idx], kind="stable")
        r = np.empty(cnt, dtype=np.float64)
        r[order] = np.arange(cnt, dtype=np.float64)
        out[i, idx] = (r / (cnt - 1) - 0.5).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 1) refit 训练窗口审计
# ---------------------------------------------------------------------------
def test_refit_training_window_uses_only_realized_labels(md, saved, inputs, monkeypatch):
    rows = inputs["rows"]
    panels = inputs["panels"]
    mask = md.eligible[rows]
    label = _independent_weekly_label(md, rows, mask, saved.cfg.horizon)

    seen: list[dict] = []
    real_fit = qfm.fit_weights

    def spy_fit(F, y, m, cfg, verbose=False, strict=None):
        seen.append({
            "n_weeks": int(y.shape[0]),
            "y_sha": hashlib.sha1(np.ascontiguousarray(np.nan_to_num(y, nan=-999.0)).tobytes()).hexdigest(),
            "y": np.array(y, dtype=np.float32, copy=True),
        })
        return real_fit(F, y, m, cfg, verbose=verbose, strict=strict)

    monkeypatch.setattr(qfm, "fit_weights", spy_fit)
    res = qfm.walk_forward(md, rows, panels, __import__("qf.factors", fromlist=["x"]).load_catalog(),
                           cfg=saved.cfg, start_date=C.VALID_START, verbose=False,
                           pool_override=saved.pool)
    monkeypatch.undo()

    refit_dates = sorted(pd.to_datetime(res.weights["refit_date"].unique()))
    # refit 所在周行号
    refit_idx = [int(np.searchsorted(rows, np.searchsorted(md.dates, d))) for d in refit_dates]
    assert len(seen) == len(refit_idx), f"fit_weights 调用 {len(seen)} 次 != refit {len(refit_idx)} 次"

    train_weeks = int(saved.cfg.train_weeks)
    for k, (r, d) in enumerate(zip(refit_idx, refit_dates)):
        tr_lo = max(0, r - train_weeks)
        tr_hi = max(0, r - 2)
        exp = label[tr_lo:tr_hi]
        got = seen[k]["y"]
        assert got.shape == exp.shape, (
            f"refit {d.date()} 训练窗口 {got.shape} != 预期 {exp.shape}（应 [r-train, r-2)）")
        if not np.array_equal(got, exp, equal_nan=True):
            n_diff = int((~((got == exp) | (np.isnan(got) & np.isnan(exp)))).sum())
            raise AssertionError(f"refit {d.date()} 训练标签与独立重算不一致（{n_diff} 格）"
                                 " → 可能用了 refit 之后的标签")
        # 训练窗口最后一个标签必须在 refit 日之前已实现（卖出日 <= refit 日）
        last_sig_row = int(rows[tr_hi - 1])
        realized_row = last_sig_row + saved.cfg.horizon + 1
        assert realized_row <= int(rows[r]), (
            f"refit {d.date()} 训练样本包含未实现标签：最后信号行 {md.dates[last_sig_row].date()} "
            f"卖出行 {md.dates[realized_row].date()} > refit {d.date()}")
    print(f"[audit] refit 训练窗口审计通过：{len(refit_idx)} 次 refit，训练集仅含"
          f" [r-{train_weeks}, r-2) 周且全部标签在 refit 前已实现")


# ---------------------------------------------------------------------------
# 2) 复现性 + 关键扰动测试
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rerun(md, saved, inputs):
    """用保存的 pool/cfg 重跑 walk_forward（不扰动），应与保存的 scores 完全一致。"""
    import qf.factors as qff
    cat = qff.load_catalog()
    res = qfm.walk_forward(md, inputs["rows"], inputs["panels"], cat,
                           cfg=saved.cfg, start_date=C.VALID_START, verbose=False,
                           pool_override=saved.pool)
    return res


def test_walk_forward_reproducible(md, saved, rerun):
    a = np.asarray(saved.scores)
    b = np.asarray(rerun.scores)
    assert a.shape == b.shape
    assert np.array_equal(a, b, equal_nan=True), \
        "重跑 walk_forward 与保存的 composite_score.npz 不一致（不可复现）"
    # refit 节奏一致
    d1 = sorted(pd.to_datetime(saved.weights["refit_date"].unique()))
    d2 = sorted(pd.to_datetime(rerun.weights["refit_date"].unique()))
    assert d1 == d2, f"refit 日期不一致：{d1} vs {d2}"
    print(f"[audit] walk_forward 复现通过：scores 逐位一致，{len(d1)} 次 refit")


def test_future_factor_perturbation_does_not_change_past_scores(md, saved, inputs, rerun):
    """把"最后一个有效 refit 之后"的因子面板替换为随机数 → refit 日前的 scores 必须逐位不变。

    注：若最后一个 refit 恰好落在最后一个周频行（无未来应用区间），则退回到倒数第二个
    有效 refit 作为切点（仍有未来区间可供扰动检验）。
    """
    rows = inputs["rows"]
    panels = inputs["panels"]
    refit_dates = sorted(pd.to_datetime(saved.weights["refit_date"].unique()))
    # 找最后一个"之后至少还有 2 个周频行"的 refit 作为切点
    cut_idx = None
    cut_date = None
    for d in reversed(refit_dates):
        wi = int(np.searchsorted(rows, np.searchsorted(md.dates, d)))
        if wi < len(rows) - 2:
            cut_idx, cut_date = wi, d
            break
    assert cut_idx is not None, "找不到可扰动的未来区间（检验无法进行）"
    assert cut_idx > 0

    rng = np.random.default_rng(C.RANDOM_SEED + 2024)
    pert = {}
    for k, v in panels.items():
        w = np.array(v, dtype=np.float32, copy=True)
        fin = np.isfinite(w[cut_idx:])
        rnd = rng.standard_normal(size=w[cut_idx:].shape).astype(np.float32)
        w[cut_idx:] = np.where(fin, rnd, np.nan)
        pert[k] = w

    import qf.factors as qff
    cat = qff.load_catalog()
    res2 = qfm.walk_forward(md, rows, pert, cat, cfg=saved.cfg,
                            start_date=C.VALID_START, verbose=False,
                            pool_override=saved.pool)

    cut_rows = rows[:cut_idx]
    a = np.asarray(rerun.scores)[cut_rows]
    b = np.asarray(res2.scores)[cut_rows]
    n_diff = int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
    assert n_diff == 0, (
        f"把 {cut_date.date()} 之后的因子面板全部替换为随机数后，"
        f"refit 日之前的 scores 有 {n_diff} 个格子改变 → 存在未来函数")
    # 反向 sanity：refit 之后的分数必须变化（否则扰动无效/模型未使用因子）
    later = rows[cut_idx:]
    a2 = np.asarray(rerun.scores)[later]
    b2 = np.asarray(res2.scores)[later]
    assert np.isfinite(a2).any(), "扰动区间原始 scores 全为 NaN，无法比较"
    assert not np.array_equal(a2, b2, equal_nan=True), "扰动后更新区间的 scores 未变化，扰动无效"
    print(f"[audit] 未来因子扰动通过：{cut_date.date()} 前 {len(cut_rows)} 周 scores 逐位不变，"
          f"之后 {len(later)} 周发生变化")


# ---------------------------------------------------------------------------
# 3) 候选池 / 权重结构审计
# ---------------------------------------------------------------------------
def test_pool_selected_from_dev_period_only(saved, md):
    rows = saved.rows_weekly
    dev_weeks = int((md.dates[rows] <= pd.Timestamp(C.DEV_END)).sum())
    pm = saved.pool_meta
    assert len(pm) == len(saved.pool)
    assert (pm["n"] <= dev_weeks).all(), "候选池统计用了 DEV 期以外的标签"
    assert saved.cfg.max_factors >= len(saved.pool) > 0
    # 类别配额
    if "category" in pm.columns:
        assert pm["category"].value_counts().max() <= saved.cfg.max_per_category
    # |ICIR| 排序（选池只在相关性剪枝后可能乱序，这里只检查非空）
    assert np.isfinite(pm["icir"].to_numpy(dtype=float)).all()


def test_weights_normalized_and_schedule(saved, md):
    w = saved.weights
    assert set(w.columns) >= {"refit_date", "factor_key", "weight"}
    for d, g in w.groupby("refit_date"):
        l1 = float(np.abs(g["weight"].to_numpy()).sum())
        assert l1 == pytest.approx(1.0, abs=1e-9), f"{d} 权重 L1 范数 {l1}"
    dates = sorted(pd.to_datetime(w["refit_date"].unique()))
    assert len(dates) >= 10
    assert dates[0].year == 2020 or dates[0].year == 2021, f"首次 refit {dates[0]}"
    gaps = np.array([(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)])
    # 审计记录：cfg.refit_weeks=13，但季度初与周频网格错位 + min-gap 判据会让相邻
    # 季度被跳过，实际平均间隔 ~176 天（约半年），与“每 3 个月再估计”文档口径不符。
    print(f"[audit] refit 间隔：mean={gaps.mean():.1f}d min={gaps.min()} max={gaps.max()} "
          f"（cfg.refit_weeks={saved.cfg.refit_weeks} → 约季度；实际约半年，见报告 P2）")
    assert gaps.min() > 60, f"refit 间隔过短：{gaps.min()} 天"
    assert gaps.max() < 220, f"refit 间隔过长：{gaps.max()} 天（超过半年）"
    # 复合分只在周频行有值
    sc = np.asarray(saved.scores)
    n_finite_rows = int(np.isfinite(sc).any(axis=1).sum())
    assert n_finite_rows <= len(saved.rows_weekly), "scores 在非周频行有值"
