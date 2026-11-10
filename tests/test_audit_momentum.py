# -*- coding: utf-8 -*-
"""独立审计：额外因子族 qf/momentum.py（中期动量）无未来函数 + 定义正确性。

* 定义：mom{k} = price[t]/price[t-k] - 1；mom{k}s{s} = price[t-s]/price[t-k-s] - 1；
  全部只用 t 及以前价格；
* 扰动 t0 之后的全部价格，t<=t0 的因子值必须逐位不变；
* 抽查定义数值（与手算 numpy 对齐）；
* 模型集成：qf.model.build_feature_block 对动量面板只做当日横截面 rank/中性化/zscore，
  扰动未来行的面板值不改变历史行特征（逐位）。
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C                      # noqa: E402
from qf import data as qfdata                   # noqa: E402
from qf import momentum as qfmom                # noqa: E402
from qf import model as qfmodel                 # noqa: E402

CUT_DATE = "2023-06-15"


@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def rows(md):
    return np.arange(md.date_index(C.DEV_START), md.T, C.REBALANCE_DAYS, dtype=np.int64)


def _perturb_prices_after(md, t0: int, seed: int = C.RANDOM_SEED + 5) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    fake = SimpleNamespace()
    fake.T, fake.N = md.T, md.N
    fake.alive = md.alive
    p = np.array(md.price, dtype=np.float32, copy=True)
    seg = p[t0 + 1:]
    fin = np.isfinite(seg)
    noise = np.exp(rng.normal(0.0, 0.3, size=seg.shape)).astype(np.float32)
    p[t0 + 1:] = np.where(fin, seg * noise, seg)
    fake.price = p
    return fake


def test_momentum_no_lookahead(md, rows):
    t0 = md.date_index(CUT_DATE)
    fake = _perturb_prices_after(md, t0)
    base = qfmom.build_panels(md, rows)
    pert = qfmom.build_panels(fake, rows)
    assert len(base) == len(qfmom.DEFS) >= 8
    for k in base:
        a, b = base[k], pert[k]
        # 行 t 的信号在 t 收盘后可算：扰动 t0 之后价格，t0 及以前的周频行必须逐位相同
        keep = rows <= t0
        assert np.array_equal(a[keep], b[keep], equal_nan=True), (
            f"{k} 在 t<={CUT_DATE} 的因子值被未来价格改变 → 未来函数")
    print(f"[audit] momentum 无未来函数通过：{len(base)} 个因子 / {int((rows <= t0).sum())} 周")


def test_momentum_definition(md, rows):
    p = md.price.astype(np.float64)
    t = rows
    for fid, k, skip, _ in qfmom.DEFS:
        key = f"{qfmom.LIB}:{fid}"
        got = qfmom.build_panels(md, rows)[key]
        valid = (t - k - skip) >= 0
        cur_idx = np.maximum(t - skip, 0)
        prev_idx = np.maximum(t - k - skip, 0)
        cur = np.where(valid[:, None], p[cur_idx], np.nan)
        prev = np.where(valid[:, None], p[prev_idx], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            exp = cur / prev - 1.0
        exp = np.where(md.alive[t] & np.isfinite(exp), exp, np.nan).astype(np.float32)
        assert np.array_equal(got, exp, equal_nan=True), f"{key} 定义数值不符"


def test_momentum_live_matches_full_panel(md):
    t = md.T - 1
    for fid, _, _, _ in qfmom.DEFS:
        key = f"{qfmom.LIB}:{fid}"
        full = qfmom.build_live(md, key)
        assert full.shape == (md.T, md.N)
        assert np.array_equal(full[t], qfmom.build_panels(md)[key][t], equal_nan=True)


def test_momentum_feature_block_no_lookahead(md, rows):
    """build_feature_block 对动量面板只做逐行横截面处理：扰动未来行不改变历史特征。"""
    t0 = md.date_index(CUT_DATE)
    fake = _perturb_prices_after(md, t0, seed=C.RANDOM_SEED + 11)
    base = qfmom.build_panels(md, rows)
    pert = qfmom.build_panels(fake, rows)
    mask = md.eligible[rows]
    key = "momentum-extra:M001"
    f1 = qfmodel.build_feature_block(md, rows, base, key, mask)
    f2 = qfmodel.build_feature_block(md, rows, pert, key, mask)
    keep = rows <= t0
    assert np.array_equal(f1[keep], f2[keep], equal_nan=True), (
        "build_feature_block 的历史特征受未来面板值影响 → 存在未来函数")
