# -*- coding: utf-8 -*-
"""新因子族「历史面板 ↔ 最新一日行」同源性的回归测试。

背景（2026-11-03，真实故障）：把换手族（``newf-extra:N001..N004``）接进生产池后，
``scripts/daily_signal.py`` 直接崩在::

    qf/model.py: build_live_features → cat.loc[k] → KeyError: 'newf-extra:N001'

原因是 walk-forward 走 ``--extra`` 加载 npz（能拿到这些因子），而 live 打分走
``load_catalog()``（基础 catalog 里没有这些键）。**今日信号完全出不来**，而且
管道前面的 model/backtest 全部「成功」，只有最后一步 signal 失败。

修法：算式集中到 ``qf.newf.build_panels``，落盘（build_newf_family.py）与
live（build_live_features → qf.newf.live_row）共用同一份代码。
本文件守住三件事：
1. 落盘面板 == 现算面板（逐元素，含 NaN 位置）
2. live 行 == 面板末行
3. 含 ``newf-extra:`` 的池能真的算出特征（不再 KeyError）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qf import config as C
from qf import data as qfdata
from qf import newf as qfn

NPZ = C.BUILD_FACTORS / "newf_turnover_extra.npz"
ROWS = C.BUILD_FACTORS / "weekly_meta.npz"


@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def panels(md):
    return qfn.build_panels(md)


# ---------------------------------------------------------------- 同源性
def test_live_row_matches_last_panel_row(panels, md):
    lr = qfn.live_row(md)
    assert set(lr) == {f"{qfn.PREFIX}{qfn.SLUG_TO_ID[s]}" for s in qfn.SLUG_TO_ID}
    for slug, arr in panels.items():
        key = f"{qfn.PREFIX}{qfn.SLUG_TO_ID[slug]}"
        assert lr[key].shape == (md.N,)
        assert np.array_equal(lr[key], arr[-1], equal_nan=True), f"{slug} live 行与面板末行不一致"


@pytest.mark.skipif(not (NPZ.exists() and ROWS.exists()), reason="换手族面板未构建")
def test_saved_npz_matches_live_computation(panels):
    """落盘 npz 必须能被现算复现（防止有人手改 npz 或改算式后没重算）。"""
    rows = np.load(ROWS)["rows"].astype(np.int64)
    z = np.load(NPZ)
    for slug in qfn.FAMILY["turnover"]:
        key = f"{qfn.PREFIX}{qfn.SLUG_TO_ID[slug]}"
        if key not in z.files:
            continue
        assert np.array_equal(z[key], panels[slug][rows], equal_nan=True), \
            f"{key} 落盘值与 qf.newf 现算值不一致，请重跑 build_newf_family.py"


@pytest.mark.skipif(not NPZ.exists(), reason="换手族面板未构建")
def test_catalog_path_points_to_existing_file():
    cat_p = NPZ.with_name(NPZ.stem.replace("_extra", "_catalog") + ".parquet")
    assert cat_p.exists(), f"缺 catalog：{cat_p.name}"
    cat = pd.read_parquet(cat_p)
    for k in ("factor_key", "slug", "category_en", "base", "path"):
        assert k in cat.columns
    assert set(cat["factor_key"]) <= set(np.load(NPZ).files)
    for p in cat["path"]:
        f = C.BUILD.parent / str(p).split("#")[0]
        assert f.exists(), f"catalog.path 指向不存在的文件：{p}"


# ---------------------------------------------------------------- 故障回归
def test_live_features_accept_newf_pool(md):
    """原先这里 KeyError → 今日信号整体失败。必须能算出 (K, N) 特征。"""
    from qf import factors as qff
    from qf import model as qfm

    inp = qff.FactorInputs(md)
    cfg = qfm.ModelConfig()
    pool = [f"{qfn.PREFIX}N001", "risk-pattern:R080"]
    F = qfm.build_live_features(md, inp, pool, cfg)
    assert F.shape == (len(pool), md.N)
    for i, k in enumerate(pool):
        n = int(np.isfinite(F[i]).sum())
        assert n > 1000, f"{k} 的有效股票只有 {n} 只"


def test_prod_model_pool_is_live_computable(md):
    """生产模型池里每个因子都必须能算 live 行（新族入池后的守门测试）。"""
    from qf import factors as qff
    from qf import model as qfm

    res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
    inp = qff.FactorInputs(md)
    F = qfm.build_live_features(md, inp, res.pool, res.cfg)
    assert F.shape == (len(res.pool), md.N)
    assert np.isfinite(F).any(axis=1).all(), "池内存在完全算不出数的因子"
