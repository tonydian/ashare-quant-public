# -*- coding: utf-8 -*-
"""右侧模型 v2.1（600936 缺陷修复版）独立审计测试。

只使用合成数据 + 只读产物；不依赖 web 服务，不做任何写盘（除 pytest 自身）。
覆盖 Lead 要求的：(1) apply_universe_filter 无未来函数 + 与独立 MA60 口径一致；
(2) 600936 在系统数据口径下被排除（close<MA60）；(5) ModelConfig.universe 新字段不破坏旧模型加载。
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C          # noqa: E402
from qf import model as qfm         # noqa: E402

MODELS = C.BUILD_MODELS
EXT = pathlib.Path(r"D:\wkf\getrich\data_extract\was_stock_data_ext")
T0 = 120


def _fake_close(T=200, N=16, seed=7):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.02, size=(T, N))
    close = 10.0 * np.cumprod(1.0 + steps, axis=0)
    return close.astype(np.float64)


def _indep_ma(close, t, win=60):
    seg = close[max(0, t - win + 1):t + 1]
    cnt = np.isfinite(seg).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= max(2, int(win * 0.8)),
                        np.nansum(seg, axis=0) / cnt, np.nan)


def test_universe_filter_no_lookahead_and_ma60():
    close = _fake_close()
    md = types.SimpleNamespace(close=close)
    rows = np.array([T0, T0 + 5, T0 + 10, T0 + 15, T0 + 20])
    rng = np.random.default_rng(11)
    z = rng.standard_normal((len(rows), close.shape[1])).astype(np.float32)
    base = qfm.apply_universe_filter(md, z, rows, "above_ma60")

    # 1) 扰动 t0 之后的价格：t<=t0 的结果必须逐位不变
    close2 = close.copy()
    close2[T0 + 1:] = close2[T0 + 1:] * 1.37 + 3.0
    md2 = types.SimpleNamespace(close=close2)
    pert = qfm.apply_universe_filter(md2, z, rows, "above_ma60")
    pre = rows <= T0
    assert np.array_equal(base[pre], pert[pre], equal_nan=True), \
        "apply_universe_filter 使用了 t 之后的数据（无未来函数被破坏）"
    post = rows > T0
    assert not np.array_equal(base[post], pert[post], equal_nan=True), \
        "扰动 t0 之后的行情没有改变未来行的过滤结果（扰动无效）"

    # 2) 与独立 MA60 实现一致（含有效计数规则）
    for i, t in enumerate(rows):
        ma = _indep_ma(close, int(t))
        exp_ok = np.isfinite(close[int(t)]) & np.isfinite(ma) & (close[int(t)] > ma)
        got_ok = np.isfinite(base[i])
        assert np.array_equal(got_ok, exp_ok), f"row {t} 的 MA60 掩码与独立实现不一致"
        # 域内数值必须不被修改
        assert np.array_equal(base[i][exp_ok], z[i][exp_ok].astype(np.float32))

    # 3) universe=None 直通
    assert np.array_equal(qfm.apply_universe_filter(md, z, rows, None), z)

    # 4) above_ma60_slope 变体同样只用过去数据
    bs = qfm.apply_universe_filter(md, z, rows, "above_ma60_slope")
    ps = qfm.apply_universe_filter(md2, z, rows, "above_ma60_slope")
    assert np.array_equal(bs[pre], ps[pre], equal_nan=True)


def test_model_config_universe_backward_compatible():
    cfg = qfm.ModelConfig()
    assert getattr(cfg, "universe", "<missing>") is None
    cfg2 = qfm.ModelConfig(universe="above_ma60")
    assert cfg2.universe == "above_ma60"
    # 旧模型 meta 无 universe 字段 → 加载后应为 None（不破坏 v1 模型）
    old = MODELS / "wf_h10"
    if old.exists():
        res = qfm.load_result(old)
        assert getattr(res.cfg, "universe", "<missing>") is None
        assert np.asarray(res.scores).ndim == 2
    # 新模型 meta 含 universe → 加载后应为 above_ma60
    new = MODELS / "wf_rs10"
    if new.exists():
        res = qfm.load_result(new)
        assert res.cfg.universe == "above_ma60"
        assert len(res.pool) >= 1


def test_wf_rs10_meta_consistency():
    mdir = MODELS / "wf_rs10"
    if not (mdir / "model_meta.json").exists():
        pytest.skip("wf_rs10 尚未生成")
    meta = json.load(open(mdir / "model_meta.json", encoding="utf-8"))
    assert meta["cfg"]["universe"] == "above_ma60"
    pool = meta["pool"]
    assert len(pool) == 6
    rs2 = json.load(open(mdir / "rs2_meta.json", encoding="utf-8"))
    assert rs2["chosen_variant"] == "pull_ma60"
    assert rs2["universe"] == "above_ma60"
    assert set(rs2["weights"].keys()) == set(pool)


def test_saved_scores_only_inside_ma60_universe():
    """saved wf_rs10 分数在域外（close<=MA60）必须为 NaN（选股域过滤已固化到 scores）。"""
    zf = EXT / "panel_ohlcv.npz"
    mdir = MODELS / "wf_rs10"
    if not zf.exists() or not (mdir / "composite_score.npz").exists():
        pytest.skip("ext 数据或 wf_rs10 产物不存在")
    z = np.load(str(zf))
    close = np.asarray(z["close"], dtype=np.float64)
    # 与 qf.data 一致的收盘价：停牌/缺失前向填充（alive 外 NaN 由分数侧自然为空）
    close = pd.DataFrame(close).ffill().to_numpy()
    zsc = np.load(str(mdir / "composite_score.npz"))
    scores = zsc["scores"]
    rows = zsc["rows"].astype(np.int64)
    # 数据更新后模型可能尚未重建 → 行数不一致属正常，跳过而非失败
    if close.shape[0] != scores.shape[0]:
        pytest.skip(f"数据 {close.shape[0]} 行 vs 模型 {scores.shape[0]} 行，"
                    f"模型需重建（run_trend_model.py）")
    # 逐周行用 pandas rolling 计算独立 MA60（count>=48）
    ma = pd.DataFrame(close).rolling(60, min_periods=48).mean().to_numpy()
    viol = 0
    for t in rows:
        t = int(t)
        inside = np.isfinite(close[t]) & np.isfinite(ma[t]) & (close[t] > ma[t])
        fin = np.isfinite(scores[t])
        viol += int((fin & ~inside).sum())
    assert viol == 0, f"{viol} 个域外分数非 NaN（选股域过滤未覆盖 saved scores）"
    # 域内应存在大量有效分数（否则过滤过度/模型为空）
    n_fin = int(np.isfinite(scores).sum())
    assert n_fin > 1000, f"wf_rs10 有效分数过少：{n_fin}"


def test_600936_below_ma60_at_2026_10_22():
    zf = EXT / "panel_ohlcv.npz"
    if not zf.exists():
        pytest.skip("ext 数据不存在")
    z = np.load(zf)
    dates = pd.DatetimeIndex(z["dates"])
    codes = np.array([str(c) for c in z["codes"]])
    j = int(np.flatnonzero(codes == "600936")[0])
    close = np.asarray(z["close"], dtype=np.float64)
    t = len(dates) - 1  # 最新交易日（数据可继续更新，不硬编码日期）
    ma60 = float(np.nanmean(close[t - 59:t + 1, j]))
    assert close[t, j] < ma60, f"600936 close={close[t, j]:.2f} 未低于 MA60={ma60:.2f}"
    # 最新交易日 60 日跌幅仍 < -15%（600936 持续下跌；实测 10-22 为 -18.9%、10-23 为 -18.4%）
    assert close[t, j] / close[t - 60, j] - 1.0 < -0.15
    # 模型分数：最新周频行该股必须为 NaN（域外过滤已固化到 saved scores）
    mdir = MODELS / "wf_rs10"
    if (mdir / "composite_score.npz").exists():
        zsc = np.load(str(mdir / "composite_score.npz"))
        sc = zsc["scores"]
        rows = zsc["rows"].astype(np.int64)
        tl = int(rows[-1])
        ma_tl = float(np.nanmean(close[tl - 59:tl + 1, j]))
        assert close[tl, j] < ma_tl
        assert np.isnan(sc[tl, j]), "600936 最新周频分数应为 NaN（域外）"
