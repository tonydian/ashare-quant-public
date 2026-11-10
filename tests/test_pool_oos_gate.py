# -*- coding: utf-8 -*-
"""选池第三判据「DEV→OOS 不衰减」（2026-11-03 加）。

判据要解决的问题：只按 DEV |ICIR| 排序选池，会选中「越往后越弱」的因子
（ep 0.0436→0.0221、ROE 0.0467→0.0072），而 bp 是唯一 OOS 反超 DEV 的。

两条硬约束（本测试守的就是这两条）：
1. 衰减因子必须被挡在池外，持续有效的因子不能误伤；
2. 衰减检验窗**不准越过 DEV_END**——越过去就等于用回测窗口挑因子（前视）。
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

from qf import config as C  # noqa: E402
from qf import model as qfm  # noqa: E402

DEV_N = 69          # 0..68 视为 DEV
BASE_N = 45         # 0..44 基线窗
T, N = 120, 420


def _fake_market():
    rng = np.random.default_rng(7)
    industry = np.tile(np.arange(N) % 6, (T, 1)).astype(np.int32)
    logmcap = rng.normal(10.0, 0.8, size=(T, N)).astype(np.float32)
    eligible = np.ones((T, N), dtype=bool)
    dates = pd.bdate_range("2017-01-02", periods=T)
    return SimpleNamespace(industry=industry, logmcap=logmcap, eligible=eligible,
                           buyable=eligible.copy(), N=N, T=T, dates=pd.DatetimeIndex(dates))


def _catalog(keys):
    return pd.DataFrame({"factor_key": keys, "category_en": ["Pattern"] * len(keys),
                         "lib": ["synthetic"] * len(keys), "base": ["x"] * len(keys)})


def _panels_and_label():
    rng = np.random.default_rng(11)
    label = rng.normal(size=(T, N))
    keep = label + rng.normal(0, 0.5, size=(T, N))          # 全程有效
    decay = label.copy()
    decay[BASE_N:DEV_N] = rng.normal(0, 1.0, size=(DEV_N - BASE_N, N))  # 只在基线窗有效
    noise = rng.normal(size=(T, N))                          # 全程无效
    panels = {"synthetic:A_keep": keep, "synthetic:B_decay": decay,
              "synthetic:C_noise": noise}
    return panels, label


def _windows(rows, label, mask):
    base = rows < BASE_N
    oos = (rows >= BASE_N) & (rows < DEV_N)
    return {"oos": (rows[oos], {}, label[oos], mask[oos]),
            "base": (rows[base], {}, label[base], mask[base])}


def _run(cfg, with_oos: bool):
    md = _fake_market()
    panels, label = _panels_and_label()
    rows = np.arange(T)
    mask = md.eligible.copy()
    dev = rows < DEV_N
    kwargs = {}
    if with_oos:
        win = _windows(rows, label, mask)
        # 只给 DEV 期的面板切片
        win["oos"] = (win["oos"][0], {k: v[win["oos"][0]] for k, v in panels.items()},
                      win["oos"][2], win["oos"][3])
        win["base"] = (win["base"][0], {k: v[win["base"][0]] for k, v in panels.items()},
                       win["base"][2], win["base"][3])
        kwargs["oos"] = win
    pool, meta = qfm.select_pool(
        md, rows[dev], {k: v[dev] for k, v in panels.items()}, label[dev], mask[dev],
        cfg, _catalog(list(panels)), verbose=False, **kwargs)
    return pool, meta


def test_oos_gate_drops_decayed_factor():
    """衰减因子必须出局，持续有效的因子必须留下。"""
    cfg = qfm.ModelConfig(max_factors=3, max_per_category=3, min_ic_obs=10,
                          pool_guard=False, require_oos_nondecay=True, oos_decay_tol=0.0)
    pool, meta = _run(cfg, with_oos=True)
    assert "synthetic:A_keep" in pool, f"持续因子被误伤：{pool}"
    assert "synthetic:B_decay" not in pool, f"衰减因子没被挡住：{pool}"
    # 判据列必须落到 pool_meta 里，供报告溯源
    assert {"oos_icir", "oos_base_icir"} <= set(meta.columns)
    for k in pool:
        r = meta.loc[k]
        assert abs(r["oos_icir"]) >= abs(r["oos_base_icir"])


def test_oos_gate_off_is_backward_compatible():
    """默认关闭时，即使传了 oos 窗口也必须与不传逐项相同（生产零影响）。"""
    cfg_off = qfm.ModelConfig(max_factors=3, max_per_category=3, min_ic_obs=10,
                              pool_guard=False)
    pool_plain, _ = _run(cfg_off, with_oos=False)
    pool_with_arg, _ = _run(cfg_off, with_oos=True)
    assert pool_plain == pool_with_arg
    # 关闭时不应多出判据列
    _, meta = _run(cfg_off, with_oos=False)
    assert "oos_icir" not in meta.columns


def test_oos_window_cannot_enter_backtest_period():
    """衰减检验窗越过 DEV_END 必须直接报错——否则就是前视。"""
    md = _fake_market()
    panels, label = _panels_and_label()
    md.fwd_open_ret = lambda h: label        # weekly_label 需要它
    rows = np.arange(T)
    cfg = qfm.ModelConfig(max_factors=3, max_per_category=3, min_ic_obs=10,
                          pool_guard=False, require_oos_nondecay=True,
                          oos_window_start=str(pd.Timestamp(C.DEV_END).date()),
                          oos_window_end=str(pd.Timestamp(C.DEV_END).date()))
    # 把 DEV_END 之前的窗口强行改成之后：构造一个越界的起点
    cfg.oos_window_start = str((pd.Timestamp(C.DEV_END) + pd.Timedelta(days=1)).date())
    with pytest.raises(ValueError, match="前视"):
        qfm.walk_forward(md, rows, panels, _catalog(list(panels)), cfg=cfg,
                         start_date="2017-01-02", verbose=False)
