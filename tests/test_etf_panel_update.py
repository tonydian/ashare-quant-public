# -*- coding: utf-8 -*-
"""ETF 面板增量合并的回归测试（qfq 基准漂移必须按重叠系数接续，不能直接 concat）。

背景：--update 分支原先算了 existing 却从未使用，等于把历史清空成 NaN；
且面板布局是「2D 矩阵 + codes」，按 close_510300 去取键永远取不到。
直连 concat 也不行：qfq 以最新价为基准，除息后整条历史等比缩放，接缝会假跳变。
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _mod():
    p = ROOT / "scripts" / "build_etf_data.py"
    spec = importlib.util.spec_from_file_location("_etf_mod", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _old(n=60):
    idx = pd.date_range("2026-01-05", periods=n, freq="B")
    base = np.linspace(10.0, 20.0, n)
    return pd.DataFrame({"open": base, "high": base * 1.01, "low": base * 0.99,
                         "close": base, "raw_close": base,
                         "volume": 1e6, "amount": 1e8}, index=idx)


def _new(old, scale, tail=3, shift=1.002):
    """用老面板尾部 5 日 + 3 个新日构造"新抓窗口"，并整体乘 scale（模拟 qfq 重定基）。"""
    idx = list(old.index[-5:])
    last = old.index[-1]
    idx += list(pd.bdate_range(last + pd.Timedelta(days=1), periods=tail))
    vals = list(old["close"].to_numpy()[-5:]) + [old["close"].iloc[-1] * shift] * tail
    vals = [v * scale for v in vals]
    return pd.DataFrame({"date": idx, "close": vals, "raw_close": vals,
                         "open": vals, "high": vals, "low": vals,
                         "volume": 1e6, "amount": 1e8})


def test_merge_rescales_old_history():
    m = _mod()
    old = _old()
    merged = m._merge_on_basis(old, _new(old, scale=1.1), "510300")
    assert merged is not None
    assert len(merged) == len(old) + 3
    # 老历史被缩放到新基准：首日 10 → 11
    assert merged["close"].iloc[0] == pytest.approx(old["close"].iloc[0] * 1.1, rel=1e-9)
    # 接缝处无跳变：重叠日收益率的量级与历史一致
    r = merged["close"].pct_change().dropna()
    assert r.abs().max() < 0.05, f"接缝跳变：max |r|={r.abs().max():.3f}"
    assert not merged["close"].isna().any()


def test_merge_keeps_raw_close_unscaled():
    m = _mod()
    old = _old()
    merged = m._merge_on_basis(old, _new(old, scale=1.1), "510300")
    # raw_close（不复权）不能跟着缩放
    assert merged["raw_close"].iloc[0] == pytest.approx(old["raw_close"].iloc[0], rel=1e-9)


def test_merge_rejects_unstable_basis():
    m = _mod()
    old = _old()
    new = _new(old, scale=1.0)
    new.loc[:2, "close"] = new.loc[:2, "close"] * 1.3   # 部分重叠日不一致 → 基准不稳
    assert m._merge_on_basis(old, new, "510300") is None


def test_merge_needs_overlap():
    m = _mod()
    old = _old()
    new = _new(old, scale=1.0).iloc[-3:].copy()          # 完全不重叠
    assert m._merge_on_basis(old, new, "510300") is None


def test_panel_layout_is_matrix_with_codes():
    """面板必须是 2D 矩阵 + codes；若退回 close_510300 这类键，读取逻辑会静默清空历史。"""
    p = ROOT / "build" / "etf" / "etf_panel.npz"
    if not p.exists():
        pytest.skip("ETF 面板不存在")
    z = np.load(p, allow_pickle=True)
    assert "codes" in z.files
    assert z["close"].ndim == 2
    assert z["close"].shape[1] == len(z["codes"])
    assert not any(f.startswith("close_") for f in z.files)
