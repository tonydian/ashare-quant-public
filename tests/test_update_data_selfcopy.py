# -*- coding: utf-8 -*-
"""update_data.py 回归测试：源目录=输出目录的自我拷贝、indices.csv 缺口修复。

背景（2026-10-29 / 10-30 两次"数据更新失败"）：
C.DATA_RAW 自动选行数更多的目录，ext 建好后它就等于 ext；而 --out 默认仍派生到
ext，导致 SRC == out。第 5 步 shutil.copy2(SRC/f, out/f) 变成自我拷贝，Windows
抛 PermissionError[WinError 32] 退出，indices.csv / UPDATE_INFO 没写 → 个股面板
已到最新、指数却停在上周，bench 被 ffill 成"三天零涨跌"。
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
    p = ROOT / "scripts" / "update_data.py"
    spec = importlib.util.spec_from_file_location("_update_data_mod", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 输出目录
def test_default_out_is_inplace_when_src_is_ext():
    m = _mod()
    ext = pathlib.Path("D:/x/data_extract/was_stock_data_ext")
    assert m.default_out_for(ext) == ext, "源已是 ext 时应原地更新，否则会自我拷贝"


def test_default_out_derives_ext_from_original():
    m = _mod()
    base = pathlib.Path("D:/x/data_extract/was_stock_data")
    out = m.default_out_for(base)
    assert out.name == "was_stock_data_ext"
    assert out != base
    # 不能被反复累加后缀
    assert not out.name.endswith("_ext_ext")


def test_no_self_copy_guards_in_source():
    s = (ROOT / "scripts" / "update_data.py").read_text(encoding="utf-8")
    assert "default_out_for" in s
    assert "inplace" in s, "缺原地更新标记"
    # 复制固定文件与 fund 目录都必须先比较 resolve()
    assert "s.resolve() == d.resolve()" in s
    assert "fund_s.resolve() != fund_d.resolve()" in s


# ---------------------------------------------------------------- 指数合并
def _idx_old():
    return pd.DataFrame(
        {"hs300": [1.0, 2.0], "csi500": [1.0, 2.0],
         "csi1000": [1.0, 2.0], "csi_all": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-10-27", "2026-10-28"]))


def test_merge_indices_fills_gap():
    m = _mod()
    d29, d30 = pd.Timestamp("2026-10-29"), pd.Timestamp("2026-10-30")
    idx_new = {k: pd.Series({d29: 3.0, d30: 4.0}) for k in
               ("hs300", "csi500", "csi1000", "csi_all")}
    out, filled = m.merge_indices(_idx_old(), idx_new, [d29, d30])
    assert filled == [d29, d30]
    assert len(out) == 4
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    assert out.loc[d30, "csi_all"] == pytest.approx(4.0)


def test_merge_indices_dedups_existing():
    m = _mod()
    d28, d29 = pd.Timestamp("2026-10-28"), pd.Timestamp("2026-10-29")
    idx_new = {k: pd.Series({d28: 9.9, d29: 3.0}) for k in
               ("hs300", "csi500", "csi1000", "csi_all")}
    out, filled = m.merge_indices(_idx_old(), idx_new, [d28, d29])
    assert filled == [d28, d29]
    assert len(out) == 3, "已存在的日期必须去重"
    assert out.loc[d28, "csi_all"] == pytest.approx(9.9), "重复日保留新值"


def test_merge_indices_skips_incomplete_date():
    """四条指数缺任一条就不补，宁缺勿造。"""
    m = _mod()
    d29, d30 = pd.Timestamp("2026-10-29"), pd.Timestamp("2026-10-30")
    idx_new = {"hs300": pd.Series({d29: 3.0}),
               "csi500": pd.Series({d29: 3.0, d30: 4.0}),
               "csi1000": pd.Series({d29: 3.0, d30: 4.0}),
               "csi_all": pd.Series({d29: 3.0, d30: 4.0})}
    out, filled = m.merge_indices(_idx_old(), idx_new, [d29, d30])
    assert filled == [d29]
    assert d30 not in out.index


def test_merge_indices_empty_when_nothing_to_add():
    m = _mod()
    out, filled = m.merge_indices(_idx_old(), {}, [])
    assert filled == []
    assert len(out) == 2


# ---------------------------------------------------------------- 真实数据自检
def test_ext_indices_covers_panel_tail():
    """现有 ext 目录：indices.csv 必须覆盖面板最后交易日（缺口=bench 被 ffill）。"""
    ext = ROOT.parent / "data_extract" / "was_stock_data_ext"
    panel, idxf = ext / "panel_ohlcv.npz", ext / "indices.csv"
    if not (panel.exists() and idxf.exists()):
        pytest.skip("ext 数据不存在")
    dates = pd.DatetimeIndex(np.load(panel)["dates"])
    idx = pd.read_csv(idxf, index_col=0, parse_dates=True)
    missing = [d for d in dates[-20:] if d not in idx.index]
    assert not missing, f"面板最后 20 日有指数缺失：{[str(d.date()) for d in missing]}"
    tail = idx.reindex(dates[-5:])
    assert tail.notna().all().all(), "面板尾部 5 日存在 NaN 指数"


def test_aux_datasets_not_behind_panel():
    """国债/黄金/ETF 面板必须与个股面板同尾——10-19 vs 11-02 的脱档就是本 bug 的余波。"""
    ext = ROOT.parent / "data_extract" / "was_stock_data_ext"
    panel = ext / "panel_ohlcv.npz"
    if not panel.exists():
        pytest.skip("ext 数据不存在")
    last = pd.Timestamp(np.load(panel)["dates"][-1])
    for name, f, col in (("bond", "bond_000012.parquet", "date"),
                         ("gold", "gold_518880.parquet", "date")):
        p = ext / f
        if not p.exists():
            pytest.skip(f"{f} 不存在")
        got = pd.to_datetime(pd.read_parquet(p)[col]).max()
        assert got >= last, f"{name} 落后个股面板（{got.date()} < {last.date()}）"
    etf = ROOT / "build" / "etf" / "etf_panel.npz"
    if etf.exists():
        etf_last = pd.Timestamp(np.load(etf, allow_pickle=True)["dates"][-1])
        assert etf_last >= last, f"ETF 面板落后（{etf_last.date()} < {last.date()}）"
