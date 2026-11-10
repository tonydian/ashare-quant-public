# -*- coding: utf-8 -*-
"""滑点校准工具的口径测试（`scripts/fill_log.py`）。

钉三件事：
1. 滑点的**符号**（正数 = 吃亏），买/卖方向不能写反；
2. 年化换算（单笔 bp × 年双边换手），这是唯一与配置直接对话的数字；
3. 模板里的价格必须是**真实可下单**价，不是后复权价 —— 否则填进去没法核对。
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import fill_log as fl      # noqa: E402


def _row(side, open_px, filled, close=None):
    return {"side": side, "open_px": open_px, "filled_px": filled,
            "ref_close": close if close is not None else open_px}


def test_slip_sign_buy():
    """买入成交价高于开盘 = 吃亏 = 正滑点。"""
    assert fl._slip(_row("买入", 10.0, 10.04)) == pytest.approx(40.0, abs=1e-6)
    assert fl._slip(_row("买入", 10.0, 9.96)) == pytest.approx(-40.0, abs=1e-6)


def test_slip_sign_sell():
    """卖出成交价低于开盘 = 吃亏 = 正滑点（方向不能和买入写成一样）。"""
    assert fl._slip(_row("卖出", 10.0, 9.96)) == pytest.approx(40.0, abs=1e-6)
    assert fl._slip(_row("卖出", 10.0, 10.04)) == pytest.approx(-40.0, abs=1e-6)


def test_slip_missing_price_is_nan():
    assert np.isnan(fl._slip(_row("买入", np.nan, 10.0)))
    assert np.isnan(fl._slip(_row("买入", 0.0, 10.0)))


def test_total_includes_overnight_gap():
    """相对信号日收盘的执行总成本，含隔夜跳空，与相对开盘的滑点不同。"""
    r = {"side": "买入", "open_px": 10.10, "filled_px": 10.15, "ref_close": 10.00}
    assert fl._total(r) == pytest.approx(150.0, abs=1e-6)
    assert fl._slip(r) == pytest.approx(10.15 / 10.10 * 1e4 - 1e4, abs=1e-6)


def _write(tmp_path, rows):
    d = tmp_path / "fills_2026-11-06.csv"
    pd.DataFrame(rows, columns=fl.COLS).to_csv(d, index=False, encoding="utf-8-sig")
    return d


def _mkrows(pairs):
    out = []
    for side, op, f in pairs:
        out.append({"signal_date": "2026-11-05", "trade_date": "2026-11-06",
                    "code": "600000", "name": "x", "kind": "stock", "side": side,
                    "plan_shares": 100, "unit": "股", "ref_close": op,
                    "open_px": op, "filled_px": f, "filled_shares": 100, "note": ""})
    return out


def test_report_mean_and_annualisation(tmp_path, monkeypatch, capsys):
    """10 笔都在开盘价上方 0.20% → 均值 20bp；年化 = 20bp × 换手 1.5139 = 0.303%。"""
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    _write(tmp_path, _mkrows([("买入", 10.0, 10.02)] * 10))

    fl.cmd_report(type("A", (), {"turnover": 1.5139344571299536})())
    out = capsys.readouterr().out

    assert "均值        +20.00 bp" in out or "+20.00 bp" in out
    assert "0.303%" in out, out
    assert "n=10" in out
    assert "n=10 < 20" in out, "样本不足 20 笔必须警告，不能让人直接改配置"


def test_report_refuses_when_nothing_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    _write(tmp_path, _mkrows([("买入", 10.0, "")]))
    with pytest.raises(SystemExit):
        fl.cmd_report(type("A", (), {"turnover": 1.5})())


def test_report_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    with pytest.raises(SystemExit):
        fl.cmd_report(type("A", (), {"turnover": 1.5})())


def test_new_template_uses_real_price_space(tmp_path, monkeypatch):
    """模板里的 ref_close 必须等于信号价（真实下单价），不能是后复权价。"""
    import json

    sig = json.loads((ROOT / "signals" / "latest_combo.json").read_text(encoding="utf-8"))
    main = (sig.get("main") or {}).get("targets") or []
    if not main:
        pytest.skip("当前信号没有主策略持仓")

    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    fl.cmd_new(type("A", (), {"date": "2026-11-06", "force": True})())

    files = list(tmp_path.glob("fills_*.csv"))
    assert files, "没生成模板"
    df = pd.read_csv(files[0], dtype={"code": str})
    assert list(df.columns) == fl.COLS
    assert len(df) >= len(main)

    px = {str(t["code"]).zfill(6): float(t["price"]) for t in main}
    checked = 0
    for r in df.itertuples():
        if r.code in px and np.isfinite(r.ref_close):
            assert r.ref_close == pytest.approx(px[r.code], rel=1e-4), \
                f"{r.code} 模板价 {r.ref_close} 与信号价 {px[r.code]} 不一致（后复权没换算）"
            checked += 1
    assert checked >= 10, f"只核对了 {checked} 只，样本太少"


def test_new_template_shares_match_signal(tmp_path, monkeypatch):
    import json

    sig = json.loads((ROOT / "signals" / "latest_combo.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    fl.cmd_new(type("A", (), {"date": "2026-11-06", "force": True})())
    df = pd.read_csv(list(tmp_path.glob("fills_*.csv"))[0], dtype={"code": str})

    want = {}
    for key in ("main", "lowturn", "etf", "cb"):
        seg = sig.get(key) or {}
        for x in (seg.get("targets") or seg.get("legs") or []):
            if int(x.get("shares") or 0) > 0:
                want[str(x["code"]).zfill(6)] = int(x["shares"])
    assert want, "信号里没有可下单标的"
    got = dict(zip(df["code"], df["plan_shares"]))
    for code, sh in want.items():
        assert got.get(code) == sh, f"{code} 计划股数 {got.get(code)} != 信号 {sh}"


def test_annualisation_sign_is_positive_when_measured_below_assumption(tmp_path, monkeypatch, capsys):
    """实测 0bp < 假设 40bp → 真实成本更低 → 收益应**上调**，符号不能写反。"""
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    _write(tmp_path, _mkrows([("买入", 10.0, 10.0)] * 10))     # 零滑点

    fl.cmd_report(type("A", (), {"turnover": 1.5139344571299536})())
    out = capsys.readouterr().out
    assert "年化收益 +0.606pp" in out, out
    assert "年化收益 -" not in out


# ================================================================ 执行闭环（2026-11-08 新增）
#
# 背景：170 篇报告 / 129 条运行记录 / **0 笔真实成交**。补闭环时加了三个命令，
# 每个都对应一个具体的失败模式，测试守住它们各自的红线。

from types import SimpleNamespace as NS      # noqa: E402


def _w(dirp, name, rows):
    p = dirp / name
    pd.DataFrame(rows, columns=fl.COLS).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def _r(code="600000", leg="main", kind="stock", shares=100, ref=10.0, op=10.2):
    unit = {"stock": "股", "cb": "张", "etf": "份"}[kind]
    return {"signal_date": "2026-11-06", "trade_date": "2026-11-09", "code": code,
            "name": "测试", "kind": kind, "leg": leg, "side": "买入",
            "plan_shares": shares, "unit": unit, "ref_close": ref,
            "open_px": op, "filled_px": "", "filled_shares": "", "note": ""}


def test_order_list_filter_keeps_only_requested():
    """分批建仓：--legs cb 不能把别的腿带进来，也不能漏掉 cb。"""
    _, allrows = fl.order_list()
    _, only_cb = fl.order_list(["cb"])
    assert only_cb, "过滤后空了"
    assert {r["leg"] for r in only_cb} == {"cb"}
    assert len(only_cb) < len(allrows)


def test_order_list_filter_order_independent():
    _, a = fl.order_list(["lowturn", "cb"])
    _, b = fl.order_list(["cb", "lowturn"])
    assert [r["code"] for r in a] == [r["code"] for r in b]


def test_order_list_skips_zero_shares(tmp_path, monkeypatch):
    import json

    sig = tmp_path / "sig.json"
    sig.write_text(json.dumps({"signal_date": "2026-11-06", "main": {"targets": [
        {"code": "600000", "name": "a", "shares": 0, "price": 10.0},
        {"code": "600001", "name": "b", "shares": 200, "price": 10.0}]}}),
        encoding="utf-8")
    monkeypatch.setattr(fl, "SIG", sig)
    _, rows = fl.order_list()
    assert [r["code"] for r in rows] == ["600001"]


def test_find_files_uses_prefix(tmp_path, monkeypatch):
    """分批模板叫 fills_<日期>_<腿组>.csv；精确匹配会漏掉它们。"""
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    _w(tmp_path, "fills_2026-11-09_lowturn-cb.csv", [_r(leg="lowturn")])
    _w(tmp_path, "fills_2026-11-09_main.csv", [_r()])
    _w(tmp_path, "fills_2026-11-10_main.csv", [_r()])
    got = fl._find_files("2026-11-09")
    assert len(got) == 2 and all("11-10" not in f.name for f in got)


def test_fill_never_overwrites_measured(tmp_path, monkeypatch):
    """已有真实成交价的行必须原样保留 —— 一次手滑让校准永久失真，不可逆。"""
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    p = _w(tmp_path, "fills_2026-11-09_main.csv", [_r(), _r(code="600001", op=5.0)])
    df = pd.read_csv(p, dtype={"code": str})
    df.loc[0, "filled_px"] = 10.33
    df.loc[0, "filled_shares"] = 100
    df.to_csv(p, index=False, encoding="utf-8-sig")

    fl.cmd_fill(NS(date="2026-11-09", mode="open", force=False))
    df = pd.read_csv(p, dtype={"code": str})
    assert float(df.loc[0, "filled_px"]) == pytest.approx(10.33)
    assert float(df.loc[1, "filled_px"]) == pytest.approx(5.0)


def test_fill_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    p = _w(tmp_path, "fills_2026-11-09_main.csv", [_r(op=10.2)])
    df = pd.read_csv(p, dtype={"code": str})
    df.loc[0, "filled_px"] = 99.0
    df.to_csv(p, index=False, encoding="utf-8-sig")

    fl.cmd_fill(NS(date="2026-11-09", mode="open", force=True))
    assert float(pd.read_csv(p, dtype={"code": str}).loc[0, "filled_px"]) == pytest.approx(10.2)


def test_fill_marks_assumed(tmp_path, monkeypatch):
    """回填必须打标记，否则 report 会把假设成交当实测。"""
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    p = _w(tmp_path, "fills_2026-11-09_main.csv", [_r()])
    fl.cmd_fill(NS(date="2026-11-09", mode="open", force=False))
    assert fl.ASSUMED_TAG in str(pd.read_csv(p, dtype={"code": str}).loc[0, "note"])


def test_fill_skips_missing_price(tmp_path, monkeypatch):
    """交易日未到时开盘价是空的，不能填 0 冒充成交。"""
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    p = _w(tmp_path, "fills_2026-11-09_main.csv", [_r(op=np.nan)])
    fl.cmd_fill(NS(date="2026-11-09", mode="open", force=False))
    df = pd.read_csv(p, dtype={"code": str})
    assert not (pd.to_numeric(df["filled_px"], errors="coerce").fillna(0) > 0).any()


def test_fill_no_files_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    with pytest.raises(SystemExit):
        fl.cmd_fill(NS(date="1999-01-01", mode="open", force=False))


def test_status_shows_unfilled(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    _w(tmp_path, "fills_2026-11-09_main.csv", [_r()])
    fl.cmd_status(NS())
    out = capsys.readouterr().out
    assert "未执行" in out
    assert "fills_2026-11-09_main.csv" in out


def test_report_refuses_when_only_assumed(tmp_path, monkeypatch, capsys):
    """只有假设成交时必须拒绝出结论，不能报「滑点 0bp」。"""
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    _w(tmp_path, "fills_2026-11-09_main.csv", [_r()])
    fl.cmd_fill(NS(date="2026-11-09", mode="open", force=False))

    class Boom:
        def read_text(self, *a, **k):
            raise AssertionError("没有实测数据时不该读 metrics 算年化")

    monkeypatch.setattr(fl, "METRICS", Boom())
    fl.cmd_report(NS(turnover=None))
    out = capsys.readouterr().out
    assert "滑点无法校准" in out
    assert "假设成交 1 笔" in out


def test_list_outputs_limit_price(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    _w(tmp_path, "fills_2026-11-09_main.csv", [_r(ref=10.0)])
    fl.cmd_list(NS(date="2026-11-09", limit_bp=200.0, min_fee=5.0))
    out = capsys.readouterr().out
    assert "600000" in out and "建议限价" in out
    assert "10.20" in out, "10.0 × 1.02 应为 10.20"


def test_list_min_fee_affects_bp(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fl, "FILL_DIR", tmp_path)
    _w(tmp_path, "fills_2026-11-09_main.csv", [_r(ref=10.0)])
    fl.cmd_list(NS(date="2026-11-09", limit_bp=200.0, min_fee=0.1))
    cheap = capsys.readouterr().out
    fl.cmd_list(NS(date="2026-11-09", limit_bp=200.0, min_fee=5.0))
    assert cheap != capsys.readouterr().out
