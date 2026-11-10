# -*- coding: utf-8 -*-
"""回测口径与今日推荐的回归测试（对应 2026-11-02 审计发现的三个问题）。

这些断言都很「浅」，但每一条都对应一个真实发生过的错：
  * 个股回测默认调仓 10 而非生产 40 → 界面 6.13% 对不上报告 7.11%
  * 起点没做相位对齐 → 初始建仓日跟着用户选的日期走，净值路径分叉
  * 低换手成本面板 delta=0/ann_cost=0 → 实际扣了 4.01pp 却显示「没扣费」
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "web" / "services.py").read_text(encoding="utf-8")
DAILY = (ROOT / "scripts" / "daily_signal.py").read_text(encoding="utf-8")
LOWTURN = (ROOT / "qf" / "lowturn.py").read_text(encoding="utf-8")


def test_stock_backtest_default_rebalance_is_production_40():
    """`_bt_stock` 的兜底调仓周期必须是 40（production `run_backtest.py` 默认值）。"""
    seg = SRC.split("def _bt_stock(")[1].split("def _bt_etf(")[0]
    assert 'p.get("rebalance_days") or 40' in seg, "默认调仓周期又变回 10 了"
    assert 'p.get("rebalance_days") or 10' not in seg


def test_stock_backtest_aligns_start_to_score_phase():
    """起点要推到打分相位日，否则初始建仓日与生产不可比。"""
    seg = SRC.split("def _bt_stock(")[1].split("def _bt_etf(")[0]
    assert "while start_row % 5 != phase" in seg, "缺少起点相位对齐"
    # 相位对齐必须发生在算 offset 之前
    assert seg.index("while start_row % 5 != phase") < seg.index("offset = (phase - start_row)")


def test_lowturn_cost_panel_is_double_book():
    """低换手是双账本：不能报 delta=0，ann_cost 也不能把双边换手再乘 2。"""
    seg = SRC.split("def _bt_lowturn(")[1].split("def _combo_inputs(")[0]
    assert '"basis": "double_book"' in seg
    assert '"delta_bp": None' in seg, "delta_bp 不能填 0（会显示成「没扣费」）"
    assert "tur * 2 * target" not in seg, "turnover 已是双边成交额，不能再 ×2"
    assert "tur * target / 1e4" in seg


def test_lowturn_gross_net_follow_selected_range():
    """毛/净两个数字要跟所选区间走，不能一直报全程。"""
    seg = SRC.split("def _bt_lowturn(")[1].split("def _combo_inputs(")[0]
    assert "def _seg(" in seg or "gross[gross.index >= start]" in seg


def test_cb_cost_applied_after_range_cut():
    """转债先截区间再扣费，否则 factor 会从 2019 累积、区间内 drag 偏小。"""
    seg = SRC.split("def _bt_cb(")[1].split("def _lowturn_bt(")[0]
    assert "raw = nav[band]" in seg
    assert "raw = nav\n" not in seg


def test_today_recommend_supports_lowturn():
    seg = SRC.split("def today_recommend(")[1].split("def today_export_csv(")[0]
    assert '"lowturn"' in seg, "今日推荐没接低换手"
    assert '"kind": "lowturn"' in seg
    # 分组循环必须覆盖所有腿（含 2026-11-05 新增的 cb）：无标的的侧也要出卡片。
    # 不写死整条元组，只断言"这几个 key 必须在里面"，以后加腿不会误报。
    import re
    m = re.search(r'for key in \(([^)]+)\)', seg)
    assert m, "today_recommend 里找不到分组循环"
    keys = set(re.findall(r'"(\w+)"', m.group(1)))
    missing = {"main", "lowturn", "etf", "cb"} - keys
    assert not missing, f"分组循环漏了 {missing}（现有 {sorted(keys)}）"
    assert 'r["kind"] in ("stock", "lowturn")' in seg, "n_stock 要把低换手算进个股数"


def test_daily_signal_has_four_strategy_mode():
    assert "--four" in DAILY
    assert "def _lowturn_signal(" in DAILY
    assert '"lowturn": 0.30' in DAILY, "四策略默认权重里没有低换手"
    # 章节号要按实际侧数编号，不能写死「三/四」
    assert 'lines += ["", "## 三、板块 ETF 轮动侧目标", ""]' not in DAILY


def test_month_role_moved_to_qf():
    """月度时点判定必须只有一份实现（三处共用），脚本里只做委托。"""
    assert "def month_role(" in LOWTURN
    seg = (ROOT / "scripts" / "lowturn_signal.py").read_text(encoding="utf-8")
    assert "return lt.month_role(dates, t)" in seg


def test_today_recommend_rows_shape():
    """真实跑一次：今日推荐里的低换手行字段齐全、金额为正、可下整手。"""
    from web.services import DataService

    d = DataService().today_recommend()
    lt_rows = [r for r in (d.get("rows") or []) if r.get("kind") == "lowturn"]
    meta = d.get("lowturn")
    if not meta:
        return                     # 老 combo 文件没有低换手侧，跳过
    if not lt_rows:
        assert d["idle_notes"] and any("低换手" in n for n in d["idle_notes"])
        return
    for r in lt_rows:
        assert r["code"] and len(str(r["code"])) == 6
        assert r["name"]
        assert r["unit"] == "股"
        assert (r["shares"] or 0) > 0 and (r["shares"] or 0) % 100 == 0
        assert (r["amount"] or 0) > 0
        assert "换手" in (r["reason"] or "")
    assert d["n_stock"] >= len(lt_rows)
