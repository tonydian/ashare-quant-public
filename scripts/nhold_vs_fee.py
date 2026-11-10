# -*- coding: utf-8 -*-
"""在真实费率下，测试不同持仓数的表现。

核心矛盾：
  · 持仓越多 → 单笔越小 → 最低 5 元佣金占比越高 → 成本越高
  · 持仓越多 → 分散越好 → 波动越低

  需要找到最优持仓数（在真实费率下）。

费率：
  佣金 万0.1，最低 5 元
  印花税 5bp（卖出）
  过户费 万0.1（双向）
  滑点：按市值映射（15bp 基准 + 小市值加成）

方法：
  用 run_backtest 的 slippage_bp 参数，
  把"佣金+印花税+过户"的额外部分折算进 slippage_bp。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, metrics as qfm_metrics  # noqa: E402

OUT = ROOT / "build" / "eval" / "nhold_vs_fee.json"
START, END = "2020-01-08", "2026-12-31"
CAP = 300000.0
MIN_COMM = 5.0
COMM_RATE = 0.00001     # 万0.1

# 滑点（按持仓市值估算，取中位数档 40bp；保守）
SLIP_BASE = 40.0


def effective_comm_bp(n_hold: float) -> float:
    """单笔实际佣金率（bp），考虑 5 元最低。"""
    per = CAP / max(n_hold, 1)
    by_rate = per * COMM_RATE
    actual = max(MIN_COMM, by_rate)
    return actual / per * 1e4


def total_roundtrip_bp(n_hold: float) -> dict:
    comm = effective_comm_bp(n_hold)
    return {
        "commission": comm * 2,      # 买卖各一次
        "stamp": C.STAMP_TAX_SELL * 1e4,
        "transfer": C.COMMISSION_RATE * 0 + 0.2,   # 过户费万0.1双向 ≈ 0.2bp
        "slippage": SLIP_BASE * 2,
    }


def main():
    print("=" * 92)
    print("真实费率下：持仓数 vs 收益")
    print("=" * 92)
    print(f"  本金 {CAP:,.0f} | 佣金万0.1(最低5元) | 印花5bp | 滑点{SLIP_BASE:.0f}bp/边")

    md = qfdata.load_market(verbose=False)
    zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
    phase = int(zm["rows"][0]) % 5
    cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)
    res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
    sc = np.asarray(res.scores)
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])

    start = int(md.date_index(START))
    while start % 5 != phase:
        start += 1
    end = int(md.date_index(END))
    offset = (phase - start) % 10

    NS = [10, 15, 20, 30, 40]
    out = {"slip_base": SLIP_BASE, "results": {}}

    print("\n" + "=" * 92)
    print(f"{'持仓':>5}{'单笔金额':>11}{'佣金bp':>9}{'往返bp':>9}"
          f"{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>8}{'年化成本':>10}")
    print("-" * 92)

    for n in NS:
        tb = total_roundtrip_bp(n)
        rt = sum(tb.values())

        # 把额外成本折算成滑点：run_backtest 内部已含 commission/stamp
        # 我们用 slippage_bp 注入"差额"
        # 默认 engine 用 COMMISSION_RATE=2.5bp、STAMP=5bp、TRANSFER
        # 真实佣金 = tb["commission"]/2 单边
        cur_comm_side = C.COMMISSION_RATE * 1e4          # 2.5bp
        real_comm_side = tb["commission"] / 2            # 实际单边
        delta_comm = real_comm_side - cur_comm_side      # 需要补的差额

        # 额外滑点 = 基础滑点 + 佣金差额
        slip_arg = SLIP_BASE + max(delta_comm, 0)

        old = C.N_HOLDINGS
        C.N_HOLDINGS = n
        try:
            r = qfbt.run_backtest(
                md, sc, start, end, risk_overlay=True, stop_loss=True,
                rebalance_days=10, rebalance_offset=offset,
                trade_band_rel=0.30, trade_band_abs=6000.0,
                max_new_entries=6, cash_asset_returns=cash,
                slippage_bp=float(slip_arg), verbose=False)
        finally:
            C.N_HOLDINGS = old

        sub = r.nav[r.nav.index >= START]
        p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)

        tr = r.trades
        yrs = len(sub) / 252
        amt = float(tr["amount"].sum())
        avg_nav = float(sub.mean())
        to = amt / avg_nav / yrs
        ann_cost = to * rt * 1e-4

        out["results"][str(n)] = {
            "cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
            "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
            "turnover": float(to), "roundtrip_bp": float(rt),
            "comm_bp": float(tb["commission"] / 2),
            "slip_arg": float(slip_arg),
            "ann_cost": float(ann_cost),
            "n_trades": int(len(tr)),
        }
        print(f"{n:>5}{CAP/n:>10,.0f}{tb['commission']/2:>8.2f}"
              f"{rt:>8.0f}{p['cagr']:>9.2%}{p['sharpe']:>9.3f}"
              f"{p['max_dd']:>9.2%}{p['calmar']:>8.2f}{ann_cost:>9.2%}")

    # ---- 最优
    print("\n" + "=" * 92)
    print("最优持仓数（按 Calmar 排序）")
    print("=" * 92)
    ranked = sorted(out["results"].items(),
                    key=lambda kv: -kv[1]["calmar"])
    print(f"  {'排名':>4}{'持仓':>6}{'Calmar':>9}{'Sharpe':>9}{'CAGR':>9}{'年化成本':>10}")
    print("-" * 50)
    for i, (n, v) in enumerate(ranked, 1):
        print(f"  {i:>4}{n:>6}{v['calmar']:>9.2f}{v['sharpe']:>9.3f}"
              f"{v['cagr']:>9.2%}{v['ann_cost']:>9.2%}")

    best = ranked[0]
    print(f"\n  → 最优: {best[0]} 只（Calmar {best[1]['calmar']:.2f}）")
    out["best_n"] = int(best[0])

    # ---- 对比：40 只 vs 最优
    if "40" in out["results"]:
        cur = out["results"]["40"]
        print(f"\n  40 只（当前）: Calmar {cur['calmar']:.2f}, "
              f"CAGR {cur['cagr']:.2%}, 年化成本 {cur['ann_cost']:.2%}")
        print(f"  {best[0]} 只（最优）: Calmar {best[1]['calmar']:.2f}, "
              f"CAGR {best[1]['cagr']:.2%}, 年化成本 {best[1]['ann_cost']:.2%}")
        print(f"  → 成本节省 {cur['ann_cost']-best[1]['ann_cost']:+.2%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
