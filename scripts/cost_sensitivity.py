# -*- coding: utf-8 -*-
"""成本敏感性分析：滑点从 0 到 60bp，看各策略还能不能赚钱。

背景：
  当前回测假设 SLIPPAGE_BP=15（单边），往返成本约 40bp。
  但主策略持仓偏小市值（60~120 亿），盘口薄，真实滑点可能 30~50bp。
  如果成本被低估，回测的 7.3% 可能是虚的。

方法：
  1. 用同一套代码、同一份数据，只改 slippage_bp
  2. 扫描 0 / 5 / 10 / 15 / 20 / 25 / 30 / 40 / 50 / 60 bp
  3. 找出"成本临界点"：Sharpe 跌破 0.3 / CAGR 归零 时的滑点
  4. 同时算：换手率、年化成本占比（验证成本是否敏感）

严格说明：
  · 这不是调参，是压力测试（不改任何策略逻辑）
  · 主策略的因子池/权重全部固定
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, metrics as qfm_metrics  # noqa: E402

OUT = ROOT / "build" / "eval" / "cost_sensitivity.json"
SLIPS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]
START, END = "2020-01-08", "2026-12-31"


def main():
    print("=" * 94)
    print("成本敏感性分析")
    print("=" * 94)
    print(f"  当前假设: SLIPPAGE_BP={C.SLIPPAGE_BP}（单边）")
    print(f"    佣金 {C.COMMISSION_RATE*1e4:.1f}bp（双边）+ 印花税 {C.STAMP_TAX_SELL*1e4:.0f}bp（卖出）")
    print(f"    往返成本 ≈ {(C.SLIPPAGE_BP*2 + C.COMMISSION_RATE*1e4*2 + C.STAMP_TAX_SELL*1e4):.1f} bp")
    print(f"  扫描: {SLIPS}")

    md = qfdata.load_market(verbose=False)
    zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
    rows = zm["rows"].astype(np.int64)
    phase = int(rows[0]) % 5
    cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)

    res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
    sc = np.asarray(res.scores)
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    elif sc.shape[0] > md.T:
        sc = sc[:md.T]
    print(f"\n  模型: wf_h10，池 {len(res.pool)} 个因子，scores {sc.shape}")
    print(f"  数据: {md.T} 日 × {md.N} 只")

    start = int(md.date_index(START))
    while start % 5 != phase:
        start += 1
    end = int(md.date_index(END))
    offset = (phase - start) % 40

    out = {"slips": SLIPS, "results": {}, "note": "仅改 slippage，策略逻辑不变"}
    navs = {}

    print("\n" + "=" * 94)
    print(f"{'滑点':>6}{'往返成本':>10}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>10}"
          f"{'Calmar':>8}{'年化成本':>10}")
    print("-" * 94)

    for slip in SLIPS:
        old = C.N_HOLDINGS
        C.N_HOLDINGS = 40
        try:
            r = qfbt.run_backtest(
                md, sc, start, end, risk_overlay=True, stop_loss=True,
                rebalance_days=40, rebalance_offset=offset,
                trade_band_rel=0.30, trade_band_abs=6000.0,
                max_new_entries=6, cash_asset_returns=cash,
                slippage_bp=float(slip), verbose=False)
        finally:
            C.N_HOLDINGS = old

        sub = r.nav[r.nav.index >= START]
        p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)

        # 成本占比 = 累计成本 / 累计收益（简单口径）
        cost = float(getattr(r, "cost_total", 0.0) or 0.0)
        tr = getattr(r, "trades", None)
        n_tr = len(tr) if tr is not None else 0
        years = max(len(sub) / 252, 0.1)
        # 年化成本（相对初始净值 30 万）
        cap = 300000.0
        ann_cost = (cost / cap) / years if cap > 0 else 0.0

        rt_bp = slip * 2 + C.COMMISSION_RATE * 1e4 * 2 + C.STAMP_TAX_SELL * 1e4
        rec = {
            "cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
            "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
            "n_trades": n_tr, "cost_total": cost,
            "ann_cost_pct": float(ann_cost), "roundtrip_bp": float(rt_bp),
        }
        out["results"][str(slip)] = rec
        navs[slip] = sub

        print(f"{slip:>4}bp{rt_bp:>9.0f}bp{p['cagr']:>10.2%}{p['sharpe']:>9.3f}"
              f"{p['max_dd']:>10.2%}{p['calmar']:>8.2f}{ann_cost:>9.2%}")

    # ---- 换手率
    print("\n" + "=" * 94)
    print("换手率分析（用 15bp 基准的成交记录）")
    print("=" * 94)
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r15 = qfbt.run_backtest(
            md, sc, start, end, risk_overlay=True, stop_loss=True,
            rebalance_days=40, rebalance_offset=offset,
            trade_band_rel=0.30, trade_band_abs=6000.0,
            max_new_entries=6, cash_asset_returns=cash,
            slippage_bp=15.0, verbose=False)
    finally:
        C.N_HOLDINGS = old

    tr = getattr(r15, "trades", None)
    if tr is not None and len(tr) > 0:
        amt = float(tr["amount"].sum())
        sub0 = r15.nav[r15.nav.index >= START]
        yrs = len(sub0) / 252
        avg_nav = float(sub0.mean())
        turnover = amt / avg_nav / yrs
        print(f"  成交笔数: {len(tr)}")
        print(f"  总成交额: {amt/1e8:.2f} 亿")
        print(f"  平均净值: {avg_nav/1e4:.1f} 万")
        print(f"  **年换手率: {turnover:.2f}x**")
        print(f"  → 每 1bp 滑点 = 年化 {(turnover*2*1e-4):.3%} 成本")
        out["turnover"] = float(turnover)
        out["n_trades_15bp"] = int(len(tr))

        # 分年成交额
        tr2 = tr.copy()
        tr2["dt"] = pd.to_datetime(tr2["date"])
        tr2["year"] = tr2["dt"].dt.year
        yr = tr2.groupby("year")["amount"].sum()
        print("\n  分年成交额:")
        for y, v in yr.items():
            print(f"    {y}: {v/1e8:>6.2f} 亿")

    # ---- 临界点
    print("\n" + "=" * 94)
    print("成本临界点")
    print("=" * 94)
    rs = out["results"]
    base = rs["15"]
    print(f"  基准（15bp）：CAGR {base['cagr']:.2%}，Sharpe {base['sharpe']:.3f}")

    # Sharpe 跌破 0.3
    crit_sharpe = None
    for slip in SLIPS:
        if rs[str(slip)]["sharpe"] < 0.30:
            crit_sharpe = slip
            break
    # CAGR 归零
    crit_zero = None
    for slip in SLIPS:
        if rs[str(slip)]["cagr"] <= 0:
            crit_zero = slip
            break

    print(f"  Sharpe 跌破 0.30 的滑点: "
          f"{f'{crit_sharpe}bp' if crit_sharpe else '扫描范围内未跌破'}")
    print(f"  CAGR 归零的滑点:         "
          f"{f'{crit_zero}bp' if crit_zero else '扫描范围内未归零'}")
    out["crit_sharpe_030"] = crit_sharpe
    out["crit_cagr_zero"] = crit_zero

    # 每增加 10bp 滑点的影响
    print("\n  每增加 10bp 滑点的影响:")
    for a, b in ((15, 25), (25, 40), (15, 30)):
        if str(a) in rs and str(b) in rs:
            d = (rs[str(b)]["cagr"] - rs[str(a)]["cagr"]) * 100
            print(f"    {a}bp → {b}bp: CAGR {d:+.2f}pp")

    # ---- 保存
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    print(f"\n→ {OUT}")

    # 保存净值曲线供画图
    nav_out = ROOT / "build" / "eval" / "cost_navs.parquet"
    df = pd.DataFrame({f"slip_{k}": v for k, v in navs.items()})
    df.to_parquet(nav_out)
    print(f"→ {nav_out}")


if __name__ == "__main__":
    main()
