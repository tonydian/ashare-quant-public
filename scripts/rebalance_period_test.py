# -*- coding: utf-8 -*-
"""实测：调仓周期 10 日 → 20 日，能否降低换手/成本并保住收益。

严格的验证纪律（防止过拟合）：
  · 只在 **DEV 期（2020-2021）** 比较不同调仓周期
  · 选定后，**OOS 期（2022-2026）只看一次**，用于确认
  · 参数网格不扩大，只测 5/10/15/20/30 日

成本口径（真实费率）：
  佣金 万0.1（最低 5 元）→ 40 只时实际 6.67bp
  印花税 5bp（卖出）
  过户费 0.2bp（双向）
  滑点 40bp/边（按市值估算的中性假设）

实现方式：
  run_backtest 内部用 cfg.COMMISSION_RATE / MIN_COMMISSION，
  已更新为真实值。额外滑点用 slippage_bp 传入。
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

OUT = ROOT / "build" / "eval" / "rebalance_test.json"
DEV = ("2020-01-08", "2021-12-31")
OOS = ("2022-01-01", "2026-12-31")
PERIODS = [5, 10, 15, 20, 30]
SLIP = 40.0       # 中性滑点假设（按持仓市值估算）


def main():
    print("=" * 96)
    print("调仓周期测试（真实费率 + 中性滑点）")
    print("=" * 96)
    print(f"  佣金 {C.COMMISSION_RATE*1e4:.2f}bp（万0.1，最低{C.MIN_COMMISSION:.0f}元）")
    print(f"  印花税 {C.STAMP_TAX_SELL*1e4:.0f}bp | 滑点 {SLIP:.0f}bp/边")
    print(f"  持仓 40 只 | 本金 30 万")
    print(f"  DEV {DEV[0]}~{DEV[1]}（选参数） → OOS {OOS[0]}~{OOS[1]}（只验证一次）")

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

    print(f"\n  数据 {md.T} 日 × {md.N} 只，模型池 {len(res.pool)} 因子")

    out = {"slip": SLIP, "fee": {"comm_rate": C.COMMISSION_RATE,
                                  "min_comm": C.MIN_COMMISSION,
                                  "stamp": C.STAMP_TAX_SELL},
           "grid": {}, "dev_best": None, "oos_confirm": None}

    # ---------------- 第一步：DEV 期选参数
    print("\n" + "=" * 96)
    print("第一步：DEV 期（2020-2021）比较调仓周期")
    print("=" * 96)
    print(f"{'周期':>5}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>8}"
          f"{'年换手':>9}{'年化成本':>10}{'交易笔数':>10}")
    print("-" * 96)

    dev_res = {}
    for rb in PERIODS:
        rec = run_one(md, sc, cash, rb, DEV[0], DEV[1], phase)
        if rec is None:
            continue
        dev_res[rb] = rec
        print(f"{rb:>5}{rec['cagr']:>9.2%}{rec['sharpe']:>9.3f}"
              f"{rec['max_dd']:>9.2%}{rec['calmar']:>8.2f}"
              f"{rec['turnover']:>8.2f}x{rec['ann_cost']:>9.2%}{rec['n_trades']:>10}")

    out["grid"]["dev"] = {str(k): v for k, v in dev_res.items()}

    # 选参数：按 Calmar（因为目标是"高收益低回撤"）
    if not dev_res:
        print("无有效结果")
        return
    best_rb = max(dev_res.items(), key=lambda kv: kv[1]["calmar"])[0]
    print(f"\n  → DEV 期最优（按 Calmar）: **{best_rb} 日**"
          f"（Calmar {dev_res[best_rb]['calmar']:.2f}）")
    out["dev_best"] = int(best_rb)

    # ---------------- 第二步：OOS 验证（只看一次）
    print("\n" + "=" * 96)
    print(f"第二步：OOS 期（2022-2026）验证 —— 只比较 10日（现状）vs {best_rb}日")
    print("=" * 96)
    print(f"{'周期':>5}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>8}"
          f"{'年换手':>9}{'年化成本':>10}")
    print("-" * 96)

    oos_res = {}
    for rb in sorted({10, best_rb}):
        rec = run_one(md, sc, cash, rb, OOS[0], OOS[1], phase)
        if rec is None:
            continue
        oos_res[rb] = rec
        print(f"{rb:>5}{rec['cagr']:>9.2%}{rec['sharpe']:>9.3f}"
              f"{rec['max_dd']:>9.2%}{rec['calmar']:>8.2f}"
              f"{rec['turnover']:>8.2f}x{rec['ann_cost']:>9.2%}")

    out["grid"]["oos"] = {str(k): v for k, v in oos_res.items()}

    if 10 in oos_res and best_rb in oos_res and best_rb != 10:
        a, b = oos_res[10], oos_res[best_rb]
        print(f"\n  OOS 对比：10日 CAGR {a['cagr']:.2%} → "
              f"{best_rb}日 {b['cagr']:.2%}（{(b['cagr']-a['cagr'])*100:+.2f}pp）")
        print(f"           Sharpe {a['sharpe']:.3f} → {b['sharpe']:.3f}")
        print(f"           Calmar {a['calmar']:.2f} → {b['calmar']:.2f}")
        print(f"           成本   {a['ann_cost']:.2%} → {b['ann_cost']:.2%}"
              f"（省 {(a['ann_cost']-b['ann_cost'])*100:+.2f}pp）")
        out["oos_confirm"] = {
            "delta_cagr_pp": float((b["cagr"] - a["cagr"]) * 100),
            "delta_sharpe": float(b["sharpe"] - a["sharpe"]),
            "delta_calmar": float(b["calmar"] - a["calmar"]),
            "delta_cost_pp": float((a["ann_cost"] - b["ann_cost"]) * 100),
        }

    # ---------------- 第三步：全期
    print("\n" + "=" * 96)
    print("第三步：全期（2020-2026）")
    print("=" * 96)
    print(f"{'周期':>5}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>8}"
          f"{'年换手':>9}{'年化成本':>10}")
    print("-" * 96)
    full_res = {}
    for rb in sorted({10, best_rb}):
        rec = run_one(md, sc, cash, rb, DEV[0], OOS[1], phase)
        if rec is None:
            continue
        full_res[rb] = rec
        print(f"{rb:>5}{rec['cagr']:>9.2%}{rec['sharpe']:>9.3f}"
              f"{rec['max_dd']:>9.2%}{rec['calmar']:>8.2f}"
              f"{rec['turnover']:>8.2f}x{rec['ann_cost']:>9.2%}")
    out["grid"]["full"] = {str(k): v for k, v in full_res.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    print(f"\n→ {OUT}")


def run_one(md, sc, cash, rb: int, start_s: str, end_s: str, phase: int):
    start = int(md.date_index(start_s))
    while start % 5 != phase:
        start += 1
    end = int(md.date_index(end_s))
    offset = (phase - start) % rb if rb % 5 == 0 else 0

    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(
            md, sc, start, end, risk_overlay=True, stop_loss=True,
            rebalance_days=rb, rebalance_offset=offset,
            trade_band_rel=0.30, trade_band_abs=6000.0,
            max_new_entries=6, cash_asset_returns=cash,
            slippage_bp=SLIP, verbose=False)
    except Exception as e:
        print(f"  [{rb}d] 失败: {type(e).__name__}: {e}")
        return None
    finally:
        C.N_HOLDINGS = old

    sub = r.nav[r.nav.index >= pd.Timestamp(start_s)]
    if len(sub) < 60:
        return None
    p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)

    tr = r.trades
    yrs = len(sub) / 252
    amt = float(tr["amount"].sum())
    avg_nav = float(sub.mean())
    to = amt / avg_nav / yrs if avg_nav > 0 else 0.0

    # 年化成本（往返 = 滑点*2 + 佣金实际*2 + 印花 + 过户）
    per = 300000.0 / 40
    comm_bp = max(C.MIN_COMMISSION, per * C.COMMISSION_RATE) / per * 1e4
    rt_bp = SLIP * 2 + comm_bp * 2 + C.STAMP_TAX_SELL * 1e4 + 0.2
    ann_cost = to * rt_bp * 1e-4

    return {
        "cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
        "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
        "turnover": float(to), "ann_cost": float(ann_cost),
        "n_trades": int(len(tr)), "roundtrip_bp": float(rt_bp),
        "comm_bp": float(comm_bp),
    }


if __name__ == "__main__":
    main()
