# -*- coding: utf-8 -*-
"""最终回测：加载 walk-forward 复合打分 → 组合/执行/风控 → 指标/图/报告数据。

用法:
    python scripts/run_backtest.py --model build/models/wf_v1 --out build/runs/main --start 2020-01-03
    python scripts/run_backtest.py --model build/models/wf_v1 --out build/runs/main_rebal10 \
        --rebalance-days 10
（风控 overlay 与个股止损默认开启，可用 --no-risk-overlay / --no-stop-loss 关闭）

输出（out 目录）:
    nav.parquet / metrics.json / yearly.csv / monthly.csv / trades.parquet /
    holdings.parquet / exposures.parquet / summary.md / figures/*.png
并在 journal/runs.jsonl 追加实验记录。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C
from qf import data as qfdata
from qf import model as qfm
from qf import metrics as qfmetrics
from qf import backtest as qfbt
from qf import report as qfreport


def _append_journal(rec: dict) -> None:
    C.JOURNAL.mkdir(parents=True, exist_ok=True)
    with open(C.JOURNAL / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(C.BUILD_MODELS / "wf_h10"),
                    help="主模型（报告口径为 wf_h10：34 因子 / 10 日标签）")
    ap.add_argument("--out", default=str(C.BUILD_RUNS / "main"))
    ap.add_argument("--start", default=C.VALID_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-risk-overlay", action="store_true")
    ap.add_argument("--no-stop-loss", action="store_true")
    ap.add_argument("--rebalance-days", type=int, default=40,
                    help="调仓周期（交易日）；报告口径为 10（config 库默认 5）")
    ap.add_argument("--n-holdings", type=int, default=40)
    ap.add_argument("--band-rel", type=float, default=0.30)
    ap.add_argument("--band-abs", type=float, default=6000.0)
    ap.add_argument("--max-new", type=int, default=6, help="每期新进上限；<0 表示不限制")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--cash-blend", default=f"{C.CASH_BOND_WEIGHT},{C.CASH_GOLD_WEIGHT}",
                    help="闲置资金替代资产（国债指数,黄金ETF）内部权重，如 '0.8,0.2'；'0,0' 表示纯现金计息")
    args = ap.parse_args()

    t0 = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载行情 ...")
    md = qfdata.load_market(verbose=True)
    print("[2/4] 加载模型打分 ...")
    res = qfm.load_result(args.model)
    scores = res.scores
    # 对齐：从第一个有打分的周频行开始（调仓网格与打分网格必须同相位）
    valid_rows = np.flatnonzero(np.isfinite(scores).any(axis=1))
    if len(valid_rows) == 0:
        raise SystemExit("打分面板为空")
    first_score = int(valid_rows[0])
    start_row = max(md.date_index(args.start), first_score)
    # 保持与周频行同相位（rows_weekly = arange(first, T, 5)）
    rows_w = res.rows_weekly
    if len(rows_w):
        phase = int(rows_w[0]) % 5
        while start_row % 5 != phase and start_row < md.T - 1:
            start_row += 1
    end_row = md.T - 1 if args.end is None else md.date_index(args.end)
    # 调仓相位对齐：让首个调仓日落在周频打分行上（rb 为 5 的倍数时长期保持对齐）
    rb = int(args.rebalance_days)
    phase = int(rows_w[0]) % 5 if len(rows_w) else 0
    if rb % 5 == 0:
        rebalance_offset = (phase - start_row) % rb
        # rb 是 5 的倍数时，调仓日的 (mod 5) 是调仓相位的不变量：offset%5!=0 会让
        # 每个调仓日都落在没有打分的交易日上 → 组合永远建不起来、净值退化成纯现金，
        # 且**不报错**（2026-11-03 在相位实验里整批撞到，8 个无效相位全给 4.17%）。
        if rebalance_offset % 5 != 0:
            raise SystemExit(
                f"rebalance_days={rb} 且相位 offset={rebalance_offset} 未与周频打分行对齐，"
                f"回测会退化成全现金。请用 rb 为 5 的倍数并保证 offset%5==0")
    else:
        rebalance_offset = 0
        print(f"    [warn] rebalance_days={rb} 不是 5 的倍数，调仓网格会逐渐偏离周频打分行")
    n_valid = int(np.isfinite(scores[start_row:end_row + 1]).any(axis=1).sum())
    print(f"    打分有效期行数 {n_valid}；区间 {md.dates[start_row].date()} ~ {md.dates[end_row].date()}；"
          f"调仓周期 {rb}，相位 offset={rebalance_offset}")

    print("[3/4] 回测 ...")
    max_new = None if args.max_new < 0 else args.max_new
    # 闲置现金替代资产（国债指数/黄金ETF）：默认 80/20；'0,0' 退回纯现金计息
    try:
        _bw, _gw = [float(x) for x in str(args.cash_blend).split(",")]
    except ValueError:
        raise SystemExit(f"--cash-blend 格式应为 'bond,gold'，收到 {args.cash_blend!r}")
    cash_asset_returns = None
    if _bw + _gw > 0:
        cash_asset_returns = qfbt.load_cash_asset_returns(md, _bw, _gw)
        _days = cash_asset_returns[int(start_row):int(end_row) + 1]
        print(f"    闲置资金：{_bw:.0%} 国债指数 + {_gw:.0%} 黄金ETF（年化 "
              f"{(np.prod(1.0 + _days) ** (252.0 / max(len(_days), 1)) - 1):.2%}）")
    old_nh = C.N_HOLDINGS
    C.N_HOLDINGS = int(args.n_holdings)
    try:
        result = qfbt.run_backtest(
            md, scores, start_row, end_row,
            risk_overlay=not args.no_risk_overlay,
            stop_loss=not args.no_stop_loss,
            rebalance_days=args.rebalance_days,
            rebalance_offset=rebalance_offset,
            trade_band_rel=0.0 if args.band_rel <= 0 else args.band_rel,
            trade_band_abs=0.0 if args.band_abs <= 0 else args.band_abs,
            max_new_entries=max_new,
            verbose=True,
            cash_asset_returns=cash_asset_returns,
        )
    finally:
        C.N_HOLDINGS = old_nh
    print("[4/4] 指标 / 图表 / 落盘 ...")
    m = result.metrics
    # 基准同区间
    idx = md.dates[start_row:end_row + 1]
    bench = {}
    for name in C.BENCHMARKS + ["all_eq"]:
        r = md.bench[name].reindex(idx).fillna(0.0).to_numpy()
        nav_b = np.cumprod(1.0 + r)
        bench[name] = pd.Series(nav_b, index=idx)
    bm = {}
    for name, nav_b in bench.items():
        bm[name] = qfmetrics.performance(nav_b, rf=C.CASH_YIELD)
    path = qfreport.plot_run(result, bench, out)
    qfreport.write_summary(result, bm, out)

    result.nav.to_frame("nav").to_parquet(out / "nav.parquet")
    result.daily_ret.to_frame("ret").to_parquet(out / "daily_ret.parquet")
    for name, df in [("weights", result.weights), ("trades", result.trades),
                     ("holdings", result.holdings), ("exposures", result.exposures)]:
        if df is not None and len(df):
            df.to_parquet(out / f"{name}.parquet")
    json.dump(m, open(out / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    json.dump(bm, open(out / "benchmark_metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    json.dump({"cash_bond_weight": _bw, "cash_gold_weight": _gw,
               "cash_sleeve": "国债指数+黄金ETF" if _bw + _gw > 0 else "纯现金计息"},
              open(out / "cash_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    qfmetrics.yearly_returns(result.nav).to_csv(out / "yearly.csv", encoding="utf-8-sig")
    qfmetrics.monthly_returns_table(result.nav).to_csv(out / "monthly.csv", encoding="utf-8-sig")
    stress = qfmetrics.stress_periods(result.nav)
    stress.to_csv(out / "stress.csv", encoding="utf-8-sig")
    _append_journal({"run_id": f"{pd.Timestamp.now():%Y-%m-%dT%H:%M:%S}_{args.tag}", "stage": "backtest",
                     "status": "ok", "tag": args.tag, "start": str(md.dates[start_row].date()),
                     "end": str(md.dates[end_row].date()),
                     "risk_overlay": not args.no_risk_overlay,
                     "stop_loss": not args.no_stop_loss,
                     "rebalance_days": int(args.rebalance_days),
                     "rebalance_offset": int(rebalance_offset),
                     "trade_band_rel": float(args.band_rel),
                     "trade_band_abs": float(args.band_abs),
                     "max_new_entries": None if max_new is None else int(max_new),
                     "cash_bond_weight": float(_bw), "cash_gold_weight": float(_gw),
                     "metrics": {k: m.get(k) for k in
                                 ["cagr", "ann_vol", "sharpe", "max_dd", "calmar", "turnover_ann"] if k in m},
                     "artifacts": [str(out)], "elapsed_sec": round(time.time() - t0, 1)})

    print("\n===== 绩效 =====")
    for k, v in m.items():
        if isinstance(v, (int, float)):
            print(f"  {k:28s} {v:>12.4f}")
    print("\n===== 基准 =====")
    print(pd.DataFrame(bm).T[["cagr", "ann_vol", "sharpe", "max_dd", "calmar"]].round(4).to_string())
    print(f"\n完成，用时 {time.time()-t0:.1f}s → {out}")


if __name__ == "__main__":
    main()
