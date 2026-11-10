# -*- coding: utf-8 -*-
"""稳健性 / 敏感性测试：模型超参数 + 回测设定 + 成本压力。

用法:
    python scripts/run_robustness.py --factors build/factors/weekly_top.npz \
        --meta build/factors/weekly_meta.npz --out build/runs/robustness

输出:
    build/runs/robustness/robustness.csv   每个变体的绩效（全期 2020-2026 / 样本外 2022-2026）
    build/runs/robustness/<variant>/metrics.json
说明：所有变体都使用 walk-forward（权重只用当时之前数据），DEV 期选池只发生在 2017-2019；
      backtest-only 变体复用 base 模型的打分，不改权重。
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
from qf import factors as qff
from qf import model as qfm
from qf import backtest as qfbt
from qf import metrics as qfmetrics


def perf_slice(nav: pd.Series, start: str) -> dict:
    sub = nav[nav.index >= pd.Timestamp(start)]
    if len(sub) < 30:
        return {}
    return qfmetrics.performance(sub, rf=C.CASH_YIELD)


def run_variant(md, scores, tag, out_root, risk_overlay=True, stop_loss=True,
                rebalance_days=10, n_holdings=40, slippage_bp=None,
                trade_band_rel=None, trade_band_abs=None, max_new_entries=None,
                start_row=None, end_row=None, cash_asset_returns=None):
    old_n = C.N_HOLDINGS
    old_slip = C.SLIPPAGE_BP
    try:
        if n_holdings:
            C.N_HOLDINGS = int(n_holdings)
        if slippage_bp is not None:
            C.SLIPPAGE_BP = float(slippage_bp)
        t0 = time.time()
        kw = {}
        if trade_band_rel is not None:
            kw["trade_band_rel"] = float(trade_band_rel)
        if trade_band_abs is not None:
            kw["trade_band_abs"] = float(trade_band_abs)
        if max_new_entries is not None:
            kw["max_new_entries"] = int(max_new_entries)
        res = qfbt.run_backtest(md, scores, start_row, end_row,
                                risk_overlay=risk_overlay, stop_loss=stop_loss,
                                rebalance_days=rebalance_days, verbose=False,
                                cash_asset_returns=cash_asset_returns, **kw)
    finally:
        C.N_HOLDINGS = old_n
        C.SLIPPAGE_BP = old_slip
    out = pathlib.Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    m = dict(res.metrics)
    m["tag"] = tag
    m["risk_overlay"] = risk_overlay
    m["stop_loss"] = stop_loss
    m["rebalance_days"] = int(rebalance_days or C.REBALANCE_DAYS)
    m["n_holdings"] = int(n_holdings or C.N_HOLDINGS)
    m["slippage_bp"] = float(slippage_bp if slippage_bp is not None else C.SLIPPAGE_BP)
    m["full_cagr"] = m.get("cagr")
    m["full_sharpe"] = m.get("sharpe")
    m["full_maxdd"] = m.get("max_dd")
    m["full_calmar"] = m.get("calmar")
    p22 = perf_slice(res.nav, "2022-01-01")
    for k, v in p22.items():
        m[f"oos_{k}"] = v
    m["elapsed_sec"] = round(time.time() - t0, 1)
    res.nav.to_frame("nav").to_parquet(out / "nav.parquet")
    json.dump(m, open(out / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    print(f"[{tag:28s}] CAGR={m.get('cagr', np.nan):7.2%} Sharpe={m.get('sharpe', np.nan):5.2f} "
          f"MaxDD={m.get('max_dd', np.nan):7.2%} Calmar={m.get('calmar', np.nan):5.2f} "
          f"OOS_CAGR={m.get('oos_cagr', np.nan):7.2%} OOS_MaxDD={m.get('oos_max_dd', np.nan):7.2%} "
          f"({m['elapsed_sec']:.0f}s)")
    return res, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default=str(C.BUILD_FACTORS / "weekly_top.npz"))
    ap.add_argument("--meta", default=str(C.BUILD_FACTORS / "weekly_meta.npz"))
    ap.add_argument("--model-dir", default=str(C.BUILD_MODELS / "wf_v1"))
    ap.add_argument("--out", default=str(C.BUILD_RUNS / "robustness"))
    ap.add_argument("--skip-model-variants", action="store_true")
    ap.add_argument("--cash-blend", default=None,
                    help="闲置资金替代资产（国债,黄金）权重；缺省用 config 的 0.8,0.2；'0,0'=纯现金")
    args = ap.parse_args()
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[0] 加载数据与因子面板 ...")
    md = qfdata.load_market(verbose=True)
    _bw, _gw = (C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)
    if args.cash_blend:
        _bw, _gw = [float(x) for x in args.cash_blend.split(",")]
    cash_ret = None
    if _bw + _gw > 0:
        cash_ret = qfbt.load_cash_asset_returns(md, _bw, _gw)
        print(f"    闲置资金替代资产：{_bw:.0%} 国债指数 + {_gw:.0%} 黄金ETF")
    cat = qff.load_catalog()
    zf = np.load(args.factors)
    zm = np.load(args.meta)
    rows_weekly = zm["rows"].astype(np.int64)
    panels = {k: zf[k] for k in zf.files if k in set(cat["factor_key"])}
    print(f"    {len(panels)} 个因子 × {len(rows_weekly)} 周")

    model_dir = pathlib.Path(args.model_dir)
    if model_dir.exists() and (model_dir / "composite_score.npz").exists():
        base = qfm.load_result(model_dir)
        print(f"    复用已有 base 模型（pool={len(base.pool)}）")
    else:
        cfg = qfm.ModelConfig()
        base = qfm.walk_forward(md, rows_weekly, panels, cat, cfg=cfg, verbose=True)
        qfm.save_result(base, model_dir)

    valid_rows = np.flatnonzero(np.isfinite(base.scores).any(axis=1))
    start_row = int(max(valid_rows[0], md.date_index(C.VALID_START)))
    phase = int(rows_weekly[0]) % 5
    while start_row % 5 != phase:
        start_row += 1
    end_row = md.T - 1
    base_start = start_row
    rows = []

    if not args.skip_model_variants:
        # ---- 模型变体：不同方法 / 因子数 / 相关性阈值（重新跑 walk-forward）
        variants = [
            ("model_icir", dict(cfg=qfm.ModelConfig(method="icir"), pool=base.pool)),
            ("model_ridge", dict(cfg=qfm.ModelConfig(method="ridge"), pool=base.pool)),
            ("model_smooth100", dict(cfg=qfm.ModelConfig(score_smooth_alpha=1.0), pool=base.pool)),
            ("model_K60", dict(cfg=qfm.ModelConfig(max_factors=60), pool=None)),
        ]
        for tag, kw in variants:
            print(f"\n== {tag} ==")
            t0 = time.time()
            res = qfm.walk_forward(md, rows_weekly, panels, cat, cfg=kw["cfg"],
                                   verbose=False, pool_override=kw["pool"])
            print(f"  walk-forward {time.time()-t0:.0f}s pool={len(res.pool)}")
            save_dir = out_root / tag
            qfm.save_result(res, save_dir)
            _, m = run_variant(md, res.scores, tag, out_root,
                               start_row=base_start, end_row=end_row,
                               cash_asset_returns=cash_ret)
            m["n_factors"] = len(res.pool)
            m["method"] = kw["cfg"].method
            rows.append(m)

    # ---- 回测设定变体：复用 base 打分
    print("\n== 回测设定变体（复用 base 打分）==")
    settings = [
        ("base", dict()),
        ("no_overlay", dict(risk_overlay=False)),
        ("no_stoploss", dict(stop_loss=False)),
        ("rebal5", dict(rebalance_days=5)),
        ("rebal20", dict(rebalance_days=20)),
        ("n15", dict(n_holdings=15)),
        ("n30", dict(n_holdings=30)),
        ("n50", dict(n_holdings=50)),
        ("band_off", dict(trade_band_rel=0.0, trade_band_abs=0.0, max_new_entries=100)),
        ("maxnew3", dict(max_new_entries=3)),
        ("maxnew100", dict(max_new_entries=100)),
        ("slip25bp", dict(slippage_bp=25.0)),
        ("slip30bp", dict(slippage_bp=30.0)),
        ("no_overlay_no_stop", dict(risk_overlay=False, stop_loss=False)),
    ]
    for tag, kw in settings:
        print(f"\n== {tag} ==")
        _, m = run_variant(md, base.scores, tag, out_root,
                           start_row=base_start, end_row=end_row,
                           cash_asset_returns=cash_ret, **kw)
        m["n_factors"] = len(base.pool)
        m["method"] = "blend"
        rows.append(m)

    # ---- 选股域变体：重新加载 md（数据层按 config 计算 eligible）
    print("\n== 选股域变体（重新加载 md）==")
    univ_variants = [
        ("univ_mcap3b", {"MIN_MCAP": 3e9}),
        ("univ_mcap8b", {"MIN_MCAP": 8e9}),
        ("univ_liq50m", {"MIN_AMOUNT_20D": 5e7}),
        ("univ_price5", {"MIN_PRICE": 5.0}),
    ]
    import copy
    for tag, over in univ_variants:
        print(f"\n== {tag} ==")
        old_vals = {k: getattr(C, k) for k in over}
        try:
            for k, v in over.items():
                setattr(C, k, v)
            md2 = qfdata.load_market(verbose=False)
        finally:
            for k, v in old_vals.items():
                setattr(C, k, v)
        _, m = run_variant(md2, base.scores, tag, out_root,
                           start_row=base_start, end_row=end_row,
                           cash_asset_returns=cash_ret)
        m["n_factors"] = len(base.pool)
        m["method"] = "blend"
        m["universe"] = str(over)
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "robustness.csv", index=False, encoding="utf-8-sig")
    cols = ["tag", "cagr", "ann_vol", "sharpe", "max_dd", "calmar",
            "oos_cagr", "oos_sharpe", "oos_max_dd", "oos_calmar",
            "turnover_ann", "cost_ratio", "avg_holdings"]
    print("\n===== 稳健性汇总 =====")
    print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False))
    print(f"\n→ {out_root/'robustness.csv'}")


if __name__ == "__main__":
    main()
