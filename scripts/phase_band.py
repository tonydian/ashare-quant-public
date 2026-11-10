# -*- coding: utf-8 -*-
"""主策略「相位带」：把单点回测绩效换成 8 个对齐调仓相位的分布。

为什么需要
----------
``run_backtest.py`` 的 offset 由 ``(phase - start_row) % rb`` 决定，等价于「拿到分数
就立刻建仓」。2026-11-03 的相位实验发现：同一份 scores、只改调仓起始日，
40 日调仓下 CAGR 从 3.15% 铺到 7.21%，而生产恰好落在**最高的那一档**。
单点披露会把「调仓日历的运气」当成策略能力，因此主策略绩效必须同时给出：

    相位均值 / 中位数 / 最差-最好区间，以及生产 offset 在分布中的分位

做法
----
rb=40 时与周频打分行对齐的 offset 只有 ``0,5,...,35`` 共 8 个（offset%5!=0 会
退化全现金，见 run_backtest.py 的硬校验）。对每个 offset 跑一次完整回测
（参数与生产完全一致），输出 ``build/eval/phase_band.json``。

用法::

    python scripts/phase_band.py                     # 默认 build/models/wf_h10
    python scripts/phase_band.py --model build/models/wf_clone --out xxx.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C
from qf import data as qfdata
from qf import model as qfm
from qf import backtest as qfbt
from qf import metrics as qfmetrics


def _summary(vals: np.ndarray) -> dict:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if not len(v):
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"),
                "max": float("nan"), "p25": float("nan"), "p75": float("nan"),
                "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(C.BUILD_MODELS / "wf_h10"))
    ap.add_argument("--out", default=str(C.BUILD / "eval" / "phase_band.json"))
    ap.add_argument("--start", default=C.VALID_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--rebalance-days", type=int, default=40)
    ap.add_argument("--step", type=int, default=5,
                    help="相位步长；offset%5!=0 会退化全现金，故只能取 5 的倍数")
    ap.add_argument("--n-holdings", type=int, default=C.N_HOLDINGS)
    ap.add_argument("--band-rel", type=float, default=0.30)
    ap.add_argument("--band-abs", type=float, default=6000.0)
    ap.add_argument("--max-new", type=int, default=6)
    ap.add_argument("--cash-blend", default=f"{C.CASH_BOND_WEIGHT},{C.CASH_GOLD_WEIGHT}")
    ap.add_argument("--no-risk-overlay", action="store_true")
    ap.add_argument("--no-stop-loss", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="复用上次跑出的相位明细（同 out 旁的 *.rows.json），不重跑回测")
    args = ap.parse_args()

    t0 = time.time()
    rb = int(args.rebalance_days)
    if rb % 5 != 0:
        raise SystemExit(f"rebalance_days={rb} 必须是 5 的倍数（周频打分只落在 mod5 同相位行）")
    if args.step % 5 != 0:
        raise SystemExit(f"--step={args.step} 必须是 5 的倍数，否则回测退化为全现金且不报错")

    print("[1/3] 加载模型与行情 ...")
    res = qfm.load_result(pathlib.Path(args.model))
    md = qfdata.load_market(verbose=False)
    scores = res.scores
    rows_w = res.rows_weekly
    valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
    start_row = max(md.date_index(args.start), int(valid[0]))
    phase = int(rows_w[0]) % 5 if len(rows_w) else 0
    while start_row % 5 != phase and start_row < md.T - 1:
        start_row += 1
    end_row = md.T - 1 if args.end is None else md.date_index(args.end)
    # 与 run_backtest.py 完全一致的生产相位
    prod_offset = (phase - start_row) % rb
    if prod_offset % 5 != 0:
        raise SystemExit(f"生产相位 offset={prod_offset} 未与周频打分行对齐")
    print(f"    区间 {md.dates[start_row].date()} ~ {md.dates[end_row].date()}；"
          f"rb={rb}；生产 offset={prod_offset}")

    try:
        _bw, _gw = [float(x) for x in str(args.cash_blend).split(",")]
    except ValueError:
        raise SystemExit(f"--cash-blend 格式应为 'bond,gold'，收到 {args.cash_blend!r}")
    cash = qfbt.load_cash_asset_returns(md, _bw, _gw) if _bw + _gw > 0 else None

    offsets = list(range(0, rb, args.step))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 明细另存一份：8 个相位约 100s，统计段出错时不至于白白重跑回测
    rows_path = out.with_suffix(".rows.json")
    recs: list[dict] = []
    _sig = {"model": str(args.model), "start": str(md.dates[start_row].date()),
            "end": str(md.dates[end_row].date()), "rb": rb, "step": int(args.step),
            "n_holdings": int(args.n_holdings), "band_rel": float(args.band_rel),
            "band_abs": float(args.band_abs), "max_new": int(args.max_new),
            "cash": str(args.cash_blend), "overlay": not args.no_risk_overlay,
            "stop": not args.no_stop_loss}
    if args.reuse and rows_path.exists():
        try:
            _old = json.load(open(rows_path, encoding="utf-8"))
            if _old.get("sig") == _sig:
                recs = _old["rows"]
                print(f"[2/3] 复用 {rows_path.name}（{len(recs)} 个相位，--reuse）")
        except Exception as _e:
            print(f"    [warn] 复用失败（{_e}），改为重跑")
    if not recs:
        print(f"[2/3] 回测 {len(offsets)} 个对齐相位 {offsets} ...")
        old_nh = C.N_HOLDINGS
        C.N_HOLDINGS = int(args.n_holdings)
        try:
            for off in offsets:
                t1 = time.time()
                r = qfbt.run_backtest(
                    md, scores, start_row, end_row,
                    risk_overlay=not args.no_risk_overlay,
                    stop_loss=not args.no_stop_loss,
                    rebalance_days=rb, rebalance_offset=off,
                    trade_band_rel=max(0.0, float(args.band_rel)),
                    trade_band_abs=max(0.0, float(args.band_abs)),
                    max_new_entries=None if args.max_new < 0 else args.max_new,
                    cash_asset_returns=cash, verbose=False)
                nav = pd.Series(r.nav).dropna()
                p = qfmetrics.performance(nav, rf=C.CASH_YIELD)
                recs.append({"offset": off, "cagr": float(p["cagr"]),
                             "sharpe": float(p["sharpe"]), "max_dd": float(p["max_dd"]),
                             "calmar": float(p["calmar"]), "ann_vol": float(p["ann_vol"]),
                             "elapsed_sec": round(time.time() - t1, 1)})
                print(f"    offset={off:>2}  CAGR {p['cagr']*100:>6.2f}%  "
                      f"Sharpe {p['sharpe']:>5.3f}  MaxDD {p['max_dd']*100:>6.2f}%"
                      f"{'   ← 生产' if off == prod_offset else ''}", flush=True)
        finally:
            C.N_HOLDINGS = old_nh
        rows_path.write_text(json.dumps({"sig": _sig, "rows": recs},
                                        ensure_ascii=False, indent=1), encoding="utf-8")

    df = pd.DataFrame(recs)
    cagrs = df["cagr"].to_numpy()
    prod_row = df[df["offset"] == prod_offset]
    prod_cagr = float(prod_row["cagr"].iloc[0]) if len(prod_row) else float("nan")
    # 生产相位的分位（0=最好，越小越说明生产站在最幸运一侧）
    better = int((cagrs > prod_cagr + 1e-12).sum())
    pctile = float(better) / max(len(cagrs) - 1, 1) if len(cagrs) > 1 else 0.0

    band = {
        "model": str(pathlib.Path(args.model).relative_to(ROOT))
        if pathlib.Path(args.model).is_absolute() else str(args.model),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "start": str(md.dates[start_row].date()),
        "end": str(md.dates[end_row].date()),
        "rebalance_days": rb,
        "prod_offset": int(prod_offset),
        "n_phases": int(len(df)),
        "cagr": _summary(cagrs),
        "sharpe": _summary(df["sharpe"].to_numpy()),
        "max_dd": _summary(df["max_dd"].to_numpy()),
        "calmar": _summary(df["calmar"].to_numpy()),
        "prod": {"cagr": prod_cagr,
                 "sharpe": float(prod_row["sharpe"].iloc[0]) if len(prod_row) else float("nan"),
                 "max_dd": float(prod_row["max_dd"].iloc[0]) if len(prod_row) else float("nan"),
                 # pct_better: 比生产更好的相位占比；0 表示生产就是最好的那个相位
                 "pct_better": pctile,
                 "n_better": better},
        "rows": recs,
    }
    out.write_text(json.dumps(band, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[3/3] 相位带（{len(df)} 个对齐相位）")
    print(f"    CAGR  均值 {band['cagr']['mean']*100:>6.2f}%  "
          f"中位 {band['cagr']['median']*100:>6.2f}%  "
          f"区间 [{band['cagr']['min']*100:.2f}%, {band['cagr']['max']*100:.2f}%]")
    print(f"    Sharpe 均值 {band['sharpe']['mean']:>5.3f}  中位 {band['sharpe']['median']:>5.3f}")
    print(f"    生产 offset={prod_offset} CAGR {prod_cagr*100:.2f}%，"
          f"比它更好的相位 {better}/{len(cagrs)-1}"
          f"{'（即最优档）' if better == 0 else ''}")
    print(f"    → {out}   用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
