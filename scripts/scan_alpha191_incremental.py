# -*- coding: utf-8 -*-
"""Alpha191 的"含金量"检验：剔除规模/价格/流动性后还剩多少 IC。

Alpha191 里有相当一批因子**不是尺度无关的**——例如
`alpha_013 = SQRT(HIGH*LOW) - VWAP` 的绝对值就是"股价水平"，
`alpha_070 = STD(AMOUNT,6)`、`alpha_132 = MEAN(AMOUNT,20)` 是"成交额水平"（≈规模）。
这类因子在 IC 表上会排到最前面，但抓到的是 A 股的**低价股/小市值溢价**，
不是新的选股信号。

做法（逐周频行，全部在横截面秩空间上做最小二乘）：
1. ``ic_raw``：原始 Rank IC；
2. ``ic_style``：对 `[log 市值, log 股价, log 20日均额]` 的秩回归后的残差 IC；
3. ``ic_full``：再叠加现有因子库最强 K 个因子后（可选）的残差 IC。

用法::

    python scripts/scan_alpha191_incremental.py --top 30 --n-base 0     # 只剔风格
    python scripts/scan_alpha191_incremental.py --top 30 --n-base 30    # 再剔现有库
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qf import alpha191 as A          # noqa: E402
from qf import data as qfdata         # noqa: E402
from qf import evaluate as qfe        # noqa: E402


def _rank_rows(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    xx = np.where(ok, x, np.inf)
    order = np.argsort(xx, axis=1, kind="stable")
    r = np.empty(x.shape, dtype=np.float64)
    np.put_along_axis(r, order, np.broadcast_to(np.arange(x.shape[1], dtype=float), x.shape), axis=1)
    return np.where(ok, r, np.nan)


def _ic(y: np.ndarray, a: np.ndarray) -> float:
    yc = y - y.mean()
    ac = a - a.mean()
    d = np.sqrt((yc @ yc) * (ac @ ac))
    return float(ac @ yc / d) if d > 0 else np.nan


def _resid(a: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if Z.shape[1] == 0:
        return a
    Zc = Z - Z.mean(axis=0)
    coef, *_ = np.linalg.lstsq(Zc, a, rcond=None)
    return a - Zc @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--n-base", type=int, default=0, help="再叠加现有库最强 N 个因子；0=不做")
    ap.add_argument("--out", default=str(ROOT / "build" / "alpha191"))
    args = ap.parse_args()
    out = Path(args.out)

    md = qfdata.load_market(verbose=False)
    rows = qfe.weekly_rows(md)
    elig = np.asarray(md.eligible)[rows]
    label = np.asarray(md.fwd_open_ret(5))[rows]

    # 风格变量（横截面秩）
    close = np.asarray(md.close, np.float64)[rows]
    mcap = np.asarray(md.mcap, np.float64)[rows]
    with np.errstate(all="ignore"):
        amt20 = np.asarray(md.amount_20d, np.float64)[rows]
        amt20 = np.where(amt20 > 0, amt20, np.asarray(md.amount, np.float64)[rows])
        style = np.stack([np.log(np.where(mcap > 0, mcap, np.nan)),
                          np.log(np.where(close > 0, close, np.nan)),
                          np.log(np.where(amt20 > 0, amt20, np.nan))], axis=-1)
    style_r = [_rank_rows(style[..., j], elig & np.isfinite(style[..., j])) for j in range(3)]
    style_name = ["log市值", "log股价", "log20日均额"]

    rl = _rank_rows(label, elig & np.isfinite(label))

    met = pd.read_csv(out / "metrics.csv")
    met = met.sort_values("ic_5d", key=lambda s: s.abs(), ascending=False)
    ks = [int(n.split("_")[1]) for n in met["name"].head(args.top)]

    base_r, base_names = [], []
    if args.n_base > 0:
        z = np.load(ROOT / "build" / "factors" / "weekly_top.npz", allow_pickle=True)
        bm = pd.read_parquet(ROOT / "build" / "eval" / "factor_metrics.parquet")
        bm = bm[bm["name"].isin(z.files)].sort_values("ic_5d", key=lambda s: s.abs(), ascending=False)
        base_names = list(bm["name"].head(args.n_base))
        for n in base_names:
            b = np.asarray(z[n], np.float64)
            base_r.append(_rank_rows(b, elig & np.isfinite(b)).astype(np.float32))
        print(f"叠加现有库 {len(base_names)} 个因子", flush=True)

    print(f"重算 IC 前 {len(ks)} 个 Alpha191 因子 ...", flush=True)
    panels = A.compute_panel(md, ks, rows=rows, verbose=False)

    recs = []
    for k in ks:
        x = panels[k]
        ok = elig & np.isfinite(x) & np.isfinite(label)
        rx = _rank_rows(x, ok)
        ic_raw, ic_style, ic_full = [], [], []
        corr_style = [0.0] * 3
        for i in range(len(rows)):
            m = ok[i] & np.isfinite(rx[i])
            if m.sum() < 200:
                continue
            yl = rl[i][m]
            a = rx[i][m]
            Zs = np.column_stack([style_r[j][i, m] for j in range(3)])
            good = np.all(np.isfinite(Zs), axis=1)
            if good.sum() < 200:
                continue
            yl, a, Zs = yl[good], a[good], Zs[good]
            ic_raw.append(_ic(yl, a))
            ic_style.append(_ic(yl, _resid(a, Zs)))
            if base_r:
                Zb = np.column_stack([np.where(np.isfinite(base_r[j][i, m][good]), base_r[j][i, m][good], 0.0).astype(np.float64)
                                      for j in range(len(base_r))])
                ic_full.append(_ic(yl, _resid(a, np.column_stack([Zs, Zb]))))
            for j in range(3):
                pair = np.isfinite(style_r[j][i, m]) & np.isfinite(rx[i, m])
                if pair.sum() > 200:
                    corr_style[j] = max(corr_style[j],
                                        abs(_ic(style_r[j][i, m][pair], rx[i, m][pair])))
        if not ic_raw:
            continue
        ir, istyle = np.array(ic_raw), np.array(ic_style)
        rec = {
            "alpha": f"alpha_{k:03d}",
            "ic_raw": float(np.nanmean(ir)),
            "ic_style_adj": float(np.nanmean(istyle)),
            "keep": float(np.nanmean(istyle) / np.nanmean(ir)) if np.nanmean(ir) != 0 else np.nan,
            "ic_full_adj": float(np.nanmean(ic_full)) if ic_full else np.nan,
            "t_full": (float(np.nanmean(ic_full) / np.nanstd(ic_full) * np.sqrt(len(ic_full)))
                       if ic_full else np.nan),
            "t_style": float(np.nanmean(istyle) / np.nanstd(istyle) * np.sqrt(len(istyle))),
            "max_corr_size": corr_style[0],
            "max_corr_price": corr_style[1],
            "max_corr_amt": corr_style[2],
            "n_weeks": len(ir),
        }
        recs.append(rec)
        print(f"  {rec['alpha']}: raw {rec['ic_raw']:+.4f} → 剔风格 {rec['ic_style_adj']:+.4f} "
              f"→ 再剔现有库 {rec['ic_full_adj']:+.4f}  "
              f"(|ρ| 市值 {rec['max_corr_size']:.2f} 股价 {rec['max_corr_price']:.2f} "
              f"额 {rec['max_corr_amt']:.2f})", flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(out / "incremental.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写出 {out / 'incremental.csv'}", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
