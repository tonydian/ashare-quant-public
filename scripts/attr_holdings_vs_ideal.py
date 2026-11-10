# -*- coding: utf-8 -*-
"""实际持仓 vs 当期理想 top-k：区分「模型选错」与「执行滞后损耗」。

背景
----
主回测换手只有 0.70/年（40 日调仓 + trade_band 0.3 + 每期最多新建 6 只），
持仓是"陈旧组合"。若模型分数的有效期短于持仓期，实际持仓会系统性偏离
当期排名，alpha 在持有过程中被磨掉。

本脚本逐期比较三条收益：
    R_domain   域等权（选股域基准）
    R_actual   实际持仓（holdings 权重）
    R_ideal    当期模型 top-k 等权（完全按当期分数重建，理想执行）

用法::

    python scripts/attr_holdings_vs_ideal.py --model build/models/wf_h10 --k 40
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import data as qfdata       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "build" / "runs" / "main"))
    ap.add_argument("--model", default=str(ROOT / "build" / "models" / "wf_h10"))
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "build" / "attr_bridge"))
    args = ap.parse_args()
    run, mdl = pathlib.Path(args.run), pathlib.Path(args.model)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md = qfdata.load_market(verbose=False)
    hold = pd.read_parquet(run / "holdings.parquet")
    hold["date"] = pd.to_datetime(hold["date"])
    hold["code"] = hold["code"].astype(str).str.zfill(6)
    z = np.load(mdl / "composite_score.npz")
    sc = z["scores"]
    # 分数只在部分周频行有值（回测区间内），取 <= t1 的最近一个「有分数」的行
    _fnz = np.flatnonzero(np.isfinite(sc).sum(axis=1) > 0)
    def _srow(t1: int) -> int:
        k = int(np.searchsorted(_fnz, t1, side="right")) - 1
        return int(_fnz[max(k, 0)])
    cl = np.asarray(md.close, dtype=np.float64)
    ci = {c: j for j, c in enumerate(md.codes)}

    dates = sorted(hold["date"].unique())
    recs = []
    for i, d in enumerate(dates):
        t1 = int(md.dates.searchsorted(pd.Timestamp(d)))
        t2 = (int(md.dates.searchsorted(pd.Timestamp(dates[i + 1])))
              if i + 1 < len(dates) else md.T - 2)
        g = hold[hold["date"] == d]
        hcodes = list(g["code"])
        hidx = np.array([ci.get(c, -1) for c in hcodes])
        w = g["weight"].to_numpy(float)
        okh = hidx >= 0
        hidx, w = hidx[okh], w[okh]

        with np.errstate(invalid="ignore", divide="ignore"):
            r = cl[t2] / np.where(cl[t1] > 0, cl[t1], np.nan) - 1.0
        rok = np.isfinite(r) & (r > -0.99)
        r_use = np.where(rok, r, 0.0)
        elig = np.asarray(md.eligible[t1], dtype=bool)
        base = elig & rok
        R_dom = float(r[base].mean())

        # 实际持仓（含现金）
        W = float(w.sum())
        R_act_stock = float((w * r_use[hidx]).sum()) / max(W, 1e-9)

        # 当期理想 top-k：用 t1 的模型分数，域内排序
        st = _srow(t1)
        s = sc[st]
        cand = np.flatnonzero(elig & np.isfinite(s))
        if len(cand) < args.k:
            continue
        order = cand[np.argsort(-s[cand])]
        top = order[:args.k]
        R_ideal = float(r_use[top].mean())
        # 重合度
        overlap = len(set(top.tolist()) & set(hidx.tolist())) / max(len(hidx), 1)
        # 分数分位（1.0 = 当期排名最高）
        rp = pd.Series(s[cand]).rank(pct=True).to_numpy()
        t_rank = float(rp[np.argsort(-s[cand])[:args.k]].mean())
        hpos = np.searchsorted(cand, hidx)
        hpos_c = np.clip(hpos, 0, len(cand) - 1)
        hok = cand[hpos_c] == hidx
        h_rank = float(rp[hpos_c[hok]].mean()) if hok.any() else np.nan

        recs.append(dict(date=d, year=pd.Timestamp(d).year, n=len(hidx), W=W,
                         R_dom=R_dom, R_act=R_act_stock, R_ideal=R_ideal,
                         overlap=overlap, h_rank=h_rank, t_rank=t_rank,
                         L=t2 - t1))

    df = pd.DataFrame(recs)
    ppy = 244.0 / df["L"].mean()
    # 统一到「股票部分满仓」口径
    df["ex_act"] = df["R_act"] - df["R_dom"]          # 实际持仓超额（满仓）
    df["ex_ideal"] = df["R_ideal"] - df["R_dom"]      # 理想 top-k 超额（满仓）
    df["gap"] = df["ex_ideal"] - df["ex_act"]         # 执行滞后损耗
    df["ex_act_net"] = df["ex_act"] * df["W"]         # 折算到净值

    yr = df.groupby("year").agg({c: "mean" for c in df.columns if c != "date"})
    for c in ("ex_act", "ex_ideal", "gap", "ex_act_net"):
        yr[c] = yr[c] * ppy * 100
    yr["overlap"] = yr["overlap"] * 100
    yr["W"] = yr["W"] * 100

    pd.set_option("display.width", 240)
    print(f"周频期数 {len(df)}；每年 {ppy:.2f} 期；k={args.k}")
    print("\n=== 分年（超额为年化 %，满仓口径；ex_act_net 已乘仓位）===")
    print(yr[["n", "W", "ex_act", "ex_ideal", "gap", "ex_act_net",
              "overlap", "h_rank", "t_rank"]].round(2).to_string())
    tot = df[["ex_act", "ex_ideal", "gap", "ex_act_net", "overlap", "h_rank", "t_rank"]].mean()
    print("\n=== 全期 ===")
    print(f"实际持仓超额（满仓）   {tot['ex_act']*ppy*100:+.2f}%/年")
    print(f"理想 top{args.k} 超额      {tot['ex_ideal']*ppy*100:+.2f}%/年")
    print(f"执行滞后损耗          {tot['gap']*ppy*100:+.2f}%/年")
    print(f"实际超额折算净值       {tot['ex_act_net']*ppy*100:+.2f}%/年")
    print(f"持仓与 top{args.k} 重合度  {tot['overlap']*100:.1f}%")
    print(f"持仓平均分数分位        {tot['h_rank']:.3f}（理想 {tot['t_rank']:.3f}）")

    df.to_csv(out / "holdings_vs_ideal.csv", index=False, encoding="utf-8-sig")
    yr.to_csv(out / "holdings_vs_ideal_yearly.csv", encoding="utf-8-sig")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
