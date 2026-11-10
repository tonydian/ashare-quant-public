# -*- coding: utf-8 -*-
"""复合分的「方向拆解」：多头端 alpha 与空头端 alpha 分年对比。

动机
----
组合是**纯多头**。如果复合分的信息主要集中在空头端（低分组显著跑输），
那么 IC 可以很稳定，但 top-k 多头组合的超额会长期接近于零。
本脚本把每期分组收益拆成两段：

    多头端（long）  = Q5 - Q3     组合实际能吃到的部分
    空头端（short） = Q3 - Q1     纯多头组合吃不到的部分

用法::

    python scripts/attr_alpha_direction.py --model build/models/wf_h10 --horizon 10
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C          # noqa: E402
from qf import data as qfdata       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "build" / "models" / "wf_h10"))
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "build" / "attr_bridge"))
    args = ap.parse_args()
    mdl = pathlib.Path(args.model)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md = qfdata.load_market(verbose=False)
    z = np.load(mdl / "composite_score.npz")
    sc, rows = z["scores"], z["rows"].astype(np.int64)
    lab = md.fwd_open_ret(args.horizon)

    print(f"周频行 {len(rows)}，标签 {args.horizon} 日开盘口径")

    recs = []
    for r, t in enumerate(rows):
        s = sc[t]
        y = lab[t]
        elig = np.asarray(md.eligible[t], dtype=bool)
        m = elig & np.isfinite(s) & np.isfinite(y)
        if m.sum() < 200:
            continue
        sv, yv = s[m], y[m]
        # 5 组（分位）
        q = np.full(m.sum(), -1, dtype=int)
        try:
            edges = np.quantile(sv, [0.2, 0.4, 0.6, 0.8])
        except Exception:
            continue
        q = np.searchsorted(edges, sv, side="right")
        g = np.array([yv[q == k].mean() if (q == k).sum() >= 20 else np.nan for k in range(5)])
        # top-k 多头（模拟实际持仓宽度）
        order = np.argsort(-sv)
        univ = yv.mean()
        tk = {}
        for k in (20, 40, 60, 100):
            if len(order) > k:
                tk[f"top{k}"] = yv[order[:k]].mean() - univ
        # Spearman IC
        ic = float(pd.Series(sv).corr(pd.Series(yv), method="spearman"))
        recs.append(dict(date=md.dates[t], n=int(m.sum()), univ=univ, ic=ic,
                         q1=g[0], q2=g[1], q3=g[2], q4=g[3], q5=g[4],
                         long=g[4] - g[2], short=g[2] - g[0], ls=g[4] - g[0],
                         **tk))

    df = pd.DataFrame(recs)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    ppy = 244.0 / args.horizon          # 每年期数

    def agg(g):
        return pd.Series({
            "n": len(g),
            "ic": g["ic"].mean(),
            "ic_t": g["ic"].mean() / g["ic"].std(ddof=1) * np.sqrt(len(g)) if len(g) > 2 else np.nan,
            "Q1": g["q1"].mean() * ppy * 100,
            "Q2": g["q2"].mean() * ppy * 100,
            "Q3": g["q3"].mean() * ppy * 100,
            "Q4": g["q4"].mean() * ppy * 100,
            "Q5": g["q5"].mean() * ppy * 100,
            "univ": g["univ"].mean() * ppy * 100,
            "long": g["long"].mean() * ppy * 100,
            "short": g["short"].mean() * ppy * 100,
            "ls": g["ls"].mean() * ppy * 100,
            "top20": g["top20"].mean() * ppy * 100,
            "top40": g["top40"].mean() * ppy * 100,
            "top60": g["top60"].mean() * ppy * 100,
            "top100": g["top100"].mean() * ppy * 100,
        })

    yr = df.groupby("year").apply(agg, include_groups=False)
    tot = agg(df).to_frame("全期").T

    pd.set_option("display.width", 220)
    print("\n=== 分组年化收益（%，Q5=最高分；多头端=Q5-Q3，空头端=Q3-Q1）===")
    print(yr[["n", "ic", "ic_t", "Q1", "Q3", "Q5", "univ", "long", "short", "ls"]].round(2).to_string())
    print("\n=== 全期 ===")
    print(tot[["n", "ic", "ic_t", "Q1", "Q3", "Q5", "univ", "long", "short", "ls"]].round(2).to_string())
    print("\n=== top-k 相对域等权的年化超额（%）===")
    print(yr[["top20", "top40", "top60", "top100"]].round(2).to_string())
    print("\n=== 全期 top-k ===")
    print(tot[["top20", "top40", "top60", "top100"]].round(2).to_string())

    df.to_csv(out / "alpha_direction_periods.csv", index=False, encoding="utf-8-sig")
    yr.to_csv(out / "alpha_direction_yearly.csv", encoding="utf-8-sig")
    tot.to_csv(out / "alpha_direction_total.csv", encoding="utf-8-sig")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
