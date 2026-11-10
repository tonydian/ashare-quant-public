# -*- coding: utf-8 -*-
"""低换手 vs 红利：净值 / 分年度 / 累计超额 三联图。

输出 reports/shots/低换手_vs_红利.png
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf.data import load_market   # noqa: E402
from qf import lowturn            # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DIV = ROOT / "build" / "etf" / "div_panel.parquet"
OUT = ROOT / "reports" / "shots" / "低换手_vs_红利.png"


def load_div(idx):
    d = pd.read_parquet(DIV)
    out = {}
    for code, g in d.groupby("code"):
        s = g.sort_values("date").set_index("date")["close_adj"].astype(float)
        out[code] = s[~s.index.duplicated()].reindex(idx.union(s.index)).ffill().reindex(idx)
    return out


def main():
    md = load_market(verbose=False)
    res = lowturn.backtest(md, start="2020-01-02")
    nav, nav_g = res["nav"], res["nav_gross"]
    idx = nav.index
    div = load_div(idx)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0),
                             gridspec_kw={"width_ratios": [1.5, 1, 1.2]})
    fig.patch.set_facecolor("white")

    # ---- 1) 净值 ----
    ax = axes[0]
    ax.plot(idx, nav / nav.iloc[0], lw=1.9, color="#c0392b", label=f"低换手（净）")
    ax.plot(idx, nav_g / nav_g.iloc[0], lw=1.1, color="#e08e0b", ls="--", label="低换手（毛，零成本）")
    cfgs = [("515080", "#2c6fbb", "中证红利ETF"), ("512890", "#1f8a70", "红利低波ETF"),
            ("510880", "#8e7cc3", "上证红利ETF")]
    for code, col, lab in cfgs:
        s = div[code].dropna()
        if len(s) < 250:
            continue
        ax.plot(s.index, s / s.iloc[0], lw=1.4, color=col, label=lab)
    b = md.bench_prices["csi_all"].reindex(idx).ffill()
    ax.plot(idx, b / b.iloc[0], lw=1.1, color="#999999", ls=":", label="中证全指")
    ax.set_title("净值对比（2020-01 起，红利ETF 为后复权含股息）", fontsize=12)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=.9)
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=8)

    # ---- 2) 分年度 ----
    ax = axes[1]
    yl = nav.pct_change().groupby(nav.index.year).apply(lambda x: (1 + x).prod() - 1)
    bl = div["515080"].dropna()
    yb = bl.pct_change().groupby(bl.index.year).apply(lambda x: (1 + x).prod() - 1)
    df = pd.concat([yl.rename("lt"), yb.rename("div")], axis=1).dropna()
    x = np.arange(len(df))
    ax.bar(x - .2, df["lt"] * 100, .38, color="#c0392b", label="低换手（净）")
    ax.bar(x + .2, df["div"] * 100, .38, color="#2c6fbb", label="中证红利ETF")
    ax.axhline(0, color="#333", lw=.8)
    ax.set_xticks(x, [str(i) for i in df.index], fontsize=8.5)
    ax.set_title("分年度收益：2024 一年定胜负", fontsize=12)
    ax.set_ylabel("%", fontsize=9)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=.25)
    ax.tick_params(labelsize=8)

    # ---- 3) 相对净值（复利口径） ----
    ax = axes[2]
    a = pd.concat([nav, bl], axis=1).dropna()
    a.columns = ["s", "b"]
    # 几何口径：净值比，而不是日频超额累加（后者忽略波动拖累，会低估低波策略）
    ratio = (a["s"] / a["s"].iloc[0]) / (a["b"] / a["b"].iloc[0])
    ax.plot(ratio.index, (ratio - 1) * 100, lw=1.7, color="#6d4c9f")
    ax.axhline(0, color="#333", lw=.8, ls="--")
    ax.fill_between(ratio.index, 0, (ratio - 1) * 100, color="#6d4c9f", alpha=.15)
    ax.set_title("相对中证红利ETF 的净值比（复利口径，>0 跑赢）", fontsize=12)
    ax.set_ylabel("相对净值 %", fontsize=9)
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=8)
    ax.annotate("2020-23 成长风格：累计落后 25pp\n2024-09 银行行情一次拉回\n全期末尾 ≈ -6%（年化差 -0.04pp）",
                xy=(0.52, 0.06), xycoords="axes fraction", va="bottom", fontsize=8.5,
                color="#6d4c9f",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c9bcd9", lw=.8, alpha=.95))

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor="white", bbox_inches="tight")
    print("[out]", OUT)


if __name__ == "__main__":
    main()
