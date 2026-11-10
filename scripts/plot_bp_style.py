# -*- coding: utf-8 -*-
"""bp 增量的风格剥离图（2026-11-03）。

两个面板：
  左：两池差值的分年贡献（21 池 − 20 池，周复利）—— 增量是否集中在某一年
  右：bp 的原始 IC vs 残差 IC（剔除行业+市值+20池复合分）—— 多少是重复信息

数据来源：build/eval/bp_style.json、bp_style_detail.json
输出：reports/shots/bp_风格剥离.png
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "eval"
SHOT = ROOT / "reports" / "shots"
SHOT.mkdir(parents=True, exist_ok=True)

for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(x.name == f for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

C_POS = "#2f6f9f"
C_NEG = "#b0b6bd"
C_RAW = "#95a5a6"
C_RES = "#c0392b"
INK = "#2b2b2b"
GRID = "#d8d8d8"


def main() -> None:
    st = json.loads((EVAL / "bp_style.json").read_text(encoding="utf-8"))
    dt = json.loads((EVAL / "bp_style_detail.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(13.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22,
                          left=0.065, right=0.975, top=0.80, bottom=0.20)

    # ---------------- 左：差值分年贡献
    ax = fig.add_subplot(gs[0, 0])
    yrs = sorted(int(y) for y in dt["diff_by_year"])
    vals = [dt["diff_by_year"][str(y)] * 100 for y in yrs]
    x = np.arange(len(yrs))
    ax.bar(x, vals, width=0.62,
           color=[C_POS if v >= 0 else C_NEG for v in vals],
           edgecolor="white", linewidth=0.7, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.28 if v >= 0 else -0.42), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.8, color=INK, zorder=4)
    ax.axhline(0, color="#5a5a5a", lw=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in yrs], fontsize=8.4)
    ax.set_ylabel("21 池 − 20 池  周复利累计  %", fontsize=9)
    ax.set_title("① 增量集中在 2024 一年，不是逐年均匀", fontsize=10.5, pad=8, color=INK)
    ax.set_ylim(min(vals) - 1.6, max(vals) + 1.6)
    ax.grid(axis="y", ls=":", lw=0.55, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    share = dt.get("top4_share_of_pos", 0)
    ax.text(0.5, -0.155,
            f"全期累计 +{st['diff_total']*100:.2f}%，其中 2022-24 占 {st['diff_share_2224']*100:.0f}% ；"
            f"最强 4 周只解释正贡献的 {share*100:.0f}% → 不是单周事件",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="#555")

    # ---------------- 右：原始 IC vs 残差 IC
    ax = fig.add_subplot(gs[0, 1])
    y1 = st["bp_ic_by_year"]["mean"]
    y2 = st["resid_ic_by_year"]["mean"]
    common = sorted(set(int(k) for k in y1) & set(int(k) for k in y2))
    common = [y for y in common if y >= 2020]
    raw = [y1[str(y)] for y in common]
    res = [y2[str(y)] for y in common]
    xx = np.arange(len(common))
    ax.bar(xx - 0.19, raw, width=0.36, color=C_RAW, edgecolor="white", lw=0.6,
           label=f"原始 IC（均值 {st['bp_ic_all']:+.4f}）", zorder=3)
    ax.bar(xx + 0.19, res, width=0.36, color=C_RES, edgecolor="white", lw=0.6,
           label=f"残差 IC（均值 {st['resid_ic_all']:+.4f}，保留 29%）", zorder=3)
    ax.axhline(0, color="#5a5a5a", lw=0.9, zorder=2)
    ax.set_xticks(xx)
    ax.set_xticklabels([str(y) for y in common], fontsize=8.4)
    ax.set_ylabel("RankIC", fontsize=9)
    ax.set_title("② 原始 IC 的 71% 是行业/市值/现有因子的重复", fontsize=10.5, pad=8, color=INK)
    ax.grid(axis="y", ls=":", lw=0.55, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.6, frameon=False, ncol=1,
              bbox_to_anchor=(0.005, 1.005))

    fig.suptitle("bp 的价值暴露是真的，但它不是风格 beta —— 是弱而集中在小样本上的选股 alpha",
                 fontsize=12.2, x=0.028, ha="left", y=0.955, color=INK)
    fig.text(0.028, 0.885,
             "两池差值回归：β价值 −0.016(t=−1.19)、β市值 +0.004、β动量 +0.007，三腿合计 R²=1.3%；"
             "alpha 1.65%/年(t=1.72)。持仓 bp 分位 0.63→0.74（40/40 期更高）但没拿到对应溢价。",
             fontsize=8.4, color="#4a4a4a", ha="left")

    out = SHOT / "bp_风格剥离.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"已保存 → {out}")


if __name__ == "__main__":
    main()
