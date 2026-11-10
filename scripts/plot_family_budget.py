# -*- coding: utf-8 -*-
"""方案 C（族预算）的相位带对比图（2026-11-03）。

一张图要说清的事：
  族预算在生产相位（offset=10）拿到 8.71%，比现状的 8.38% 高；
  但同一个分数在另外 7 个相位几乎全面更低 → 均值 6.08% vs 7.01%。
  这正是「单点最优、分布更差」的样子。

输出：reports/shots/族预算_相位带.png
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

C_BASE = "#2f6f9f"
C_FAM = "#c0392b"
INK = "#2b2b2b"

PAIRS = [("phase_band.json", "现状：因子级权重（21 池）", C_BASE, "o"),
         ("phase_band_dimC.json", "方案 C：族等预算 1.0", C_FAM, "s")]


def main() -> None:
    fig = plt.figure(figsize=(12.6, 5.3))
    ax = fig.add_axes([0.065, 0.155, 0.585, 0.655])
    bx = fig.add_axes([0.72, 0.155, 0.245, 0.655])

    offs = None
    for fn, lab, clr, mk in PAIRS:
        d = json.loads((EVAL / fn).read_text(encoding="utf-8"))
        rows = sorted(d["rows"], key=lambda r: r["offset"])
        offs = [r["offset"] for r in rows]
        c = np.array([r["cagr"] for r in rows]) * 100
        ax.plot(offs, c, marker=mk, ms=6.2, lw=1.9, color=clr, label=lab, zorder=3)
        ax.axhline(c.mean(), color=clr, ls=":", lw=1.0, alpha=0.75, zorder=2)
        ax.text(offs[-1] + 1.0, c.mean(), f"均值 {c.mean():.2f}%", color=clr,
                fontsize=7.8, va="center", ha="left")

        # 生产相位单独标注
        i = [k for k, r in enumerate(rows) if r["offset"] == d["prod_offset"]][0]
        ax.scatter([offs[i]], [c[i]], s=150, facecolor="white", edgecolor=clr,
                   lw=2.0, zorder=5)

    # 生产相位竖线
    p = 10
    ax.axvline(p, color="#9aa0a6", ls="--", lw=1.0, zorder=1)
    ax.text(p, ax.get_ylim()[1] * 0.995, "  生产相位 offset=10", fontsize=8.0,
            color="#5f6368", va="top", ha="left")

    ax.set_xlabel("调仓相位 offset（同一份分数的不同建仓日）", fontsize=9)
    ax.set_ylabel("CAGR  %", fontsize=9)
    ax.set_xticks(offs)
    ax.grid(axis="y", ls=":", lw=0.55, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=8.2, frameon=False)
    ax.set_title("① 8 个相位的 CAGR：族预算只在生产相位赢，其余 7 个几乎全输",
                 fontsize=10.6, pad=8, color=INK)

    # ---------------- 右：均值 / 最差 / 中位 三根对比
    labels, vals_base, vals_fam = ["均值", "中位", "最差"], [], []
    for fn in ("phase_band.json", "phase_band_dimC.json"):
        d = json.loads((EVAL / fn).read_text(encoding="utf-8"))
        c = d["cagr"]
        v = [c["mean"] * 100, c["median"] * 100, c["min"] * 100]
        (vals_base if fn.startswith("phase_band.") else vals_fam).extend(v)
    x = np.arange(len(labels))
    bx.bar(x - 0.19, vals_base, width=0.36, color=C_BASE, edgecolor="white", lw=0.7,
           label="现状", zorder=3)
    bx.bar(x + 0.19, vals_fam, width=0.36, color=C_FAM, edgecolor="white", lw=0.7,
           label="族预算 1.0", zorder=3)
    for xi, a, b in zip(x, vals_base, vals_fam):
        bx.text(xi - 0.19, a + 0.12, f"{a:.2f}", ha="center", va="bottom", fontsize=7.4, color=INK)
        bx.text(xi + 0.19, b + 0.12, f"{b:.2f}", ha="center", va="bottom", fontsize=7.4, color=INK)
    bx.set_xticks(x)
    bx.set_xticklabels(labels, fontsize=8.6)
    bx.set_ylabel("CAGR  %", fontsize=8.8)
    bx.set_ylim(0, max(vals_base + vals_fam) * 1.22)
    bx.grid(axis="y", ls=":", lw=0.55, alpha=0.6)
    bx.set_axisbelow(True)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.legend(loc="upper right", fontsize=7.6, frameon=False, ncol=1)
    bx.set_title("② 三项分布指标全线下移", fontsize=10.0, pad=8, color=INK)

    fig.suptitle("方案 C（族预算）不该落地：生产相位 8.71% 是 8 档里的最高值，不是能力",
                 fontsize=12.0, x=0.028, ha="left", y=0.945, color=INK)
    fig.text(0.028, 0.885,
             "族预算把 bp 权重从 7.1% 抬到 16.7%、持仓 bp 分位 0.744→0.833，"
             "但价值暴露已被证明没有变现（见 bp 风格剥离），加大暴露只加大未定价风险。",
             fontsize=8.3, color="#4a4a4a", ha="left")

    out = SHOT / "族预算_相位带.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"已保存 → {out}")


if __name__ == "__main__":
    main()
