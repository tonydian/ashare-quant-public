# -*- coding: utf-8 -*-
"""因子池精简的权衡图：8 个调仓相位的 CAGR 分布 vs 因子个数。

每行一个方案：横线 = 8 个相位的 [最差, 最好]，圆点 = 均值，叉 = 生产相位；
右侧一栏画平均最大回撤。用来回答「减到几个因子最划算、左尾有没有变好」。

用法::
    python scripts/plot_pool_slim_tradeoff.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from qf import config as C  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = C.BUILD / "eval"
SHOT = ROOT / "reports" / "shots"

# (标签, 相位带 json, 颜色, 是否基准)
CASES = [
    ("34 个（现状）", "phase_band.json", "#7f8c8d", True),
    ("24 个｜每类 3 个", "phase_band_cap3_24.json", "#2f6f9f", False),
    ("20 个｜每类 2 个", "phase_band_cap2_20.json", "#2f6f9f", False),
    ("20 个｜按权重", "phase_band_slim20.json", "#4a8f6c", False),
    ("16 个｜按权重", "phase_band_slim16.json", "#4a8f6c", False),
    ("16 个｜按 DEV ICIR", "phase_band_dev16.json", "#c0392b", False),
    ("16 个｜取消族配额", "phase_band_nocap16.json", "#c0392b", False),
    ("12 个｜按权重", "phase_band_slim12.json", "#d9a13a", False),
]


def main():
    rows = []
    for label, fn, color, base in CASES:
        p = EVAL / fn
        if not p.exists():
            print(f"[skip] {fn} 不存在")
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "label": label, "color": color, "base": base,
            "mean": b["cagr"]["mean"], "med": b["cagr"]["median"],
            "min": b["cagr"]["min"], "max": b["cagr"]["max"],
            "prod": b["prod"]["cagr"], "mdd": b["max_dd"]["mean"],
            "sharpe": b["sharpe"]["mean"],
            "k": int(b["prod"].get("n_pool") or 0) or None,
        })
    # 按均值降序排，视觉上从好到差
    rows.sort(key=lambda r: -r["mean"])

    fig = plt.figure(figsize=(12.4, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.06,
                          left=0.175, right=0.975, top=0.855, bottom=0.135)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)

    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.plot([r["min"] * 100, r["max"] * 100], [i, i], color=r["color"],
                lw=6.0 if r["base"] else 4.2, alpha=0.30, solid_capstyle="round",
                zorder=2)
        ax.scatter([r["mean"] * 100], [i], s=64 if r["base"] else 78,
                   color=r["color"], zorder=4, edgecolor="white", linewidth=1.0)
        ax.scatter([r["prod"] * 100], [i], marker="x", s=46, color=r["color"],
                   zorder=5, linewidth=1.5)
        ax.text(r["max"] * 100 + 0.13, i, f"均值 {r['mean']*100:.2f}%",
                va="center", ha="left", fontsize=8.2, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("8 个调仓相位的 CAGR（%）：横线为 [最差, 最好]，圆点为均值，叉为生产相位", fontsize=9)
    ax.set_xlim(2.2, 12.6)
    ax.grid(axis="x", ls=":", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # 右：平均最大回撤
    for i, r in enumerate(rows):
        v = abs(r["mdd"]) * 100
        ax2.barh(i, v, height=0.52, color=r["color"], alpha=0.30 if r["base"] else 0.72,
                 edgecolor="none")
        ax2.text(v + 0.22, i, f"-{v:.2f}%", va="center", fontsize=8, color="#333")
    ax2.set_xlabel("平均最大回撤（%）", fontsize=9)
    ax2.set_xlim(0, 20.5)
    ax2.grid(axis="x", ls=":", lw=0.6, alpha=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", length=0, labelleft=False)

    base = [r for r in rows if r["base"]][0]
    fig.suptitle("34 个因子删到 16 个：相位均值 +1.5pp，最差相位 +1.6pp，回撤还小了一点",
                 fontsize=12.6, y=0.955)
    fig.text(0.5, 0.895,
             f"基准 34 个：均值 {base['mean']*100:.2f}%、最差 {base['min']*100:.2f}%、"
             f"Sharpe {base['sharpe']:.3f}　→　"
             f"最好方案：均值 {rows[0]['mean']*100:.2f}%、最差 {rows[0]['min']*100:.2f}%、"
             f"Sharpe {rows[0]['sharpe']:.3f}",
             ha="center", fontsize=8.8, color="#555")

    out = SHOT / "因子池精简_相位权衡.png"
    fig.savefig(out, dpi=160)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
