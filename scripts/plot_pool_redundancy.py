# -*- coding: utf-8 -*-
"""因子池冗余度图：34 个因子按相关结构聚类后，实际只有 8 个族 / 5.5 个独立信号。

左：34×34 平均相关矩阵（按族排序，白线为族边界），右侧附每个因子的平均 |权重|
右：族规模与「槽位 / 独立信号」的落差

用法::
    python scripts/plot_pool_redundancy.py
"""
from __future__ import annotations

import json
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from qf import config as C  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EVAL = C.BUILD / "eval"
SHOT = pathlib.Path(__file__).resolve().parents[1] / "reports" / "shots"
SHOT.mkdir(parents=True, exist_ok=True)

# 族名（依 2026-11-03 聚类结果；成员变化时会退化为「族N」）
FAM_NAME = {
    7: "趋势 / 反转 / 突破",
    2: "波动率 + 换手",
    5: "跳空动量",
    6: "量价相关",
    1: "缺口累积",
    4: "收益偏度",
    3: "趋势效率",
    8: "换手不稳定",
}
# 画图用的短名（底部空间窄）
FAM_SHORT = {
    7: "趋势/反转", 2: "波动+换手", 5: "跳空", 6: "量价相关",
    1: "缺口", 4: "偏度", 3: "效率", 8: "换手波动",
}


def short(k: str) -> str:
    return k.split(":")[-1]


def main():
    d = json.loads((EVAL / "pool_slim.json").read_text(encoding="utf-8"))
    corr = pd.read_csv(EVAL / "pool_corr.csv", index_col=0)
    w = d["weights"]

    order, fams, bounds = [], [], []
    for f in sorted(d["families"], key=lambda r: (-r["n"], r["family"])):
        mem = [k for k, _ in f["members"]]
        start = len(order)
        order += mem
        fams.append((f["family"], start, len(mem)))
        bounds.append(len(order))
    corr = corr.loc[order, order]

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.52], wspace=0.30,
                          left=0.075, right=0.985, top=0.855, bottom=0.185)

    # ---------------- 左：相关热图
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(order)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([""] * n)          # 矩阵对称，x 轴不再重复打标签
    ax.set_yticklabels([short(k) for k in order], fontsize=6.4)
    # 族边界
    for i, b in enumerate(bounds[:-1]):
        ax.axhline(b - 0.5, color="white", lw=2.2)
        ax.axvline(b - 0.5, color="white", lw=2.2)
    # 族标签放底部，两行短名；窄族（<=2 个）下沉一行，避免与相邻族挤在一起
    for fid, start, cnt in fams:
        nm = FAM_SHORT.get(fid, f"族{fid}")
        dy = 0.30 if cnt > 2 else 1.15
        ax.text(start + cnt / 2 - 0.5, n - 0.5 + dy,
                f"{nm}\n{cnt} 个", ha="center", va="top", fontsize=6.7,
                color="#2b2b2b", linespacing=1.35)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    cb.set_label("平均横截面相关 ρ（477 周）", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ---------------- 右：权重条 + 族规模对比
    ax2 = fig.add_subplot(gs[0, 1])
    ypos = np.arange(n)
    wv = np.array([w[k]["mean_abs"] for k in order])
    colors = []
    pal = ["#c0392b", "#e08a3c", "#d9b23a", "#4a8f6c", "#3f7fa8", "#6a6fb0", "#9a6aa8", "#8d8d8d"]
    for i, (fid, start, cnt) in enumerate(fams):
        colors += [pal[i % len(pal)]] * cnt
    ax2.barh(ypos, wv, color=colors, height=0.72, edgecolor="white", lw=0.4)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"{short(k)}  {w[k]['mean_abs']:.3f}" for k in order], fontsize=6.4)
    ax2.invert_yaxis()
    ax2.set_xlabel("平均 |权重|（23 次 refit）", fontsize=8.5)
    ax2.set_title("权重是分散的：前 10 个只占 55%", fontsize=9.6, pad=8)
    ax2.grid(axis="x", ls=":", lw=0.6, alpha=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    for i, b in enumerate(bounds[:-1]):
        ax2.axhline(b - 0.5, color="#cccccc", lw=0.8, ls="-")

    fig.suptitle("34 个因子槽，8 个族，5.5 个真正独立的信号", fontsize=13, y=0.965)
    fig.text(0.5, 0.915,
             f"平均 |ρ| = {d['avg_abs_rho']:.3f}　|ρ|≥0.7 有 {d['n_pairs_rho70']} 对　"
             f"特征值>1 的主成分 {d['n_pc_eig_gt1']} 个　"
             f"有效独立因子数（参与率）{d['n_effective']:.1f}",
             ha="center", fontsize=9, color="#555")
    out = SHOT / "因子池冗余结构.png"
    fig.savefig(out, dpi=160)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
