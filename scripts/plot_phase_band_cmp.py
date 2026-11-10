# -*- coding: utf-8 -*-
"""相位带对比图：换手族接入前后，8 个调仓相位的 CAGR 分布。

左：分组柱（旧池 34 个 R 因子 vs 新池 +4 换手），虚线标各自相位均值，箭头指生产 offset
右：斜率图（同一 offset 前后连线），直观看出 7/8 个相位被抬高、离散度收窄

用法::
    python scripts/plot_phase_band_cmp.py
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from qf import config as C  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLR_OLD = "#9aa0a6"
CLR_NEW = "#c0392b"   # 改善用红（A 股习惯：涨为红）
CLR_DOWN = "#2e8b74"  # 变差用绿


def load(p):
    with open(C.BUILD / "eval" / p, encoding="utf-8") as f:
        return json.load(f)


def main():
    old = load("phase_band_bak.json")
    new = load("phase_band.json")
    o = {r["offset"]: r for r in old["rows"]}
    n = {r["offset"]: r for r in new["rows"]}
    offs = sorted(set(o) & set(n))
    xo = np.array([o[k]["cagr"] * 100 for k in offs])
    xn = np.array([n[k]["cagr"] * 100 for k in offs])
    prod = new["prod_offset"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    # ---------------- 左：分组柱
    ax = axes[0]
    x = np.arange(len(offs))
    w = 0.38
    b1 = ax.bar(x - w / 2, xo, w, label="旧池（34 个量价因子）", color=CLR_OLD,
                edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, xn, w, label="新池（+4 个换手因子）", color=CLR_NEW,
                edgecolor="white", linewidth=0.6)
    for b, v in list(zip(b1, xo)) + list(zip(b2, xn)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.10, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7.6, color="#333")
    ax.axhline(old["cagr"]["mean"] * 100, color=CLR_OLD, ls="--", lw=1.1, alpha=0.9)
    ax.axhline(new["cagr"]["mean"] * 100, color=CLR_NEW, ls="--", lw=1.1, alpha=0.9)
    # 均值标签放在柱子区右侧的留白里，避免压住 35 号柱子
    ax.set_xlim(-0.62, len(offs) + 0.95)
    for val, col, txt, dy in ((old["cagr"]["mean"] * 100, CLR_OLD,
                              f"旧均值\n{old['cagr']['mean']*100:.2f}%", -0.16),
                             (new["cagr"]["mean"] * 100, CLR_NEW,
                              f"新均值\n{new['cagr']['mean']*100:.2f}%", 0.16)):
        ax.text(len(offs) + 0.16, val + dy, txt, ha="left",
                va="top" if dy < 0 else "bottom", fontsize=8, color=col,
                linespacing=1.25)
    # 生产相位
    ip = offs.index(prod)
    ax.annotate("生产相位", xy=(ip + w / 2, xn[ip] + 0.05), xytext=(ip - 1.45, max(xn) + 0.95),
                fontsize=8, color="#1a1a1a",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#1a1a1a",
                                connectionstyle="arc3,rad=-0.12"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}" for k in offs])
    ax.set_xlabel("调仓相位 offset（40 日调仓，仅 5 的倍数与周频打分行对齐）")
    ax.set_ylabel("CAGR  %")
    ax.set_title("同一份因子分数，只改建仓日历：8 个相位的 CAGR", fontsize=10.5)
    ax.set_ylim(0, max(xn.max(), xo.max()) + 2.1)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.2, frameon=False, ncol=2,
              bbox_to_anchor=(0.0, 1.0))

    # ---------------- 右：斜率图
    ax = axes[1]
    for k, a, b in zip(offs, xo, xn):
        up = b >= a
        ax.plot([0, 1], [a, b], color=CLR_NEW if up else CLR_DOWN,
                lw=1.9 if k == prod else 1.0,
                alpha=0.95 if k == prod else 0.5, zorder=3 if k == prod else 2)
    # 右图只保留点，不标 offset 数字：8 个标签在 5-7% 这一小段必然互相压住，
    # 而 offset → CAGR 的明细左图已经给全了
    ax.scatter([0] * len(offs), xo, s=18, color=CLR_OLD, zorder=4)
    ax.scatter([1] * len(offs), xn, s=22,
               color=[CLR_NEW if b >= a else CLR_DOWN for a, b in zip(xo, xn)], zorder=4)
    ax.scatter([1], [n[prod]["cagr"] * 100], s=78, facecolors="none",
               edgecolors="#1a1a1a", lw=1.2, zorder=5)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(min(xo.min(), xn.min()) - 1.45, max(xo.max(), xn.max()) + 0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"旧池\n均值 {old['cagr']['mean']*100:.2f}%",
                        f"新池\n均值 {new['cagr']['mean']*100:.2f}%"], fontsize=9)
    ax.set_ylabel("CAGR  %")
    d = xn - xo
    ax.set_title(f"同相位配对：{int((d>0).sum())}/{len(d)} 个相位改善，"
                 f"均值 +{d.mean():.2f}pp", fontsize=10.5)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # 说明放底部留白区（ylim 已下压，不会压到线）
    ax.text(0.5, 0.015,
            f"每点 = 一个 offset（明细见左图）；粗线/圈 = 生产相位 offset {prod}\n"
            f"相位离散度 {old['cagr']['std']*100:.2f}pp → {new['cagr']['std']*100:.2f}pp"
            f"　最差相位 {old['cagr']['min']*100:.2f}% → {new['cagr']['min']*100:.2f}%",
            transform=ax.transAxes, fontsize=7.6, color="#444", va="bottom", ha="center",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#dcdfe3", lw=0.7))

    fig.suptitle("", fontsize=1)
    fig.tight_layout()
    out = C.BUILD.parent / "reports" / "shots"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "相位带_换手族接入前后.png"
    fig.savefig(dest, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"→ {dest}")
    print(f"  旧均值 {old['cagr']['mean']*100:.2f}%  新均值 {new['cagr']['mean']*100:.2f}%  "
          f"配对 Δ 均值 {d.mean():+.2f}pp  胜率 {(d>0).mean():.0%}")


if __name__ == "__main__":
    main()
