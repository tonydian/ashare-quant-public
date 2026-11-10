# -*- coding: utf-8 -*-
"""族上限实验的结论图（2026-11-03）。

一张图要说清的事：
  干预强度单调地伤害收益——不干预 0 → cap0.30/0.25 约 −0.1pp → cap0.20 −0.74pp →
  族等分 −0.93pp。同时 cap0.20 的生产相位 9.17% 是 8 相位最高值，
  与族预算的 8.71% 一样是「单点最优、分布更差」的陷阱。

输出：reports/shots/族上限_干预强度.png
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
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

INK, GRID = "#2b2f36", "#e4e7ec"
BLUE, RED, GREY = "#3b6fd4", "#c8483e", "#9aa2ad"

base = json.load(open(EVAL / "phase_band.json"))
base_c = {r["offset"]: r["cagr"] for r in base["rows"]}
base_mean = base["cagr"]["mean"] * 100
base_worst = base["cagr"]["min"] * 100

cases = [
    ("cap 0.30\n(17/23 期生效)", "phase_band_wf_cap030"),
    ("cap 0.25\n(24 族·期)", "phase_band_wf_cap025"),
    ("cap 0.20\n(42 族·期)", "phase_band_wf_cap020"),
    ("族等分\n(方案C)", None),
]
budget = json.load(open(EVAL / "phase_band_dimC.json"))

labels, d_mean, d_worst, prod, best = [], [], [], [], []
for tag, f in cases:
    d = budget if f is None else json.load(open(EVAL / f"{f}.json"))
    diffs = np.array([r["cagr"] - base_c[r["offset"]] for r in d["rows"]]) * 100
    labels.append(tag)
    d_mean.append(diffs.mean())
    d_worst.append(d["cagr"]["min"] * 100 - base_worst)
    prod.append(d["prod"]["cagr"] * 100)
    best.append(int(d["prod"]["n_better"]) == 0)   # 生产相位是否为 8 相位最高

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
bw = 0.36
b1 = ax.bar(x - bw / 2, d_mean, bw, color=RED, label="Δ 相位均值 (pp)")
b2 = ax.bar(x + bw / 2, d_worst, bw, color=GREY, label="Δ 最差相位 (pp)")
for b in list(b1) + list(b2):
    v = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, v - 0.045 if v < 0 else v + 0.02,
            f"{v:+.2f}", ha="center", va="top" if v < 0 else "bottom",
            fontsize=8, color=INK)

# 生产相位标注：越强的干预，生产相位反而越好看 —— 陷阱本身
for i, (p, is_best) in enumerate(zip(prod, best)):
    ax.annotate(f"生产相位 {p:.2f}%\n({ '8 相位最高!' if is_best else '排名中等' })",
                xy=(i, -1.52), ha="center", fontsize=7.6,
                color=RED if is_best else "#666",
                bbox=dict(boxstyle="round,pad=0.28",
                          fc="#fdf3f2" if is_best else "#f6f7f9",
                          ec=RED if is_best else "#d8dce2", lw=0.7))

ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(x, labels, fontsize=8.6)
ax.set_ylabel("相对现状的变化（百分点）", fontsize=9.5, color=INK)
ax.set_ylim(-1.86, 0.28)
ax.set_title("干预强度越大，相位分布越差；但生产相位反而越好看 —— 别被右下角骗了",
             fontsize=10.5, color=INK, pad=26)
ax.text(0.0, 1.045, "同池对比（21 因子不变，只改权重分配）· 8 个对齐调仓相位 · 现状均值 7.01% / 最差 5.46%",
        transform=ax.transAxes, fontsize=8.2, color="#666")
ax.legend(loc="upper right", fontsize=8.2, frameon=False)
ax.grid(axis="y", color=GRID, lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
out = SHOT / "族上限_干预强度.png"
fig.savefig(out, bbox_inches="tight")
print("→", out)
