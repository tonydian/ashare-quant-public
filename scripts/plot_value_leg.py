# -*- coding: utf-8 -*-
"""独立价值腿实验图（2026-11-03）。

左：bp 分位 10 档 → 10 日前向收益年化（全市场 / 小盘 / 大盘），标出「价值陷阱区」
右：组合层配比的有效前沿（CAGR vs MDD），三条价值腿轨迹 + 直接降仓位参考线

数据：build/eval/value_bins.json、legr_*.json、value_leg_variants.json
输出：reports/shots/价值腿_实验图.png
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
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

INK = "#2b2b2b"
GRID = "#dcdcdc"
BLUE = "#2f6f9f"
RED = "#b03a2e"
TEAL = "#1f7a6b"
GREY = "#8a9099"
TRAP = "#f3d9d5"

EDGES = [0, .2, .4, .5, .6, .7, .8, .9, .95, .98, 1.0]


def compute_bins() -> dict:
    from qf import data as qfdata
    md = qfdata.load_market(verbose=False)
    bp = np.load(ROOT / "build" / "factors" / "value_extra.npz")[
        "fundamental-extra:F002"].astype(np.float64)
    zm = np.load(ROOT / "build" / "factors" / "weekly_meta.npz")
    rows = zm["rows"].astype(np.int64)
    elig = zm["eligible"]
    fwd = md.fwd_open_ret(10)[rows]
    lmc = md.logmcap[rows]
    k = len(EDGES) - 1
    out = {}
    for scope in ("all", "small", "large"):
        acc, cnt = np.zeros(k), np.zeros(k)
        for i in range(len(rows)):
            m = elig[i] & np.isfinite(bp[i]) & np.isfinite(fwd[i]) & np.isfinite(lmc[i])
            if m.sum() < 300:
                continue
            if scope == "small":
                m = m & (lmc[i] <= np.quantile(lmc[i][m], 0.34))
            elif scope == "large":
                m = m & (lmc[i] >= np.quantile(lmc[i][m], 0.67))
            if m.sum() < 100:
                continue
            v, y = bp[i][m], fwd[i][m]
            qs = np.quantile(v, EDGES)
            idx = np.clip(np.searchsorted(qs, v, side="right") - 1, 0, k - 1)
            for g in range(k):
                s = idx == g
                if s.sum() >= 5:
                    acc[g] += y[s].mean()
                    cnt[g] += 1
        out[scope] = (acc / np.maximum(cnt, 1) * 25.0).tolist()
    (EVAL / "value_bins.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def mix_curve(leg: str, ws) -> list[dict]:
    m = json.load(open(EVAL / "legr_main.json", encoding="utf-8"))
    v = json.load(open(EVAL / f"legr_{leg}.json", encoding="utf-8"))
    from qf import metrics as qfmetrics
    rows = []
    for w in ws:
        cs, ds, ss = [], [], []
        for off in m:
            a = pd.Series(m[off]["r"], index=pd.to_datetime(m[off]["idx"]))
            b = pd.Series(v[off]["r"], index=pd.to_datetime(v[off]["idx"]))
            j = a.index.intersection(b.index)
            x = (1 - w) * a.reindex(j) + w * b.reindex(j)
            nav = (1 + x).cumprod()
            p = qfmetrics.performance(nav, rf=0.015)
            cs.append(p["cagr"]); ds.append(p["max_dd"]); ss.append(p["sharpe"])
        rows.append({"w": w, "cagr": float(np.mean(cs)), "mdd": float(np.mean(ds)),
                     "sharpe": float(np.mean(ss))})
    return rows


def cash_curve(ws) -> list[dict]:
    from qf import metrics as qfmetrics
    m = json.load(open(EVAL / "legr_main.json", encoding="utf-8"))
    rows = []
    for eq in ws:
        cs, ds = [], []
        for off in m:
            r = pd.Series(m[off]["r"], index=pd.to_datetime(m[off]["idx"]))
            x = eq * r + (1 - eq) * (0.015 / 252)
            nav = (1 + x).cumprod()
            p = qfmetrics.performance(nav, rf=0.015)
            cs.append(p["cagr"]); ds.append(p["max_dd"])
        rows.append({"w": 1 - eq, "cagr": float(np.mean(cs)), "mdd": float(np.mean(ds))})
    return rows


def main() -> None:
    bins = compute_bins()
    ws = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    curves = {k: mix_curve(k, ws) for k in ("value_ind", "value_mid", "value_small")}
    cash = cash_curve([1.0, 0.95, 0.9, 0.85, 0.8, 0.7])

    fig = plt.figure(figsize=(13.4, 5.6), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.975, top=0.80, bottom=0.155,
                          wspace=0.28)

    # ---------------- 左：分位收益 ----------------
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(EDGES) - 1)
    ax.axvspan(8.5, 9.5, color=TRAP, zorder=0)
    styles = {"all": (BLUE, "全市场", "-", "o"), "small": (TEAL, "小盘后 1/3", "-", "s"),
              "large": (GREY, "大盘前 1/3", "--", "^")}
    for scope, (c, lab, ls, mk) in styles.items():
        y = np.array(bins[scope]) * 100
        ax.plot(x, y, ls, color=c, lw=1.9, marker=mk, ms=4.2, label=lab, zorder=3,
                mfc="white", mew=1.3)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{EDGES[i]:.2f}-{EDGES[i+1]:.2f}" for i in range(len(EDGES) - 1)],
        fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("bp 分位（左 = 贵，右 = 便宜）", fontsize=9.5, color=INK, labelpad=4)
    ax.set_ylabel("10 日前向收益 · 年化 %", fontsize=9.5, color=INK)
    ax.set_ylim(-4.2, 18.5)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title("价值的多头端：越便宜越好，但最末 2% 掉头", fontsize=10.5,
                 color=INK, pad=8, loc="left")
    ax.annotate("价值陷阱区：最末 2%\n现行价值腿正买在这里（0.982 分位）",
                xy=(9, bins["all"][-1] * 100), xytext=(5.9, -1.9),
                fontsize=7.8, color=RED, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdf3f2", ec=TRAP, lw=0.8),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0,
                                connectionstyle="arc3,rad=-0.3"))

    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=1,
              bbox_to_anchor=(0.0, 0.99))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)

    # ---------------- 右：配比前沿 ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    cx = [-r["mdd"] * 100 for r in cash]
    cy = [r["cagr"] * 100 for r in cash]
    ax2.plot(cx, cy, "--", color=GREY, lw=1.6, label="直接降股票仓位", zorder=2)
    ax2.scatter(cx, cy, s=16, color=GREY, zorder=3, marker="|")

    pal = {"value_ind": (RED, "价值腿：最便宜（现行口径）"),
           "value_mid": (BLUE, "价值腿：便宜但不极端"),
           "value_small": (TEAL, "价值腿：小盘内最便宜")}
    for k, (c, lab) in pal.items():
        r = curves[k]
        xx = [-t["mdd"] * 100 for t in r]
        yy = [t["cagr"] * 100 for t in r]
        ax2.plot(xx, yy, "-", color=c, lw=1.9, marker="o", ms=4.2, label=lab,
                 zorder=4, mfc="white", mew=1.3)
        dy = {"value_ind": 9, "value_mid": -12, "value_small": -13}[k]
        ax2.annotate(f"混入 {r[-1]['w']:.0%}", xy=(xx[-1], yy[-1]), xytext=(3, dy),
                     textcoords="offset points", fontsize=7.2, color=c)

    ax2.scatter([-curves["value_mid"][0]["mdd"] * 100], [curves["value_mid"][0]["cagr"] * 100],
                s=70, color=INK, zorder=6, marker="D")
    ax2.annotate("主线 100%", xy=(-curves["value_mid"][0]["mdd"] * 100,
                                 curves["value_mid"][0]["cagr"] * 100),
                 xytext=(-9, 7), textcoords="offset points", fontsize=8.5,
                 color=INK, fontweight="bold")
    ax2.set_xlabel("平均最大回撤（%，越右越安全）", fontsize=9.5, color=INK, labelpad=4)
    ax2.set_ylabel("跨 8 相位平均 CAGR %", fontsize=9.5, color=INK)
    ax2.grid(color=GRID, lw=0.7, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.set_title("配比前沿：价值腿给的东西，降仓位也能给", fontsize=10.5,
                  color=INK, pad=8, loc="left")
    ax2.legend(loc="lower right", fontsize=7.8, frameon=False)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(labelsize=8, colors=INK)

    fig.text(0.065, 0.915, "独立价值腿：能不能从组合层拿到这份收益", fontsize=13.5,
             color=INK, fontweight="bold")
    fig.text(0.065, 0.868,
             "左图：bp 十分位的前向收益（2020-2026，475 周）。"
             "右图：主线与三种价值腿按权重混合，8 个调仓相位各跑一次后取均值；虚线是同等的降仓位方案。",
             fontsize=8.6, color="#5a6068")
    out = SHOT / "价值腿_实验图.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
