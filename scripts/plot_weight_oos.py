# -*- coding: utf-8 -*-
"""四侧权重的样本外体检 —— 出图 + 关键数字落盘。

图 1（两联）：左 = 各腿分区间年化（看"区间依赖"）；右 = 走查式权重优化 vs 固定权重净值
图 2（两联）：左 = 分年度贡献堆叠；右 = 滚动 36 个月年化分布（看稳定性）
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from qf import config as C  # noqa: E402
from qf import metrics as M  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LEGS = ["main", "lowturn", "etf", "sector"]
CN = {"main": "主策略", "lowturn": "低换手", "etf": "宽基ETF", "sector": "板块ETF"}
CLR = {"main": "#2f6fb5", "lowturn": "#6d4c9f", "etf": "#2e8b74", "sector": "#c9782f"}
FILES = {
    "main": ("build/runs/main/nav.parquet", "nav"),
    "lowturn": ("build/runs/lowturn/nav.parquet", "nav_real"),
    "etf": ("build/runs/etf_trend/nav_donchian55.parquet", "nav"),
    "sector": ("build/runs/sector_rotation/nav.parquet", "nav"),
}
CUR = np.array([0.40, 0.30, 0.20, 0.10])
W_GRID = None


def load():
    navs = {}
    for k, (fp, col) in FILES.items():
        s = pd.read_parquet(fp)[col].astype(float)
        s.index = pd.to_datetime(s.index)
        navs[k] = s.dropna()
    idx = None
    for s in navs.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.sort_values()
    return pd.DataFrame({k: navs[k].reindex(idx).ffill().pct_change()
                         for k in LEGS}).dropna()


def grid(step=0.05):
    n = int(round(1 / step))
    out = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                out.append((a * step, b * step, c * step, (n - a - b - c) * step))
    return np.array(out)


def perf(r):
    nav = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]),
                     (1 + r).cumprod()])
    return M.performance(nav, rf=C.CASH_YIELD)


def main():
    R = load()
    W = grid()
    SEGS = [("样本内\n20-21", "2020-01-09", "2021-12-31"),
            ("2022-23", "2022-01-04", "2023-12-31"),
            ("2024-26", "2024-01-02", None),
            ("全期", "2020-01-09", None)]

    out = {}

    # ---------- 图 1 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.3))

    ax = axes[0]
    x = np.arange(len(SEGS))
    wbar = 0.19
    for j, k in enumerate(LEGS):
        vals = []
        for nm, st, en in SEGS:
            sl = R.loc[st:en if en else R.index[-1], k]
            vals.append(perf(sl)["cagr"] * 100)
        ax.bar(x + (j - 1.5) * wbar, vals, wbar, label=CN[k], color=CLR[k], alpha=.92)
        out.setdefault("leg_cagr", {})[k] = [round(v, 2) for v in vals]
    comb = []
    for nm, st, en in SEGS:
        sl = R.loc[st:en if en else R.index[-1], LEGS]
        comb.append(perf(sl @ CUR)["cagr"] * 100)
    ax.plot(x, comb, "o-", color="#111", lw=2, ms=7, label="组合 40/30/20/10", zorder=5)
    for i, v in enumerate(comb):
        ax.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="#444", lw=.9)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SEGS], fontsize=9.5)
    ax.set_ylabel("年化收益 %", fontsize=10)
    ax.set_title("每条腿都自带一个「好区间」——单腿业绩不能外推", fontsize=12.5, pad=11)
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.09), frameon=False)
    ax.grid(axis="y", alpha=.22)
    ax.tick_params(labelsize=9)
    out["combo_cagr"] = [round(v, 2) for v in comb]

    # 走查
    pts = ["2023-01-03", "2024-01-02", "2025-01-02", "2026-01-05"]
    rows = []
    for i, p in enumerate(pts):
        t = pd.Timestamp(p)
        tr = R.loc[t - pd.DateOffset(years=3):t - pd.Timedelta(days=1)]
        te = R.loc[t:] if i == len(pts) - 1 else R.loc[t:pd.Timestamp(pts[i + 1]) - pd.Timedelta(days=1)]
        Rs = tr[LEGS].to_numpy() @ W.T
        sh = (Rs - C.CASH_YIELD / 252).mean(0) / Rs.std(0, ddof=1) * np.sqrt(252)
        rows.append((t, W[int(np.argmax(sh))], te))

    def stitch(pick):
        seg = []
        for t, w_opt, te in rows:
            w = w_opt if isinstance(pick, str) else pick
            seg.append(te[LEGS].to_numpy() @ np.asarray(w, dtype=float))
        return pd.Series(np.concatenate(seg), index=pd.concat([r[2] for r in rows]).index)

    vol = R.loc["2022-01-04":, LEGS].std() * np.sqrt(252)
    rp = ((1 / vol) / (1 / vol).sum()).to_numpy()
    curves = {
        "走查式优化（每年重选）": (stitch("opt"), "#c0392b", 2.0, "-"),
        "固定 40/30/20/10": (stitch(CUR), "#111", 2.2, "-"),
        "固定等权 25×4": (stitch(np.array([.25] * 4)), "#7f8c8d", 1.4, "--"),
        "固定风险平价": (stitch(rp), "#2e8b74", 1.4, "--"),
    }
    ax = axes[1]
    perf_tbl = {}
    for nm, (r, c, lw, ls) in curves.items():
        n = (1 + r).cumprod()
        n = n / n.iloc[0]
        ax.plot(n.index, n, color=c, lw=lw, ls=ls, label=f"{nm}（{perf(r)['cagr']*100:.2f}%）")
        p = perf(r)
        perf_tbl[nm] = {k: round(p[k], 3) if isinstance(p[k], float) else p[k]
                        for k in ("cagr", "sharpe", "max_dd", "ann_vol")}
    ax.set_title("每年按过去 3 年最优重选权重，反而更差（2023-01 ~ 2026-10）",
                 fontsize=12.5, pad=11)
    ax.set_ylabel("净值", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.grid(alpha=.22)
    ax.tick_params(labelsize=9)
    out["walkforward"] = perf_tbl
    fig.tight_layout()
    p1 = C.REPORTS / "shots" / "四侧权重_区间依赖与走查.png"
    p1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p1, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p1)

    # ---------- 图 2 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))

    ax = axes[0]
    contrib = R[LEGS].mul(CUR, axis=1) * 100
    y = contrib.groupby(contrib.index.year).sum()
    years = y.index.to_numpy()
    bottom = np.zeros(len(y))
    for k in LEGS:
        ax.bar(years, y[k].to_numpy(), .62, bottom=bottom, label=CN[k], color=CLR[k], alpha=.92)
        bottom += y[k].to_numpy()
    for i, v in enumerate(bottom):
        ax.annotate(f"{v:.1f}", (years[i], v), textcoords="offset points",
                    xytext=(0, 5) if v >= 0 else (0, -14), ha="center",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=.85))
    ax.axhline(0, color="#444", lw=.9)
    ax.set_ylabel("对组合的贡献（pp，按 40/30/20/10 加权）", fontsize=10)
    ax.set_title("分年度贡献：2021 靠主策略+板块，2024 靠低换手", fontsize=12.5, pad=11)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.08), frameon=False)
    ax.grid(axis="y", alpha=.22)
    ax.tick_params(labelsize=9)
    out["year_contrib"] = {int(k): [round(v, 2) for v in y.loc[k].to_numpy()] for k in y.index}

    # 滚动 36 月
    ax = axes[1]
    rolls = {}
    for k in LEGS:
        m = (1 + R[k]).resample("ME").prod() - 1
        roll = (1 + m).rolling(36).apply(np.prod, raw=True) ** (12 / 36) - 1
        ax.plot(roll.index, roll * 100, lw=1.6, color=CLR[k], label=CN[k], alpha=.85)
        rolls[k] = roll.dropna()
    pm = (1 + R[LEGS] @ CUR).resample("ME").prod() - 1
    rcomb = ((1 + pm).rolling(36).apply(np.prod, raw=True) ** (12 / 36) - 1).dropna()
    ax.plot(rcomb.index, rcomb * 100, lw=2.6, color="#111", label="组合 40/30/20/10")
    ax.fill_between(rcomb.index, 0, rcomb * 100, color="#111", alpha=.06)
    ax.axhline(0, color="#444", lw=.9, ls="--")
    ax.set_ylabel("滚动 36 个月年化 %", fontsize=10)
    ax.set_title("组合的滚动 3 年最差 3.3%，比任何单腿都稳——分散是真的", fontsize=12.5, pad=11)
    ax.set_ylim(top=max(32, float(rcomb.max()) * 100 + 4))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.08), frameon=False)
    ax.grid(alpha=.22)
    ax.tick_params(labelsize=9)
    out["roll36"] = {k: {"min": round(float(v.min()) * 100, 2), "median": round(float(v.median()) * 100, 2),
                         "pos_share": round(float((v > 0).mean()) * 100, 1)} for k, v in rolls.items()}
    out["roll36"]["combo"] = {"min": round(float(rcomb.min()) * 100, 2),
                              "median": round(float(rcomb.median()) * 100, 2),
                              "pos_share": round(float((rcomb > 0).mean()) * 100, 1)}
    fig.tight_layout()
    p2 = C.REPORTS / "shots" / "四侧权重_贡献与稳定性.png"
    fig.savefig(p2, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p2)

    with open(C.BUILD / "weight_oos_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("summary →", C.BUILD / "weight_oos_summary.json")


if __name__ == "__main__":
    main()
