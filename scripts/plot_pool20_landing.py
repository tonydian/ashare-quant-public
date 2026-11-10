# -*- coding: utf-8 -*-
"""20 个去冗余因子的落地对比图（2026-11-03）。

三个面板：
  左：全期（2020-01~2026-11）8 个调仓相位的 CAGR，32 个 → 20 个
  中：样本外（2022-01~）同样的 8 个相位
  右：池内两两 |ρ| 的分布（排序后曲线），标出 0.70 剪枝阈值
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

from qf import config as C
from qf import data as qfdata
from qf import model as qfm

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "eval"
SHOT = ROOT / "reports" / "shots"
SHOT.mkdir(parents=True, exist_ok=True)

OLD_MODEL = "wf_drop2"      # 32 个（删掉 R737/R738 后、去冗余前）
NEW_MODEL = "wf_h10"        # 20 个（去冗余后，生产）
FULL = ("phase_band_drop2.json", "phase_band.json")
OOS = ("phase_band_drop2_oos.json", "phase_band_h10_oos.json")

CLR_OLD = "#95a5a6"
CLR_NEW = "#2f6f9f"
CLR_BAD = "#c0392b"

for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(x.name == f for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False


def dev_corr(model: str) -> np.ndarray:
    """池内两两 |ρ|（DEV 期同口径：逐周横截面 rank + 中性化 + z-score）。"""
    md = qfdata.load_market(verbose=False)
    rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
    pool = json.loads((C.BUILD_MODELS / model / "model_meta.json")
                      .read_text(encoding="utf-8"))["pool"]
    zs = [np.load(C.BUILD_FACTORS / "weekly_top.npz")]
    zs += [np.load(p) for p in sorted(C.BUILD_FACTORS.glob("*_extra.npz"))]
    pan = {}
    for k in pool:
        for z in zs:
            if k in z.files:
                a = z[k]
                if a.shape[0] < len(rows):
                    pad = np.full((len(rows) - a.shape[0], a.shape[1]), np.nan, np.float32)
                    a = np.concatenate([a, pad], axis=0)
                pan[k] = a
                break
        else:
            raise KeyError(k)
    dev = md.dates[rows] <= pd.Timestamp(C.DEV_END)
    rd = rows[dev]
    mk = md.eligible[rd]
    A = np.stack([qfm.build_feature_block(md, rd, {k: pan[k][dev]}, k, mk) for k in pool], 0)
    V = np.isfinite(A) & mk[None]
    A[~V] = 0.0
    G = np.einsum("kwn,lwn->kl", A.astype(np.float64), A.astype(np.float64), optimize=True)
    J = np.einsum("kwn,lwn->kl", V.astype(np.float32), V.astype(np.float32), optimize=True)
    c = np.clip(G / np.maximum(J, 1.0), -1, 1)
    iu = np.triu_indices(len(pool), 1)
    return np.sort(np.abs(c[iu]))[::-1]


def load(name: str):
    d = json.loads((EVAL / name).read_text(encoding="utf-8"))
    m = {r["offset"]: r["cagr"] * 100 for r in d["rows"]}
    offs = sorted(m)
    return np.array([m[k] for k in offs]), d["cagr"]["mean"] * 100, offs


def panel_bars(ax, old_j, new_j, title, ylab=True):
    xo, mo, offs = load(old_j)
    xn, mn, _ = load(new_j)
    x = np.arange(len(offs))
    w = 0.38
    b1 = ax.bar(x - w / 2, xo, w, color=CLR_OLD, edgecolor="white", linewidth=0.5,
                label=f"32 个（均值 {mo:.2f}%）")
    b2 = ax.bar(x + w / 2, xn, w, color=CLR_NEW, edgecolor="white", linewidth=0.5,
                label=f"20 个（均值 {mn:.2f}%）")
    for b, v in list(zip(b1, xo)) + list(zip(b2, xn)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.07, f"{v:.1f}",
                ha="center", va="bottom", fontsize=6.6, color="#3a3a3a")
    ax.axhline(mo, color=CLR_OLD, ls="--", lw=0.9)
    ax.axhline(mn, color=CLR_NEW, ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in offs], fontsize=8)
    ax.set_xlabel("调仓相位 offset", fontsize=8.8)
    if ylab:
        ax.set_ylabel("CAGR  %", fontsize=8.8)
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_ylim(0, max(xo.max(), xn.max()) + 1.5)
    ax.grid(axis="y", ls=":", lw=0.55, alpha=0.55)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.4, frameon=False, ncol=2)
    return xo, xn, mo, mn


def main() -> None:
    rho_o = dev_corr(OLD_MODEL)
    rho_n = dev_corr(NEW_MODEL)

    fig = plt.figure(figsize=(15.6, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.18, 1.0], wspace=0.24,
                          left=0.055, right=0.985, top=0.795, bottom=0.145)

    ax = fig.add_subplot(gs[0, 0])
    xo, xn, mo, mn = panel_bars(ax, *FULL, "全期 2020-01 ~ 2026-11")

    ax = fig.add_subplot(gs[0, 1])
    xo2, xn2, mo2, mn2 = panel_bars(ax, *OOS, "样本外 2022-01 ~", ylab=False)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(np.arange(1, len(rho_o) + 1), rho_o, color=CLR_OLD, lw=1.5,
            label=f"32 个：{(rho_o>0.7).sum()} 对 >0.7（共 {len(rho_o)} 对）")
    ax.plot(np.arange(1, len(rho_n) + 1), rho_n, color=CLR_NEW, lw=1.5,
            label=f"20 个：{(rho_n>0.7).sum()} 对 >0.7（共 {len(rho_n)} 对）")
    ax.axhline(0.70, color=CLR_BAD, ls="--", lw=0.9)
    ax.text(len(rho_o) * 0.99, 0.715, "剪枝阈值 0.70", ha="right", va="bottom",
            fontsize=7.4, color=CLR_BAD)
    ax.fill_between(np.arange(1, len(rho_o) + 1), 0.70, rho_o,
                    where=rho_o > 0.70, color=CLR_BAD, alpha=0.10)
    ax.set_xlim(0, len(rho_o) + 2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("因子对的排序（|ρ| 从大到小）", fontsize=8.8)
    ax.set_ylabel("|ρ|（DEV 期，逐周横截面）", fontsize=8.8)
    ax.set_title("池内「说同一件事」的因子对", fontsize=10, pad=7)
    ax.grid(ls=":", lw=0.55, alpha=0.55)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=7.2, frameon=False)

    d = xn - xo
    fig.suptitle(f"因子池 32 → 20：全期相位均值 {mo:.2f}% → {mn:.2f}%"
                 f"（{d.mean():+.2f}pp，{int((d>0).sum())}/{len(d)} 改善）"
                 f"　样本外 {mo2:.2f}% → {mn2:.2f}%（{xn2.mean()-xo2.mean():+.2f}pp，基本持平）",
                 fontsize=11.8, y=0.975)
    fig.text(0.055, 0.905,
             "IC 0.0911→0.0907（持平）　ICIR 0.767→0.812　有效独立信号 5.9→7.6 个　"
             "冗余对 49→0：换的是「同样收益里少 12 个重复槽」",
             fontsize=8.4, color="#5f6368")

    out = SHOT / "因子池_32到20相位对比.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
