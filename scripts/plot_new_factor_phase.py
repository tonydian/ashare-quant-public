# -*- coding: utf-8 -*-
"""新因子正式接入实验的出图。

图 1（两联）：左 = 8 个调仓相位的 CAGR（含生产相位标记）；右 = 配对散点（ctl vs 加因子）
图 2（两联）：左 = 复合分分年度 IC；右 = 族级消融的 ΔCAGR（含 t 值）
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLR = {"ctl": "#8a8f98", "turnover": "#2f6fb5", "newf_fund": "#c0392b", "newf": "#2e8b74"}
CN = {"ctl": "33 因子基线", "turnover": "基线+换手族", "newf_fund": "基线+换手+隔夜+基本面",
      "newf": "基线+全部新因子", "fund": "仅基本面(15)", "overnight": "仅隔夜(3)",
      "liquidity": "仅流动性(2)", "pricevolume": "仅量价形态(2)"}
MODELS = {"ctl": "wf_ctl", "newf": "wf_newf", "turnover": "wf_f_turnover",
          "overnight": "wf_f_overnight", "liquidity": "wf_f_liquidity",
          "pricevolume": "wf_f_pricevolume", "fund": "wf_fund", "newf_fund": "wf_newf_fund"}


def ic_by_year(model: str) -> pd.Series:
    d = pd.read_parquet(C.BUILD_MODELS / model / "composite_ic.parquet")
    d["y"] = pd.to_datetime(d["date"]).dt.year
    return d.groupby("y")["ic"].mean()


def main():
    ph = pd.read_csv("build/runs/phase_ctl40.csv")
    dist = json.load(open("build/runs/phase_dist_rb40.json", encoding="utf-8"))
    delta = pd.read_parquet("build/runs/phase_delta_rb40.parquet").set_index("model")

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- 左：逐相位 CAGR
    x = np.arange(len(ph))
    w = 0.27
    for i, (k, lab) in enumerate([("ctl", "ctl"), ("tno", "turnover"), ("nf", "newf_fund")]):
        ax[0].bar(x + (i - 1) * w, ph[f"{k}_cagr"] * 100, width=w,
                  color=CLR[lab], label=CN[lab], edgecolor="white", linewidth=0.6)
    ax[0].axhline(7.22, ls="--", lw=1.3, color="#c0392b", alpha=0.85)
    ax[0].axhline(ph["ctl_cagr"].mean() * 100, ls=":", lw=1.2, color="#555")
    ax[0].text(-0.42, 9.05, "红色虚线 = 生产报告值 7.22%（offset=10，8 个相位里最高的一档）",
               fontsize=9.2, color="#c0392b", ha="left", va="center")
    ax[0].text(-0.42, 8.55, f"灰色点线 = 基线 8 相位均值 {ph['ctl_cagr'].mean()*100:.2f}%",
               fontsize=9.2, color="#555", ha="left", va="center")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"off={int(o)}" for o in ph["offset"]], fontsize=9)
    ax[0].set_ylabel("CAGR（%）")
    ax[0].set_title("同一模型、8 个调仓相位的年化收益（40 日调仓）", fontsize=11.5)
    ax[0].legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                 frameon=False, handlelength=1.4, columnspacing=1.6)
    ax[0].set_ylim(0, 9.5)
    ax[0].grid(axis="y", alpha=0.25)
    ax[0].set_axisbelow(True)

    # ---- 右：配对散点
    for k, lab, col in [("tno", "turnover", CLR["turnover"]), ("nf", "newf_fund", CLR["newf_fund"])]:
        ax[1].scatter(ph["ctl_cagr"] * 100, ph[f"{k}_cagr"] * 100, s=64, color=col,
                      label=CN[lab], alpha=0.9, edgecolor="white", linewidth=0.8)
    lo, hi = 2.5, 8.8
    ax[1].plot([lo, hi], [lo, hi], ls="--", lw=1, color="#999")
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(lo, hi)
    ax[1].set_xlabel("基线 CAGR（%）")
    ax[1].set_ylabel("加因子后 CAGR（%）")
    ax[1].set_title("8 个相位配对比较（对角线上方=改善）", fontsize=11.5)
    ax[1].legend(fontsize=9, loc="lower right", frameon=False)
    ax[1].grid(alpha=0.25)
    ax[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("reports/shots/新因子_相位分布.png", dpi=150)
    plt.close(fig)

    # ---- 图 2
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    yrs = None
    for m, lab in [("wf_ctl", "ctl"), ("wf_fund", "fund"), ("wf_f_turnover", "turnover"), ("wf_newf", "newf"), ("wf_newf_fund", "newf_fund")]:
        s = ic_by_year(m)
        yrs = s.index
        ax[0].plot(s.index, s.values, marker="o", ms=4.5, lw=1.7,
                   color=CLR.get(lab, "#333"), label=CN.get(lab, lab), alpha=0.9)
    ax[0].set_xticks(list(yrs))
    ax[0].set_ylabel("复合分 RankIC（年均）")
    ax[0].set_title("复合分 IC：加因子后逐年抬升", fontsize=11.5)
    ax[0].set_ylim(0.052, 0.118)
    ax[0].legend(fontsize=8.2, ncol=2, loc="lower right", framealpha=0.92, edgecolor="white")
    ax[0].grid(alpha=0.25)
    ax[0].set_axisbelow(True)

    order = ["turnover", "newf", "newf_fund", "pricevolume", "fund", "overnight", "liquidity"]
    lab = {"turnover": "换手族(5)", "newf": "全部新因子(12)", "newf_fund": "新因子+基本面",
           "pricevolume": "量价形态(2)", "fund": "基本面(15)", "overnight": "隔夜(3)",
           "liquidity": "流动性(2)"}
    d = delta.loc[order]
    cols = ["#2f6fb5" if v > 0 else "#c0392b" for v in d["d_cagr"]]
    y = np.arange(len(d))
    ax[1].barh(y, d["d_cagr"] * 100, color=cols, alpha=0.88, edgecolor="white")
    for i, (v, t, wn) in enumerate(zip(d["d_cagr"] * 100, d["t"], d["win"])):
        ax[1].text(v + (0.07 if v >= 0 else -0.07), i, f"{v:+.2f}pp  t={t:+.1f}  胜{wn:.0%}",
                   va="center", ha="left" if v >= 0 else "right", fontsize=8.4, color="#222")
    ax[1].axvline(0, color="#444", lw=1)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([lab[k] for k in order], fontsize=9.5)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("相对基线的 ΔCAGR（8 相位均值，%）")
    ax[1].set_xlim(-2.6, 2.5)
    ax[1].set_title("族级消融：只有换手族是稳健正贡献", fontsize=11.5)
    ax[1].grid(axis="x", alpha=0.25)
    ax[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("reports/shots/新因子_族级消融.png", dpi=150)
    plt.close(fig)
    print("图已写出：reports/shots/新因子_相位分布.png、新因子_族级消融.png")


if __name__ == "__main__":
    main()
