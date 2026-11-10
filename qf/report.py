# -*- coding: utf-8 -*-
"""回测结果可视化 + 报告数据生成（matplotlib，中文字体）。"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

COLORS = ["#1f4e79", "#c00000", "#2e7d32", "#ed7d31", "#7030a0", "#808080", "#00b0f0",
          "#a5a5a5", "#ffc000"]


def _fig_dir(out_dir) -> pathlib.Path:
    d = pathlib.Path(out_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _nav_series(result) -> pd.Series:
    nav = result.nav
    if not isinstance(nav, pd.Series):
        nav = pd.Series(np.asarray(nav).ravel(), index=result.nav_index if hasattr(result, "nav_index") else None)
    return nav.astype(float)


def plot_nav_drawdown(result, bench: dict, out_dir) -> pathlib.Path:
    nav = _nav_series(result)
    dd = nav / nav.cummax() - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3]})
    ax = axes[0]
    ax.plot(nav.index, nav / nav.iloc[0], color=COLORS[0], lw=1.8, label="策略（扣费后）")
    for i, (name, bnav) in enumerate(bench.items()):
        bnav = bnav.reindex(nav.index).ffill()
        if bnav.notna().sum() < 10:
            continue
        ax.plot(bnav.index, bnav / bnav.dropna().iloc[0], color=COLORS[(i + 1) % len(COLORS)],
                lw=1.0, alpha=0.85, label=f"{name}")
    ax.set_yscale("log")
    ax.set_ylabel("净值（对数轴）")
    ax.set_title("策略净值 vs 基准")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    ax2 = axes[1]
    ax2.fill_between(dd.index, dd * 100, 0, color=COLORS[1], alpha=0.6)
    ax2.set_ylabel("回撤 %")
    ax2.grid(alpha=0.25)
    fig.savefig(_fig_dir(out_dir) / "nav_drawdown.png")
    plt.close(fig)
    return _fig_dir(out_dir) / "nav_drawdown.png"


def plot_yearly(result, bench: dict, out_dir) -> pathlib.Path:
    from . import metrics as qfmetrics
    nav = _nav_series(result)
    yr = qfmetrics.yearly_returns(nav)
    yb = {}
    for name, bnav in bench.items():
        b = bnav.reindex(nav.index).ffill()
        yb[name] = qfmetrics.yearly_returns(b)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(yr))
    w = 0.8 / (1 + len(yb))
    ax.bar(x - 0.4 + w / 2, yr.values * 100, width=w, color=COLORS[0], label="策略")
    for i, (name, r) in enumerate(yb.items()):
        ax.bar(x - 0.4 + w * (i + 1.5), r.reindex(yr.index).values * 100, width=w,
               color=COLORS[(i + 1) % len(COLORS)], alpha=0.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(yr.index)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("年度收益 %")
    ax.set_title("逐年收益对比")
    ax.legend(fontsize=8, ncol=min(6, len(yb) + 1))
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(_fig_dir(out_dir) / "yearly_returns.png")
    plt.close(fig)
    return _fig_dir(out_dir) / "yearly_returns.png"


def plot_monthly_heatmap(result, out_dir) -> pathlib.Path:
    nav = _nav_series(result)
    me = nav.resample("ME").last()
    m = me / me.shift(1) - 1.0
    if len(me):
        m.iloc[0] = me.iloc[0] / nav.iloc[0] - 1.0
    tab = pd.DataFrame({"ret": m.values}, index=m.index)
    tab["year"] = tab.index.year
    tab["month"] = tab.index.month
    heat = tab.pivot_table(index="year", columns="month", values="ret")
    heat = heat.reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(heat) + 1.6))
    data = heat.values * 100
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 5
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{i}月" for i in range(1, 13)])
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("月度收益热力图（%）")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.savefig(_fig_dir(out_dir) / "monthly_heatmap.png")
    plt.close(fig)
    return _fig_dir(out_dir) / "monthly_heatmap.png"


def plot_rolling(result, out_dir) -> pathlib.Path:
    nav = _nav_series(result)
    ret = nav.pct_change().dropna()
    roll_sharpe = (ret.rolling(252).mean() / ret.rolling(252).std()) * np.sqrt(252)
    roll_vol = ret.rolling(60).std() * np.sqrt(252)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(roll_sharpe.index, roll_sharpe, color=COLORS[0])
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("滚动 1 年 Sharpe")
    axes[0].grid(alpha=0.25)
    axes[1].plot(roll_vol.index, roll_vol * 100, color=COLORS[3])
    axes[1].set_ylabel("滚动 60 日年化波动 %")
    axes[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))
    axes[1].grid(alpha=0.25)
    fig.suptitle("滚动风险指标")
    fig.savefig(_fig_dir(out_dir) / "rolling_metrics.png")
    plt.close(fig)
    return _fig_dir(out_dir) / "rolling_metrics.png"


def plot_exposures(result, out_dir) -> pathlib.Path | None:
    ex = getattr(result, "exposures", None)
    if ex is None or not isinstance(ex, pd.DataFrame) or ex.empty:
        return None
    ex = ex.copy()
    ex.index = pd.to_datetime(ex.index)
    meta_cols = {"cash", "stock", "growth", "n_positions", "max_single", "max_industry",
                 "target_exposure", "regime_scale", "dd_cap", "dd_1y", "dd_state",
                 "exposure_cap", "date"}
    industry_cols = [c for c in ex.columns if c not in meta_cols and ex[c].dtype.kind in "fi"]
    if len(industry_cols) > 8:
        industry_cols = ex[industry_cols].mean().sort_values(ascending=False).index[:8].tolist()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(ex.index, ex["cash"] * 100, lw=1.6, color=COLORS[5], label="现金")
    if "stock" in ex.columns:
        ax.plot(ex.index, ex["stock"] * 100, lw=1.6, color=COLORS[2], label="股票仓位")
    for i, c in enumerate(industry_cols[:8]):
        ax.plot(ex.index, ex[c] * 100, lw=0.9, alpha=0.9, label=str(c),
                color=COLORS[i % len(COLORS)])
    ax.set_ylabel("权重 %")
    ax.set_title("组合暴露（调仓日）：仓位与行业分布")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.25)
    fig.savefig(_fig_dir(out_dir) / "exposures.png")
    plt.close(fig)
    return _fig_dir(out_dir) / "exposures.png"


def plot_turnover(result, out_dir) -> pathlib.Path | None:
    tr = getattr(result, "trades", None)
    if tr is None or not isinstance(tr, pd.DataFrame) or tr.empty:
        return None
    d = tr.copy()
    datecol = "date" if "date" in d.columns else d.columns[0]
    d[datecol] = pd.to_datetime(d[datecol])
    amt = d.groupby([datecol, "side"])["amount"].sum().unstack(fill_value=0.0)
    nav = _nav_series(result)
    if len(nav) < 2:
        return None
    fig, ax = plt.subplots(figsize=(11, 3.8))
    if "buy" in amt.columns:
        ax.bar(amt.index, amt["buy"] / nav.mean() * 100, width=1.5, color=COLORS[2], label="买入额/均值净值 %")
    if "sell" in amt.columns:
        ax.bar(amt.index, -amt["sell"] / nav.mean() * 100, width=1.5, color=COLORS[1], label="卖出额/均值净值 %")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("%")
    ax.set_title("调仓成交额（占净值）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(_fig_dir(out_dir) / "turnover.png")
    plt.close(fig)
    return _fig_dir(out_dir) / "turnover.png"


def plot_run(result, bench: dict, out_dir) -> pathlib.Path:
    out = pathlib.Path(out_dir)
    plot_nav_drawdown(result, bench, out)
    plot_yearly(result, bench, out)
    plot_monthly_heatmap(result, out)
    plot_rolling(result, out)
    plot_exposures(result, out)
    plot_turnover(result, out)
    return out / "figures"


def _fmt_pct(x) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return str(x)


def write_summary(result, bm: dict, out_dir) -> pathlib.Path:
    out = pathlib.Path(out_dir)
    m = result.metrics
    lines = ["# 回测摘要", ""]
    lines += ["## 策略绩效", "", "| 指标 | 数值 |", "|---|---|"]
    labels = {
        "cagr": "年化复合收益", "ann_vol": "年化波动", "sharpe": "Sharpe",
        "sortino": "Sortino", "max_dd": "最大回撤", "max_dd_days": "最长回撤天数",
        "calmar": "Calmar", "win_rate_daily": "日胜率", "win_rate_monthly": "月胜率",
        "skew": "日收益偏度", "kurt": "日收益峰度", "turnover_ann": "年化单边换手",
        "turnover_ann_double": "年化双边换手",
        "cost_ratio": "成本/净值", "n_trades": "成交笔数", "avg_holdings": "平均持仓数",
        "beta_hs300": "Beta(HS300)", "alpha_hs300": "Alpha(HS300)", "ir_hs300": "信息比率(HS300)",
        "beta_csi1000": "Beta(CSI1000)", "alpha_csi1000": "Alpha(CSI1000)",
    }
    for k, lab in labels.items():
        if k in m:
            v = m[k]
            if k in ("cagr", "ann_vol", "max_dd", "win_rate_daily", "win_rate_monthly",
                     "alpha_hs300", "alpha_csi1000", "cost_ratio"):
                lines.append(f"| {lab} | {_fmt_pct(v)} |")
            else:
                lines.append(f"| {lab} | {v:.3f} |" if isinstance(v, (int, float)) else f"| {lab} | {v} |")
    # 基准
    if bm:
        lines += ["", "## 与基准对比", "",
                  "| 基准 | 年化 | 波动 | Sharpe | 最大回撤 | Calmar |",
                  "|---|---|---|---|---|---|"]
        for name, b in bm.items():
            lines.append(f"| {name} | {_fmt_pct(b.get('cagr', np.nan))} | "
                         f"{_fmt_pct(b.get('ann_vol', np.nan))} | {b.get('sharpe', np.nan):.2f} | "
                         f"{_fmt_pct(b.get('max_dd', np.nan))} | {b.get('calmar', np.nan):.2f} |")
    # 逐年
    from . import metrics as qfmetrics
    try:
        yr = qfmetrics.yearly_returns(result.nav)
        lines += ["", "## 逐年收益", "", "| 年份 | 策略 |", "|---|---|"]
        for y, v in yr.items():
            lines.append(f"| {y} | {_fmt_pct(v)} |")
    except Exception:
        pass
    lines += ["", "![净值与回撤](figures/nav_drawdown.png)",
              "", "![逐年收益](figures/yearly_returns.png)",
              "", "![月度热力](figures/monthly_heatmap.png)",
              "", "![滚动指标](figures/rolling_metrics.png)"]
    p = out / "summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
