# -*- coding: utf-8 -*-
"""Web UI 图表：服务端 matplotlib 渲染 PNG（深色主题、A 股红涨绿跌、中文字体）。

不依赖 pyplot 全局状态（每次 Figure + FigureCanvasAgg），可在多线程 HTTP handler 中安全调用。
"""
from __future__ import annotations

import io
import functools
import pathlib

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib import font_manager, rcParams  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ---- 深色主题（与 web/static/style.css 的 token 保持一致，见界面设计方案 v2）
BG = "#0a0e13"       # --bg
PANEL = "#111721"    # --surface
GRID = "#1c222b"     # --chart-grid (rgba(255,255,255,.06) 叠在 surface 上)
FG = "#e9eef5"       # --text
MUTED = "#808d9d"    # --text-3
ACCENT = "#5b9dff"   # --accent
UP = "#f2555a"       # A 股：涨红
DOWN = "#2fbf71"     # 跌绿
WARN = "#e0a03a"     # --warn
MA_COLORS = {"ma20": "#e0a03a", "ma60": "#808d9d", "ma120": "#5b9dff"}

_FONT_PATHS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
               r"C:\Windows\Fonts\msyhbd.ttc"]
_font_name = None
for _p in _FONT_PATHS:
    if pathlib.Path(_p).exists():
        try:
            font_manager.fontManager.addfont(_p)
            _font_name = font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:
            pass
rcParams["font.sans-serif"] = [_font_name or "DejaVu Sans", "Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False


def _new_fig(w: float, h: float, dpi: int = 110) -> Figure:
    return Figure(figsize=(w, h), dpi=dpi, facecolor=BG)


def _style(ax, title: str | None = None):
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if title:
        ax.set_title(title, color=FG, fontsize=11, loc="left", pad=8)


def _png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return buf.getvalue()


def _date_axis(ax, dates):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.tick_params(axis="x", rotation=0, colors=MUTED, labelsize=8)


# ------------------------------------------------------------------ 指数走势
def index_line(dates: pd.DatetimeIndex, close: np.ndarray, name: str,
               days: int = 250) -> bytes:
    n = min(int(days), len(close))
    d = dates[-n:]
    c = np.asarray(close[-n:], dtype=float)
    fig = _new_fig(9.2, 3.4)
    ax = fig.add_subplot(111)
    _style(ax, f"{name}  收盘 {c[-1]:,.2f}")
    ax.plot(d, c, color=ACCENT, linewidth=1.4, label="收盘")
    for win, key in ((20, "ma20"), (60, "ma60"), (120, "ma120")):
        if n >= win:
            ma = pd.Series(np.asarray(close, dtype=float)).rolling(win).mean().to_numpy()[-n:]
            ax.plot(d, ma, color=MA_COLORS[key], linewidth=0.9, alpha=0.9, label=f"MA{win}")
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=FG, ncol=4)
    _date_axis(ax, d)
    ax.set_ylabel("点位", color=MUTED, fontsize=9)
    fig.tight_layout(pad=0.6)
    return _png(fig)


# ------------------------------------------------------------------ 个股 K 线
def stock_kline(dates: pd.DatetimeIndex, o: np.ndarray, h: np.ndarray,
                low: np.ndarray, c: np.ndarray, vol: np.ndarray,
                name: str, code: str, days: int = 250) -> bytes:
    n = min(int(days), len(c))
    d = dates[-n:]
    o, h, low, c = (np.asarray(x[-n:], dtype=float) for x in (o, h, low, c))
    v = np.asarray(vol[-n:], dtype=float)
    fig = _new_fig(9.6, 4.4)
    ax = fig.add_axes([0.06, 0.28, 0.92, 0.66])
    axv = fig.add_axes([0.06, 0.05, 0.92, 0.16], sharex=ax)
    _style(ax, f"{name}（{code}）  收盘 {c[-1]:,.2f}" if np.isfinite(c[-1]) else f"{name}（{code}）")
    valid = np.isfinite(o) & np.isfinite(c) & np.isfinite(h) & np.isfinite(low) & (c > 0)
    x = mdates.date2num(d.to_pydatetime())
    for i in np.flatnonzero(valid):
        color = UP if c[i] >= o[i] else DOWN
        ax.vlines(x[i], low[i], h[i], color=color, linewidth=0.8, alpha=0.95)
        bottom = min(o[i], c[i])
        height = max(abs(c[i] - o[i]), (h[i] - low[i]) * 0.004)
        ax.add_patch(Rectangle((x[i] - 0.32, bottom), 0.64, height,
                               facecolor=color, edgecolor=color, linewidth=0.5))
    if valid.any():
        ma20 = pd.Series(c).rolling(20).mean().to_numpy()
        ax.plot(x, ma20, color=MA_COLORS["ma20"], linewidth=0.9, label="MA20")
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=FG)
    ax.set_xlim(x[valid][0] - 1 if valid.any() else x[0] - 1, x[-1] + 1)
    _date_axis(ax, d)
    ax.tick_params(labelbottom=False)
    axv.bar(x, np.nan_to_num(v), width=0.64, color=ACCENT, alpha=0.55)
    _style(axv)
    axv.set_ylabel("量", color=MUTED, fontsize=8)
    _date_axis(axv, d)
    return _png(fig)


# ------------------------------------------------------------------ 行业热力
def industry_bar(rows: list[dict], days: int = 20, topn: int = 16) -> bytes:
    rows = [r for r in rows if np.isfinite(r.get("ret_20d", np.nan))]
    if not rows:
        fig = _new_fig(6, 3)
        ax = fig.add_subplot(111)
        _style(ax, "行业表现（无数据）")
        return _png(fig)
    rows = sorted(rows, key=lambda r: r["ret_20d"], reverse=True)
    if len(rows) > topn:
        half = topn // 2
        sel = rows[:half] + rows[-half:]
    else:
        sel = rows
    seen, uniq = set(), []
    for r in sel:
        if r["industry"] not in seen:
            seen.add(r["industry"])
            uniq.append(r)
    uniq = uniq[:topn]
    labels = [r["industry"] for r in uniq][::-1]
    vals = [r["ret_20d"] * 100 for r in uniq][::-1]
    fig = _new_fig(7.4, max(3.0, 0.26 * len(labels) + 1.0))
    ax = fig.add_subplot(111)
    _style(ax, f"行业 {days} 日收益（%）")
    colors = [UP if v >= 0 else DOWN for v in vals]
    ax.barh(labels, vals, color=colors, alpha=0.9, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + (0.15 if v >= 0 else -0.15), y, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", color=FG, fontsize=7.5)
    ax.axvline(0, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.6)
    fig.tight_layout(pad=0.5)
    return _png(fig)


# ------------------------------------------------------------------ 回测净值
def nav_chart(dates: pd.DatetimeIndex, nav: np.ndarray, dd: np.ndarray,
              title: str = "回测净值与回撤") -> bytes:
    fig = _new_fig(10.2, 4.2)
    ax = fig.add_axes([0.06, 0.34, 0.92, 0.60])
    axd = fig.add_axes([0.06, 0.06, 0.92, 0.20], sharex=ax)
    _style(ax, title)
    nav = np.asarray(nav, dtype=float)
    dd = np.asarray(dd, dtype=float) * 100
    ax.plot(dates, nav, color=ACCENT, linewidth=1.3)
    ax.fill_between(dates, nav.min() * 0.995, nav, color=ACCENT, alpha=0.10)
    # 净值 Y 轴：4 档刻度 + 数值标签（原来只有网格无线索）
    lo = nav.min() - (nav.max() - nav.min()) * 0.04
    hi = nav.max() + (nav.max() - nav.min()) * 0.04
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.set_ylim(lo, hi)
        ticks = np.linspace(lo, hi, 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:.2f}" for v in ticks], color=MUTED, fontsize=8)
    _date_axis(ax, dates)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("净值", color=MUTED, fontsize=8.5)

    _style(axd)
    axd.fill_between(dates, dd, 0, color=DOWN, alpha=0.45)
    axd.axhline(0, color=GRID, linewidth=0.9)
    axd.set_ylim(min(dd.min() * 1.15, -0.5), 0.5)
    axd.set_ylabel("回撤 %", color=MUTED, fontsize=8.5)
    axd.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    # 最大回撤标注（固定左上，避免压住右侧日期刻度）
    if len(dd):
        axd.text(0.006, 0.10, f"最大回撤 {dd.min():.2f}%", transform=axd.transAxes,
                 color=DOWN, fontsize=8, va="bottom")
    _date_axis(axd, dates)
    return _png(fig)
