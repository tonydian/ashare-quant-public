# -*- coding: utf-8 -*-
"""界面设计方案 v2 落地后的契约测试。

守住四件事，避免下次改样式又回退：
1. 导航必须分组 + 全部内联 SVG 图标（不再用字符图标）
2. 长文说明必须进折叠区，不占首屏
3. 颜色只能来自设计 token（CSS 变量），组件里不许再写死主题色
4. 最小字号 11px
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "web" / "static"

NAV_PAGES = [
    "today", "monthly", "market", "tracking", "portfolio", "backtest",
    "stock", "factors", "research", "cb", "etf", "sector", "data", "reports",
]
# 设计文档点名的 6 组缺陷之一：字符图标跨字体字重不一致
CHAR_ICONS = ["★", "◧", "▤", "⌕", "▥", "ƒ", "▣", "↗", "⊞", "◈", "◑"]


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _css() -> str:
    return (STATIC / "style.css").read_text(encoding="utf-8")


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


# ------------------------------------------------------------------ 导航
def test_nav_grouped_in_three_sections():
    """14 个导航项按使用动线分 3 组，并保留全部 data-page。

    2026-11-04 由 5 组（决策/执行/分析/策略/参考）并为 3 组：
    旧分法的「决策 vs 执行」「分析 vs 策略」语义重叠（回测既分析也策略、
    ETF趋势既策略也分析），且「参考」只有 2 项却占一个组标题。
    """
    html = _html()
    nav = html[html.find('<nav class="nav"'):html.find("</nav>")]
    groups = re.findall(r'<div class="nav-group-title">([^<]+)</div>', nav)
    assert groups == ["决策", "策略", "研究"], groups
    assert nav.count('class="nav-group"') == 3


def test_nav_items_all_present_and_labelled():
    """每个导航项都必须有 nav-label（历史上 3 个 <a> 漏了这个类）。"""
    html = _html()
    nav = html[html.find('<nav class="nav"'):html.find("</nav>")]
    items = re.findall(r'<button[^>]*class="nav-item[^"]*"[^>]*>(.*?)</button>', nav, re.S)
    assert len(items) == 14, f"导航项应为 14 个，实际 {len(items)}"
    for body in items:
        assert 'class="nav-label"' in body, body[:120]
    for page in NAV_PAGES:
        assert f'data-page="{page}"' in nav, f"导航缺 {page}"


def test_nav_uses_inline_svg_icons():
    """图标全部换成 inline SVG，且不再出现字符图标。"""
    html = _html()
    nav = html[html.find('<nav class="nav"'):html.find("</nav>")]
    assert nav.count("<svg") == 14, f"SVG 图标应为 14 个，实际 {nav.count('<svg')}"
    assert nav.count('class="nav-ico"') == 14
    for ico in CHAR_ICONS:
        assert ico not in nav, f"导航仍在使用字符图标 {ico}"


# ------------------------------------------------------------------ 折叠
def test_today_long_notice_is_collapsed():
    """6 行长文必须进 <details>，默认收起。"""
    html = _html()
    i = html.find('id="page-today"')
    seg = html[i:html.find("</section>", i)]
    assert '<details class="fold card">' in seg, "重要提示未折叠"
    assert "<summary" in seg
    # 折叠块默认不能带 open
    det = seg[seg.find("<details"):seg.find(">", seg.find("<details")) + 1]
    assert "open" not in det, "折叠区默认不应展开"


def test_today_uses_split_layout():
    """今日页主栅格：左侧操作清单 + 右侧仓位栏。"""
    html = _html()
    assert 'class="today-main"' in html
    assert 'class="today-side"' in html
    for cid in ("todayKpis", "todayList", "todayGroups", "todayCash",
                "todayCandidates", "todaySignalNote"):
        assert f'id="{cid}"' in html, f"缺 {cid}"


# ------------------------------------------------------------------ token
def test_design_tokens_defined():
    css = _css()
    for tok in ("--bg", "--surface", "--surface-2", "--surface-3", "--line",
                "--text", "--text-2", "--text-3", "--accent", "--up",
                "--down", "--warn"):
        assert re.search(rf"^\s*{re.escape(tok)}:", css, re.M), f"缺 token {tok}"
    # 旧变量名保留为别名，避免 app.js 与页面层样式一次性失效
    for alias in ("--panel:", "--border:", "--muted:", "--border-soft:"):
        assert alias in css, f"缺别名 {alias}"


def test_legacy_card_head_classes_are_styled():
    """.card-head/.card-body 与 .card-hd/.card-bd 必须都定义（历史上前者完全没样式）。"""
    css = _css()
    assert ".card-hd," in css and ".card-head {" in css
    assert ".card-bd," in css and ".card-body {" in css
    assert ".page-sub" in css
    assert ".btn-ghost" in css


def test_no_hardcoded_theme_colors():
    """组件里不许再写死旧主题色，必须走 token。"""
    old = ["#38bdf8", "#22c55e", "#4ade80", "#f87171", "#fbbf24",
           "#a78bfa", "#0ea5e9", "#0b1119", "#243449"]
    css, js = _css(), _js()
    for c in old:
        assert c not in css, f"style.css 仍写死 {c}"
        assert c not in js, f"app.js 仍写死 {c}"


def test_charts_py_matches_tokens():
    """服务端 matplotlib 出图配色必须与 CSS token 一致。"""
    src = (ROOT / "web" / "charts.py").read_text(encoding="utf-8")
    for c in ("#5b9dff", "#f2555a", "#2fbf71", "#111721"):
        assert c in src, f"charts.py 缺少配色 {c}"
    assert "#38bdf8" not in src


def test_min_font_size_11px():
    """正文最小值不低于 11px（旧版有 10.5px / 10px）。"""
    css = _css()
    sizes = [float(x) for x in re.findall(r"font-size:\s*([\d.]+)px", css)]
    assert sizes, "没有解析到字号"
    assert min(sizes) >= 11, f"存在小于 11px 的字号: {sorted(set(s for s in sizes if s < 11))}"


def test_base_font_size_bumped():
    """body 字号必须 >= 13.5px。"""
    css = _css()
    m = re.search(r"body\s*\{[^}]*font-size:\s*([\d.]+)px", css, re.S)
    assert m, "找不到 body 字号"
    assert float(m.group(1)) >= 13.5, m.group(1)


# ------------------------------------------------------------------ 语义色
def test_action_labels_not_using_rise_fall_colors():
    """买入/卖出用 accent/warn，不能借用涨跌色（避免「买入=红=涨」混淆）。"""
    js = _js()
    i = js.find("var isBuy = x.side === 'buy'")
    assert i > 0, "找不到成交方向渲染"
    seg = js[i:i + 240]
    assert "var(--accent)" in seg and "var(--warn)" in seg, seg[:200]


def test_tracking_bars_zero_aligned():
    """跟踪页净值柱必须零轴对齐，负值向下生长。"""
    js = _js()
    assert "zeroPct" in js, "缺零轴位置计算"
    assert "tk-zero" in js, "缺零轴参考线"
    css = _css()
    assert ".tk-zero" in css


def test_a_share_rise_is_red():
    """涨红跌绿：accent 之外，--up 必须是红、--down 必须是绿。"""
    css = _css()

    def lum(hexs: str) -> int:
        v = hexs.lstrip("#")
        r, g, b = (int(v[i:i + 2], 16) for i in (0, 2, 4))
        return r - max(g, b)

    up = re.search(r"--up:\s*(#[0-9a-fA-F]{6})", css).group(1)
    down = re.search(r"--down:\s*(#[0-9a-fA-F]{6})", css).group(1)
    assert lum(up) > 20, f"--up 应为红色系，实际 {up}"
    assert lum(down) < 0, f"--down 应为绿色系，实际 {down}"
