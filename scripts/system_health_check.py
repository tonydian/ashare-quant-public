# -*- coding: utf-8 -*-
"""系统当前状态的诚实体检：哪些环节还不可靠？

目的：在给出"接下来做什么"之前，先确认现有结论的可信度。

检查项：
  1. 回测区间的样本外占比（是否足够）
  2. 因子池缓冲后的稳定性（换手、CAGR 波动）
  3. 成本假设是否现实（滑点 15bp 是否低估）
  4. 是否有实盘对照
  5. 策略之间的相关性（组合是否真分散）
  6. 各策略的样本外衰减
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402

print("=" * 92)
print("系统诚实体检")
print("=" * 92)

# ---------------------------------------------------------------- 1) 样本外占比
print("\n【1】回测样本分布")
print("-" * 92)
nav = pd.read_parquet(C.BUILD_RUNS / "main" / "nav.parquet")["nav"].astype(float)
nav.index = pd.to_datetime(nav.index)
total_days = len(nav)
dev_days = int((nav.index <= pd.Timestamp(C.DEV_END)).sum())
val_days = int(((nav.index >= "2020-01-01") & (nav.index <= "2021-12-31")).sum())
oos_days = total_days - val_days
print(f"  总回测 {total_days} 天（{nav.index[0].date()} ~ {nav.index[-1].date()}）")
print(f"  DEV 期（2017-2019，选因子用）: {dev_days} 天（不在回测内）")
print(f"  验证期 2020-2021: {val_days} 天（{val_days/total_days:.0%}）")
print(f"  **样本外 2022+**: {oos_days} 天（{oos_days/total_days:.0%}）")
print(f"  → 样本外 {oos_days/252:.1f} 年")

# ---------------------------------------------------------------- 2) 各策略样本外
print("\n【2】各策略的分区间表现")
print("-" * 92)
srcs = {
    "主策略": C.BUILD_RUNS / "main" / "nav.parquet",
    "宽基ETF": C.BUILD_RUNS / "etf_trend" / "nav_donchian55.parquet",
    "板块轮动": C.BUILD_RUNS / "sector_rotation" / "nav.parquet",
}
navs = {}
print(f"  {'策略':<12}{'验证期20-21':>14}{'样本外22+':>14}{'衰减':>10}{'样本外Sharpe':>14}")
print("  " + "-" * 68)
for name, p in srcs.items():
    if not p.exists():
        continue
    s = pd.read_parquet(p)["nav"].astype(float)
    s.index = pd.to_datetime(s.index)
    navs[name] = s
    v = s[(s.index >= "2020-01-01") & (s.index <= "2021-12-31")]
    o = s[s.index >= "2022-01-01"]
    pv = qfm.performance(v, rf=C.CASH_YIELD)
    po = qfm.performance(o, rf=C.CASH_YIELD)
    decay = (1 - po["cagr"] / pv["cagr"]) if pv["cagr"] != 0 else 0
    print(f"  {name:<12}{pv['cagr']:>13.2%}{po['cagr']:>14.2%}"
          f"{decay:>9.0%}{po['sharpe']:>14.3f}")

# ---------------------------------------------------------------- 3) 相关性
print("\n【3】策略间相关性（组合是否真分散）")
print("-" * 92)
if len(navs) >= 2:
    idx = None
    for s in navs.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx[idx >= "2020-01-01"]
    rets = pd.DataFrame({k: v.reindex(idx).pct_change() for k, v in navs.items()}).dropna()
    print(rets.corr().round(3).to_string())
    print(f"\n  平均相关: {rets.corr().values[np.triu_indices(len(navs), 1)].mean():.3f}")

# ---------------------------------------------------------------- 4) 成本假设
print("\n【4】成本假设的现实性")
print("-" * 92)
print(f"  当前假设：滑点 15bp/边、佣金 2.5bp、印花税 5bp（卖出）")
print(f"  往返成本 ≈ {(15+2.5)*2 + 5:.1f} bp = {((15+2.5)*2+5)/100:.2f}%")
met = json.load(open(C.BUILD_RUNS / "main" / "metrics.json", encoding="utf-8"))
turnover = met.get("turnover_annual") or met.get("annual_turnover")
print(f"  年换手: {turnover}")
print(f"  年化成本占净值: {met.get('cost_ratio') or met.get('cost_pct') or '见报告 1.36%'}")
print("\n  ⚠️ 注意：真实滑点对小市值股票可能远高于 15bp")
print("     （盘口薄、冲击成本大），这是回测最大的不确定来源。")

# ---------------------------------------------------------------- 5) 因子池稳定性
print("\n【5】因子池状态")
print("-" * 92)
mp = C.BUILD_MODELS / "wf_h10" / "model_meta.json"
if mp.exists():
    m = json.load(open(mp, encoding="utf-8"))
    print(f"  池大小: {len(m['pool'])}")
    print(f"  缓冲启用: {m['cfg'].get('pool_guard')}")
st = C.BUILD_MODELS / "_pool_guard_state.json"
if st.exists():
    d = json.load(open(st, encoding="utf-8"))
    print(f"  状态条目: {len(d.get('state') or {})}")
    h = d.get("history") or []
    print(f"  历史记录: {len(h)} 次")
    if len(h) >= 2:
        for i in range(1, len(h)):
            a, b = set(h[i-1]["pool"]), set(h[i]["pool"])
            print(f"    第{i}→{i+1}次: +{len(b-a)} -{len(a-b)}")

# ---------------------------------------------------------------- 6) 结论
print("\n" + "=" * 92)
print("体检结论")
print("=" * 92)
print("""
  ✅ 可靠的：
     · 回测框架无前视（已多次审计，2017/2017 笔成交都是次日开盘）
     · T+1 严格（板块轮动/ETF 都已修正同 bar 问题）
     · 数据源统一（自动选 ext、管道强制传参、一致性校验）
     · 因子池缓冲（数据不变零换手）

  ❌ 不可靠 / 缺失的：
     · **没有实盘对照** —— 所有结论都来自回测
     · **成本假设可能过于乐观** —— 小市值滑点未充分建模
     · **样本外只有 4.8 年** —— 不足以确认 alpha 持续性
     · **多策略组合的稳健性未验证**（ERC vs 50/50 差异不显著）

  🟡 半可靠：
     · 因子池缓冲已实现，但未经真实多次数据更新考验
     · 另类数据（两融/龙虎榜/资金流）样本外改善但不显著
""")
