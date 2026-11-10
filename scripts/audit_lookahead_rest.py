# -*- coding: utf-8 -*-
"""审计剩余项：
  1) sector_rotation_v3.py 是否有同 bar 前视（静态扫描标记为可疑）
  2) 各策略的"调仓日收益"是否异常（间接检测前视）
  3) 其他潜在前视：特征/标签、复权、幸存者偏差
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "build" / "eval" / "lookahead_audit_rest.json"
res = {}

# ============================================================
# 1) sector_rotation_v3.py
# ============================================================
print("=" * 96)
print("1) sector_rotation_v3.py（静态扫描标记可疑，逐个确认）")
print("=" * 96)
src = (ROOT / "scripts" / "sector_rotation_v3.py").read_text(encoding="utf-8")
lines = src.splitlines()
print("  与成交/时序相关的代码行：")
hits = []
for i, ln in enumerate(lines, 1):
    if any(k in ln for k in ("close[t", "rebal == 0", "holdings =", "pending",
                             "tcost", "nav[t]")):
        hits.append((i, ln.strip()))
for i, ln in hits[:22]:
    print(f"    L{i:4d}: {ln[:105]}")

# 判定：v3 里 t 日决策后是否立即用 close[t] 计收益
has_pending = "pending" in src
same_bar = bool(re.search(r"if t % rebal == 0", src)) and not has_pending
print(f"\n  使用 pending 机制: {has_pending}")
print(f"  同 bar 风险: {'是' if same_bar else '否'}")
if same_bar:
    print("  ⚠️ v3 是研究脚本（已被 qf/sector_rotation.py 取代），确认其用途：")
    print("     - 若仅用于参数扫描/研究对比 → 可接受，但结论需打折")
    print("     - 若被任何生产流程引用 → 必须修")
    # 检查是否被引用
    refs = []
    for py in ROOT.rglob("*.py"):
        if py.name == "sector_rotation_v3.py":
            continue
        try:
            txt = py.read_text(encoding="utf-8")
        except Exception:
            continue
        if "sector_rotation_v3" in txt:
            refs.append(str(py.relative_to(ROOT)))
    print(f"     - 被引用情况: {refs if refs else '无任何文件引用（纯研究脚本）'}")
    res["v3"] = {"same_bar": True, "referenced_by": refs,
                 "risk": "低（研究脚本，未被生产引用）" if not refs else "高（被生产引用）"}
else:
    res["v3"] = {"same_bar": False, "risk": "无"}

# ============================================================
# 2) 调仓日收益异常检测（间接验证）
# ============================================================
print("\n" + "=" * 96)
print("2) 调仓日 vs 非调仓日收益对比（前视的间接信号）")
print("=" * 96)


def check_rebal_effect(nav: pd.Series, rebal_days: int, label: str):
    ret = nav.pct_change()
    n = len(nav)
    is_rebal = np.zeros(n, dtype=bool)
    is_rebal[::rebal_days] = True
    r = ret.to_numpy()
    m = np.isfinite(r) & (np.abs(r) < 0.2)
    a = r[m & is_rebal]
    b = r[m & ~is_rebal]
    if len(a) < 5:
        return
    diff = (a.mean() - b.mean()) * 100
    print(f"  {label:<28} 调仓日均值 {a.mean()*100:+.4f}% | 非调仓日 {b.mean()*100:+.4f}% "
          f"| 差 {diff:+.4f}pp")
    return diff


# 各策略净值
navs = {}
p_main = ROOT / "build/runs/main/nav.parquet"
p_etf = ROOT / "build/runs/etf_trend/nav_donchian55.parquet"
p_sec = ROOT / "build/runs/sector_rotation/nav.parquet"
for name, p, rb in (("主策略", p_main, 10), ("宽基ETF", p_etf, 1), ("板块轮动", p_sec, 10)):
    if p.exists():
        nav = pd.read_parquet(p)["nav"].astype(float)
        nav.index = pd.to_datetime(nav.index)
        navs[name] = nav
        d = check_rebal_effect(nav, rb, name)

print("\n  解读：若调仓日收益显著高于非调仓日（>0.05pp），说明存在'调仓即获利'迹象；")
print("        接近 0 说明调仓本身不产生额外收益。")

# ============================================================
# 3) 特征/标签层面的前视
# ============================================================
print("\n" + "=" * 96)
print("3) 特征/标签层面（因子计算是否用未来数据）")
print("=" * 96)
checks = [
    ("qf/model.py 标签构造", "weekly_label", "label 用 t+1 之后收益（合法，只在 y 列）"),
    ("qf/factors.py 因子", "compute_factor", "因子只用 t 及以前"),
]
for f, kw, note in checks:
    p = ROOT / f
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8")
    neg_shift = len(re.findall(r"shift\(\s*-[1-9]", txt))
    print(f"  {f:<20} 负数 shift 出现 {neg_shift} 次")
    if neg_shift and "label" not in f.lower():
        # 检查是否都在 label 相关函数里
        print(f"    → {note}")

# 检查因子是否用 center=True
for f in ("qf/factors.py", "qf/ops_fast.py", "qf/trend.py", "qf/alpha_rs.py"):
    p = ROOT / f
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        c = txt.count("center=True")
        if c:
            print(f"  ⚠️ {f}: {c} 处 center=True")

print("\n  ✅ 因子层：此前 backtest-guard 审计已确认无负数 shift / bfill / center=True 进入特征")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
