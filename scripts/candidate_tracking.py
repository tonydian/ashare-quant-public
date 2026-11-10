# -*- coding: utf-8 -*-
"""候选模型前向跟踪机制（wf_h10_m_all3 等）。

思路（无法真的"等 1-2 年"，但能做两件等价的事）：
  1) **滚动样本外回溯**（walk-forward OOS）：
     模拟"在某时点上线"，只用该时点之前的数据训练，然后统计之后的表现。
     这是最接近"实盘跟踪"的严谨做法。
  2) **信号落盘 + 每日对账**：
     每次运行记录候选模型的当日信号与净值，累积成真实的前向记录。
     等真实时间流逝后，这里就是"实盘跟踪"的数据源。

产出：
  build/tracking/candidates.jsonl    —— 累积的前向记录
  build/eval/candidate_rolling_oos.json —— 滚动样本外统计
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import backtest as qfbt, config as C, data as qfdata, metrics as qfm  # noqa: E402

TRACK = C.BUILD / "tracking"
TRACK.mkdir(parents=True, exist_ok=True)
OUT = C.BUILD / "eval" / "candidate_rolling_oos.json"
S = "2020-01-08"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
phase = int(rows[0]) % 5
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)

CANDIDATES = {
    "wf_h10": "基线（34 量价）",
    "wf_h10_m_all3": "候选（量价+两融+龙虎榜+快照）",
    "wf_h10_smallvol": "候选（小市值+高波动）",
    "wf_h10_margin_flow": "候选（量价+两融+资金流）",
}

# ============================================================
print("=" * 96)
print("1) 滚动样本外：模拟「在不同时点上线」的表现")
print("=" * 96)
print("  方法：把回测区间按年切分，逐年统计累计表现（等价于各时点上线的后续表现）")


def load_scores(tag):
    p = C.BUILD_MODELS / tag / "composite_score.npz"
    if not p.exists():
        return None
    sc = np.load(p)["scores"]
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    elif sc.shape[0] > md.T:
        sc = sc[:md.T]
    return sc


def run(tag):
    sc = load_scores(tag)
    if sc is None:
        return None
    start = int(md.date_index(S))
    while start % 5 != phase:
        start += 1
    end = md.T - 1
    offset = (phase - start) % 10
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=offset,
                              trade_band_rel=0.30, trade_band_abs=6000.0,
                              max_new_entries=6, cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    return r.nav


navs = {}
for tag in CANDIDATES:
    n = run(tag)
    if n is not None:
        navs[tag] = n
        print(f"  {tag:<22} ✓ {n.index[0].date()} ~ {n.index[-1].date()}")

# ============================================================
print("\n" + "=" * 96)
print("2) 「若在某个时点上线」的后续表现（关键表格）")
print("=" * 96)
# 各起点：2021/2022/2023/2024/2025 年初上线
starts = ["2021-01-04", "2022-01-04", "2023-01-03", "2024-01-02", "2025-01-02"]
print(f"\n  {'上线时点':<12}" + "".join(f"{t.replace('wf_h10','').replace('_','') or 'base':>16}"
                                      for t in navs))
print("  " + "-" * (12 + 16 * len(navs)))
roll = {}
for st in starts:
    line = f"  {st:<12}"
    for tag in navs:
        sub = navs[tag][navs[tag].index >= st]
        if len(sub) < 20:
            line += f"{'—':>16}"
            continue
        tot = sub.iloc[-1] / sub.iloc[0] - 1
        yrs = len(sub) / 252
        ann = (1 + tot) ** (1 / yrs) - 1
        roll.setdefault(tag, {})[st] = float(ann)
        line += f"{ann:>15.2%}"
    print(line)
print("\n  → 若候选模型在每个上线时点都更好，说明稳定性好")

# ============================================================
print("\n" + "=" * 96)
print("3) 滚动 252 日超额（候选 vs 基线）")
print("=" * 96)
base = navs.get("wf_h10")
if base is not None:
    for tag in navs:
        if tag == "wf_h10":
            continue
        common = base.index.intersection(navs[tag].index)
        rb = base[common].pct_change().fillna(0)
        rc = navs[tag][common].pct_change().fillna(0)
        excess = (rc - rb).rolling(252).sum()
        print(f"\n  {tag}（滚动 252 日累计超额）:")
        print(f"    {'时段':<20}{'累计超额':>12}")
        for y in (2021, 2022, 2023, 2024, 2025, 2026):
            sub = excess[excess.index.year == y]
            if sub.notna().sum() > 10:
                print(f"    {y:<20}{sub.mean()*100:>+11.2f}%")
        n_pos = int((excess.dropna() > 0).sum())
        n_tot = int(excess.notna().sum())
        print(f"    正值占比: {n_pos}/{n_tot} = {n_pos/max(n_tot,1):.0%}")

# ============================================================
print("\n" + "=" * 96)
print("4) 信号落盘（用于未来真实跟踪）")
print("=" * 96)
today = str(md.dates[-1].date())
rec = {
    "asof": today,
    "recorded_at": datetime.now().isoformat(timespec="seconds"),
    "data_end": today,
    "candidates": {},
}
for tag in navs:
    nav = navs[tag]
    sub = nav[nav.index >= S]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    rec["candidates"][tag] = {
        "desc": CANDIDATES.get(tag, ""),
        "nav_last": float(nav.iloc[-1]),
        "cagr": float(p["cagr"]),
        "sharpe": float(p["sharpe"]),
        "max_dd": float(p["max_dd"]),
        "calmar": float(p["calmar"]),
    }

# 追加或更新当日记录
track_p = TRACK / "candidates.jsonl"
existing = []
if track_p.exists():
    with open(track_p, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
                if o.get("asof") != today:
                    existing.append(line.rstrip("\n"))
            except Exception:
                pass
with open(track_p, "w", encoding="utf-8") as f:
    for line in existing:
        f.write(line + "\n")
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"  记录 {len(rec['candidates'])} 个候选 → {track_p}")
for tag, d in rec["candidates"].items():
    print(f"    {tag:<22} CAGR {d['cagr']:6.2%} Sharpe {d['sharpe']:5.3f} "
          f"MaxDD {d['max_dd']:7.2%}")

# ============================================================
# 保存
result = {
    "rolling_start": roll,
    "latest": rec["candidates"],
    "starts": starts,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float),
               encoding="utf-8")
print(f"\n→ {OUT}")
print(f"→ {track_p}")

# ============================================================
print("\n" + "=" * 96)
print("5) 结论")
print("=" * 96)
print("""
  这个机制做了两件事：
  1) 滚动样本外回溯 —— 立刻就能看到"在不同时点上线会怎样"（已产出）
  2) 信号落盘 —— 每次数据更新后累积真实前向记录（已建立，需持续运行）

  未来判断标准（建议）：
  * 候选模型连续 6 个月跑赢基线 → 提升为默认
  * 连续 6 个月跑输 → 淘汰
  * 单月剧烈偏离（>10pp）→ 立即复查数据与逻辑
""")
