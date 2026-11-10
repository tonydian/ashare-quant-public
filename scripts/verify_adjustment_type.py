# -*- coding: utf-8 -*-
"""核查：我们的净值口径是否因复权处理而失真？

文章（《复权》）提出的三个价格口径：
  ① 动量因子 → 等比复权价
  ② 成交价   → 真实原始价
  ③ 净值     → 动态复权（分红现金入账）

我们的实现（从代码看）：
  · 全程用 md.price（后复权价）
  · 成交：open[t] × (1±滑点)
  · 净值：现金 + Σ 股数 × md.price[t]
  · 现金按 1.5%/365 计息
  · 无分红入账、无送股处理

理论上是否等价？
  ✅ 如果价格是**等比后复权**：
     md.price 已经隐含"分红再投资"，
     所以用复权价算净值 == 动态复权算净值（数学上等价，文章 §6.2 已证明）
     现金利息单独计（我们做了）

  ❌ 但如果价格是**普通后复权**（加减法）：
     涨幅会失真（文章 §3.2 例子：真实 7.50% vs 普通后复权 7.80%）
     长期累积会高估净值

关键验证：我们的数据是等比还是普通？
  方法：找一只高分红股，取除权日前后，
        比较"复权价算的涨幅"与"真实涨幅（含分红）"
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

from qf import config as C, data as qfdata  # noqa: E402

print("=" * 96)
print("验证：我们的复权数据是「等比」还是「普通」？")
print("=" * 96)

md = qfdata.load_market(verbose=False)
dates = pd.DatetimeIndex(md.dates)
print(f"  数据 {md.T} 日 × {md.N} 只，{dates[0].date()} ~ {dates[-1].date()}")

# 读分红数据（我们已抓的）
div_path = ROOT / "build" / "alt" / "dividend_raw.jsonl"
print(f"\n  分红数据: {'存在' if div_path.exists() else '缺失'}")

# 找高分红 + 有除息日落在数据区间内的股票
code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}
date_pos = {d.normalize(): i for i, d in enumerate(dates)}

events = []   # (code, j, ex_date_idx, 每股派息)
if div_path.exists():
    with open(div_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for r in obj["rows"]:
                sc = str(r.get("SECURITY_CODE") or "").zfill(6)
                j = code_idx.get(sc)
                if j is None:
                    continue
                exd = r.get("EX_DIVIDEND_DATE")
                if not exd:
                    continue
                try:
                    di = date_pos.get(pd.Timestamp(exd).normalize())
                except Exception:
                    continue
                if di is None or di < 5 or di > md.T - 5:
                    continue
                try:
                    per10 = float(r.get("PRETAX_BONUS_RMB") or 0)
                except Exception:
                    per10 = 0.0
                if per10 > 0:
                    events.append((sc, j, di, per10 / 10.0))

print(f"  可用分红事件（落在数据区间内）: {len(events)}")

if not events:
    print("无事件，退出")
    sys.exit(0)

# 对每个事件，比较：
#   复权价涨幅  r_adj = price[t] / price[t-1] - 1
#   若价格是"等比后复权"，r_adj 应等于真实涨幅（含分红）
#   真实涨幅 r_true = (P_close_t + 分红) / P_close_{t-1} - 1
#   但我们没有 raw_close 的精确对应... 用 raw_close.npz 试试
raw = None
rp = C.DATA_RAW / "raw_close.npz"
if rp.exists():
    raw = np.load(rp)
    key = "raw_close" if "raw_close" in raw.files else raw.files[0]
    raw = raw[key].astype(np.float64)
    print(f"  raw_close 可用: {raw.shape}")
else:
    print("  ⚠️ raw_close 不存在，无法直接对照")

print("\n" + "=" * 96)
print("抽样检查（前 10 个分红事件）")
print("=" * 96)
print(f"{'代码':<8}{'除息日':<12}{'每股派息':>9}{'复权涨幅':>11}"
      f"{'原始涨幅':>11}{'差异':>10}{'判定':>8}")
print("-" * 78)

n_equal = n_diff = 0
diffs = []
for sc, j, di, dps in events[:60]:
    p_prev = md.price[di - 1, j]
    p_t = md.price[di, j]
    if not (np.isfinite(p_prev) and np.isfinite(p_t) and p_prev > 0):
        continue
    r_adj = p_t / p_prev - 1

    if raw is not None and np.isfinite(raw[di - 1, j]) and np.isfinite(raw[di, j]):
        rp_prev, rp_t = raw[di - 1, j], raw[di, j]
        if rp_prev > 0:
            r_adj_raw = rp_t / rp_prev - 1
            # 真实涨幅 = (原始收盘 + 分红) / 前收 - 1
            r_true = (rp_t + dps) / rp_prev - 1
            diff = r_adj - r_true
            diffs.append(diff)
            ok = abs(diff) < 1e-4
            if ok:
                n_equal += 1
            else:
                n_diff += 1
            if len(diffs) <= 10:
                print(f"{sc:<8}{dates[di].date()!s:<12}{dps:>9.4f}"
                      f"{r_adj*100:>10.3f}%{r_true*100:>10.3f}%"
                      f"{diff*100:>9.4f}{'等比✅' if ok else '偏差❌':>8}")

print("\n" + "=" * 96)
print("统计结论")
print("=" * 96)
if diffs:
    d = np.array(diffs)
    print(f"  样本: {len(d)} 个分红事件")
    print(f"  平均偏差: {d.mean()*1e4:+.3f} bp")
    print(f"  最大偏差: {np.abs(d).max()*1e4:.3f} bp")
    print(f"  中位偏差: {np.median(np.abs(d))*1e4:.3f} bp")
    print(f"  完全一致(<1bp): {(np.abs(d) < 1e-4).sum()} / {len(d)}")
    if np.abs(d).max() < 1e-4:
        print("\n  ✅ 我们的复权是「等比后复权」")
        print("     → 净值口径正确（复权价隐含分红再投资，与动态复权等价）")
        print("     → 无需额外分红入账逻辑")
    else:
        print(f"\n  ⚠️ 存在偏差，最大 {np.abs(d).max()*1e4:.1f}bp")
        print("     → 可能是普通后复权，或 raw_close 口径不完全对应")
else:
    print("  无有效对照（raw_close 可能缺失）")

# 额外：检查复权价与原始价的比值变化（等比复权比值应恒定在分红前后跳变）
print("\n" + "=" * 96)
print("补充：复权因子检查（高分红股）")
print("=" * 96)
if raw is not None:
    for sc in ("600153", "601398", "600900"):
        j = code_idx.get(sc)
        if j is None:
            continue
        ratio = md.price[:, j] / raw[:, j]
        ok = np.isfinite(ratio) & (ratio > 0)
        if ok.sum() < 100:
            continue
        # 比值应单调递增（后复权因子累积）
        r_ok = ratio[ok]
        print(f"\n  {sc}: 复权因子 {r_ok.min():.4f} → {r_ok.max():.4f}")
        # 检查是否单调不减（后复权的特征）
        d = np.diff(r_ok)
        n_down = int((d < -1e-9).sum())
        print(f"    因子下降次数: {n_down}（后复权应接近 0）")
        if n_down < len(d) * 0.01:
            print(f"    ✅ 符合后复权特征（因子单调累积）")
