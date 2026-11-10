# -*- coding: utf-8 -*-
"""修复板块池数据质量问题 + 重跑方向 A 测试。

发现的问题：
  1) 酒 ETF（512690）等 16 条"前复权因子跳变"（单日 ±24%），实际不复权收益正常
     → 这是腾讯 qfq 接口在除权日的复权基准切换导致，需用"不复权价 + 复权因子修正"
  2) 2026-07-21 半导体：qfq +14.3% vs 不复权 -61.9%，数据源明显错误

修复策略：
  * 用 raw_close 计算真实日收益，用 qfq 的累计复权因子（close/raw_close）平滑拼接
  * 具体：日收益 = raw[t]/raw[t-1] - 1（真实价格变动，不含分红）
           再单独处理分红：当 |qfq收益 - raw收益| > 5pp 且 raw 正常时，判定为复权跳变，用 raw 收益
  * 对 raw 缺失的日期用 qfq 收益兜底
  * 最终重建 close 序列（从 1.0 累乘修复后的收益）
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "build" / "etf" / "sector_panel.npz"
FIXED = ROOT / "build" / "etf" / "sector_panel_fixed.npz"
REPORT = ROOT / "build" / "eval" / "sector_data_fix_report.json"

z = np.load(PANEL, allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
close = z["close"].astype(float)
raw = z["raw_close"].astype(float)
amount = z["amount"].astype(float)

print(f"原始面板 {len(dates)} 日 × {len(codes)} 只")

T, N = close.shape
fixed_close = np.full_like(close, np.nan)
log = []

for j, c in enumerate(codes):
    qfq = close[:, j].copy()
    rw = raw[:, j].copy()
    n_fix = 0
    # 用不复权收益作为真实价格变动（更可靠，不含复权因子跳变）
    ret_raw = np.full(T, np.nan)
    ok = np.isfinite(rw[1:]) & np.isfinite(rw[:-1]) & (rw[:-1] > 0)
    ret_raw[1:][ok] = rw[1:][ok] / rw[:-1][ok] - 1.0
    # 剔除明显的错误数据（|raw 收益| > 25% 且非真实涨跌停）
    ret_raw[np.abs(ret_raw) > 0.25] = np.nan

    ret_qfq = np.full(T, np.nan)
    ok2 = np.isfinite(qfq[1:]) & np.isfinite(qfq[:-1]) & (qfq[:-1] > 0)
    ret_qfq[1:][ok2] = qfq[1:][ok2] / qfq[:-1][ok2] - 1.0

    # 优先用 raw 收益；raw 缺失时用 qfq
    ret = np.where(np.isfinite(ret_raw), ret_raw, ret_qfq)
    # 统计修正点数
    diff = np.abs(ret_qfq - ret_raw)
    n_fix = int(np.sum(np.isfinite(diff) & (diff > 0.05)))
    if n_fix:
        log.append({"code": c, "n_fixed": n_fix})

    # 重建价格序列（从第一个有效点开始累乘）
    first = int(np.flatnonzero(np.isfinite(qfq))[0]) if np.isfinite(qfq).any() else -1
    if first < 0:
        continue
    px = np.full(T, np.nan)
    px[first] = float(qfq[first])
    for t in range(first + 1, T):
        if np.isfinite(ret[t]):
            px[t] = px[t - 1] * (1.0 + ret[t])
        else:
            px[t] = px[t - 1] if np.isfinite(px[t - 1]) else np.nan
    fixed_close[:, j] = px

print(f"修正了 {len(log)} 只标的的复权跳变：")
for r in sorted(log, key=lambda x: -x["n_fixed"])[:10]:
    print(f"  {r['code']}: {r['n_fixed']} 处")

# 校验：修复后是否还有异常日收益
print("\n修复后异常检查（|日收益|>11%）：")
n_bad = 0
for j, c in enumerate(codes):
    col = fixed_close[:, j]
    r = col[1:] / col[:-1] - 1.0
    idx = np.flatnonzero(np.abs(r) > 0.11)
    for i in idx:
        n_bad += 1
        if n_bad <= 8:
            print(f"  {dates[i+1].date()} {c} {r[i]*100:+.1f}%")
print(f"  合计 {n_bad} 条（修复前 45 条）")

np.savez_compressed(FIXED, dates=dates.values, codes=np.array(codes),
                    close=fixed_close.astype(np.float32),
                    raw_close=raw.astype(np.float32),
                    amount=amount.astype(np.float32))
REPORT.write_text(json.dumps({"n_bad_before": 45, "n_bad_after": n_bad,
                              "fixed": log}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {FIXED}")
print(f"→ {REPORT}")
