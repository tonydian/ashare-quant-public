# -*- coding: utf-8 -*-
"""把债券 ETF 并入 ETF 主面板，并重建池子。

加入：
  511260 10年国债  （2017-08 上市，日均 33 亿，波动 1.35%）
  511010 5年国债   （2013 上市，日均 51 亿，波动 0.75%）
  511220 城投债    （2014 上市，日均 16 亿）
  511090 30年国债  （2023 上市，数据结构上可用但回测区间短）

移除（按文章标准）：
  159861 环保      （日均 256 万，流动性差）
  516970 基建      （日均 0.26 亿，一般，保留观察）

注意：回测起点 2017-01-03，所以只能用 2017 年前上市的：
  511010(2013) ✅  511220(2014) ✅  511260(2017-08) ⚠️ 2017-08 才有数据
  → 为保持完整性，起点可考虑 2018-01-01
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
ETF = ROOT / "build" / "etf"

# 读现有面板
z = np.load(ETF / "etf_ohlc_panel.npz", allow_pickle=True)
old_dates = pd.DatetimeIndex(z["dates"])
old_codes = [str(c) for c in z["codes"]]
old_meta = json.load(open(ETF / "etf_ohlc_meta.json", encoding="utf-8"))

# 读债券数据
bond = json.load(open(ETF / "bond_ohlc_raw.json", encoding="utf-8"))

# 选择要加入的债券 ETF
ADD = {
    "511010": ("5年国债", "债券"),
    "511260": ("10年国债", "债券"),
    "511220": ("城投债", "债券"),
    "511020": ("活跃国债", "债券"),
}

# 移除的
DROP = {"159861"}   # 环保，流动性差

print("=" * 88)
print("重建 ETF 面板")
print("=" * 88)
print(f"  原面板: {len(old_dates)} 日 × {len(old_codes)} 只")

# 合并日期
all_dates = set(old_dates)
for code in ADD:
    if code not in bond:
        print(f"  ⚠️ {code} 无数据")
        continue
    for row in bond[code]["rows"]:
        all_dates.add(pd.Timestamp(row[0]).normalize())
dates = pd.DatetimeIndex(sorted(all_dates))
print(f"  合并后: {len(dates)} 日")

# 保留的旧标的
keep_codes = [c for c in old_codes if c not in DROP]
print(f"  移除 {len(old_codes)-len(keep_codes)} 只: {[old_meta.get(c,c) for c in DROP]}")
new_codes = keep_codes + [c for c in ADD if c in bond]
N = len(new_codes)
T = len(dates)
print(f"  新面板: {T} 日 × {N} 只（新增 {len(ADD)} 只债券）")

pos = {d.normalize(): i for i, d in enumerate(dates)}


def blank():
    return np.full((T, N), np.nan, np.float32)


O, C, H, L, V, A = blank(), blank(), blank(), blank(), blank(), blank()

# 填旧数据（向量化：用索引映射）
old_idx = np.array([pos[d.normalize()] for d in old_dates], dtype=np.int64)
keep_old_j = [old_codes.index(c) for c in keep_codes]
for k, j_old in enumerate(keep_old_j):
    O[old_idx, k] = z["open"][:, j_old]
    C[old_idx, k] = z["close"][:, j_old]
    H[old_idx, k] = z["high"][:, j_old]
    L[old_idx, k] = z["low"][:, j_old]
    V[old_idx, k] = z["volume"][:, j_old]
    A[old_idx, k] = z["amount"][:, j_old]

# 填债券
for code, (name, cat) in ADD.items():
    if code not in bond:
        continue
    k = new_codes.index(code)
    n_row = 0
    for row in bond[code]["rows"]:
        try:
            i = pos[pd.Timestamp(row[0]).normalize()]
            O[i, k] = float(row[1])
            C[i, k] = float(row[2])
            H[i, k] = float(row[3])
            L[i, k] = float(row[4])
            if row[5] not in (None, ""):
                V[i, k] = float(row[5])
            if len(row) > 8 and row[8] not in (None, ""):
                # 债券 ETF 的 amount 单位可能与股票 ETF 不同，检查量级
                A[i, k] = float(row[8])
            n_row += 1
        except Exception:
            continue
    print(f"  ✅ 加入 {code} {name}: {n_row} 条")

np.savez_compressed(
    ETF / "etf_ohlc_panel_v2.npz",
    dates=dates.values, codes=np.array(new_codes),
    open=O, close=C, high=H, low=L, volume=V, amount=A)

# 新 meta
meta = {c: old_meta.get(c, "?") for c in keep_codes}
for c, (n, _) in ADD.items():
    if c in bond:
        meta[c] = n
json.dump(meta, open(ETF / "etf_ohlc_meta_v2.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

cov = np.isfinite(C)
print(f"\n  覆盖率: {cov.mean():.1%}")
print(f"  区间: {dates[0].date()} ~ {dates[-1].date()}")

# 检查债券 ETF 的 amount 量级
print("\n  债券 ETF 成交额量级检查:")
for code in ADD:
    if code not in bond:
        continue
    k = new_codes.index(code)
    a = A[:, k]
    ok = np.isfinite(a)
    if ok.sum() > 20:
        print(f"    {code} {meta[code]}: 中位 {np.nanmedian(a):,.0f}, "
              f"最近 {np.nanmean(a[-20:]):,.0f}")

# 每只的有效期
print("\n  各标的有效期（前后 5 只）:")
for c in new_codes[:3] + new_codes[-5:]:
    k = new_codes.index(c)
    s = C[:, k]
    ok = np.isfinite(s)
    if ok.sum():
        print(f"    {c} {meta[c]:<10}: {dates[ok][0].date()} ~ {dates[ok][-1].date()}"
              f" ({ok.sum()} 天)")

print(f"\n→ {ETF / 'etf_ohlc_panel_v2.npz'}")
