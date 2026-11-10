# -*- coding: utf-8 -*-
"""验证因子池缓冲机制：模拟多次数据微变，对比"有缓冲 vs 无缓冲"的池换手率。"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

from qf import pool_guard as pg

print("=" * 84)
print("缓冲机制逻辑验证")
print("=" * 84)

# 模拟 10 次数据微变：排名尾部小幅抖动
rng = np.random.default_rng(7)
N_FACTORS = 80
K = 32
base_quality = np.sort(rng.normal(0.5, 0.3, N_FACTORS))[::-1]
base_order = [f"F{i:03d}" for i in np.argsort(-base_quality)]

print(f"\n模拟：{N_FACTORS} 个因子，取前 {K}，10 次重估")
print(f"{'次':>3}{'无缓冲换手':>12}{'有缓冲换手':>12}{'无缓冲池∩基准':>16}{'有缓冲池∩基准':>16}")
print("-" * 62)

no_guard_pool = None
guard_pool = None
guard_state = {}
stats_no, stats_guard = [], []

for trial in range(10):
    # 每次排名 = 基准质量 + 噪声（模拟数据微变）
    noise = rng.normal(0, 0.04, N_FACTORS)   # 小噪声
    q = base_quality + noise
    order = [f"F{i:03d}" for i in np.argsort(-q)]

    # 无缓冲：直接取前 K
    new_no = order[:K]
    if no_guard_pool is None:
        turn_no = 0
    else:
        turn_no = len(set(new_no) - set(no_guard_pool))
    no_guard_pool = new_no

    # 有缓冲
    new_g, guard_state, info = pg.apply_guard(
        prev_pool=guard_pool, ranked=order, k=K,
        state=guard_state, need_streak=2, verbose=False)
    turn_g = info["turnover"]
    guard_pool = new_g

    overlap_no = len(set(no_guard_pool) & set(base_order[:K]))
    overlap_g = len(set(guard_pool) & set(base_order[:K]))
    stats_no.append(turn_no)
    stats_guard.append(turn_g)
    print(f"{trial+1:>3}{turn_no:>12}{turn_g:>12}{overlap_no:>16}{overlap_g:>16}")

print("\n" + "=" * 84)
print("汇总")
print("=" * 84)
print(f"  无缓冲：10 次累计换手 {sum(stats_no)} 个（平均 {np.mean(stats_no):.1f}/次）")
print(f"  有缓冲：10 次累计换手 {sum(stats_guard)} 个（平均 {np.mean(stats_guard):.1f}/次）")
reduction = (1 - sum(stats_guard) / max(sum(stats_no), 1)) * 100
print(f"  → 换手率降低 {reduction:.0f}%")

# 验证：缓冲后是否还能跟上真正变化的因子
print("\n" + "=" * 84)
print("关键验证：如果某个因子真的变强了，缓冲能否及时纳入？")
print("=" * 84)
guard_state2 = {}
pool2 = None
# 第 1 次：正常
order0 = base_order
pool2, guard_state2, _ = pg.apply_guard(None, order0, K, guard_state2, verbose=False)
print(f"  初始池含 F000（最强）: {'F000' in pool2}")

# 第 2 次：让一个原本很弱的因子（F070）变最强
q3 = base_quality.copy()
q3[70] = 3.0     # 突然变超强
order3 = [f"F{i:03d}" for i in np.argsort(-q3)]
pool3, guard_state2, info3 = pg.apply_guard(pool2, order3, K, guard_state2,
                                            need_streak=2, verbose=False)
print(f"  变强第 1 次：F070 入池 = {'F070' in pool3}（排名 {order3.index('F070')+1}）")

# 第 3 次：继续保持强
pool4, guard_state2, info4 = pg.apply_guard(pool3, order3, K, guard_state2,
                                            need_streak=2, verbose=False)
print(f"  变强第 2 次：F070 入池 = {'F070' in pool4}")

if "F070" in pool4:
    print("  ✅ 真正变强的因子能在 1~2 次内进入")
else:
    print("  ⚠️ 未能及时纳入，需检查逻辑")
