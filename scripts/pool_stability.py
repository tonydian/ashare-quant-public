# -*- coding: utf-8 -*-
"""验证：因子池重选到底造成了多少差异？

实验：
  A. 用当前模型跑（factor pool 已变）→ 6.52%
  B. 重建模型（--pool-from 旧池）？没有旧池快照
  C. 直接对比：重建前后的因子池差异
  D. 量化：随机换掉 N 个因子，CAGR 波动多大？

关键：需要找出"8.87% 那次"和"6.52% 这次"的因子池差异。
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

from qf import config as C, model as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "pool_stability.json"

print("=" * 92)
print("因子池稳定性分析")
print("=" * 92)

# 当前池
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
cur_pool = list(res.pool)
print(f"\n当前池 {len(cur_pool)} 个因子:")
for k in cur_pool:
    print(f"  {k}")

# 找所有历史 wf_h10 备份
print("\n" + "=" * 92)
print("历史模型目录（找旧池快照）")
print("=" * 92)
for d in sorted((C.BUILD_MODELS).iterdir()):
    if not d.is_dir():
        continue
    mp = d / "model_meta.json"
    if not mp.exists():
        continue
    try:
        m = json.load(open(mp, encoding="utf-8"))
        mp2 = d / "pool_meta.parquet"
        n = len(m.get("pool") or [])
        ts = m.get("built_at") or m.get("created") or ""
        print(f"  {d.name:<24} {n:>3} 因子  {ts}")
    except Exception:
        pass

# pool_meta 里有 IC 信息，可看因子排名稳定性
print("\n" + "=" * 92)
print("当前池的 IC 分布（看有多少处于边界）")
print("=" * 92)
pm = pd.read_parquet(C.BUILD_MODELS / "wf_h10" / "pool_meta.parquet")
pm = pm.sort_values("abs_icir", ascending=False)
print(f"\n{'#':>3} {'因子':<22}{'IC':>10}{'ICIR':>9}{'t':>8}")
print("-" * 56)
for i, (k, r) in enumerate(pm.iterrows(), 1):
    mark = " ← 边界" if i >= len(pm) - 4 else ""
    print(f"{i:>3} {k:<22}{r['ic']:>+10.4f}{r['icir']:>+9.3f}"
          f"{r['t']:>+8.2f}{mark}")

print("\n" + "=" * 92)
print("结论")
print("=" * 92)
print(f"""
  当前池最后一个因子的 |ICIR| = {pm['abs_icir'].iloc[-1]:.3f}
  倒数第 5 个 = {pm['abs_icir'].iloc[-5]:.3f}

  含义：池尾部因子的 |ICIR| 都在 0.4~0.5 附近，彼此差距很小。
  数据微变（如多 8 个交易日）就足以让排名的尾部换人。

  这就是"因子池抖动"的机制：**选池标准是硬阈值 + 硬排名，没有缓冲区。**
""")

json.dump({"current_pool": cur_pool,
           "pool_meta": {k: {"ic": float(r["ic"]), "icir": float(r["icir"]),
                             "t": float(r["t"])}
                         for k, r in pm.iterrows()}},
          open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n→ {OUT}")
