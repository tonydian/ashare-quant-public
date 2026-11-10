# -*- coding: utf-8 -*-
"""Alpha191 空头端（避雷）诊断 —— Step 1 因子层。

动机
----
`docs/ALPHA191.md` 的结论是「整库正交化后多头增量 IC 归零」，但 README §1
同时诚实披露「A 股 OHLCV 因子 alpha 集中在空头端」。两者放一起意味着：

    用 Alpha191 选股（挑最好的）没有增量，
    但用它避雷（挑最差的）可能仍有价值 —— 这是完全不同的用法。

本脚本只做**因子层**判断，不做组合、不跑回测：
1. 把 q1..q5 按 IC 方向对齐成「好端 / 差端」，看多空不对称；
2. 用「差端 - 中位组」定义空头 alpha，用「好端 - 中位组」定义多头 alpha；
3. 结合换手与 IC 稳定性，筛出可用作负面清单的因子。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M = ROOT / "build" / "alpha191" / "metrics.csv"


def main() -> None:
    df = pd.read_csv(M)
    q = df[["q1_5d", "q2_5d", "q3_5d", "q4_5d", "q5_5d"]].to_numpy(dtype=float)
    ic = df["ic_5d"].to_numpy(dtype=float)
    neg = ic < 0

    # 统一方向：good = 收益最高的组，bad = 收益最低的组
    good = np.where(neg, q[:, 0], q[:, 4])
    bad = np.where(neg, q[:, 4], q[:, 0])
    mid = q[:, 2]

    df["short_alpha"] = mid - bad          # 中位组 - 差端（>0 表示差端确实更差）
    df["long_alpha"] = good - mid
    df["asym"] = df["short_alpha"] / np.where(
        np.abs(df["long_alpha"]) > 1e-9, np.abs(df["long_alpha"]), np.nan)

    ok = df["coverage"] >= 0.95
    d = df[ok].copy()

    print(f"因子 {len(df)} 个，覆盖率≥95% 的 {len(d)} 个")
    print(f"\n空头 alpha（中位组-差端，5 日）分布：")
    print(d["short_alpha"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(5).to_string())
    print(f"\n多头 alpha（好端-中位组，5 日）分布：")
    print(d["long_alpha"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(5).to_string())

    n_short = (d["short_alpha"] > 0).sum()
    n_long = (d["long_alpha"] > 0).sum()
    print(f"\n空头 alpha>0 的因子 {n_short} 个；多头 alpha>0 的因子 {n_long} 个")
    print(f"|空头alpha| > |多头alpha| 的因子 {((d['short_alpha'].abs() > d['long_alpha'].abs()).sum())} 个")

    # 空头端最强，且换手可接受
    cols = ["name", "ic_5d", "tstat", "turnover_top20", "monotonicity",
            "short_alpha", "long_alpha", "asym"]
    print("\n=== 空头端最强的 25 个（按 short_alpha 降序）===")
    t = d[d["short_alpha"] > 0].nlargest(25, "short_alpha")[cols]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.round(5).to_string(index=False))

    print("\n=== 其中换手 <=0.40 的子集（负面清单可用）===")
    t2 = d[(d["short_alpha"] > 0) & (d["turnover_top20"] <= 0.40)].nlargest(25, "short_alpha")[cols]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t2.round(5).to_string(index=False))

    print("\n=== 分年稳定性检查（空头最强 15 个，逐年 ic_5d 符号）===")
    top15 = d[d["short_alpha"] > 0].nlargest(15, "short_alpha")
    rows = []
    for _, r in top15.iterrows():
        by = eval(r["ic_by_year"]) if isinstance(r["ic_by_year"], str) else {}
        same = sum(1 for v in by.values() if (v < 0) == (r["ic_5d"] < 0))
        rows.append({
            "name": r["name"], "ic_5d": round(r["ic_5d"], 4),
            "short_alpha": round(r["short_alpha"], 5),
            "turnover": round(r["turnover_top20"], 3),
            "yrs_same_sign": f"{same}/{len(by)}",
            "worst_year_ic": round(max(by.values(), key=abs), 4) if by else np.nan,
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
