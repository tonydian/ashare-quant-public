# -*- coding: utf-8 -*-
"""板块 ETF 轮动：过拟合与稳健性严格检验。

初步结果（donchian top3: CAGR 20.24%/Sharpe 1.471）显著优于宽基（10.57%/0.912），
必须排除以下可能：
  T1 **选板块偏差**：8 只板块是我挑的（"事后看这些板块有行情"），不是时点可知的；
  T2 **样本极短**：2018 起仅 6~7 年，且 2019-2021 板块牛 + 2024-2025 轮动剧烈；
  T3 **参数尖峰**：top_n / rebal / entry_n 是否高原；
  T4 **退市/存续偏差**：ETF 选择本身有幸存者偏差（表现差的板块 ETF 可能没上市/清盘）；
  T5 **与主策略/宽基相关性**：是否只是重复暴露。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from qf import config as C, metrics as qfm  # noqa: E402
from scripts.sector_rotation_feasibility import (  # noqa: E402
    SECTORS, bt_single_or_rotation, cash_series, load_sectors, perf,
)

OUT = ROOT / "build" / "eval" / "sector_rotation_robustness.json"
S, E = "2020-01-02", "2026-10-23"


def main():
    dates, codes, close, amount = load_sectors()
    cash = cash_series(dates)
    res = {}

    print("=" * 90)
    print("T1 选板块偏差：每只板块单独做趋势，看是否每只都行")
    print("=" * 90)
    single = {}
    for j, c in enumerate(codes):
        nav = bt_single_or_rotation(dates, [c], close[:, [j]], amount[:, [j]],
                                    "donchian", top_n=1, cash_ret=cash)
        single[c] = perf(nav, S, E)
        print(f"  {c} {SECTORS[c]:6s} Sharpe {single[c]['sharpe']:6.3f} CADR {single[c]['cagr']:7.2%}")
    pos = sum(1 for v in single.values() if v["sharpe"] > 0)
    print(f"  → {pos}/8 只 Sharpe 为正，平均 {np.mean([v['sharpe'] for v in single.values()]):.3f}")
    res["t1_single"] = {"per_sector": single, "n_positive": pos,
                        "avg_sharpe": float(np.mean([v["sharpe"] for v in single.values()]))}

    print()
    print("=" * 90)
    print("T2 分年度稳健性（donchian top3 vs 宽基）")
    print("=" * 90)
    nav3 = bt_single_or_rotation(dates, codes, close, amount, "donchian", top_n=3, rebal=10, cash_ret=cash)
    y = qfm.yearly_returns(nav3)
    print("  板块轮动 top3 分年: " + " | ".join(f"{int(k)}: {v*100:6.1f}%" for k, v in y.items()))
    res["t2_yearly"] = {int(k): float(v) for k, v in y.items()}

    print()
    print("=" * 90)
    print("T3 参数高原检验（entry_n × rebal × top_n）")
    print("=" * 90)
    grid = {}
    print(f"  {'entry':>5} {'rebal':>5} {'topN':>5} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8}")
    for entry in (20, 40, 55, 70):
        for rebal in (5, 10, 20):
            for top_n in (1, 2, 3, 5):
                nav = bt_single_or_rotation(dates, codes, close, amount, "donchian",
                                            entry_n=entry, top_n=top_n, rebal=rebal, cash_ret=cash)
                p = perf(nav, S, E)
                grid[f"e{entry}_rb{rebal}_t{top_n}"] = p
                if entry == 55:
                    print(f"  {entry:>5} {rebal:>5} {top_n:>5} {p['cagr']:>7.2%} "
                          f"{p['sharpe']:>7.3f} {p['max_dd']:>7.2%}")
    sharpes = [v["sharpe"] for v in grid.values()]
    print(f"  → 全部 {len(grid)} 组: Sharpe 中位 {np.median(sharpes):.3f}，"
          f"最小 {min(sharpes):.3f}，最大 {max(sharpes):.3f}，为正 {(np.array(sharpes)>0).mean():.0%}")
    res["t3_grid"] = grid
    res["t3_summary"] = {"n": len(grid), "median_sharpe": float(np.median(sharpes)),
                         "min": float(min(sharpes)), "max": float(max(sharpes)),
                         "pct_positive": float((np.array(sharpes) > 0).mean())}

    print()
    print("=" * 90)
    print("T4 切分验证：2020-2021 选参 → 2022+ 样本外")
    print("=" * 90)
    best, best_s = None, -9
    for k, v in grid.items():
        # 只按验证期挑（用网格里的同参数重算验证期）
        pass
    # 直接对几个候选做分期
    for entry in (40, 55):
        for top_n in (2, 3):
            nav = bt_single_or_rotation(dates, codes, close, amount, "donchian",
                                        entry_n=entry, top_n=top_n, rebal=10, cash_ret=cash)
            pv = perf(nav, "2020-01-02", "2021-12-31")
            po = perf(nav, "2022-01-01", E)
            print(f"  e{entry} top{top_n}: 验证期 Sharpe {pv['sharpe']:6.3f} CAGR {pv['cagr']:7.2%} "
                  f"| 样本外 Sharpe {po['sharpe']:6.3f} CAGR {po['cagr']:7.2%}")
            if pv["sharpe"] > best_s:
                best_s, best = pv["sharpe"], (entry, top_n, pv, po)
    print(f"  → 验证期最优: e{best[0]} top{best[1]}，样本外 Sharpe {best[3]['sharpe']:.3f} / "
          f"CAGR {best[3]['cagr']:.2%}")
    res["t4_split"] = {"best_params": [best[0], best[1]], "val": best[2], "oos": best[3]}

    print()
    print("=" * 90)
    print("T5 与主策略 / 宽基 ETF 的相关性")
    print("=" * 90)
    main_nav = pd.read_parquet(ROOT / "build/runs/main/nav.parquet")["nav"].astype(float)
    etf_nav = pd.read_parquet(ROOT / "build/runs/etf_trend/nav_donchian55.parquet")["nav"].astype(float)
    sec_ret = nav3.pct_change().dropna()
    m_ret = main_nav.pct_change().dropna()
    e_ret = etf_nav.pct_change().dropna()
    print(f"  板块轮动 vs 主策略: 相关 {sec_ret.corr(m_ret):.3f}")
    print(f"  板块轮动 vs 宽基ETF: 相关 {sec_ret.corr(e_ret):.3f}")
    res["t5_corr"] = {"vs_main": float(sec_ret.corr(m_ret)),
                      "vs_etf": float(sec_ret.corr(e_ret))}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
