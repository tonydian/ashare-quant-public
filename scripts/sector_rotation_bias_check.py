# -*- coding: utf-8 -*-
"""深度怀疑：板块轮动结果强得反常，必须排查三个致命偏差。

  D1 **ETF 存续/选择偏差**：8 只板块 ETF 是"事后看有行情"的板块，
     真实可投资时点（2018）并不知道哪些板块未来会牛。测：随机选 3 只板块 vs 精选 3 只。
  D2 **前复权价格前视**：qfq 前复权以"最新价"为基准，历史价格被未来分红/拆分调整；
     对**收益率**无影响，但若用绝对价格做判断（如价格带）会有偏差。此处只用收益率，需确认。
  D3 **数据对齐/停牌**：板块 ETF 有缺失日，ffill 后可能产生虚假收益。

  D4 **最重要**：把板块轮动放在"不知道未来哪些板块会牛"的语境下重测——
     即 **时点可知的板块池**（2018 年已上市的、且当时就有一定规模的板块 ETF）。
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

OUT = ROOT / "build" / "eval" / "sector_rotation_bias_check.json"
rng = np.random.default_rng(20261024)


def main():
    dates, codes, close, amount = load_sectors()
    cash = cash_series(dates)
    S, E = "2020-01-02", "2026-10-23"
    res = {}

    print("=" * 92)
    print("D1 选择偏差：随机抽 3 只板块 vs 全部 8 只 vs 每只单独")
    print("=" * 92)
    # 随机抽 3 只（100 次），看 Sharpe 分布
    sims = []
    for _ in range(100):
        pick = rng.choice(len(codes), 3, replace=False)
        nav = bt_single_or_rotation(dates, [codes[j] for j in pick],
                                    close[:, pick], amount[:, pick],
                                    "donchian", top_n=1, rebal=10, cash_ret=cash)
        sims.append(perf(nav, S, E)["sharpe"])
    sims = np.array(sims)
    print(f"  随机 3 只板块（100 次）: Sharpe 中位 {np.median(sims):.3f} | "
          f"p10 {np.percentile(sims,10):.3f} | p90 {np.percentile(sims,90):.3f} | "
          f"为正 {(sims>0).mean():.0%}")
    res["d1_random3"] = {"median": float(np.median(sims)),
                         "p10": float(np.percentile(sims, 10)),
                         "p90": float(np.percentile(sims, 90)),
                         "pct_positive": float((sims > 0).mean())}

    # 全 8 只轮动
    nav8 = bt_single_or_rotation(dates, codes, close, amount, "donchian", top_n=3, rebal=10, cash_ret=cash)
    p8 = perf(nav8, S, E)
    print(f"  全 8 只轮动 top3: Sharpe {p8['sharpe']:.3f} CAGR {p8['cagr']:.2%}")
    res["d1_all8"] = p8

    print()
    print("=" * 92)
    print("D2 前复权价格：验证收益率是否受复权基准影响")
    print("=" * 92)
    # 用不复权价 vs 前复权价分别算收益，差异应仅来自分红（ETF 分红少）
    from scripts.build_etf_data import fetch
    for code in ("512880", "512010"):
        qfq = fetch(code, "qfq", "2020-01-01", "2026-10-23")
        raw = fetch(code, "", "2020-01-01", "2026-10-23")
        if qfq.empty or raw.empty:
            continue
        m = pd.merge(qfq[["date", "close"]], raw[["date", "close"]], on="date",
                     suffixes=("_qfq", "_raw"))
        r_q = m["close_qfq"].iloc[-1] / m["close_qfq"].iloc[0] - 1
        r_r = m["close_raw"].iloc[-1] / m["close_raw"].iloc[0] - 1
        print(f"  {code} {SECTORS[code]}: 前复权总收益 {r_q*100:+.2f}% vs 不复权 {r_r*100:+.2f}% "
              f"| 差 {(r_q-r_r)*100:+.2f}pp（分红）")
        res[f"d2_{code}"] = {"qfq": float(r_q), "raw": float(r_r), "diff": float(r_q - r_r)}

    print()
    print("=" * 92)
    print("D3 数据缺失影响：板块 ETF 缺失日如何处理")
    print("=" * 92)
    for j, c in enumerate(codes):
        col = close[:, j]
        fin = np.isfinite(col)
        print(f"  {c} {SECTORS[c]:6s}: 有效 {fin.sum():4d}/{len(col)} "
              f"| 首个有效 {dates[fin][0].date()} | 缺失 {(~fin).sum()} 天")
    print("  → 缺失日 ffill 后收益率为 0（不影响趋势方向），但会低估滚动波动")

    print()
    print("=" * 92)
    print("D4 【关键】时点可知板块池：只用 2018-01 已上市且有规模的板块")
    print("=" * 92)
    # 2018-01 时已上市：全部 8 只中除 512480(2019-06)、512690(2019-05) 外都已在
    t0 = int(np.flatnonzero(dates >= pd.Timestamp("2018-01-02"))[0])
    avail = []
    for j, c in enumerate(codes):
        col = close[:, j]
        if np.isfinite(col[t0]) and np.isfinite(amount[t0, j]) and amount[t0, j] > 1e7:
            avail.append(c)
    print(f"  2018-01 可投资板块 ETF: {[(c, SECTORS[c]) for c in avail]}（{len(avail)} 只）")
    idx_avail = [codes.index(c) for c in avail]
    nav_av = bt_single_or_rotation(dates, avail, close[:, idx_avail], amount[:, idx_avail],
                                   "donchian", top_n=3, rebal=10, cash_ret=cash)
    p_av = perf(nav_av, S, E)
    p_av2 = perf(nav_av, "2022-01-01", E)
    print(f"  时点可知池 top3: 全期 Sharpe {p_av['sharpe']:.3f} CAGR {p_av['cagr']:.2%} "
          f"MaxDD {p_av['max_dd']:.2%}")
    print(f"                 样本外(2022+) Sharpe {p_av2['sharpe']:.3f} CAGR {p_av2['cagr']:.2%}")
    res["d4_timepoint_pool"] = {"sectors": avail, "full": p_av, "oos": p_av2}

    print()
    print("=" * 92)
    print("D5 逐年拆解：收益是否集中在少数年份（2019-2020 板块牛）")
    print("=" * 92)
    yy = qfm.yearly_returns(nav_av)
    print("  时点可知池 top3 分年: " + " | ".join(f"{int(k)}: {v*100:6.1f}%" for k, v in yy.items()))
    print(f"  剔除 2019-2020 后的年化：", end="")
    sub = nav_av[(nav_av.index >= "2021-01-01")]
    p_sub = qfm.performance(sub, rf=C.CASH_YIELD)
    print(f"CAGR {p_sub['cagr']:.2%} Sharpe {p_sub['sharpe']:.3f}")
    res["d5_yearly"] = {int(k): float(v) for k, v in yy.items()}
    res["d5_ex2019_2020"] = {k: float(p_sub[k]) for k in ("cagr", "sharpe", "max_dd")}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
