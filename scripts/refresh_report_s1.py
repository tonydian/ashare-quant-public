# -*- coding: utf-8 -*-
"""自动更新 reports/回测报告.md 的 §1 摘要（数据更新后保持同步）。"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm

RUN = C.BUILD_RUNS / "main"
REPORT = ROOT / "reports" / "回测报告.md"


def perf(nav, a, b=None):
    sub = nav[(nav.index >= a)] if b is None else nav[(nav.index >= a) & (nav.index <= b)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino",
                                     "max_dd", "calmar", "win_rate_monthly")}


def main():
    nav = pd.read_parquet(RUN / "nav.parquet")["nav"].astype(float)
    nav.index = pd.to_datetime(nav.index)
    a = perf(nav, str(nav.index[0].date()))
    v = perf(nav, str(nav.index[0].date()), "2021-12-31")
    o = perf(nav, "2022-01-01", str(nav.index[-1].date()))

    txt = REPORT.read_text(encoding="utf-8")
    i = txt.find("## 1. 摘要")
    j = txt.find("## 2. 策略逻辑")
    if i < 0 or j <= i:
        print("报告结构异常，跳过")
        return

    # 相位带（2026-11-03）：表里是生产相位 offset=10 的单点值，同一份分数的 CAGR
    # 在 8 个对齐相位上从 3.85% 铺到 7.44%，必须让读者看见区间，而不是只看见最高那一档。
    pb_txt = ""
    pb_p = C.BUILD / "eval" / "phase_band.json"
    if pb_p.exists():
        try:
            pb = json.loads(pb_p.read_text(encoding="utf-8"))
            c = pb["cagr"]
            pb_txt = (
                f"> **调仓相位敏感度（2026-11-03 起必读）**：上表是**生产相位**"
                f"（`rebalance_offset={pb['prod_offset']}`，拿到分数就建仓）的单点结果。"
                f"{pb['rebalance_days']} 日调仓下与周频打分行对齐的相位共 {pb['n_phases']} 个，"
                f"同一份因子分数的 CAGR 从 **{c['min']:.2%} 铺到 {c['max']:.2%}**，"
                f"**中位 {c['median']:.2%}、均值 {c['mean']:.2%}**；"
                f"生产相位有 {pb['prod']['n_better']} 个相位比它更好。"
                f"对外表述主策略收益应报「中位 ± 区间」，单点须标明是生产相位。"
                f"明细见 `build/eval/phase_band.json`。\n\n")
        except Exception as _e:
            print(f"[warn] phase_band.json 解析失败，§1 不带相位带：{_e}")

    block = f"""## 1. 摘要

> 模型 wf_h10 · 40 只 / 40 日调仓 · 个股止损 -30% · 现金替代 **100% 国债** · 数据至 {nav.index[-1].date()}

| 指标 | 全期 {nav.index[0].year}-{nav.index[-1].year} | 验证期 2020-2021 | 样本外 2022+ |
|---|---|---|---|
| 年化复合收益 | **{a['cagr']:.2%}** | **{v['cagr']:.2%}** | **{o['cagr']:.2%}** |
| 年化波动 | {a['ann_vol']:.2%} | {v['ann_vol']:.2%} | {o['ann_vol']:.2%} |
| Sharpe (rf=1.5%) | **{a['sharpe']:.2f}** | {v['sharpe']:.2f} | {o['sharpe']:.2f} |
| Sortino | {a['sortino']:.2f} | {v['sortino']:.2f} | {o['sortino']:.2f} |
| 最大回撤 | **{a['max_dd']:.2%}** | {v['max_dd']:.2%} | {o['max_dd']:.2%} |
| Calmar | **{a['calmar']:.2f}** | {v['calmar']:.2f} | {o['calmar']:.2f} |
| 月胜率 | {a['win_rate_monthly']:.1%} | {v['win_rate_monthly']:.1%} | {o['win_rate_monthly']:.1%} |

{pb_txt}"""
    txt = txt[:i] + block + txt[j:]
    REPORT.write_text(txt, encoding="utf-8")
    print(f"报告 §1 已更新：CAGR {a['cagr']:.2%} / Sharpe {a['sharpe']:.2f} "
          f"/ MaxDD {a['max_dd']:.2%}")


if __name__ == "__main__":
    main()
