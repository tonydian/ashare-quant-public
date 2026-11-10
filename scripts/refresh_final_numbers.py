# -*- coding: utf-8 -*-
"""刷新 build/eval/final_numbers.json（各模型最新数字，供测试与文档引用）。"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm

OUT = C.BUILD / "eval" / "final_numbers.json"


def load(p):
    if not p.exists():
        return None
    s = pd.read_parquet(p)["nav"].astype(float)
    s.index = pd.to_datetime(s.index)
    return s


def perf(nav, a="2020-01-08", b=None):
    sub = nav[(nav.index >= a)] if b is None else nav[(nav.index >= a) & (nav.index <= b)]
    if len(sub) < 20:
        return None
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino",
                                     "max_dd", "calmar")}


out = {}

# 主策略
nav = load(C.BUILD_RUNS / "main" / "nav.parquet")
if nav is not None:
    out["main"] = {**perf(nav), "periods": {
        "full": perf(nav),
        "val": perf(nav, "2020-01-08", "2021-12-31"),
        "oos": perf(nav, "2022-01-01"),
    }}

# 相位带（2026-11-03）：build/runs/main 的 nav 用「拿到分数就建仓」这一个 offset，
# 实测是 8 个对齐相位里最高的一档。单点 CAGR 不能当能力披露，这里并给出分布。
_pb = C.BUILD / "eval" / "phase_band.json"
if _pb.exists():
    try:
        _b = json.load(open(_pb, encoding="utf-8"))
        out.setdefault("main", {})["phase_band"] = {
            "start": _b["start"], "end": _b["end"],
            "rebalance_days": _b["rebalance_days"], "n_phases": _b["n_phases"],
            "prod_offset": _b["prod_offset"],
            "cagr_mean": _b["cagr"]["mean"], "cagr_median": _b["cagr"]["median"],
            "cagr_min": _b["cagr"]["min"], "cagr_max": _b["cagr"]["max"],
            "sharpe_mean": _b["sharpe"]["mean"], "sharpe_median": _b["sharpe"]["median"],
            "pct_better": _b["prod"]["pct_better"], "n_better": _b["prod"]["n_better"],
            "generated_at": _b.get("generated_at"),
        }
    except Exception as _e:
        print(f"[warn] phase_band.json 解析失败: {_e}")

# ETF
nav = load(C.BUILD_RUNS / "etf_trend" / "nav_donchian55.parquet")
if nav is not None:
    out["etf"] = perf(nav)

# 板块
nav = load(C.BUILD_RUNS / "sector_rotation" / "nav.parquet")
if nav is not None:
    out["sector"] = perf(nav)

# 组合（从三个 eval json 汇总）
for key, fn in (("combo_50_50", "etf_combo_stats.json"),
                ("combo_50_30_20", "three_strategy_combo.json"),
                ("erc", "erc_deep.json")):
    p = C.BUILD / "eval" / fn
    if not p.exists():
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    if key == "erc":
        m = (d.get("methods") or {}).get("ERC 真等风险") or {}
        out["erc"] = {"full": m.get("full"), "val": m.get("val"), "oos": m.get("oos")}
    elif key == "combo_50_50":
        rs = d.get("results") or {}
        for kk in rs:
            if "50/50" in str(kk) or "5050" in str(kk):
                out["combo_50_50"] = rs[kk]
                break
    else:
        w = d.get("weights") or {}
        for kk in w:
            if "50/30/20" in str(kk):
                out["combo_50_30_20"] = w[kk]
                break

out["config"] = {
    "model": "wf_h10", "n_holdings": 40, "rebalance_days": 40,
    "cash_bond": float(C.CASH_BOND_WEIGHT), "cash_gold": float(C.CASH_GOLD_WEIGHT),
}
out["generated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"→ {OUT}")
for k in ("main", "etf", "sector", "combo_50_50", "combo_50_30_20", "erc"):
    v = out.get(k) or {}
    f = v.get("full") if "full" in v else v
    if f:
        print(f"  {k:<16} CAGR {f.get('cagr', 0):.2%} Sharpe {f.get('sharpe', 0):.3f}")
    pb = (out.get(k) or {}).get("phase_band")
    if pb:
        print(f"  {'':<16} 相位带 CAGR 均值 {pb['cagr_mean']:.2%} / 中位 {pb['cagr_median']:.2%} "
              f"/ 区间 [{pb['cagr_min']:.2%}, {pb['cagr_max']:.2%}]"
              f"（{pb['n_phases']} 相位；生产 offset={pb['prod_offset']}，"
              f"优于它的相位 {pb['n_better']} 个）")
