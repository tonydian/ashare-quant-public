# -*- coding: utf-8 -*-
"""三策略组合边际贡献测试（方案 B）。

三个策略：
  M = 主策略（个股 wf_h10，2020-01 ~ 2026-10）
  W = 宽基 ETF 趋势（Donchian55，3 只宽基）
  S = 板块 ETF 轮动（25 个行业板块，突破+动量前3）

测试：
  1) 两两相关性矩阵；
  2) 现有 50/50（M+W）vs 加入 S 的多种权重（10%/20%/30%）；
  3) 等风险贡献（ERC）近似权重 vs 固定权重；
  4) 各方案的分区间表现（验证期 / 样本外）；
  5) 边际贡献：加入 S 后 Sharpe / Calmar / MaxDD 的改善是否显著（bootstrap）。
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
from qf.stats_adjust import circular_block_bootstrap_ci, sharpe_ratio  # noqa: E402

OUT = ROOT / "build" / "eval" / "three_strategy_combo.json"


def load_navs() -> dict[str, pd.Series]:
    navs = {}
    m = ROOT / "build/runs/main/nav.parquet"
    w = ROOT / "build/runs/etf_trend/nav_donchian55.parquet"
    s = ROOT / "build/runs/sector_rotation/nav.parquet"
    if m.exists():
        navs["M_主策略"] = pd.read_parquet(m)["nav"].astype(float)
    if w.exists():
        navs["W_宽基ETF"] = pd.read_parquet(w)["nav"].astype(float)
    if s.exists():
        navs["S_板块轮动"] = pd.read_parquet(s)["nav"].astype(float)
    for k in navs:
        navs[k].index = pd.to_datetime(navs[k].index)
        navs[k] = navs[k][~navs[k].index.duplicated()].sort_index()
    return navs


def perf(nav: pd.Series, s: str, e: str) -> dict:
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    if len(sub) < 30:
        return {}
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}


def main():
    navs = load_navs()
    if len(navs) < 3:
        raise SystemExit(f"缺少净值文件，当前只有: {list(navs)}")
    print("已加载：", list(navs))
    idx = None
    for v in navs.values():
        idx = v.index if idx is None else idx.intersection(v.index)
    print(f"三者共同区间：{idx[0].date()} ~ {idx[-1].date()}，{len(idx)} 日")

    rets = pd.DataFrame({k: v.reindex(idx).pct_change().fillna(0) for k, v in navs.items()})
    res = {}

    print("\n" + "=" * 90)
    print("1) 相关性矩阵（日收益）")
    print("=" * 90)
    corr = rets.corr()
    print(corr.round(3).to_string())
    res["corr"] = corr.round(4).to_dict()

    print("\n" + "=" * 90)
    print("2) 权重方案对比（固定权重，日频再平衡、无成本）")
    print("=" * 90)
    weights = {
        "100% 主策略（基准）": {"M_主策略": 1.0},
        "50/50 主+宽基（现行）": {"M_主策略": 0.5, "W_宽基ETF": 0.5},
        "50/30/20 主+宽基+板块": {"M_主策略": 0.5, "W_宽基ETF": 0.3, "S_板块轮动": 0.2},
        "40/30/30 主+宽基+板块": {"M_主策略": 0.4, "W_宽基ETF": 0.3, "S_板块轮动": 0.3},
        "40/40/20 主+宽基+板块": {"M_主策略": 0.4, "W_宽基ETF": 0.4, "S_板块轮动": 0.2},
        "50/40/10 主+宽基+板块": {"M_主策略": 0.5, "W_宽基ETF": 0.4, "S_板块轮动": 0.1},
        "60/20/20 主+宽基+板块": {"M_主策略": 0.6, "W_宽基ETF": 0.2, "S_板块轮动": 0.2},
        "100% 板块轮动": {"S_板块轮动": 1.0},
        "50/50 主+板块": {"M_主策略": 0.5, "S_板块轮动": 0.5},
    }
    S0, E0 = str(idx[0].date()), str(idx[-1].date())
    SEGS = {"full": (S0, E0), "val": ("2020-01-02", "2021-12-31"),
            "oos": ("2022-01-01", E0)}
    table = {}
    print(f"{'方案':<26}{'全期CAGR':>10}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
          f"{'验证期':>9}{'样本外':>9}")
    for name, w in weights.items():
        combo_ret = sum(rets[k] * wt for k, wt in w.items())
        nav = (1 + combo_ret).cumprod()
        row = {}
        for seg, (s, e) in SEGS.items():
            sub = nav[(nav.index >= s) & (nav.index <= e)]
            p = qfm.performance(sub, rf=C.CASH_YIELD)
            row[seg] = {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}
        table[name] = row
        print(f"{name:<26}{row['full']['cagr']:>9.2%}{row['full']['sharpe']:>8.3f}"
              f"{row['full']['max_dd']:>8.2%}{row['full']['calmar']:>8.2f}"
              f"{row['val']['cagr']:>8.2%}{row['oos']['cagr']:>8.2%}")
    res["weights"] = table

    print("\n" + "=" * 90)
    print("3) 边际贡献：在 50/50 基础上加 S，Sharpe/Calmar 改善是否显著")
    print("=" * 90)
    base_ret = rets["M_主策略"] * 0.5 + rets["W_宽基ETF"] * 0.5
    base_nav = (1 + base_ret).cumprod()
    base_sh = qfm.performance(base_nav, rf=C.CASH_YIELD)["sharpe"]
    diffs = {}
    base_arr = base_ret.to_numpy()
    rng = np.random.default_rng(20261024)
    n = len(base_arr)

    def _block_idx(n_boot: int, block: int) -> np.ndarray:
        nb = int(np.ceil(n / block))
        st = rng.integers(0, n, size=(n_boot, nb))
        off = np.arange(block)
        idx2 = (st[:, :, None] + off[None, None, :]) % n
        return idx2.reshape(n_boot, -1)[:, :n]

    def _sharpe_arr(x: np.ndarray) -> float:
        sd = np.std(x, ddof=1)
        return float(np.mean(x) / sd * np.sqrt(252)) if sd > 0 else float("nan")

    boot_idx = _block_idx(1500, 21)
    base_boot = np.array([_sharpe_arr(base_arr[i]) for i in boot_idx])
    for s_w in (0.1, 0.2, 0.3, 0.4):
        combo = (base_ret * (1 - s_w) + rets["S_板块轮动"] * s_w)
        nav = (1 + combo).cumprod()
        p = qfm.performance(nav, rf=C.CASH_YIELD)
        d = p["sharpe"] - base_sh
        combo_arr = combo.to_numpy()
        cb = np.array([_sharpe_arr(combo_arr[i]) for i in boot_idx])
        delta = cb - base_boot
        delta = delta[np.isfinite(delta)]
        lo, hi = (float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))) \
            if len(delta) > 100 else (float("nan"), float("nan"))
        pct_pos = float((delta > 0).mean()) if len(delta) else float("nan")
        diffs[f"S权重{s_w:.0%}"] = {
            "sharpe": float(p["sharpe"]), "delta_sharpe": float(d),
            "calmar": float(p["calmar"]), "cagr": float(p["cagr"]),
            "max_dd": float(p["max_dd"]),
            "delta_ci": [lo, hi], "prob_delta_positive": pct_pos,
        }
        print(f"  S权重 {s_w:4.0%}: Sharpe {p['sharpe']:6.3f}（Δ{d:+.3f}）| "
              f"Calmar {p['calmar']:5.2f} | CAGR {p['cagr']:7.2%} | MaxDD {p['max_dd']:7.2%} | "
              f"ΔSharpe 95%CI [{lo:+.2f}, {hi:+.2f}] | P(Δ>0)={pct_pos:.0%}")
    res["marginal"] = diffs

    print("\n" + "=" * 90)
    print("4) 风险平价（ERC）近似权重：按波动率倒数分配")
    print("=" * 90)
    vols = rets.std() * np.sqrt(252)
    inv = (1.0 / vols)
    erc_w = (inv / inv.sum()).to_dict()
    print("  波动率: " + " | ".join(f"{k}: {v:.2%}" for k, v in vols.items()))
    print("  ERC 权重: " + " | ".join(f"{k}: {v:.1%}" for k, v in erc_w.items()))
    combo_ret = sum(rets[k] * w for k, w in erc_w.items())
    nav = (1 + combo_ret).cumprod()
    erc_row = {}
    for seg, (s, e) in SEGS.items():
        sub = nav[(nav.index >= s) & (nav.index <= e)]
        p = qfm.performance(sub, rf=C.CASH_YIELD)
        erc_row[seg] = {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}
    print(f"  ERC 组合: 全期 CAGR {erc_row['full']['cagr']:.2%} Sharpe {erc_row['full']['sharpe']:.3f} "
          f"MaxDD {erc_row['full']['max_dd']:.2%} Calmar {erc_row['full']['calmar']:.2f}")
    print(f"           验证期 {erc_row['val']['cagr']:.2%} | 样本外 {erc_row['oos']['cagr']:.2%}")
    res["erc"] = {"weights": erc_w, "results": erc_row,
                  "vols": {k: float(v) for k, v in vols.items()}}

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
