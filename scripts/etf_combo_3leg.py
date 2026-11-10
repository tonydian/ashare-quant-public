# -*- coding: utf-8 -*-
"""三腿组合：主策略 + ETF 趋势 + 512890 红利低波（2026-11-03）。

背景
----
`reports/低波动族实测与策略方向建议_2026-11-03.md` 的 P0 结论：自建低波 2.41%，
现成 512890 14.16%（后复权含分红），差 11.75pp 且维护为零。用户拍板把它加进组合。

预先设定的规则（不寻优）：
  * 主权重保持 50%（主策略是唯一有研究投入的腿，不让 ETF 配比喧宾夺主）；
  * 两个 ETF 腿默认均分剩余 50%（25/25）为主方案；
  * 敏感性只看两个预设变体（40/30/30、50/30/20），不做网格；
  * 每季度末把漂移的权重拉回目标，成本 Σ|Δw| × 5bp（卖出+买入双边，比旧脚本
    的单边计法保守一倍）；|Δw| ≤ 2pp 不动；
  * 512890 用后复权序列（build/etf/div_panel.parquet，含分红再投资），纯持有。

采纳闸门（预先声明，先于看结果）：
  相对现 50/50：全期 CAGR 不降 且 MaxDD 改善 且 Sharpe 不降 → 三者全过才采纳。

输出：build/eval/combo3_stats.json、build/runs/combo_3leg/nav_502525.parquet
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qf import config as C, metrics as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "combo3_stats.json"
RUN = ROOT / "build" / "runs" / "combo_3leg"
REBAL_BP = 0.0005          # 单边 5bp；双边计 Σ|Δw|×5bp
BAND = 0.02                # 季度再平衡死区

VARIANTS = {               # (主, ETF趋势, 红利低波)
    "50/25/25": (0.50, 0.25, 0.25),
    "40/30/30": (0.40, 0.30, 0.30),
    "50/30/20": (0.50, 0.30, 0.20),
}


def perf(nav: pd.Series, s: str, e: str) -> dict:
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_dd",
                                     "calmar", "win_rate_monthly")}


def run_combo(nav_legs: dict[str, pd.Series], target: tuple[float, ...]) -> pd.Series:
    """季度再平衡固定权重组合（Σ|Δw|×5bp 双边成本）。

    递推：日组合收益 = Σ w_j·r_j（w 为 drifted 权重），随后 w 按相对表现漂移；
    季末 |w−目标| 超 2pp 时拉回目标并按 Σ|Δw|×5bp 计双边成本。
    """
    names = list(nav_legs)
    tgt = np.asarray(target, dtype=float)
    idx = sorted(set.intersection(*[set(v.index) for v in nav_legs.values()]))
    idx = pd.DatetimeIndex(idx)
    rets = pd.DataFrame({n: nav_legs[n].reindex(idx).pct_change().fillna(0.0)
                         for n in names})
    qends = pd.Series(idx, index=idx).groupby([idx.year, idx.quarter]).last().values
    is_qend = pd.Series(False, index=idx)
    is_qend.loc[pd.DatetimeIndex(qends)] = True

    w = tgt.copy()
    combo = np.ones(len(idx))
    cost_tot = 0.0
    n_rebal = 0
    for i in range(len(idx)):
        r_i = rets.iloc[i].to_numpy()
        rp = float(np.dot(w, r_i))
        if i > 0:
            combo[i] = combo[i - 1] * (1.0 + rp)
        # 权重漂移
        if rp > -0.9999:
            w = w * (1.0 + r_i) / (1.0 + rp)
        if is_qend.iloc[i]:
            delta = tgt - w
            if np.abs(delta).max() > BAND:
                cost = float(np.abs(delta).sum()) * REBAL_BP
                combo[i] *= (1.0 - cost)
                cost_tot += cost
                n_rebal += 1
                w = tgt.copy()
    return pd.Series(combo, index=idx, name="nav"), cost_tot, n_rebal


def main():
    main_nav = pd.read_parquet(ROOT / "build/runs/main/nav.parquet")["nav"].astype(float)
    main_nav.index = pd.to_datetime(main_nav.index)
    etf_nav = pd.read_parquet(ROOT / "build/runs/etf_trend/nav_donchian55.parquet")["nav"].astype(float)
    etf_nav.index = pd.to_datetime(etf_nav.index)

    div = pd.read_parquet(ROOT / "build/etf/div_panel.parquet")
    s = div[div.code == "512890"].set_index(pd.to_datetime(div[div.code == "512890"].date))["close_adj"].sort_index()
    div_nav = s / s.iloc[0]

    legs = {"主策略": main_nav, "ETF趋势": etf_nav, "红利低波512890": div_nav}
    idx = sorted(set.intersection(*[set(v.index) for v in legs.values()]))
    idx = pd.DatetimeIndex(idx)
    print(f"三腿对齐区间 {idx[0].date()} ~ {idx[-1].date()}，{len(idx)} 日")

    rel = pd.DataFrame({n: legs[n].reindex(idx) for n in legs})
    rel = rel / rel.iloc[0]
    rets = rel.pct_change().dropna()
    print("\n腿间日收益相关：")
    print(rets.corr().round(3).to_string())

    results: dict[str, dict] = {}
    navs: dict[str, pd.Series] = {}
    for name, tgt in VARIANTS.items():
        combo, cost, nrb = run_combo(legs, tgt)
        navs[name] = combo
        results[name] = {"weights": tgt, "rebal_cost_pct": cost * 100, "n_rebal": nrb,
                         "full": perf(combo, str(idx[0].date()), str(idx[-1].date())),
                         "val": perf(combo, "2020-01-02", "2021-12-31"),
                         "oos": perf(combo, "2022-01-01", str(idx[-1].date()))}
    combo2, cost2, nrb2 = run_combo({"主策略": main_nav, "ETF趋势": etf_nav}, (0.5, 0.5))
    results["50/50(现状)"] = {"weights": (0.5, 0.5), "rebal_cost_pct": cost2 * 100,
                              "n_rebal": nrb2,
                              "full": perf(combo2, str(idx[0].date()), str(idx[-1].date())),
                              "val": perf(combo2, "2020-01-02", "2021-12-31"),
                              "oos": perf(combo2, "2022-01-01", str(idx[-1].date()))}

    print(f"\n{'方案':<16}{'全期CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}"
          f"{'验证期':>9}{'样本外':>9}{'成本%':>7}")
    for name, r in results.items():
        f = r["full"]
        print(f"{name:<15}{f['cagr']*100:>8.2f}%{f['sharpe']:>8.3f}{f['max_dd']*100:>7.2f}%"
              f"{f['calmar']:>8.2f}{r['val']['cagr']*100:>8.2f}%{r['oos']['cagr']*100:>8.2f}%"
              f"{r['rebal_cost_pct']:>7.2f}")

    # 采纳闸门：相对 50/50 现状
    b = results["50/50(现状)"]["full"]
    print("\n采纳闸门（三腿 50/25/25 vs 现状 50/50，三者全过才采纳）：")
    p = results["50/25/25"]["full"]
    for k, ok in [("cagr 不降", p["cagr"] >= b["cagr"]),
                  ("MaxDD 改善", p["max_dd"] > b["max_dd"]),
                  ("Sharpe 不降", p["sharpe"] >= b["sharpe"])]:
        print(f"  {k:<10} {'✅' if ok else '❌'}")

    RUN.mkdir(parents=True, exist_ok=True)
    navs["50/25/25"].to_frame("nav").to_parquet(RUN / "nav_502525.parquet")
    OUT.write_text(json.dumps({"legs_corr": rets.corr().to_dict(),
                               "results": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
