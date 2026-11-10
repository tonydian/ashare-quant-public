# -*- coding: utf-8 -*-
"""四腿组合生产口径：主策略 + ETF 趋势 + 512890 红利低波 + 可转债（2026-11-04）。

背景
----
`reports/第四条腿可转债_2026-11-04.md`：转债腿 ρ=0.063、净 SR 1.296，接入后
20% 档 SR 0.782→1.041、波动 8.60%→7.32%、MaxDD −8.05%→−7.73%，8/8 相位、
3/3 分段全胜。但 G1（DSR@71 = 63.4%）未过 → 按预登记规则「DSR 不过而其余三条
全过 → 最高按 15% 降权试行」，生产口径取 **转债 15%**。

权重规则（不寻优）
------------------
转债固定 w_cb，剩余 (1−w_cb) 按原三腿 50/25/25 的**比例**分配：
    w_cb=15% → 主策略 42.5% / ETF 趋势 21.25% / 红利低波 21.25%
只报 5 个预设档（0/10/15/20/25%），不做网格搜索。

与 `scripts/etf_combo_3leg.py` 的差异
------------------------------------
1. **多一条腿**：转债，日收益按报告口径扣 2.52pp/年（17.97% 毛 → 15.45% 净，
   15bp 滑点 + 万0.5 佣金）。`build/runs/cb/nav_prod.parquet` 本身是**毛**序列，
   直接拼接会重犯报告 §0.1 自纠过的「毛口径当净口径」错误。
2. **区间取四腿交集** 2020-01-08 ~ 2026-10-29（转债数据止于 10-29），比三腿短
   3 个交易日；为公平对比，基线（w_cb=0）在同一区间上重算。
3. 再平衡规则与成本口径**完全沿用**三腿：季度末、|Δw| ≤ 2pp 不动、Σ|Δw|×5bp 双边。
4. **主策略腿按缩小后的真实资金重跑**（`--capital-aware`，默认开）。理由：回测
   `CAPITAL=300,000` 是按全资金跑的，四腿 15% 时主策略只分到 42.5%×30万 = 12.75 万，
   单只持仓从 4,794 元掉到 1,937 元，`trade_band_abs=6,000` 会跳过大量等权微调。
   实测（`build/verify_combo4_capital.py`）：主策略单腿 CAGR 8.38% → 7.54%（−0.84pp）。
   线性拼接会把这 0.84pp × 42.5% ≈ **0.36pp** 的损失漏掉。
   ETF / 红利 / 转债三条腿规模不敏感（整手金额小），仍按净值线性缩放。

输出
----
`build/eval/combo4_stats.json`、`build/runs/combo_4leg/nav_cb{0,10,15,20,25}.parquet`
（每一档都落 NAV —— P6 教训：不落 NAV 就永远只能假设试验间相关性）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qf import backtest as qfbt   # noqa: E402
from qf import config as C        # noqa: E402
from qf import data as qfdata     # noqa: E402
from qf import metrics as qfm     # noqa: E402
from qf import model as qfm_model  # noqa: E402

OUT = ROOT / "build" / "eval" / "combo4_stats.json"
RUN = ROOT / "build" / "runs" / "combo_4leg"
REBAL_BP = 0.0005          # 单边 5bp；双边计 Σ|Δw|×5bp
BAND = 0.02                # 季度再平衡死区
CB_COST_ANN = 0.1797 - 0.1545   # 可转债毛→净，报告口径

PROD_W_CB = 0.15           # 生产口径（DSR 未过 → 降权试行）
W_CB_LIST = (0.00, 0.10, 0.15, 0.20, 0.25)
TRIPOD = (0.50, 0.25, 0.25)     # 原三腿比例，剩余额度按此比例分


def perf(nav: pd.Series, s: str, e: str) -> dict:
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_dd",
                                     "calmar", "win_rate_monthly")}


def run_combo(nav_legs: dict[str, pd.Series], target: tuple[float, ...]
              ) -> tuple[pd.Series, float, int]:
    """季度再平衡固定权重组合（Σ|Δw|×5bp 双边成本，|Δw|≤2pp 不动）。

    与 etf_combo_3leg.run_combo 逻辑逐行一致，只是腿数不同。
    """
    names = list(nav_legs)
    tgt = np.asarray(target, dtype=float)
    idx = pd.DatetimeIndex(sorted(set.intersection(*[set(v.index) for v in nav_legs.values()])))
    rets = pd.DataFrame({n: nav_legs[n].reindex(idx).pct_change().fillna(0.0)
                         for n in names})
    qends = pd.Series(idx, index=idx).groupby([idx.year, idx.quarter]).last().values
    is_qend = pd.Series(False, index=idx)
    is_qend.loc[pd.DatetimeIndex(qends)] = True

    w = tgt.copy()
    combo = np.ones(len(idx))
    cost_tot, n_rebal = 0.0, 0
    for i in range(len(idx)):
        r_i = rets.iloc[i].to_numpy()
        rp = float(np.dot(w, r_i))
        if i > 0:
            combo[i] = combo[i - 1] * (1.0 + rp)
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


def weights_for(w_cb: float) -> tuple[float, float, float, float]:
    """转债 w_cb，其余 (1−w_cb) 按 50/25/25 比例分。"""
    rest = 1.0 - w_cb
    return tuple(rest * x for x in TRIPOD) + (w_cb,)


def load_legs() -> dict[str, pd.Series]:
    main_nav = pd.read_parquet(ROOT / "build/runs/main/nav.parquet")["nav"].astype(float)
    main_nav.index = pd.to_datetime(main_nav.index)

    etf_nav = pd.read_parquet(
        ROOT / "build/runs/etf_trend/nav_donchian55.parquet")["nav"].astype(float)
    etf_nav.index = pd.to_datetime(etf_nav.index)

    div = pd.read_parquet(ROOT / "build/etf/div_panel.parquet")
    d = div[div.code == "512890"]
    s = d.set_index(pd.to_datetime(d.date))["close_adj"].sort_index()
    div_nav = s / s.iloc[0]

    raw = pd.read_parquet(ROOT / "build/runs/cb/nav_prod.parquet")["nav"].astype(float)
    raw.index = pd.to_datetime(raw.index)
    r = raw.pct_change().fillna(0.0) - CB_COST_ANN / 252.0
    cb_nav = (1.0 + r).cumprod()

    return {"主策略": main_nav, "ETF趋势": etf_nav,
            "红利低波512890": div_nav, "可转债": cb_nav}


def load_main_legs_by_capital(capitals: list[float]) -> dict[float, pd.Series]:
    """按给定本金分别重跑主策略（生产相位 offset=10），返回 {本金: nav}。"""
    res = qfm_model.load_result(C.BUILD_MODELS / "wf_h10")
    md = qfdata.load_market(verbose=False)
    scores, rows_w = res.scores, res.rows_weekly
    valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
    start_row = max(md.date_index(C.VALID_START), int(valid[0]))
    phase = int(rows_w[0]) % 5 if len(rows_w) else 0
    while start_row % 5 != phase and start_row < md.T - 1:
        start_row += 1
    end_row = md.T - 1
    prod_offset = (phase - start_row) % 40
    cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)

    old = C.CAPITAL
    out: dict[float, pd.Series] = {}
    try:
        for cap in capitals:
            C.CAPITAL = float(cap)
            r = qfbt.run_backtest(
                md, scores, start_row, end_row, risk_overlay=True, stop_loss=True,
                rebalance_days=40, rebalance_offset=prod_offset, trade_band_rel=0.30,
                trade_band_abs=6000.0, max_new_entries=6, cash_asset_returns=cash,
                verbose=False)
            nav = np.asarray(r.nav, dtype=float)
            dates = pd.DatetimeIndex(md.dates[start_row:end_row + 1])[:len(nav)]
            out[float(cap)] = pd.Series(nav / nav[0], index=dates, name="nav")
            print(f"    主策略 @ 本金 {cap:>9,.0f} 元 → 终值 {nav[-1] / nav[0]:.4f}", flush=True)
    finally:
        C.CAPITAL = old
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital-aware", dest="capital_aware", action="store_true",
                    default=True, help="主策略腿按缩小资金重跑（默认开，更真实）")
    ap.add_argument("--no-capital-aware", dest="capital_aware", action="store_false",
                    help="退回线性拼接（与旧口径可比，但会漏掉规模损失）")
    args = ap.parse_args()

    legs = load_legs()
    idx = pd.DatetimeIndex(sorted(set.intersection(*[set(v.index) for v in legs.values()])))
    print(f"四腿对齐区间 {idx[0].date()} ~ {idx[-1].date()}，{len(idx)} 日")
    print(f"转债按报告口径扣成本 {CB_COST_ANN:.2%}/年（毛→净）")

    rel = pd.DataFrame({n: legs[n].reindex(idx) for n in legs})
    rel = rel / rel.iloc[0]
    corr = rel.pct_change().dropna().corr()
    print("\n腿间日收益相关：")
    print(corr.round(3).to_string())

    main_by_cap: dict[float, pd.Series] = {}
    if args.capital_aware:
        caps = sorted({round(C.CAPITAL * (1.0 - w) * TRIPOD[0], 2) for w in W_CB_LIST})
        print(f"\n按缩小资金重跑主策略腿（{len(caps)} 档："
              f"{', '.join(f'{c:,.0f}' for c in caps)} 元）...", flush=True)
        main_by_cap = load_main_legs_by_capital(caps)

    results: dict[str, dict] = {}
    navs: dict[str, pd.Series] = {}
    for w in W_CB_LIST:
        tgt = weights_for(w)
        use = dict(legs)
        if args.capital_aware:
            use["主策略"] = main_by_cap[round(C.CAPITAL * (1.0 - w) * TRIPOD[0], 2)]
        nav, cost, nrb = run_combo(use, tgt)
        key = f"cb{int(round(w * 100)):02d}"
        navs[key] = nav
        results[key] = {
            "w_cb": w, "weights": tgt, "label": f"转债{w:.0%}",
            "rebal_cost_pct": cost * 100, "n_rebal": nrb,
            "full": perf(nav, str(idx[0].date()), str(idx[-1].date())),
            "val": perf(nav, "2020-01-02", "2021-12-31"),
            "oos": perf(nav, "2022-01-01", str(idx[-1].date())),
        }

    base = results["cb00"]["full"]
    print(f"\n{'方案':<10}{'主策略':>8}{'ETF':>8}{'红利':>8}{'转债':>7}"
          f"{'全期CAGR':>10}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}"
          f"{'验证期':>9}{'样本外':>9}{'ΔSR':>8}{'成本%':>7}")
    for k, r in results.items():
        f, w = r["full"], r["weights"]
        print(f"{r['label']:<9}{w[0]:>8.1%}{w[1]:>8.1%}{w[2]:>8.1%}{w[3]:>7.0%}"
              f"{f['cagr'] * 100:>9.2f}%{f['sharpe']:>8.3f}{f['max_dd'] * 100:>7.2f}%"
              f"{f['calmar']:>8.2f}{r['val']['cagr'] * 100:>8.2f}%"
              f"{r['oos']['cagr'] * 100:>8.2f}%"
              f"{f['sharpe'] - base['sharpe']:>+8.3f}{r['rebal_cost_pct']:>7.2f}")

    prod = results[f"cb{int(round(PROD_W_CB * 100)):02d}"]
    print(f"\n生产口径 = 转债 {PROD_W_CB:.0%}：主策略 {prod['weights'][0]:.2%} / "
          f"ETF {prod['weights'][1]:.2%} / 红利 {prod['weights'][2]:.2%} / "
          f"转债 {prod['weights'][3]:.2%}")
    print(f"  vs 三腿基线：CAGR {base['cagr'] * 100:.2f}% → {prod['full']['cagr'] * 100:.2f}%，"
          f"Sharpe {base['sharpe']:.3f} → {prod['full']['sharpe']:.3f}，"
          f"MaxDD {base['max_dd'] * 100:.2f}% → {prod['full']['max_dd'] * 100:.2f}%，"
          f"波动 {base['ann_vol'] * 100:.2f}% → {prod['full']['ann_vol'] * 100:.2f}%")

    RUN.mkdir(parents=True, exist_ok=True)
    for k, nav in navs.items():
        nav.to_frame("nav").to_parquet(RUN / f"nav_{k}.parquet")

    OUT.write_text(json.dumps({
        "prod_w_cb": PROD_W_CB,
        "capital_aware": bool(args.capital_aware),
        "capital": float(C.CAPITAL),
        "cb_cost_ann": CB_COST_ANN,
        "window": {"start": str(idx[0].date()), "end": str(idx[-1].date()), "n": len(idx)},
        "legs_corr": corr.to_dict(),
        "results": results,
        "notes": [
            "转债权重 15% 是降权试行：DSR@71 = 63.4% 未过 95%，按预登记规则取上限的一半档。",
            "可转债 nav_prod 为毛序列，本脚本按报告口径扣 2.52pp/年转净；"
            "转债腿成本未做换手重估，与真实成交差异见 reports/可转债前视审计与成本重估_2026-11-01.md。",
            "最大风险：当前转债极贵（YTM 中位 −3.04%、价格中位 125.85），"
            "历史同类环境后一年收益中位仅 6.46%。回测的 8.18%→9.26% 在坏环境下大概率达不到。",
            "区间止于 2026-10-29（转债数据末端），比三腿少 3 个交易日。",
        ],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")
    print(f"→ {RUN}/nav_cb{{00,10,15,20,25}}.parquet")


if __name__ == "__main__":
    main()
