# -*- coding: utf-8 -*-
"""独立审计：指标口径复算（不依赖 qf.metrics，纯 numpy 独立实现）。

对照 build/runs/main/metrics.json / yearly.csv / monthly.csv / trades.parquet：
* n_days / final_nav / total_return / CAGR / 年化波动 / Sharpe / Sortino /
  MaxDD / MaxDD 水下天数 / Calmar / 日·月胜率 / skew / kurt；
* 换手（单边/双边）、成本合计、成本占比、每笔成本 bp；
* 逐年收益、月度收益与 YTD；
* Sharpe 是否扣无风险利率（审计记录：主指标 rf=0，但现金按 1.5% 计息）。
误差要求 < 1e-6。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C                      # noqa: E402

RUN = C.BUILD_RUNS / "main"
TOL = 1e-6


def _load_artifacts():
    if not (RUN / "metrics.json").exists():
        pytest.skip(f"{RUN}/metrics.json 尚未生成（等 Lead 跑完主回测）")
    nav = pd.read_parquet(RUN / "nav.parquet")["nav"].astype(float)
    with open(RUN / "metrics.json", encoding="utf-8") as f:
        met = json.load(f)
    trades = pd.read_parquet(RUN / "trades.parquet") if (RUN / "trades.parquet").exists() else None
    yearly = pd.read_csv(RUN / "yearly.csv") if (RUN / "yearly.csv").exists() else None
    monthly = pd.read_csv(RUN / "monthly.csv") if (RUN / "monthly.csv").exists() else None
    return nav, met, trades, yearly, monthly


@pytest.fixture(scope="module")
def arts():
    return _load_artifacts()


# ---------------------------------------------------------------------------
# 独立指标实现（与 qf.metrics 无关）
# ---------------------------------------------------------------------------
def _perf(nav: pd.Series, rf: float = 0.0) -> dict:
    n = len(nav)
    nav0, nav1 = float(nav.iloc[0]), float(nav.iloc[-1])
    years = (n - 1) / 252.0
    cagr = (nav1 / nav0) ** (1.0 / years) - 1.0
    r = nav.pct_change().dropna().to_numpy(dtype=np.float64)
    rf_d = float(rf) / 252.0
    excess = r - rf_d
    sd = float(np.std(r, ddof=1))
    mu = float(np.mean(excess))
    dd = (nav / nav.cummax() - 1.0).to_numpy(dtype=np.float64)
    max_dd = float(dd.min())
    # 最长水下天数
    under = dd < 0
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    down = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    return {
        "n_days": int(n),
        "final_nav": nav1,
        "total_return": nav1 / nav0 - 1.0,
        "cagr": float(cagr),
        "ann_vol": sd * np.sqrt(252.0),
        "sharpe": mu / sd * np.sqrt(252.0),
        "sortino": mu / down * np.sqrt(252.0) if down > 0 else np.nan,
        "max_dd": max_dd,
        "max_dd_days": int(best),
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "win_rate_daily": float((r > 0).mean()),
        "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurt()),
        "years": years,
        "mean_nav": float(nav.mean()),
    }


def _close(a, b, name: str, tol: float = TOL, rel: float = 0.0):
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        assert np.isnan(a) and np.isnan(b), f"{name}: {a} vs {b}"
        return abs(a - b) if np.isfinite(a) else 0.0
    diff = abs(a - b)
    assert diff <= tol + rel * abs(b), f"{name} 不一致：独立={a!r} 系统={b!r} diff={diff:.3e}"
    return diff


# ---------------------------------------------------------------------------
# 1. 主指标逐项复算
# ---------------------------------------------------------------------------
def test_metrics_recompute(arts):
    nav, met, *_ = arts
    # Lead 已统一 rf=C.CASH_YIELD（1.5%）作为无风险利率（verifier P2#6）
    p = _perf(nav, rf=C.CASH_YIELD)
    checks = {}
    for k in ["n_days", "final_nav", "total_return", "cagr", "ann_vol", "sharpe",
              "sortino", "max_dd", "max_dd_days", "calmar", "win_rate_daily",
              "skew", "kurt"]:
        checks[k] = _close(p[k], met[k], k)
    print("[audit] 指标复算最大偏差：", {k: f"{v:.2e}" for k, v in checks.items()})
    assert max(checks.values()) < TOL


def test_sharpe_riskfree_convention(arts):
    """口径记录：主指标 Sharpe 现在扣除无风险利率 rf=1.5%（与基准口径一致）。
    同时给出 rf=0 的 Sharpe 以便报告对照，两者差 ≈ rf/年化波动。"""
    nav, met, *_ = arts
    p0 = _perf(nav, rf=0.0)
    p15 = _perf(nav, rf=C.CASH_YIELD)
    assert abs(met["sharpe"] - p15["sharpe"]) < TOL, (
        f"metrics.sharpe={met['sharpe']} 与 rf=1.5% 独立复算 {p15['sharpe']} 不一致")
    print(f"[audit] Sharpe(rf={C.CASH_YIELD:.1%})={p15['sharpe']:.4f}  "
          f"Sharpe(rf=0)={p0['sharpe']:.4f}  差={p0['sharpe'] - p15['sharpe']:.4f}")
    assert p0["sharpe"] >= p15["sharpe"]


# ---------------------------------------------------------------------------
# 2. 换手 / 成本口径
# ---------------------------------------------------------------------------
def test_turnover_and_cost_recompute(arts):
    nav, met, trades, *_ = arts
    assert trades is not None, "缺 trades.parquet"
    years = (len(nav) - 1) / 252.0
    avg_nav = float(nav.mean())
    buy = float(trades.loc[trades["side"] == "buy", "amount"].sum())
    sell = float(trades.loc[trades["side"] == "sell", "amount"].sum())
    cost = float((trades["commission"] + trades["stamp"]
                  + trades["transfer"] + trades["slip"]).sum())
    exp = {
        "cost_total": cost,
        "turnover_ann_single": (buy + sell) / 2.0 / avg_nav / years,
        "turnover_ann_double": (buy + sell) / avg_nav / years,
        "cost_ratio": cost / avg_nav / years,
        "cost_bp_per_trade": cost / (buy + sell) * 1e4,
        "n_buy_trades": int((trades["side"] == "buy").sum()),
        "n_sell_trades": int((trades["side"] == "sell").sum()),
        "trade_amount_buy": buy,
        "trade_amount_sell": sell,
        "years": years,
    }
    exp["turnover_ann"] = exp["turnover_ann_single"]
    dev = {}
    for k, v in exp.items():
        if k in ("n_buy_trades", "n_sell_trades"):
            assert int(met[k]) == int(v), f"{k}: {met[k]} vs {v}"
            dev[k] = 0.0
        else:
            dev[k] = _close(v, met[k], k)
    print("[audit] 换手/成本复算最大偏差：", {k: f"{v:.2e}" for k, v in dev.items()})
    assert max(dev.values()) < TOL


# ---------------------------------------------------------------------------
# 3. 逐年 / 月度收益
# ---------------------------------------------------------------------------
def test_yearly_recompute(arts):
    nav, met, trades, yearly, monthly = arts
    if yearly is None:
        pytest.skip("缺 yearly.csv")
    y = yearly.copy()
    # 第一列可能是 index（年份）
    col = "year" if "year" in y.columns else y.columns[0]
    y = y.set_index(col)["return"].astype(float)
    years = nav.index.year
    exp = {}
    prev = None
    for yr in sorted(set(years)):
        seg = nav[years == yr]
        base = float(nav.iloc[0]) if prev is None else prev
        exp[int(yr)] = float(seg.iloc[-1]) / base - 1.0
        prev = float(seg.iloc[-1])
    assert set(y.index.astype(int)) == set(exp), (y.index.tolist(), sorted(exp))
    diffs = {yr: abs(y.loc[yr] - exp[yr]) for yr in exp}
    print("[audit] 逐年收益最大偏差：", max(diffs.values()))
    assert max(diffs.values()) < TOL


def test_monthly_recompute(arts):
    nav, met, trades, yearly, monthly = arts
    if monthly is None:
        pytest.skip("缺 monthly.csv")
    m = monthly.copy()
    col = "year" if "year" in m.columns else m.columns[0]
    m = m.set_index(col)
    # 月列可能是 str 或 int，建立 int→原列名映射
    mcol = {}
    for c in m.columns:
        try:
            mcol[int(str(c))] = c
        except (TypeError, ValueError):
            pass
    # 独立按月复算
    s = nav
    me = s.resample("ME").last()
    ret = me.pct_change()
    ret.iloc[0] = me.iloc[0] / s.iloc[0] - 1.0
    maxdiff = 0.0
    for ts, r in ret.items():
        if ts.year not in m.index or ts.month not in mcol:
            continue
        got = float(m.loc[ts.year, mcol[ts.month]])
        if np.isfinite(got):
            assert abs(got - float(r)) < TOL, f"{ts} 月收益不一致 {got} vs {r}"
            maxdiff = max(maxdiff, abs(got - float(r)))
    # YTD = 月度复合
    for yr in m.index:
        vals = np.array([float(m.loc[yr, mcol[c]]) for c in range(1, 13) if c in mcol],
                        dtype=float)
        vals = vals[np.isfinite(vals)]
        ytd = float(np.prod(1.0 + vals) - 1.0)
        assert abs(float(m.loc[yr, "YTD"]) - ytd) < 1e-9, f"{yr} YTD 不一致"
    print("[audit] 月度收益最大偏差：", maxdiff)


# ---------------------------------------------------------------------------
# 4. 净值连续性：nav 与 daily_ret 一致
# ---------------------------------------------------------------------------
def test_nav_daily_ret_consistency(arts):
    nav, met, *_ = arts
    fp = RUN / "daily_ret.parquet"
    if not fp.exists():
        pytest.skip("缺 daily_ret.parquet")
    dr = pd.read_parquet(fp)["ret"].astype(float)
    assert dr.index.equals(nav.index)
    exp = nav.pct_change()
    exp.iloc[0] = nav.iloc[0] / C.CAPITAL - 1.0
    exp = exp.fillna(0.0)
    d = np.abs(dr.to_numpy() - exp.to_numpy()).max()
    assert d < 1e-12, f"daily_ret 与 nav 不一致：{d:.2e}"
