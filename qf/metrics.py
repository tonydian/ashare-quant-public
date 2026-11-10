# -*- coding: utf-8 -*-
"""回测绩效指标（净值序列 → 指标字典 / 年·月收益 / 回撤 / 滚动 / 压力期）。

口径（docs/INTERFACES.md §3.4、§5）：

* 年化收益 CAGR = ``(nav_end/nav_start)^(252/交易日数) - 1``；
* 年化波动 = 日收益标准差 × √252（样本标准差 ddof=1）；
* Sharpe = 日超额收益均值 / 日收益标准差 × √252；
* Sortino = 日超额收益均值 / 下行标准差 × √252（下行标准差对 0 取 max）；
* 最大回撤用日净值；``max_dd_days`` = 最长"水下"持续交易日数（净值低于前高）；
* Calmar = CAGR / |MaxDD|；
* 基准对比：beta = Cov(r, b)/Var(b)，alpha 年化 = (mean(r) - rf_d - beta×(mean(b) - rf_d))×252，
  跟踪误差 = std(r - b)×√252，信息比率 = mean(r - b)/std(r - b)×√252。

所有函数接受 ``pd.Series``（index=日期）或等长数组；bench_returns 为
``pd.DataFrame``（index=日期，columns=基准名），来自 ``md.bench``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SQRT_252 = float(np.sqrt(TRADING_DAYS))

__all__ = [
    "performance", "yearly_returns", "monthly_returns_table",
    "drawdown_series", "rolling_metrics", "stress_periods",
]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _as_series(nav) -> pd.Series:
    if isinstance(nav, pd.Series):
        s = nav.astype(float).copy()
    else:
        s = pd.Series(np.asarray(nav, dtype=float))
    return s[np.isfinite(s.to_numpy(dtype=float))]


def _max_underwater_days(dd: pd.Series) -> int:
    """最长水下（净值低于前高）连续交易日数。"""
    under = (dd < 0).to_numpy()
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        if cur > best:
            best = cur
    return int(best)


def _monthly_returns(s: pd.Series) -> pd.Series:
    """月度收益（首月用序列起点，末月为部分月）。"""
    if len(s) < 2:
        return pd.Series(dtype=float)
    m = s.resample("ME").last()
    m = m[np.isfinite(m.to_numpy(dtype=float))]
    if len(m) == 0:
        return pd.Series(dtype=float)
    ret = m.pct_change()
    ret.iloc[0] = m.iloc[0] / s.iloc[0] - 1.0
    return ret


# ---------------------------------------------------------------------------
# 主指标
# ---------------------------------------------------------------------------
def performance(nav, bench_returns=None, rf: float = 0.0) -> dict:
    """净值表现指标字典（含 vs 各基准的 beta/alpha/跟踪误差/信息比率）。"""
    s = _as_series(nav)
    out: dict = {}
    if len(s) < 2:
        return {"n_days": int(len(s)), "cagr": np.nan, "ann_vol": np.nan,
                "sharpe": np.nan, "sortino": np.nan, "max_dd": np.nan,
                "max_dd_days": 0, "calmar": np.nan, "win_rate_daily": np.nan,
                "win_rate_monthly": np.nan, "skew": np.nan, "kurt": np.nan,
                "total_return": np.nan, "final_nav": float(s.iloc[-1]) if len(s) else np.nan}

    n = len(s)
    total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
    years = (n - 1) / TRADING_DAYS
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan

    ret = s.pct_change().dropna()
    r = ret.to_numpy(dtype=float)
    rf_d = float(rf) / TRADING_DAYS
    excess = r - rf_d
    mu = float(np.mean(excess)) if len(r) else np.nan
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    ann_vol = sd * SQRT_252 if np.isfinite(sd) else np.nan
    sharpe = mu / sd * SQRT_252 if (np.isfinite(sd) and sd > 0) else np.nan
    down = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))) if len(r) else np.nan
    sortino = mu / down * SQRT_252 if (np.isfinite(down) and down > 0) else np.nan

    dd = s / s.cummax() - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if (np.isfinite(cagr) and max_dd < 0) else np.nan
    trough = dd.idxmin() if len(dd) else None
    peak_idx = s.loc[:trough].idxmax() if trough is not None else None
    # 回撤修复天数（从谷底回到前高；未修复为 NaN）
    recovery_days = np.nan
    if trough is not None:
        after = s.loc[trough:]
        peak_val = s.loc[peak_idx]
        rec = after[after >= peak_val]
        if len(rec):
            recovery_days = int((rec.index[0] - trough).days)

    monthly = _monthly_returns(s)
    out.update({
        "n_days": int(n),
        "start": str(s.index[0].date()) if isinstance(s.index, pd.DatetimeIndex) else str(s.index[0]),
        "end": str(s.index[-1].date()) if isinstance(s.index, pd.DatetimeIndex) else str(s.index[-1]),
        "final_nav": float(s.iloc[-1]),
        "total_return": total_ret,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "max_dd_days": _max_underwater_days(dd),
        "max_dd_recovery_days": recovery_days,
        "calmar": calmar,
        "win_rate_daily": float((ret > 0).mean()) if len(ret) else np.nan,
        "win_rate_monthly": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "skew": float(ret.skew()) if len(ret) > 2 else np.nan,
        "kurt": float(ret.kurt()) if len(ret) > 2 else np.nan,
        "best_day": float(ret.max()) if len(ret) else np.nan,
        "worst_day": float(ret.min()) if len(ret) else np.nan,
    })
    if trough is not None:
        out["max_dd_trough_date"] = str(trough.date()) if isinstance(trough, pd.Timestamp) else str(trough)
        out["max_dd_peak_date"] = (str(peak_idx.date())
                                   if isinstance(peak_idx, pd.Timestamp) else str(peak_idx))

    # ------------------------------------------------------------ 基准对比
    if bench_returns is not None and len(ret):
        if isinstance(bench_returns, pd.Series):
            bench = bench_returns.to_frame()
        else:
            bench = pd.DataFrame(bench_returns)
        bench = bench.reindex(s.index).ffill()
        for col in bench.columns:
            b_all = bench[col].astype(float)
            df = pd.concat([ret.rename("r"), b_all.rename("b")], axis=1,
                           sort=False).dropna()
            if len(df) < 20:
                continue
            rr = df["r"].to_numpy(dtype=float)
            bb = df["b"].to_numpy(dtype=float)
            var_b = float(np.var(bb, ddof=1))
            beta = float(np.cov(rr, bb, ddof=1)[0, 1] / var_b) if var_b > 0 else np.nan
            alpha = float((np.mean(rr) - rf_d - beta * (np.mean(bb) - rf_d)) * TRADING_DAYS) \
                if np.isfinite(beta) else np.nan
            diff = rr - bb
            sd_diff = float(np.std(diff, ddof=1))
            te = sd_diff * SQRT_252
            ir = float(np.mean(diff) / sd_diff * SQRT_252) if sd_diff > 0 else np.nan
            corr = float(np.corrcoef(rr, bb)[0, 1]) if np.std(bb) > 0 else np.nan
            b_total = float(np.prod(1.0 + bb) - 1.0)
            b_years = len(bb) / TRADING_DAYS
            b_cagr = float((1.0 + b_total) ** (1.0 / b_years) - 1.0) if b_years > 0 and b_total > -1 else np.nan
            out[f"beta_{col}"] = beta
            out[f"alpha_{col}"] = alpha
            out[f"te_{col}"] = te
            out[f"ir_{col}"] = ir
            out[f"corr_{col}"] = corr
            out[f"bench_total_ret_{col}"] = b_total
            out[f"bench_cagr_{col}"] = b_cagr
            out[f"excess_cagr_{col}"] = (cagr - b_cagr) if np.isfinite(b_cagr) else np.nan
    return out


# ---------------------------------------------------------------------------
# 年 / 月
# ---------------------------------------------------------------------------
def yearly_returns(nav) -> pd.Series:
    """逐年收益率（index=年份 int；首年以序列起点为基期，末年为部分年）。"""
    s = _as_series(nav)
    if len(s) < 2:
        return pd.Series(dtype=float)
    years = np.asarray([d.year for d in s.index])
    out: dict[int, float] = {}
    prev: float | None = None
    for y in np.unique(years):
        vals = s[years == y]
        last = float(vals.iloc[-1])
        base = float(s.iloc[0]) if prev is None else prev
        out[int(y)] = last / base - 1.0
        prev = last
    return pd.Series(out, dtype=float, name="return")


def monthly_returns_table(nav) -> pd.DataFrame:
    """月度收益表：index=年份，columns=1..12 + 'YTD'。"""
    s = _as_series(nav)
    if len(s) < 2:
        return pd.DataFrame()
    years = sorted({d.year for d in s.index})
    df = pd.DataFrame(index=years, columns=list(range(1, 13)) + ["YTD"], dtype=float)
    monthly = _monthly_returns(s)
    for ts, r in monthly.items():
        if ts.year not in df.index:
            continue
        df.loc[ts.year, ts.month] = float(r)
    # YTD = 年内各月复合
    for y in years:
        vals = df.loc[y, list(range(1, 13))].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        df.loc[y, "YTD"] = float(np.prod(1.0 + vals) - 1.0) if len(vals) else np.nan
    df.index.name = "year"
    return df


# ---------------------------------------------------------------------------
# 回撤 / 滚动
# ---------------------------------------------------------------------------
def drawdown_series(nav) -> pd.Series:
    """回撤序列（<=0，相对历史最高净值）。"""
    s = _as_series(nav)
    return (s / s.cummax() - 1.0).astype(float)


def rolling_metrics(nav, window: int = 252) -> pd.DataFrame:
    """滚动指标：ann_ret / ann_vol / sharpe / max_dd / calmar（窗口默认 1 年）。"""
    s = _as_series(nav)
    if len(s) < 2:
        return pd.DataFrame()
    w = int(window)
    minp = max(20, w // 2)
    r = s.pct_change()
    ann_ret = (s / s.shift(w)) ** (TRADING_DAYS / w) - 1.0
    ann_vol = r.rolling(w, min_periods=minp).std() * SQRT_252
    sharpe = (r.rolling(w, min_periods=minp).mean() /
              r.rolling(w, min_periods=minp).std() * SQRT_252)
    maxdd = s.rolling(w, min_periods=minp).apply(
        lambda x: float((x / np.maximum.accumulate(x) - 1.0).min()), raw=True)
    calmar = ann_ret.abs() / maxdd.abs().replace(0.0, np.nan)
    return pd.DataFrame({
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": maxdd,
        "calmar": calmar,
    })


# ---------------------------------------------------------------------------
# 压力期
# ---------------------------------------------------------------------------
_STRESS_PERIODS = [
    ("2018全年", "2018-01-01", "2018-12-31"),
    ("2020Q1疫情", "2020-01-01", "2020-03-31"),
    ("2022全年", "2022-01-01", "2022-12-31"),
    ("2024初小微盘崩", "2024-01-01", "2024-02-29"),
    ("2024-09-24~10-10急涨", "2024-09-24", "2024-10-10"),
    ("2025-04关税冲击", "2025-04-01", "2025-04-30"),
]


def stress_periods(nav, bench_returns=None) -> pd.DataFrame:
    """预设压力区间表现（区间自动裁剪到净值覆盖范围；最后一段=最近 6 个月）。

    返回 DataFrame，列：period / start / end / n_days / ret / max_dd
    （若给 bench_returns，另加 bench_ret_<name> 与 excess_<name>）。
    """
    s = _as_series(nav)
    rows: list[dict] = []
    if len(s) < 2:
        return pd.DataFrame()
    first_ts, last_ts = s.index[0], s.index[-1]
    periods = list(_STRESS_PERIODS)
    periods.append(("最近6个月", str((last_ts - pd.DateOffset(months=6)).date()),
                    str(last_ts.date())))

    bench = None
    if bench_returns is not None and len(bench_returns):
        bench = (bench_returns.to_frame() if isinstance(bench_returns, pd.Series)
                 else pd.DataFrame(bench_returns))
        bench = bench.reindex(s.index).ffill()

    for name, a, b in periods:
        st = max(pd.Timestamp(a), first_ts)
        en = min(pd.Timestamp(b), last_ts)
        if st > en:
            continue
        seg = s.loc[st:en]
        if len(seg) < 1:
            continue
        ret = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        dd = float((seg / seg.cummax() - 1.0).min())
        row = {"period": name, "start": str(st.date()), "end": str(en.date()),
               "n_days": int(len(seg)), "ret": ret, "max_dd": dd}
        if bench is not None:
            for col in bench.columns:
                col_vals = bench[col].astype(float)
                if np.isfinite(col_vals.reindex(seg.index)).sum() < 2:
                    continue
                bseg = col_vals.loc[seg.index[1]:seg.index[-1]] if len(seg) > 1 else col_vals.iloc[0:0]
                bret = float(np.prod(1.0 + bseg.fillna(0.0).to_numpy(dtype=float)) - 1.0)
                row[f"bench_ret_{col}"] = bret
                row[f"excess_{col}"] = ret - bret
        rows.append(row)
    return pd.DataFrame(rows)
