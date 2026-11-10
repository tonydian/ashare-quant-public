# -*- coding: utf-8 -*-
"""风格归因（Barra-lite）：把组合收益拆成风格 + 行业 + 特异 + 现金 + 未解释。

为什么需要
----------
回测只告诉我们赚了多少，不告诉我们**赚在哪**。若收益主要来自小市值 / 高波动等
风格暴露，风格反转时会系统性回吐；若主要来自特异收益（选股残差），才是可持续
的 alpha；若主要来自行业押注，则是行业轮动的运气与风险。

方法
----
对每个调仓期 ``[t1, t2)``：

1. **因子暴露**（全部 PIT，只用 t1 及之前数据 + t1 收盘价）：
   - 风格：SIZE = ln(总市值)；VALUE = 0.5·z(TTM盈利收益率) + 0.5·z(账面市值比)；
     MOM = 60 日动量（跳过最近 5 日）；VOL = 60 日日收益标准差；
     LIQ = 20 日平均成交额 / 总市值（换手率代理）。
   - 行业：``md.industry[t1]`` 哑变量（丢弃股票数最多的那一类作为基线）。
2. **因子收益**：横截面加权最小二乘
   ``r_i(t1→t2) = f_c + Σ_k X_ik·f_k + u_i``，权重 ``√总市值``（Barra 惯例）。
3. **组合贡献**：``E_k = Σ_i w_i·X_ik``，贡献 ``= E_k·f_k``；
   特异 ``= Σ_i w_i·u_i``；市场 ``= (Σ w_i)·f_c``；
   现金 ``= (1 - Σ w_i)·现金套件区间收益``。
4. **未解释** ``= R_实际 − (上述各项之和)``：交易成本、整手取整、涨跌停未成交、
   t→t+1 执行时点差、样本外持仓。这一栏不隐藏，用于核对账目闭合。

口径与限制
----------
* 收益用后复权收盘 t1→t2，实际执行在 t1+1 开盘，一天错位落在「未解释」。
* 基准暴露用「股票池内总市值加权」代理，非真实指数成分。
* 行业哑变量丢弃基线行业后，``MARKET`` 的截距即为该基线行业的收益，
  因此 ``MARKET`` 应理解为「基线行业 + 市场整体」，不是纯指数收益。
* 风格收益为样本内横截面估计，含估计误差（报告中给 t 值）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["STYLES", "build_styles", "wls_factor_returns", "attribute_periods", "summarize"]

#: 风格分量
STYLES = ("SIZE", "VALUE", "MOM", "VOL", "LIQ")
#: 全部收益分量（MARKET 为国家/基线行业因子，由回归截距给出）
COMPONENTS = ("MARKET",) + STYLES + ("INDUSTRY",)


# ---------------------------------------------------------------------------
# 因子构造
# ---------------------------------------------------------------------------
def _row_z(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """单行横截面 z-score（mask 外一律 NaN）。"""
    v = np.asarray(x, dtype=np.float64)
    ok = mask & np.isfinite(v)
    out = np.full(v.shape, np.nan, dtype=np.float64)
    if ok.sum() < 20:
        return out
    mu = float(np.mean(v[ok]))
    sd = float(np.std(v[ok]))
    if not np.isfinite(sd) or sd <= 1e-12:
        return out
    out[ok] = (v[ok] - mu) / sd
    return out


def _nanmean_stack(arrs: list[np.ndarray]) -> np.ndarray:
    """逐元素 nan 平均（全 NaN 处保留 NaN）。"""
    stack = np.stack(arrs, axis=0)
    cnt = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.nansum(stack, axis=0)
        out = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out


def build_styles(md, rows, value_panels: dict | None = None,
                 mom_win: int = 60, mom_skip: int = 5,
                 vol_win: int = 60, liq_win: int = 20) -> dict[str, np.ndarray]:
    """在指定行上构造风格暴露面板（每行横截面 z-score）。

    参数
    ----
    md : MarketData
    rows : array-like int，需计算的交易日行号（通常是调仓日）
    value_panels : dict，含 ``ep_ttm`` / ``bp`` 两个 (T, N) 面板
        （``qf.fundamentals.FundamentalData(md).build_factors()``）。缺省则 VALUE 为 NaN。

    返回
    ----
    dict[str, (len(rows), N) float32]，键为 ``STYLES``。
    """
    rows = np.asarray(rows, dtype=int)
    price = np.asarray(md.price, dtype=np.float64)
    mcap = np.asarray(md.mcap, dtype=np.float64)
    amount = np.asarray(md.amount, dtype=np.float64)
    T, N = price.shape

    ret = np.full((T, N), np.nan, dtype=np.float64)
    ret[1:] = price[1:] / price[:-1] - 1.0
    ret[~np.isfinite(ret)] = np.nan

    ep = None if value_panels is None else np.asarray(value_panels.get("ep_ttm"), dtype=np.float64)
    bp = None if value_panels is None else np.asarray(value_panels.get("bp"), dtype=np.float64)

    out = {k: np.full((len(rows), N), np.nan, dtype=np.float64) for k in STYLES}

    for r, t in enumerate(rows):
        elig = np.asarray(md.eligible[t], dtype=bool)

        with np.errstate(invalid="ignore", divide="ignore"):
            size_raw = np.log(np.where(mcap[t] > 0, mcap[t], np.nan))
        out["SIZE"][r] = _row_z(size_raw, elig)

        i1, i0 = t - mom_skip, t - mom_skip - mom_win
        if i0 >= 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                mom_raw = price[i1] / np.where(price[i0] > 0, price[i0], np.nan) - 1.0
            out["MOM"][r] = _row_z(mom_raw, elig)

        j0 = max(t - vol_win + 1, 1)
        if t - j0 >= vol_win // 2:
            w = ret[j0:t + 1]
            if w.size:
                with np.errstate(invalid="ignore"):
                    sd = np.nanstd(w, axis=0)
                out["VOL"][r] = _row_z(sd, elig)

        k0 = max(t - liq_win + 1, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            amt = np.nanmean(amount[k0:t + 1], axis=0)
            liq_raw = amt / np.where(mcap[t] > 0, mcap[t], np.nan)
        out["LIQ"][r] = _row_z(liq_raw, elig)

        if ep is not None and bp is not None:
            out["VALUE"][r] = _nanmean_stack([_row_z(ep[t], elig), _row_z(bp[t], elig)])

    return {k: v.astype(np.float32) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 横截面回归
# ---------------------------------------------------------------------------
def wls_factor_returns(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                       std_cols: np.ndarray | None = None, min_obs: int = 100):
    """横截面加权最小二乘 ``r_i = c + Σ coef_k·Z_ik + u_i``。

    ``Z`` 为标准化后的设计矩阵：``std_cols`` 为 True 的列做 (x−μ)/σ，其余列原样
    （行业哑变量必须原样，否则系数失去「行业收益」含义）。

    返回 ``(coef, mu, sd, u, ok)``。注意 ``coef`` 是 **Z 空间**系数：
    贡献必须用 ``E_Z · coef``（E_Z 为标准化暴露）；如果要报告「每单位原始暴露的
    收益」，用 ``coef / sd`` 换算。样本不足时 coef/u 为 None。
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.all(np.isfinite(X), axis=1)
    if int(ok.sum()) < max(min_obs, X.shape[1] * 10):
        return None, None, None, None, ok

    Xs, ys, ws = X[ok], y[ok], w[ok]
    k = Xs.shape[1]
    if std_cols is None:
        std_cols = np.ones(k, dtype=bool)
    mu = np.where(std_cols, Xs.mean(axis=0), 0.0)
    sd = np.where(std_cols, Xs.std(axis=0), 1.0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xs - mu) / sd

    A = np.column_stack([np.ones(len(ys)), Z])
    sw = np.sqrt(ws)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], ys * sw, rcond=None)

    u_full = np.full(y.shape, np.nan, dtype=np.float64)
    u_full[ok] = ys - A @ coef
    return coef, mu, sd, u_full, ok


def _industry_design(md, t: int, base_ok: np.ndarray, mcap_row: np.ndarray,
                     min_stocks: int = 5):
    """构造行业哑变量设计矩阵。

    基线行业取**样本内总市值最大**的行业（使截距≈市场组合的行业基准），
    其余行业各占一列。返回 ``(D, keep_codes, drop_code, ind_row)``。
    """
    ind = np.asarray(md.industry[t]).astype(np.int64) + 1     # -1 → 0（未知）
    codes, counts = np.unique(ind[base_ok], return_counts=True)
    keep = [(int(c), int(n)) for c, n in zip(codes, counts) if n >= min_stocks]
    if len(keep) < 2:
        return np.zeros((ind.shape[0], 0)), [], None, ind

    cap_by_code = {}
    for c, _n in keep:
        m = base_ok & (ind == c)
        cap_by_code[c] = float(np.nansum(mcap_row[m]))
    keep.sort(key=lambda x: -cap_by_code.get(x[0], 0.0))
    drop_code = keep[0][0]
    others = [c for c, _ in keep[1:]]
    D = np.zeros((ind.shape[0], len(others)), dtype=np.float64)
    col = {c: i for i, c in enumerate(others)}
    for i, c in enumerate(ind):
        j = col.get(int(c))
        if j is not None:
            D[i, j] = 1.0
    return D, others, int(drop_code), ind


# ---------------------------------------------------------------------------
# 逐期归因
# ---------------------------------------------------------------------------
def attribute_periods(md, rows, weights: pd.DataFrame, nav: pd.Series,
                      styles: dict[str, np.ndarray], cap_weight: bool = True,
                      use_industry: bool = True,
                      cash_daily_ret: np.ndarray | None = None,
                      ret_mode: str = "close"):
    """逐期归因。

    参数
    ----
    md : MarketData
    rows : array-like int，区间边界行号（升序，相邻两行构成一期）
    weights : DataFrame[date, code, weight]，调仓日实际持仓权重
    nav : Series，index 为交易日、值为净值
    styles : ``build_styles`` 输出（行序与 rows 一致）
    cap_weight : True 用 √总市值做回归权重，False 等权
    use_industry : 是否加入行业哑变量
    cash_daily_ret : (T,) 闲置现金替代资产日收益（``qf.backtest.load_cash_asset_returns``）

    返回
    ----
    (exposures, contrib, ind_exp)
    """
    rows = np.asarray(rows, dtype=int)
    dates = md.dates
    if ret_mode not in ("close", "open"):
        raise ValueError("ret_mode 只支持 'close' 或 'open'")
    # close: t1 收盘 → t2 收盘；open: t1+1 开盘 → t2+1 开盘（贴近回测执行时点）
    px = np.asarray(md.open if ret_mode == "open" else md.close, dtype=np.float64)
    shift = 1 if ret_mode == "open" else 0
    mcap = np.asarray(md.mcap, dtype=np.float64)
    n_ind = len(getattr(md, "industry_cats", []))

    code_idx = {c: j for j, c in enumerate(md.codes)}
    w_all = weights.copy()
    w_all["code"] = w_all["code"].astype(str).str.zfill(6)
    by_date = {pd.Timestamp(d): g for d, g in w_all.groupby("date")}
    X_cache = {k: np.asarray(v, dtype=np.float64) for k, v in styles.items()}

    exp_rows: list[dict] = []
    con_rows: list[dict] = []
    ind_rows: list[dict] = []

    def _ind_name(code: int) -> str:
        if code <= 0:
            return "未知"
        if code - 1 < n_ind:
            return str(md.industry_cats[code - 1])
        return f"行业{code}"

    for r in range(len(rows) - 1):
        t1, t2 = int(rows[r]), int(rows[r + 1])
        d1, d2 = dates[t1], dates[t2]
        g = by_date.get(pd.Timestamp(d1))
        if g is None or len(g) == 0:
            continue

        elig = np.asarray(md.eligible[t1], dtype=bool)
        p1, p2 = min(t1 + shift, md.T - 1), min(t2 + shift, md.T - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            y = px[p2] / np.where(px[p1] > 0, px[p1], np.nan) - 1.0

        Xs_all = np.column_stack([X_cache[k][r] for k in STYLES])
        # 覆盖不足的风格列（如早期无财报数据时的 VALUE）直接剔除，避免整期样本被判空
        active = [k for i, k in enumerate(STYLES)
                  if np.isfinite(Xs_all[elig, i]).mean() >= 0.6]
        if not active:
            continue
        Xs = np.column_stack([X_cache[k][r] for k in active])
        base_ok = (elig & np.isfinite(y) & np.all(np.isfinite(Xs), axis=1))

        if use_industry:
            D, keep_codes, drop_code, ind_row = _industry_design(
                md, t1, base_ok, mcap[t1])
        else:
            D, keep_codes, drop_code, ind_row = np.zeros((len(y), 0)), [], None, None
        X = np.column_stack([Xs, D]) if D.shape[1] else Xs

        if cap_weight:
            with np.errstate(invalid="ignore"):
                w_reg = np.sqrt(np.where(mcap[t1] > 0, mcap[t1], np.nan))
            w_reg = np.where(elig, w_reg, np.nan)
        else:
            w_reg = np.where(elig, 1.0, np.nan)

        std_cols = np.array([True] * len(active) + [False] * D.shape[1], dtype=bool)
        coef, mu, sd, u, ok = wls_factor_returns(X, y, w_reg, std_cols=std_cols)
        if coef is None:
            continue

        # 持仓权重（区分样本内 / 样本外）
        idx = np.array([code_idx.get(c, -1) for c in g["code"]], dtype=int)
        wts = g["weight"].to_numpy(dtype=np.float64)
        keep = idx >= 0
        idx, wts = idx[keep], wts[keep]
        in_sample = ok[idx]
        w_in = np.where(in_sample, wts, 0.0)
        invested = float(wts.sum())
        w_out = float(np.where(~in_sample, wts, 0.0).sum())

        Z = (X - mu) / sd                       # 标准化设计矩阵（哑变量列 sd=1）
        Z_h = Z[idx]
        exp_style = {f"E_{k}": 0.0 for k in STYLES}
        ben_style = {f"B_{k}": 0.0 for k in STYLES}
        w_cap = np.where(ok, np.sqrt(np.where(mcap[t1] > 0, mcap[t1], np.nan)), np.nan)
        w_cap = w_cap / np.nansum(w_cap)
        for i, k in enumerate(active):
            exp_style[f"E_{k}"] = float(np.nansum(w_in * Z_h[:, i]))
            ben_style[f"B_{k}"] = float(np.nansum(w_cap * Z[:, i]))

        # 贡献 = 标准化暴露 × Z 空间系数（与回归恒等式一致，不能混用原始系数）
        contrib = {k: 0.0 for k in STYLES}
        for i, k in enumerate(active):
            contrib[k] = float(exp_style[f"E_{k}"] * coef[1 + i])

        # 行业：哑变量系数（原始单位）= 各行业相对基线行业的收益差
        invested_in = float(np.nansum(w_in))
        if keep_codes:
            n_style = len(active)
            r_ind = {int(drop_code): float(coef[0])}
            for col_i, c in enumerate(keep_codes):
                r_ind[int(c)] = float(coef[0] + coef[1 + n_style + col_i])
            cap_ok = np.where(ok & (mcap[t1] > 0), mcap[t1], 0.0)
            tot_cap = float(cap_ok.sum())
            mkt_ret = 0.0
            if tot_cap > 0:
                for c, rv in r_ind.items():
                    m = ok & (ind_row == c)
                    mkt_ret += float(cap_ok[m].sum()) / tot_cap * rv
            contrib["MARKET"] = invested_in * mkt_ret
            ind_contrib = 0.0
            w_ind: dict[str, float] = {}
            for j, wj in zip(idx, w_in):
                if wj <= 0:
                    continue
                c = int(ind_row[j]) if ind_row is not None else 0
                nm = _ind_name(c)
                w_ind[nm] = w_ind.get(nm, 0.0) + float(wj)
                ind_contrib += float(wj) * (r_ind.get(c, mkt_ret) - mkt_ret)
            contrib["INDUSTRY"] = float(ind_contrib)
            ind_rows.append({"date": d1, "investment": invested,
                             **{f"W_{k}": v for k, v in w_ind.items()}})
        else:
            contrib["MARKET"] = invested_in * float(coef[0])
            contrib["INDUSTRY"] = 0.0

        contrib["SPECIFIC"] = float(np.nansum(w_in * np.nan_to_num(u[idx])))

        r_actual = float(nav.get(d2, np.nan) / nav.get(d1, np.nan) - 1.0)
        cash_ret = 0.0
        if cash_daily_ret is not None:
            sl = cash_daily_ret[t1 + 1:t2 + 1]
            cash_ret = float(np.prod(1.0 + sl) - 1.0)
        contrib["CASH"] = float(max(1.0 - invested, 0.0) * cash_ret)

        r_est = float(np.nansum(wts * np.nan_to_num(y[idx])))
        gap = r_est - sum(contrib[k] for k in list(COMPONENTS) + ["SPECIFIC"])
        contrib["UNEXPLAINED"] = r_actual - (sum(contrib[k] for k in list(COMPONENTS) +
                                                 ["SPECIFIC", "CASH"]))

        exp_rows.append({"date": d1, "end_date": d2, "invested": invested,
                         "n_positions": int(len(wts)), "w_out_of_sample": w_out,
                         **exp_style, **ben_style})
        con_rows.append({"date": d1, "end_date": d2, "R_actual": r_actual, "R_est": r_est,
                         **{k: contrib[k] for k in COMPONENTS},
                         "SPECIFIC": contrib["SPECIFIC"], "CASH": contrib["CASH"],
                         "UNEXPLAINED": contrib["UNEXPLAINED"], "IDENTITY_GAP": gap})

    exp_df = pd.DataFrame(exp_rows).set_index("date") if exp_rows else pd.DataFrame()
    con_df = pd.DataFrame(con_rows).set_index("date") if con_rows else pd.DataFrame()
    ind_df = pd.DataFrame(ind_rows).set_index("date") if ind_rows else pd.DataFrame()
    return exp_df, con_df, ind_df


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(con: pd.DataFrame, exp: pd.DataFrame) -> dict:
    """逐期贡献 → 年化口径 + 账目闭合检查。"""
    if con.empty:
        return {}
    n = len(con)
    years = n / 12.0
    try:
        span = (pd.Timestamp(con["end_date"].iloc[-1]) - pd.Timestamp(con.index[0])).days
        years = max(span / 365.25, 1e-6)
    except Exception:
        pass

    comps = list(COMPONENTS) + ["SPECIFIC", "CASH", "UNEXPLAINED"]
    rows = []
    for k in comps:
        s = con[k]
        rows.append({"component": k,
                     "mean_per_period": float(s.mean()),
                     "sum": float(s.sum()),
                     "annual": float(s.sum() / years),
                     "std": float(s.std()),
                     "t_stat": float(s.mean() / s.std() * np.sqrt(n)) if s.std() > 0 else np.nan,
                     "win_rate": float((s > 0).mean())})
    tbl = pd.DataFrame(rows).set_index("component")
    total_actual = float((1.0 + con["R_actual"]).prod() - 1.0)
    total_decomp = float((1.0 + con[comps].sum(axis=1)).prod() - 1.0)
    return {"n_periods": n, "years": years, "table": tbl,
            "total_actual": total_actual, "total_decomp_check": total_decomp,
            "avg_exposure": {k: float(exp[f"E_{k}"].mean()) for k in STYLES},
            "avg_bench_exposure": {k: float(exp[f"B_{k}"].mean()) for k in STYLES}}
