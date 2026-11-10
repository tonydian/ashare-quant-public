# -*- coding: utf-8 -*-
"""ETF 弹性周期 + 双动量轮动策略（按次方量化文章标准实现）。

设计依据（文章核心思想）：

【弹性周期】（《ETF动量轮动深度拆解》）
  · 排名和信号**每个交易日**重新计算
  · 但买入后必须满足**最小持有天数**才允许被换出
  · 这样把"检查频率"和"交易频率"拆开
  · 止损可打破最小持有期，止盈不可

【双动量】（《双动量轮动策略详解》）
  · 第一动量：短周期（如 20 日）简单动量 → 用于**排名**
  · 第二动量：长周期（如 60 日）斜率动量 → 用于**过滤**
  · 只有同时满足"短期领先"和"长期健康"才入选

【动量计算】（《四种动量计算方式详解》）
  · 简单动量：(P_now - P_N) / P_N
  · 斜率动量：log(price) 线性回归斜率年化 × R²
  · 加权斜率：权重 1→2 线性增长

【动量的价】（《复权对ETF轮动策略的影响》）
  · 动量用等比复权价 → 我们数据是后复权，符合

【排名阈值】（《轮动策略系统详解》）
  · 动量下限：得分 > 阈值才参与排名（解决"矮子里拔将军"）
  · 动量上限：过热剔除

【大盘择时】
  · ETF 动量 > 大盘基准动量 才允许买入

【备选产品】
  · 全池无合格标的 → 买入国债 ETF（我已加入 511260）

【极端过滤】
  · 涨停/跌停当日剔除

【成本】
  · 分标的滑点：宽基 10bp / 行业 15bp / 债券 5bp
  · 佣金 万0.1，最低 5 元
  · ETF 免印花税

DEV/OOS 分离：
  · DEV（选参数）：2018-01-01 ~ 2021-12-31
  · OOS（只验证一次）：2022-01-01 ~ 2026-10-27
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402

# ================================================================ 参数
PARAMS = {
    # 动量
    "mom1_n": 20,              # 第一动量（简单）周期
    "mom2_n": 60,              # 第二动量（斜率）周期
    "mom_floor": 0.0,          # 第一动量下限（>0 才参与排名）
    "mom_ceil": 0.50,          # 第一动量上限（过热剔除）
    "mom2_floor": 0.0,         # 第二动量下限（斜率年化×R² > 0）
    # 持仓
    "top_n": 3,                # 买入前 N 名
    "min_hold": 8,             # 最小持有天数（交易日）
    "max_pos_ratio": 1.0,      # 总仓位
    # 过滤
    "ma_filter": 60,           # 价格 > MA60 才可买（0=关闭）
    "timing": True,            # 大盘择时
    "timing_code": "510300",   # 大盘基准
    # 止损
    "stop_loss": 0.10,         # 跌幅止损 10%（可打破最小持有期）
    "trail_stop": 0.08,        # 高点回撤止损 8%
    # 备选
    "fallback": "511260",      # 国债 ETF
    # 成本（分标的）
    "slip_base": {"宽基": 10.0, "行业": 15.0, "债券": 5.0, "海外": 12.0},
    "comm_rate": 0.0001,       # 万0.1
    "min_comm": 5.0,
    "capital": 300000.0,
}

DEV = ("2018-01-01", "2021-12-31")
OOS = ("2022-01-01", "2026-12-31")

# 标的分类（用于滑点）
CATEGORY = {
    "沪深300": "宽基", "中证500": "宽基", "中证1000": "宽基",
    "创业板": "宽基", "科创50": "宽基",
    "5年国债": "债券", "10年国债": "债券", "城投债": "债券",
    "活跃国债": "债券", "红利": "宽基",
    "纳指": "海外", "标普": "海外", "日经": "海外",
    "港股": "海外", "恒生科技": "海外",
}


# ================================================================ 数据
def load_panel():
    p = ROOT / "build" / "etf" / "etf_ohlc_panel_v2.npz"
    if not p.exists():
        p = ROOT / "build" / "etf" / "etf_ohlc_panel.npz"
    z = np.load(p, allow_pickle=True)
    meta_p = ROOT / "build" / "etf" / "etf_ohlc_meta_v2.json"
    if not meta_p.exists():
        meta_p = ROOT / "build" / "etf" / "etf_ohlc_meta.json"
    meta = json.load(open(meta_p, encoding="utf-8"))
    P = {
        "dates": pd.DatetimeIndex(z["dates"]),
        "codes": [str(c) for c in z["codes"]],
        "meta": meta,
        "open": z["open"].astype(np.float64),
        "close": z["close"].astype(np.float64),
        "high": z["high"].astype(np.float64),
        "low": z["low"].astype(np.float64),
        "volume": z["volume"].astype(np.float64),
        "amount": z["amount"].astype(np.float64),
    }
    P["T"] = len(P["dates"])
    P["N"] = len(P["codes"])
    print(f"[data] {P['T']} 日 × {P['N']} 只，"
          f"{P['dates'][0].date()} ~ {P['dates'][-1].date()}")
    return P


# ================================================================ 动量
def simple_mom(px: np.ndarray, n: int) -> np.ndarray:
    """简单动量：(P_t - P_{t-n}) / P_{t-n}。返回 (T, N)。"""
    T, N = px.shape
    out = np.full((T, N), np.nan)
    if T > n:
        out[n:] = px[n:] / px[:-n] - 1.0
    return out


def slope_mom(px: np.ndarray, n: int, weighted: bool = False) -> np.ndarray:
    """斜率动量：log 价格回归斜率年化 × R²。返回 (T, N)。"""
    T, N = px.shape
    out = np.full((T, N), np.nan)
    if T <= n:
        return out
    x = np.arange(n, dtype=np.float64)
    if weighted:
        w = np.linspace(1, 2, n)
    else:
        w = np.ones(n)
    # 预计算回归分量
    w_sum = w.sum()
    x_bar = (w * x).sum() / w_sum
    sxx = (w * (x - x_bar) ** 2).sum()

    logpx = np.log(np.where(px > 0, px, np.nan))
    for t in range(n, T):
        y = logpx[t - n + 1:t + 1]          # (n, N)
        ok = np.isfinite(y).all(axis=0)
        if not ok.any():
            continue
        yy = y[:, ok]
        y_bar = (w[:, None] * yy).sum(axis=0) / w_sum
        sxy = (w[:, None] * (x[:, None] - x_bar) * (yy - y_bar)).sum(axis=0)
        slope = sxy / sxx
        pred = slope[None, :] * (x[:, None] - x_bar) + y_bar[None, :]
        ss_res = (w[:, None] * (yy - pred) ** 2).sum(axis=0)
        ss_tot = (w[:, None] * (yy - y_bar[None, :]) ** 2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, np.nan)
        r2 = np.clip(np.nan_to_num(r2, nan=0.0), 0.0, 1.0)
        ann = np.expm1(slope * 250.0)
        vals = ann * r2
        row = np.full(N, np.nan)
        row[ok] = vals
        out[t] = row
    return out


def rolling_mean(px: np.ndarray, n: int) -> np.ndarray:
    return pd.DataFrame(px).rolling(n, min_periods=max(2, n // 2)).mean().to_numpy()


# ================================================================ 回测
def backtest(P, mom1, mom2, ma, start, end, tag, verbose=True):
    dates = P["dates"]
    close, openp, high, low = P["close"], P["open"], P["high"], P["low"]
    amt = P["amount"]
    codes, meta = P["codes"], P["meta"]
    T, N = P["T"], P["N"]

    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    t1 = min(t1, T - 1)

    # 大盘基准动量
    base_j = codes.index(PARAMS["timing_code"]) if PARAMS["timing_code"] in codes else None

    # 备选
    fb_j = codes.index(PARAMS["fallback"]) if PARAMS["fallback"] in codes else None

    # 每只标的的滑点
    slips = np.array([
        PARAMS["slip_base"].get(CATEGORY.get(meta.get(c, ""), "行业"), 15.0)
        for c in codes
    ]) / 1e4

    cash = 1.0
    nav = np.full(T, np.nan)
    nav[t0] = 1.0
    hold: dict[int, dict] = {}
    pend_sell = set()
    pend_buy = []
    n_tr = 0
    reasons = {"stop": 0, "trail": 0, "rotate": 0, "empty": 0}

    per_amt = PARAMS["capital"] / max(PARAMS["top_n"], 1)
    comm = max(PARAMS["min_comm"], per_amt * PARAMS["comm_rate"]) / per_amt

    for t in range(t0 + 1, t1 + 1):
        # ---- 执行卖（开盘）
        for j in list(pend_sell):
            if j in hold:
                px = openp[t, j]
                if np.isfinite(px) and px > 0:
                    cash += hold[j]["sh"] * px * (1 - slips[j])
                    del hold[j]
                    n_tr += 1
        pend_sell = set()

        # ---- 执行买（开盘）
        if pend_buy:
            slots = PARAMS["top_n"] - len(hold)
            if slots > 0:
                tot = cash + sum(hold[k]["sh"] * openp[t, k]
                                 for k in hold if np.isfinite(openp[t, k]))
                per = tot / PARAMS["top_n"]
                for j in pend_buy[:slots]:
                    px = openp[t, j]
                    if not (np.isfinite(px) and px > 0):
                        continue
                    budget = min(per, cash)
                    if budget <= 1e-9:
                        break
                    sh = budget / (px * (1 + slips[j]) * (1 + comm))
                    cash -= sh * px * (1 + slips[j]) * (1 + comm)
                    hold[j] = {"sh": sh, "cost": px, "day": t,
                               "peak": px}
                    n_tr += 1
            pend_buy = []

        # ---- 净值
        val = cash
        for j, h in hold.items():
            px = close[t, j]
            if np.isfinite(px):
                val += h["sh"] * px
                if px > h["peak"]:
                    h["peak"] = px
        nav[t] = val

        # ---- 决策
        # ① 止损/回撤（可打破最小持有期）
        for j, h in hold.items():
            c = close[t, j]
            if not np.isfinite(c):
                continue
            if c / h["cost"] - 1 <= -PARAMS["stop_loss"]:
                pend_sell.add(j)
                reasons["stop"] += 1
            elif h["peak"] > 0 and c / h["peak"] - 1 <= -PARAMS["trail_stop"]:
                pend_sell.add(j)
                reasons["trail"] += 1

        # ② 排名（每日）
        m1 = mom1[t]
        m2 = mom2[t]
        mf = ma[t]
        ok = (np.isfinite(m1) & np.isfinite(m2) & np.isfinite(mf)
              & (close[t] > 0))
        # 阈值过滤
        ok &= (m1 > PARAMS["mom_floor"]) & (m1 < PARAMS["mom_ceil"])
        ok &= (m2 > PARAMS["mom2_floor"])
        # 均线过滤
        if PARAMS["ma_filter"] > 0:
            ok &= (close[t] > mf)
        # 极端过滤（涨停/跌停）
        prev = close[t - 1] if t > 0 else close[t]
        with np.errstate(divide="ignore", invalid="ignore"):
            day_ret = close[t] / np.where(prev > 0, prev, np.nan) - 1
        ok &= ~(np.abs(day_ret) > 0.095)     # ETF 涨跌停近似

        # 大盘择时
        if PARAMS["timing"] and base_j is not None:
            bm = m1[base_j]
            if np.isfinite(bm):
                ok &= (m1 > bm)

        ranked = np.flatnonzero(ok)
        if len(ranked):
            ranked = ranked[np.argsort(-m1[ranked])]
        top = list(ranked[:PARAMS["top_n"]])

        # ③ 换仓：持有中不在 top 且满足最小持有期 → 卖
        for j in list(hold):
            if j in pend_sell:
                continue
            if t - hold[j]["day"] < PARAMS["min_hold"]:
                continue          # 最小持有期未到
            if j not in top:
                pend_sell.add(j)
                reasons["rotate"] += 1

        # ④ 买入（次日开盘）
        need = [j for j in top if j not in hold and j not in pend_sell]
        if need:
            pend_buy = need
        elif not top and not hold and fb_j is not None:
            # 全池无合格 → 备选
            if fb_j not in hold:
                pend_buy = [fb_j]
                reasons["empty"] += 1

    s = pd.Series(nav, index=dates).dropna()
    s = s[s.index <= pd.Timestamp(end)]
    if len(s) < 60:
        return None, {}
    p = qfm.performance(s, rf=C.CASH_YIELD)
    res = {k: float(p[k]) for k in
           ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}
    if verbose:
        print(f"\n  [{tag}] {s.index[0].date()} ~ {s.index[-1].date()}"
              f"（{len(s)} 天）")
        print(f"    CAGR {res['cagr']:7.2%} | Sharpe {res['sharpe']:6.3f} | "
              f"MaxDD {res['max_dd']:7.2%} | Calmar {res['calmar']:5.2f}")
        print(f"    交易 {n_tr} 笔 | 卖出原因 {reasons}")
    return s, res


# ================================================================ 主流程
def main():
    print("=" * 96)
    print("ETF 弹性周期 + 双动量策略")
    print("=" * 96)
    for k, v in PARAMS.items():
        if k == "slip_base":
            print(f"  {k:<16} {v}")
        else:
            print(f"  {k:<16} {v}")

    P = load_panel()

    print("\n[计算动量] …")
    mom1 = simple_mom(P["close"], PARAMS["mom1_n"])
    print(f"  第一动量（{PARAMS['mom1_n']}日简单）就绪")
    mom2 = slope_mom(P["close"], PARAMS["mom2_n"], weighted=False)
    print(f"  第二动量（{PARAMS['mom2_n']}日斜率）就绪")
    ma = rolling_mean(P["close"], PARAMS["ma_filter"])
    print(f"  MA{PARAMS['ma_filter']} 就绪")

    print("\n" + "=" * 96)
    print("DEV 期（2018-2021）")
    print("=" * 96)
    dev_nav, dev_res = backtest(P, mom1, mom2, ma, DEV[0], DEV[1], "DEV")

    print("\n" + "=" * 96)
    print("OOS 期（2022-2026，只看一次）")
    print("=" * 96)
    oos_nav, oos_res = backtest(P, mom1, mom2, ma, OOS[0], OOS[1], "OOS")

    print("\n" + "=" * 96)
    print("全期（2018-2026）")
    print("=" * 96)
    full_nav, full_res = backtest(P, mom1, mom2, ma, DEV[0], OOS[1], "FULL")

    # 基准
    print("\n" + "=" * 96)
    print("基准（同期买入持有）")
    print("=" * 96)
    bench = {}
    for name in ("沪深300", "中证500", "10年国债"):
        js = [j for j, c in enumerate(P["codes"]) if P["meta"].get(c) == name]
        if not js:
            continue
        j = js[0]
        s = pd.Series(P["close"][:, j], index=P["dates"]).dropna()
        s = s[s.index >= pd.Timestamp(DEV[0])]
        if len(s) < 100:
            continue
        p = qfm.performance(s, rf=C.CASH_YIELD)
        bench[name] = {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd")}
        print(f"  {name:<10} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%}")

    # 保存
    out = ROOT / "build" / "runs" / "etf_flex_dual"
    out.mkdir(parents=True, exist_ok=True)
    if full_nav is not None:
        pd.DataFrame({"nav": full_nav}).to_parquet(out / "nav.parquet")
    json.dump({"params": PARAMS, "dev": dev_res, "oos": oos_res,
               "full": full_res, "bench": bench},
              open(out / "stats.json", "w"), ensure_ascii=False, indent=1,
              default=float)
    print(f"\n→ {out}")

    # 结论
    print("\n" + "=" * 96)
    print("结论")
    print("=" * 96)
    if dev_res and oos_res:
        print(f"  DEV CAGR {dev_res['cagr']:.2%} → OOS CAGR {oos_res['cagr']:.2%}"
              f"（{(oos_res['cagr']-dev_res['cagr'])*100:+.2f}pp）")
        print(f"  DEV Sharpe {dev_res['sharpe']:.3f} → "
              f"OOS Sharpe {oos_res['sharpe']:.3f}")
        print(f"  全期 CAGR {full_res['cagr']:.2%}, Sharpe {full_res['sharpe']:.3f}")


if __name__ == "__main__":
    main()
