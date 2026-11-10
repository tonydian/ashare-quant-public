# -*- coding: utf-8 -*-
"""基本面（稳健版）+ RSI 择时策略。

与用户原版的区别（原版实测不可行，见《命中率检验》）：
  原版                        稳健版
  ROE > 30%（单期）      →   ROE > 15%（连续 2 年）
  净利同比 > 150%（单期） →   净利增速 > 20%（连续 2 年）
  市现率 < 5（与 ROE 互斥）→  市现率 < 30
  PB 0~10                 →   不变
  毛利率 > 30%             →   不变
  小市值前 50              →   不变
  RSI14 <50 买 / >70 卖    →   不变
  月频调仓、10 只等权       →   不变

严格的时点处理（无前视）：
  · 财报按法定披露截止日可用
    一季报(3-31) → 4-30
    半年报(6-30) → 8-31
    三季报(9-30) → 10-31
    年报(12-31)  → 次年 4-30
  · "连续 2 年" = 当前可用年报 与 上一期年报 都满足
  · PB / 市现率用调仓日收盘价
  · RSI 用调仓日前 N 日数据

RSI 择时逻辑（用户规则）：
  · 买入条件：RSI14 < 50
  · 卖出条件：RSI14 > 70
  · 每月第一个交易日调仓
  · 持仓 10 只，等权

DEV/OOS：
  · DEV（选参数）：2018-01-01 ~ 2021-12-31
  · OOS（只验证）：2022-01-01 ~ 2026-10-31
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

from qf import backtest as qfbt, config as C, data as qfdata, metrics as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "fund_rsi_robust.json"
OUT_RUN = ROOT / "build" / "runs" / "fund_rsi"

PARAMS = {
    # 基本面
    "pb_range": (0, 10),
    "pcf_max": 30.0,
    "roe_min": 15.0,
    "roe_years": 2,          # 连续 N 年年报满足
    "gpm_min": 30.0,
    "yoy_min": 20.0,
    "yoy_years": 2,
    "pool_top": 50,          # 小市值前 N 进候选池
    # RSI
    "rsi_n": 14,
    "rsi_buy": 999.0,        # 不限制买入（实测 RSI<50 才买是负贡献）
    "rsi_sell": 70.0,        # > 70 卖
    # 持仓
    "n_hold": 10,
    "rebal_monthly": True,
    # 成本
    "slip_bp": 40.0,
    "comm_rate": 0.0001,
    "min_comm": 5.0,
    "capital": 300000.0,
}

DEV = ("2018-01-01", "2021-12-31")
OOS = ("2022-01-01", "2026-10-31")


# ================================================================ 数据
def load_all():
    md = qfdata.load_market(verbose=False)
    dates = pd.DatetimeIndex(md.dates)
    code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}
    print(f"[data] {md.T} 日 × {md.N} 只")

    fdir = C.DATA_RAW / "fund"
    periods = sorted([p.stem for p in fdir.glob("*.parquet")])

    def usable(period):
        y, m = int(period[:4]), int(period[4:6])
        return {3: pd.Timestamp(y, 4, 30), 6: pd.Timestamp(y, 8, 31),
                9: pd.Timestamp(y, 10, 31)}.get(m, pd.Timestamp(y + 1, 4, 30))

    fund = {}
    for p in periods:
        d = pd.read_parquet(fdir / f"{p}.parquet")
        d["code6"] = d["code"].astype(str).str.zfill(6)
        d = d[d["code6"].isin(code_idx)]
        fund[p] = {"usable": usable(p), "is_annual": p.endswith("1231"),
                   "data": d.set_index("code6")[
                       ["bps", "ocfps", "roe", "gpm", "profit_yoy", "name"]]}
    print(f"[fund] {len(fund)} 期财报")
    return md, dates, code_idx, fund


def rsi_panel(px: np.ndarray, n: int) -> np.ndarray:
    """RSI14（Wilder 平滑）。返回 (T, N)。"""
    T, N = px.shape
    diff = np.diff(px, axis=0, prepend=np.nan)
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    out = np.full((T, N), np.nan)

    # Wilder 平滑：alpha = 1/n
    avg_up = np.full(N, np.nan)
    avg_dn = np.full(N, np.nan)
    for t in range(1, T):
        if t == 1:
            avg_up = up[t].copy()
            avg_dn = dn[t].copy()
        else:
            avg_up = (avg_up * (n - 1) + up[t]) / n
            avg_dn = (avg_dn * (n - 1) + dn[t]) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_dn > 1e-12, avg_up / avg_dn,
                          np.where(avg_up > 1e-12, 100.0, 50.0))
            out[t] = 100.0 - 100.0 / (1.0 + rs)
    return out


# ================================================================ 选股
def fundamental_mask(md, code_idx, fund, t, period_id) -> pd.DataFrame:
    """在 t 日按稳健规则筛选，返回通过的股票 DataFrame。

    "连续 N 年"：用当前可用年报 + 前一期年报。
    """
    dates = pd.DatetimeIndex(md.dates)
    cur = fund[period_id]["data"]

    # 找当前与前一期年报
    annual_ids = sorted([p for p, v in fund.items() if v["is_annual"]])
    usable_annual = [p for p in annual_ids if fund[p]["usable"] <= dates[t]]
    if len(usable_annual) < PARAMS["roe_years"]:
        return pd.DataFrame()
    use_years = usable_annual[-PARAMS["roe_years"]:]

    px, mcap = md.price[t], md.mcap[t]
    rows = []
    for code, r in cur.iterrows():
        j = code_idx.get(code)
        if j is None:
            continue
        p_, m_ = px[j], mcap[j]
        if not (np.isfinite(p_) and p_ > 0 and np.isfinite(m_) and m_ > 0):
            continue

        bps, ocfps = r["bps"], r["ocfps"]
        pb = p_ / bps if (np.isfinite(bps) and bps > 0) else np.nan
        pcf = p_ / ocfps if (np.isfinite(ocfps) and ocfps > 0) else np.nan

        lo, hi = PARAMS["pb_range"]
        if not (np.isfinite(pb) and lo < pb <= hi):
            continue
        if not (np.isfinite(pcf) and 0 < pcf < PARAMS["pcf_max"]):
            continue
        if not (np.isfinite(r["gpm"]) and r["gpm"] > PARAMS["gpm_min"]):
            continue

        # 连续 N 年年报都要满足 ROE / 净利增速
        ok_years = True
        for ay in use_years:
            if code not in fund[ay]["data"].index:
                ok_years = False
                break
            ar = fund[ay]["data"].loc[code]
            if not (np.isfinite(ar["roe"]) and ar["roe"] > PARAMS["roe_min"]):
                ok_years = False
                break
            if not (np.isfinite(ar["profit_yoy"])
                    and ar["profit_yoy"] > PARAMS["yoy_min"]):
                ok_years = False
                break
        if not ok_years:
            continue

        rows.append({"code": code, "name": r["name"], "j": j,
                     "pb": pb, "pcf": pcf, "gpm": r["gpm"],
                     "roe": r["roe"], "yoy": r["profit_yoy"],
                     "mcap_yi": m_ / 1e8})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.nsmallest(PARAMS["pool_top"], "mcap_yi")


# ================================================================ 回测
def run(md, dates, code_idx, fund, rsi, start, end, tag, verbose=True):
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    t1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    t1 = min(t1, md.T - 1)

    # 月初调仓日
    ds = pd.Series(dates)
    fom = (ds.dt.month != ds.shift(1).dt.month).to_numpy()
    rebal_days = set(int(x) for x in np.flatnonzero(fom) if t0 <= x <= t1)

    slip = PARAMS["slip_bp"] / 1e4
    per_amt = PARAMS["capital"] / PARAMS["n_hold"]
    comm = max(PARAMS["min_comm"], per_amt * PARAMS["comm_rate"]) / per_amt

    cash_ret = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT,
                                            C.CASH_GOLD_WEIGHT)

    cash = 1.0
    nav = np.full(md.T, np.nan)
    nav[t0] = 1.0
    hold: dict[int, dict] = {}
    pend_sell, pend_buy = set(), []
    n_tr = 0
    reasons = {"rsi_sell": 0, "rotate": 0, "empty": 0}
    diag = []

    for t in range(t0 + 1, t1 + 1):
        # ---- 执行（开盘）
        for j in list(pend_sell):
            if j in hold:
                px = md.price_open[t, j]
                if np.isfinite(px) and px > 0:
                    cash += hold[j]["sh"] * px * (1 - slip) * (1 - comm)
                    del hold[j]
                    n_tr += 1
        pend_sell = set()

        if pend_buy:
            slots = PARAMS["n_hold"] - len(hold)
            if slots > 0 and cash > 1e-9:
                tot = cash + sum(hold[k]["sh"] * md.price_open[t, k]
                                 for k in hold
                                 if np.isfinite(md.price_open[t, k]))
                per = tot / PARAMS["n_hold"]
                for j in pend_buy[:slots]:
                    if j in hold:
                        continue
                    px = md.price_open[t, j]
                    if not (np.isfinite(px) and px > 0):
                        continue
                    b = min(per, cash)
                    if b <= 1e-9:
                        break
                    sh = b / (px * (1 + slip) * (1 + comm))
                    cash -= sh * px * (1 + slip) * (1 + comm)
                    hold[j] = {"sh": sh, "cost": px, "day": t}
                    n_tr += 1
            pend_buy = []

        # ---- 现金计息 + 净值
        cr = float(cash_ret[t]) if np.isfinite(cash_ret[t]) else 0.0
        cash *= (1 + cr)
        v = cash
        for j, h in hold.items():
            p_ = md.price[t, j]
            if np.isfinite(p_):
                v += h["sh"] * p_
        nav[t] = v

        # ---- 调仓日决策
        if t not in rebal_days:
            continue

        # 找可用财报
        avail = [(p, vv["usable"]) for p, vv in fund.items()
                 if vv["usable"] <= dates[t]]
        if not avail:
            continue
        pid = max(avail, key=lambda x: x[1])[0]

        pool = fundamental_mask(md, code_idx, fund, t, pid)
        r_t = rsi[t]

        # 卖出：RSI > 70
        for j in list(hold):
            if np.isfinite(r_t[j]) and r_t[j] > PARAMS["rsi_sell"]:
                pend_sell.add(j)
                reasons["rsi_sell"] += 1

        # 买入候选：在池内 且 RSI < 50
        cands = []
        for _, r in pool.iterrows():
            j = int(r["j"])
            if j in hold or j in pend_sell:
                continue
            if np.isfinite(r_t[j]) and r_t[j] < PARAMS["rsi_buy"]:
                cands.append((j, r_t[j]))

        # 已有持仓中 RSI 仍 <50 的继续持有
        keep = set(hold.keys()) - pend_sell
        n_slots = PARAMS["n_hold"] - len(keep)
        if n_slots > 0 and cands:
            # 按 RSI 升序（越低越买）
            cands.sort(key=lambda x: x[1])
            pend_buy = [j for j, _ in cands[:n_slots]]

        # 池为空 → 卖出不满足的（已在上面处理）
        if pool.empty:
            reasons["empty"] += 1

        diag.append({"date": dates[t], "period": pid,
                     "pool": len(pool), "cands": len(cands),
                     "hold": len(hold)})

    s = pd.Series(nav, index=dates).dropna()
    s = s[s.index <= pd.Timestamp(end)]
    if len(s) < 60:
        return None, {}, diag
    p = qfm.performance(s, rf=C.CASH_YIELD)
    res = {k: float(p[k]) for k in
           ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}
    if verbose:
        print(f"\n  [{tag}] {s.index[0].date()} ~ {s.index[-1].date()}"
              f"（{len(s)} 天）")
        print(f"    CAGR {res['cagr']:7.2%} | Sharpe {res['sharpe']:6.3f} | "
              f"MaxDD {res['max_dd']:7.2%} | Calmar {res['calmar']:5.2f}")
        print(f"    交易 {n_tr} 笔 | 卖出 {reasons}")
        if diag:
            avg_pool = np.mean([d["pool"] for d in diag])
            avg_c = np.mean([d["cands"] for d in diag])
            avg_h = np.mean([d["hold"] for d in diag])
            print(f"    月均：池 {avg_pool:.1f} 只 / 合格候选 {avg_c:.1f} / "
                  f"持仓 {avg_h:.1f}")
    return s, res, diag


# ================================================================ 主流程
def main():
    print("=" * 96)
    print("基本面（稳健版）+ RSI 择时")
    print("=" * 96)
    for k, v in PARAMS.items():
        print(f"  {k:<16} {v}")

    md, dates, code_idx, fund = load_all()

    print("\n[计算 RSI] …")
    rsi = rsi_panel(md.price, PARAMS["rsi_n"])
    print(f"  RSI{'' + str(PARAMS['rsi_n'])} 就绪，"
          f"有效值 {np.isfinite(rsi).mean():.1%}")

    print("\n" + "=" * 96)
    print("DEV 期（2018-2021）")
    print("=" * 96)
    dev_nav, dev_res, dev_diag = run(md, dates, code_idx, fund, rsi,
                                     DEV[0], DEV[1], "DEV")

    print("\n" + "=" * 96)
    print("OOS 期（2022-2026）")
    print("=" * 96)
    oos_nav, oos_res, oos_diag = run(md, dates, code_idx, fund, rsi,
                                     OOS[0], OOS[1], "OOS")

    print("\n" + "=" * 96)
    print("全期（2018-2026）")
    print("=" * 96)
    full_nav, full_res, full_diag = run(md, dates, code_idx, fund, rsi,
                                        DEV[0], OOS[1], "FULL")

    # 基准
    print("\n" + "=" * 96)
    print("基准（同期买入持有）")
    print("=" * 96)
    bench = {}
    for name in ("沪深300", "中证500", "中证1000"):
        js = [j for j, c in enumerate(md.codes) if md.names[j] == name] \
            if hasattr(md, "names") else []
        # 用指数
        key = {"沪深300": "hs300", "中证500": "csi500",
               "中证1000": "csi1000"}[name]
        b = md.bench.get(key)
        if b is None:
            continue
        # md.bench 存的是**日收益率**，需先转成净值
        rr = b[b.index >= pd.Timestamp(DEV[0])]
        if len(rr) < 100:
            continue
        s = (1.0 + rr.fillna(0.0)).cumprod()
        s = s * 100.0
        p = qfm.performance(s, rf=C.CASH_YIELD)
        bench[name] = {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd")}
        print(f"  {name:<10} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%}")

    # 保存
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    if full_nav is not None:
        pd.DataFrame({"nav": full_nav}).to_parquet(OUT_RUN / "nav.parquet")
    res_all = {"params": PARAMS, "dev": dev_res, "oos": oos_res,
               "full": full_res, "bench": bench}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res_all, ensure_ascii=False, indent=1,
                              default=float), encoding="utf-8")

    # 月度诊断
    if full_diag:
        dg = pd.DataFrame(full_diag)
        dg.to_csv(OUT_RUN / "monthly_diag.csv", index=False)
        print(f"\n  月度诊断已保存（{len(dg)} 个月）")
        print(f"    池大小: 均值 {dg['pool'].mean():.1f}, "
              f"最大 {dg['pool'].max()}")
        print(f"    合格候选: 均值 {dg['cands'].mean():.1f}")
        print(f"    平均持仓: {dg['hold'].mean():.1f} 只")
        print(f"    空仓月数: {int((dg['hold'] == 0).sum())} / {len(dg)}")

    print(f"\n→ {OUT}")

    # 结论
    print("\n" + "=" * 96)
    print("结论")
    print("=" * 96)
    if dev_res and oos_res:
        print(f"  DEV CAGR {dev_res['cagr']:.2%} → OOS CAGR "
              f"{oos_res['cagr']:.2%}"
              f"（{(oos_res['cagr']-dev_res['cagr'])*100:+.2f}pp）")
        print(f"  DEV Sharpe {dev_res['sharpe']:.3f} → "
              f"OOS Sharpe {oos_res['sharpe']:.3f}")


if __name__ == "__main__":
    main()
