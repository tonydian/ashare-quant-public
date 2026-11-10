# -*- coding: utf-8 -*-
"""主回测收益桥归因：把年化收益拆到「仓位 / 择时 / 行业配置 / 行业内选股 / 成本」。

口径
----
期 = 相邻两个调仓日（主力 40 交易日一期）。每期用 t1 的**实际持仓权重**，
收益用**复权开盘价** t1+1 → t2+1（与回测"次日开盘成交"的执行时点一致）。
基准 = 选股域等权（``md.eligible[t1]`` 内等权），与模型选股域一致，
因此"选股效应"不含规模/风格偏好。

分解（逐期算术）::

    R_p = W̄·R_u          # 基础 beta：平均仓位 × 域收益
        + (W_t-W̄)·R_u    # 择时：仓位偏离平均
        + W_t·(R_s-R_u)  # 选股：持仓加权 vs 域等权（= 行业配置 + 行业内选股）
        + (1-W_t)·R_c    # 现金替代收益
        + 残差           # 成本 + 权重漂移 + 执行差异

用法::

    python scripts/attr_main_bridge.py
    python scripts/attr_main_bridge.py --run build/runs/main --out build/attr_bridge
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C          # noqa: E402
from qf import data as qfdata       # noqa: E402


def _periods(md, hold: pd.DataFrame, end_date) -> list[tuple[int, int, pd.Timestamp]]:
    dates = sorted(pd.to_datetime(hold["date"]).unique())
    out = []
    for i, d in enumerate(dates):
        t1 = int(md.dates.searchsorted(pd.Timestamp(d)))
        t2 = (int(md.dates.searchsorted(pd.Timestamp(dates[i + 1])))
              if i + 1 < len(dates) else int(md.dates.searchsorted(pd.Timestamp(end_date))))
        t2 = min(t2, md.T - 2)
        if t2 > t1:
            out.append((t1, t2, pd.Timestamp(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "build" / "runs" / "main"))
    ap.add_argument("--out", default=str(ROOT / "build" / "attr_bridge"))
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md = qfdata.load_market(verbose=False)
    hold = pd.read_parquet(run / "holdings.parquet")
    hold["date"] = pd.to_datetime(hold["date"])
    hold["code"] = hold["code"].astype(str).str.zfill(6)
    nav = pd.read_parquet(run / "nav.parquet")["nav"]
    nav.index = pd.to_datetime(nav.index)
    dret = pd.read_parquet(run / "daily_ret.parquet")["ret"]
    dret.index = pd.to_datetime(dret.index)

    code_idx = {c: j for j, c in enumerate(md.codes)}
    px = np.asarray(md.price_open, dtype=np.float64)   # 复权开盘
    industry = np.asarray(md.industry)
    ind_cats = list(getattr(md, "industry_cats", []))
    non_ind = len(ind_cats)
    n_groups = non_ind + 1

    periods = _periods(md, hold, nav.index[-1])
    print(f"期数 {len(periods)}，区间 {md.dates[periods[0][0]].date()} ~ {md.dates[periods[-1][1]].date()}")

    rows = []
    for t1, t2, d in periods:
        g = hold[hold["date"] == d]
        w = np.zeros(md.N)
        for code, wt in zip(g["code"], g["weight"]):
            j = code_idx.get(code, -1)
            if j >= 0:
                w[j] = float(wt)
        W = float(w.sum())

        elig = np.asarray(md.eligible[t1], dtype=bool)
        # holdings 的日期已是「执行日」（调仓日 +1），故期初价取 t1 当日开盘
        b1, b2 = min(t1, md.T - 1), min(t2, md.T - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = px[b2] / np.where(px[b1] > 0, px[b1], np.nan) - 1.0
        # 只剔退市/异常（跌超 99%），不能用 abs(r)<1 —— 那会把 40 日翻倍的股票当成异常丢掉
        r_ok = np.isfinite(r) & (r > -0.99)
        r_use = np.where(r_ok, r, 0.0)

        base = elig & r_ok
        R_u = float(r[base].mean()) if base.sum() >= 50 else np.nan

        R_s = float((w * r_use).sum()) / W if W > 1e-9 else np.nan
        ndays = max(t2 - t1, 1)
        R_c = (1.0 + C.CASH_YIELD) ** (ndays / 252.0) - 1.0

        # ---- 行业内 Brinson（只看股票部分，归一化到 W）
        ind_t1 = industry[t1]
        ind_t1 = np.where((ind_t1 >= 0) & (ind_t1 < non_ind), ind_t1, non_ind)
        Wk = np.zeros(non_ind)
        for j in np.flatnonzero(w > 0):
            k = ind_t1[j]
            if k < non_ind:
                Wk[k] += w[j]
        wsum_ind = Wk.sum()
        Wk_p = Wk / wsum_ind if wsum_ind > 1e-9 else Wk
        cnt = np.bincount(ind_t1[base], minlength=n_groups)[:non_ind].astype(float)
        wt_b = cnt / max(cnt.sum(), 1)

        Rb_k = np.full(non_ind, np.nan)
        Rp_k = np.full(non_ind, np.nan)
        for k in range(non_ind):
            m = base & (ind_t1 == k)
            if m.sum() >= 3:
                Rb_k[k] = float(r[m].mean())
            mk = m & (w > 0)
            if mk.sum() >= 1:
                Rp_k[k] = float((w[mk] * r_use[mk]).sum()) / w[mk].sum()
        Rb_f = np.where(np.isfinite(Rb_k), Rb_k, R_u)
        Rp_f = np.where(np.isfinite(Rp_k), Rp_k, Rb_f)
        alloc = float(np.nansum((Wk_p - wt_b) * (Rb_f - R_u)))
        sel = float(np.nansum(Wk_p * (Rp_f - Rb_f)))

        seg = dret.reindex(pd.date_range(md.dates[t1], md.dates[t2])[1:])
        R_net = float((1.0 + seg.fillna(0.0)).prod() - 1.0) if len(seg) else np.nan
        # 独立的选股度量：持仓**等权** vs 域等权（剔权重影响）
        hold_idx = np.flatnonzero(w > 0)
        R_hold_eq = float(r_use[hold_idx].mean()) if len(hold_idx) else np.nan

        rows.append(dict(
            date=d, t1=t1, t2=t2, ndays=ndays, n_pos=len(g), W=W,
            R_u=R_u, R_s=R_s, R_c=R_c, R_net=R_net, R_hold_eq=R_hold_eq,
            alloc=alloc, sel=sel,
        ))
    df = pd.DataFrame(rows)
    Wbar = float((df["W"] * df["ndays"]).sum() / df["ndays"].sum())
    df["R_beta"] = Wbar * df["R_u"]
    df["R_timing"] = (df["W"] - Wbar) * (df["R_u"] - df["R_c"])
    df["R_pick"] = df["W"] * (df["R_s"] - df["R_u"])
    df["R_cash"] = (1.0 - df["W"]) * df["R_c"]
    df["bridge_sum"] = df[["R_beta", "R_timing", "R_pick", "R_cash"]].sum(axis=1)
    df["resid"] = df["R_net"] - df["bridge_sum"]
    df["R_pick_eq"] = df["W"] * (df["R_hold_eq"] - df["R_u"])
    df["year"] = df["date"].dt.year
    df["w_days"] = df["ndays"]

    def wavg(s, w):
        m = np.isfinite(s) & np.isfinite(w)
        return float((s[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan

    ppy = 244.0 / df["ndays"].mean()
    cols = ["R_net", "R_beta", "R_timing", "R_pick", "R_pick_eq", "R_cash", "resid"]
    yr = df.groupby("year").apply(
        lambda g: pd.Series({c: wavg(g[c], g["w_days"]) for c in cols}
                            | {"W": wavg(g["W"], g["w_days"]), "n": len(g)}),
        include_groups=False)
    yr_disp = yr.copy()
    for c in cols:
        yr_disp[c] = yr[c] * ppy * 100.0
    yr_disp["W"] = yr["W"] * 100.0

    tot = pd.Series({c: wavg(df[c], df["w_days"]) for c in cols}
                    | {"W": wavg(df["W"], df["w_days"])})
    tot_disp = (tot * ppy * 100.0).copy()
    tot_disp["W"] = tot["W"] * 100.0

    # 行业拆解（区分净值货币：把 R_pick 拆成 W*alloc 与 W*sel）
    br = df.groupby("year").apply(
        lambda g: pd.Series({
            "pick": wavg(g["R_pick"], g["w_days"]) * ppy * 100,
            "alloc": wavg(g["W"] * g["alloc"], g["w_days"]) * ppy * 100,
            "sel": wavg(g["W"] * g["sel"], g["w_days"]) * ppy * 100,
        }), include_groups=False)

    pd.set_option("display.width", 200)
    print(f"\n每年约 {ppy:.2f} 期；平均股票仓位 {Wbar*100:.1f}%")
    print("\n=== 全期（算术年化，%）===")
    print(tot_disp.round(2).to_string())
    print("\n=== 分年（算术年化，%）===")
    print(yr_disp.round(2).to_string())
    print("\n=== 选股效应拆解（占净值，%）===")
    br["alloc+sel"] = br["alloc"] + br["sel"]
    print(br.round(2).to_string())

    df.to_csv(out / "bridge_periods.csv", index=False, encoding="utf-8-sig")
    yr_disp.to_csv(out / "bridge_yearly.csv", encoding="utf-8-sig")
    tot_disp.to_frame("all").T.to_csv(out / "bridge_total.csv", encoding="utf-8-sig")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
