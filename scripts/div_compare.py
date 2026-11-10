# -*- coding: utf-8 -*-
"""低换手策略 vs 红利指数/红利 ETF。

回答一个问题：低换手策略的超额，是不是「红利 beta」换了个包装？
判据不是收益率高低，而是：
  1. 与红利 ETF 的日收益相关性（>0.8 基本就是同一风险源）；
  2. 同区间年化差（净口径，两边都含成本）；
  3. 分年度胜负是否稳定（只赢某一年 = 运气）；
  4. 信息比率 IR = 超额均值/超额波动（>0.5 才算有独立信息）。

用法：
    python scripts/div_compare.py                # 默认 2020-01-02 起
    python scripts/div_compare.py --start 2018-01-02
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
from qf import lowturn              # noqa: E402
from qf import metrics as qfm       # noqa: E402
from qf.data import load_market     # noqa: E402

DIV = ROOT / "build" / "etf" / "div_panel.parquet"


def load_div(idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    d = pd.read_parquet(DIV)
    out = {}
    for code, g in d.groupby("code"):
        s = g.sort_values("date").set_index("date")["close_adj"].astype(float)
        s = s[~s.index.duplicated()]
        # 对齐到策略交易日，只在策略有、ETF 无的极少数日子前向填充
        out[code] = s.reindex(idx.union(s.index)).ffill().reindex(idx)
    return out


def stats(nav: pd.Series, label: str) -> dict:
    r = nav.pct_change().dropna()
    q = qfm.performance(nav, rf=C.CASH_YIELD)
    return {
        "name": label, "cagr": q["cagr"], "vol": q["ann_vol"], "sharpe": q["sharpe"],
        "max_dd": q["max_dd"], "calmar": q["calmar"],
        "monthly_win": float((r.resample("ME").apply(lambda x: (1 + x).prod() - 1) > 0).mean()),
        "n": len(r),
        "start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
    }


def yearly(nav: pd.Series) -> pd.Series:
    r = nav.pct_change().dropna()
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


def rel(strat: pd.Series, bench: pd.Series) -> dict:
    df = pd.concat([strat.pct_change(), bench.pct_change()], axis=1).dropna()
    df.columns = ["s", "b"]
    ex = df["s"] - df["b"]
    beta = float(np.cov(df["s"], df["b"])[0, 1] / np.var(df["b"], ddof=1))
    return {
        "corr": float(df["s"].corr(df["b"])),
        "beta": beta,
        "ex_ann": float(ex.mean() * 252),
        "te": float(ex.std() * np.sqrt(252)),
        "ir": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else None,
        "win_d": float((ex > 0).mean()),
    }


def fmt_row(s: dict) -> str:
    return (f"| {s['name']} | {s['cagr']:7.2%} | {s['vol']:7.2%} | {s['sharpe']:6.3f} | "
            f"{s['max_dd']:7.2%} | {s['calmar']:6.2f} | {s['monthly_win']:6.1%} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--n-hold", type=int, default=10)
    args = ap.parse_args()

    md = load_market(verbose=False)
    res = lowturn.backtest(md, start=args.start, n_hold=args.n_hold)
    nav = res["nav"]
    nav_g = res["nav_gross"]
    idx = nav.index
    div = load_div(idx)

    print(f"\n=== 低换手 {args.n_hold} 只 · 只换变动（净，{res['cost_bp']}bp 单边）"
          f" | {idx[0].date()} ~ {idx[-1].date()} · {res['n_months']} 个月 ===\n")

    rows = [stats(nav, "低换手（净）"), stats(nav_g, "低换手（毛，零成本）")]
    names = {"510880": "上证红利ETF", "515080": "中证红利ETF",
             "512890": "红利低波ETF", "000922": "中证红利(价格指数)",
             "H30269": "红利低波(价格指数)"}
    for code, s in div.items():
        if s.notna().sum() < 250 or s.first_valid_index() > idx[30]:
            continue
        s = s.dropna()
        rows.append(stats(s / s.iloc[0], names.get(code, code)))
    # 中证全指作参照（注意 md.bench 存的是**日收益率**，点位在 md.bench_prices）
    b = md.bench_prices["csi_all"].reindex(idx).ffill().dropna()
    rows.append(stats(b / b.iloc[0], "中证全指"))

    print("| 组合 | 年化 | 波动 | Sharpe | 最大回撤 | Calmar | 月胜率 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(fmt_row(r))

    # ---- 相对红利 ETF ----
    print("\n=== 相对基准的超额（日频，年化） ===")
    print("| 基准 | 相关 | beta | 超额 | 跟踪误差 | IR | 日胜率 | 分年度超额 |")
    print("|---|---|---|---|---|---|---|---|")
    for code in ("510880", "515080", "512890"):
        if code not in div:
            continue
        bs = div[code].dropna()
        if len(bs) < 250:
            continue
        # 只在两者都有的区间比
        a = nav.reindex(bs.index).dropna()
        bs = bs.reindex(a.index)
        r = rel(a, bs)
        ya, yb = yearly(a), yearly(bs)
        yr = (ya - yb).dropna()
        ytxt = " ".join(f"{y}:{v:+.1%}" for y, v in yr.items())
        print(f"| {names[code]} | {r['corr']:.3f} | {r['beta']:.2f} | {r['ex_ann']:+.2%} | "
              f"{r['te']:.2%} | {r['ir']:+.2f} | {r['win_d']:.1%} | {ytxt} |")

    # ---- 分年度收益 ----
    print("\n=== 分年度（含股息再投资口径） ===")
    ys = {"低换手(净)": yearly(nav), "低换手(毛)": yearly(nav_g)}
    for code in ("510880", "515080", "512890"):
        if code in div and div[code].dropna().shape[0] > 250:
            bs = div[code].dropna()
            ys[names[code]] = yearly(bs)
    df = pd.DataFrame(ys).sort_index()
    print(df.to_string(float_format=lambda x: f"{x:6.2%}"))

    out = ROOT / "build" / "eval" / "lowturn_vs_div.json"
    payload = {"start": str(idx[0].date()), "end": str(idx[-1].date()),
               "n_months": res["n_months"], "stats": rows,
               "rel": {c: rel(nav.reindex(div[c].dropna().index).dropna(),
                              div[c].dropna().reindex(nav.reindex(div[c].dropna().index).dropna().index))
                       for c in ("510880", "515080", "512890") if c in div and div[c].notna().sum() > 250}}
    import json
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
