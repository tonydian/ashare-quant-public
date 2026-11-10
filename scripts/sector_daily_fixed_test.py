# -*- coding: utf-8 -*-
"""方向 A 重测（用修复后的数据）。

修复内容：酒 ETF(512690) 等 4 只标的的前复权因子跳变已修正（45→21 条异常，
剩余 21 条为创业板/科创 ETF 真实的 ±20% 涨跌停，合规）。

对照组：
  A1 每10日调仓（现行基准）
  A2 每日检查 + 紧出场
  A3 每日 + 最短持有 5 日
  A4 每日 + 最短持有 10 日
  A5 每日 + MA20 出场
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402

PANEL = ROOT / "build" / "etf" / "sector_panel_fixed.npz"
UNIV = ROOT / "build" / "etf" / "sector_universe.json"
OUT = ROOT / "build" / "eval" / "sector_daily_check_fixed.json"

EXCLUDE = {"纳指", "标普", "日经", "港股", "恒生科技",
           "沪深300", "中证500", "中证1000", "创业板", "科创50", "红利", "金融"}


def _blend_cash(b, g, idx):
    """按 C.CASH_BOND_WEIGHT / C.CASH_GOLD_WEIGHT 混合现金收益（黄金权重可为 0）。"""
    bw, gw = float(C.CASH_BOND_WEIGHT), float(C.CASH_GOLD_WEIGHT)
    tot = bw + gw
    if tot <= 0:
        return pd.Series(0.0, index=idx)
    r = (bw / tot) * b.reindex(idx).ffill().pct_change().fillna(0)
    if gw > 0:
        r = r + (gw / tot) * g.reindex(idx).ffill().pct_change().fillna(0)
    return r


def load():
    z = np.load(PANEL, allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    close = z["close"].astype(float)
    amount = z["amount"].astype(float)
    univ = json.load(open(UNIV, encoding="utf-8"))
    s2 = {v["code"]: k for k, v in univ["sectors"].items()}
    keep = [j for j, c in enumerate(codes) if s2.get(c) not in EXCLUDE]
    return (dates, [codes[j] for j in keep], close[:, keep], amount[:, keep],
            {codes[j]: s2[codes[j]] for j in keep})


def cash_returns(idx):
    bond = pd.read_parquet(ROOT.parent / "data_extract" / "was_stock_data" / "bond_000012.parquet")
    gold = pd.read_parquet(ROOT.parent / "data_extract" / "was_stock_data" / "gold_518880.parquet")
    b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
    g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
    return (_blend_cash(b, g, idx))


def bt(dates, codes, close, amount, sector_of, *, entry_n=55, mom_n=60, top_n=3,
       min_amount=2.5e7, min_hold=0, exit_ma=0, check_every=1, start_row=500):
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    mom = (cs / cs.shift(mom_n) - 1.0).to_numpy()
    ma = cs.rolling(exit_ma, min_periods=exit_ma).mean().to_numpy() if exit_ma > 0 else None
    cash = cash_returns(dates)
    nav = np.ones(T)
    holdings: dict[int, int] = {}
    trades = []
    expo = np.zeros(T)

    for t in range(1, T):
        tcost = 0.0
        if t >= start_row and (t - start_row) % check_every == 0:
            to_sell = []
            for j, bought in holdings.items():
                if t - bought < min_hold:
                    continue
                c = close[t, j]
                if not np.isfinite(c):
                    to_sell.append(j); continue
                if exit_ma > 0:
                    out = np.isfinite(ma[t, j]) and c < ma[t, j]
                else:
                    out = not (np.isfinite(hi[t, j]) and c > hi[t, j])
                if out:
                    to_sell.append(j)
            slots = top_n - (len(holdings) - len(to_sell))
            new_buys = []
            if slots > 0:
                best = {}
                for j in range(N):
                    if j in holdings and j not in to_sell:
                        continue
                    c = close[t, j]
                    if not np.isfinite(c):
                        continue
                    if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= min_amount):
                        continue
                    if not (np.isfinite(hi[t, j]) and c > hi[t, j]):
                        continue
                    sec = sector_of.get(codes[j], codes[j])
                    m = float(mom[t, j]) if np.isfinite(mom[t, j]) else -9.0
                    if sec not in best or m > best[sec][1]:
                        best[sec] = (j, m)
                cand = sorted(best.values(), key=lambda x: -x[1])[:slots]
                new_buys = [j for j, _ in cand]
            changed = len(to_sell) + len(new_buys)
            if changed:
                tcost = (changed / max(top_n, 1)) * 5.0 / 1e4
            for j in to_sell:
                trades.append({"date": str(dates[t].date()), "code": codes[j],
                               "sector": sector_of.get(codes[j], ""), "side": "sell",
                               "price": float(close[t, j])})
                del holdings[j]
            for j in new_buys:
                trades.append({"date": str(dates[t].date()), "code": codes[j],
                               "sector": sector_of.get(codes[j], ""), "side": "buy",
                               "price": float(close[t, j])})
                holdings[j] = t
        rs = []
        for j in holdings:
            if np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0:
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        invest = len(rs) / max(top_n, 1)
        port_r = float(np.mean(rs)) if rs else 0.0
        cr = float(cash.iloc[t]) if np.isfinite(cash.iloc[t]) else 0.0
        nav[t] = nav[t - 1] * (1 + invest * port_r + (1 - invest) * cr - tcost)
        expo[t] = invest
    return pd.Series(nav, index=dates), pd.DataFrame(trades), pd.Series(expo, index=dates)


def perf(nav, s, e):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    if len(sub) < 30:
        return {}
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


def main():
    dates, codes, close, amount, sector_of = load()
    print(f"修复后面板：{len(codes)} 只 / {len(set(sector_of.values()))} 板块")
    semi_j = next((j for j, c in enumerate(codes) if sector_of.get(c) == "半导体"), None)
    semi = close[:, semi_j]
    S, E = "2020-01-02", "2026-10-23"

    cfg = [
        ("A1 每10日调仓（基准）", dict(check_every=10)),
        ("A2 每日检查+紧出场", dict(check_every=1)),
        ("A3 每日+最短持有5日", dict(check_every=1, min_hold=5)),
        ("A4 每日+最短持有10日", dict(check_every=1, min_hold=10)),
        ("A5 每日+MA20出场", dict(check_every=1, exit_ma=20)),
        ("A6 每日+MA20+最短5日", dict(check_every=1, exit_ma=20, min_hold=5)),
    ]

    res = {}
    print("\n" + "=" * 100)
    print(f"{'方案':<24}{'全期CAGR':>10}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
          f"{'2026年':>9}{'交易数':>8}{'平均仓位':>9}")
    print("-" * 100)
    for tag, kw in cfg:
        nav, trades, expo = bt(dates, codes, close, amount, sector_of, **kw)
        pa = perf(nav, S, E)
        p26 = perf(nav, "2026-01-01", E)
        res[tag] = {**pa, "y2026": p26.get("cagr"), "y2026_dd": p26.get("max_dd"),
                    "n_trades": len(trades), "avg_exposure": float(expo[expo.index >= S].mean()),
                    "cfg": kw}
        print(f"{tag:<24}{pa['cagr']:>9.2%}{pa['sharpe']:>8.3f}{pa['max_dd']:>8.2%}"
              f"{pa['calmar']:>8.2f}{p26.get('cagr', 0):>8.1%}{len(trades):>8}"
              f"{res[tag]['avg_exposure']:>8.1%}")

    # 半导体基准
    m26 = (dates >= pd.Timestamp("2026-01-01")) & np.isfinite(semi)
    i26 = np.flatnonzero(m26)
    r_semi = semi[i26[-1]] / semi[i26[0]] - 1 if len(i26) > 1 else None
    print("-" * 100)
    if r_semi is not None:
        print(f"{'半导体ETF 2026 基准':<24}{'':>10}{'':>8}{'':>9}{'':>8}{r_semi:>8.1%}")
    res["semi_bench_2026"] = r_semi

    # 2026 年买半导体的记录
    print("\n" + "=" * 100)
    print("各方案 2026 年买入半导体记录")
    print("=" * 100)
    for tag, kw in cfg:
        nav, trades, expo = bt(dates, codes, close, amount, sector_of, **kw)
        if trades.empty:
            continue
        b = trades[(trades["date"] >= "2026-01-01") &
                   (trades["sector"] == "半导体") & (trades["side"] == "buy")]
        if b.empty:
            print(f"  {tag:<24} 未买入")
            continue
        items = []
        for r in b.itertuples():
            k = int(dates.get_indexer([pd.Timestamp(r.date)])[0])
            fwd = (semi[k + 20] / semi[k] - 1) if k + 20 < len(semi) and np.isfinite(semi[k + 20]) else np.nan
            items.append(f"{r.date}@{r.price:.3f}({fwd*100:+.0f}%)")
        print(f"  {tag:<24} {len(b)} 次: " + "; ".join(items))
        res[tag]["semi_buys"] = items

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
