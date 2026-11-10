# -*- coding: utf-8 -*-
"""方向 A 测试：每日检查突破 vs 10 日调仓。

对照组（都用同一套板块池、流动性门槛、成本口径）：
  A1 每10日调仓（现行基准）
  A2 每日检查突破，触发即买（持仓到期或不再创新高才卖）
  A3 每日检查，但只在"当日突破"时买入，卖出仍按每日检查
  A4 每日检查 + 最短持有期 5 日（避免买完立刻被甩）
  A5 每日检查 + 最短持有期 10 日
  A6 每日检查 + 出场改为"跌破 20 日均线"（给趋势空间）

关键指标：2026 年收益、能否抓住半导体 4 月那波、全期 CAGR/Sharpe/MaxDD、换手与成本。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
os.environ.setdefault("QF_DATA_RAW", str(ROOT.parent / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm, sector_rotation as qfsr  # noqa: E402

OUT = ROOT / "build" / "eval" / "sector_daily_check.json"
PANEL = ROOT / "build" / "etf" / "sector_panel.npz"


def load_pool():
    return qfsr.load_pool()


def daily_backtest(dates, codes, close, amount, sector_of, *,
                   entry_n=55, mom_n=60, top_n=3, min_amount=2.5e7,
                   min_hold=0, exit_ma=0, check_every=1, start_row=500):
    """通用回测：check_every=1 表示每日检查；exit_ma>0 用均线出场替代紧出场。

    规则：
      买入：当日突破 entry_n 日新高且成交额达标，按 momentum 排序取前 top_n（板块去重）
      卖出：min_hold 满后，若不满足出场条件则保留
            - exit_ma=0：紧出场（不再符合突破条件即卖）
            - exit_ma>0：收盘 < exit_ma 日均线则卖
      成本：换手比例 × 5bp
    """
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    mom = (cs / cs.shift(mom_n) - 1.0).to_numpy()
    ma = cs.rolling(exit_ma, min_periods=exit_ma).mean().to_numpy() if exit_ma > 0 else None

    cash = qfsr.cash_returns(dates)
    nav = np.ones(T)
    holdings: dict[int, int] = {}      # j -> 买入日行号
    trades = []
    expo = np.zeros(T)

    for t in range(1, T):
        tcost = 0.0

        # ---- 每日（或每 check_every 日）决策
        if t >= start_row and (t - start_row) % check_every == 0:
            # 1) 出场
            to_sell = []
            for j, bought in holdings.items():
                held = t - bought
                if held < min_hold:
                    continue
                c = close[t, j]
                if not np.isfinite(c):
                    to_sell.append(j)
                    continue
                if exit_ma > 0:
                    out = np.isfinite(ma[t, j]) and c < ma[t, j]
                else:
                    out = not (np.isfinite(hi[t, j]) and c > hi[t, j])
                if out:
                    to_sell.append(j)

            # 2) 买入（补空缺）
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

        # ---- 当日收益
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
    dates, codes, close, amount, sector_of = load_pool()
    semi_j = next((j for j, c in enumerate(codes) if sector_of.get(c) == "半导体"), None)
    semi = close[:, semi_j] if semi_j is not None else None
    y26 = pd.Timestamp("2026-01-01")
    S, E = "2020-01-02", "2026-10-23"

    cfg = [
        ("A1 现行：每10日调仓", dict(check_every=10)),
        ("A2 每日检查+紧出场", dict(check_every=1)),
        ("A3 每日+最短持有5日", dict(check_every=1, min_hold=5)),
        ("A4 每日+最短持有10日", dict(check_every=1, min_hold=10)),
        ("A5 每日+MA20出场", dict(check_every=1, exit_ma=20)),
        ("A6 每日+MA20出场+最短5日", dict(check_every=1, exit_ma=20, min_hold=5)),
    ]

    res = {}
    print(f"{'方案':<26}{'全期CAGR':>10}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
          f"{'2026年':>9}{'交易数':>8}{'平均仓位':>9}")
    print("-" * 96)
    for tag, kw in cfg:
        nav, trades, expo = daily_backtest(dates, codes, close, amount, sector_of, **kw)
        p_all = perf(nav, S, E)
        p_26 = perf(nav, "2026-01-01", E)
        n_tr = len(trades)
        avg_expo = float(expo[(expo.index >= S)].mean())
        res[tag] = {**p_all, "y2026": p_26.get("cagr"), "n_trades": n_tr,
                    "avg_exposure": avg_expo, "cfg": kw}
        print(f"{tag:<26}{p_all['cagr']:>9.2%}{p_all['sharpe']:>8.3f}"
              f"{p_all['max_dd']:>8.2%}{p_all['calmar']:>8.2f}"
              f"{p_26.get('cagr', 0):>8.1%}{n_tr:>8}{avg_expo:>8.1%}")

    # 半导体收益基准
    print("-" * 96)
    m26 = (dates >= y26) & np.isfinite(semi)
    idx26 = np.flatnonzero(m26)
    r_semi_26 = semi[idx26[-1]] / semi[idx26[0]] - 1 if len(idx26) > 1 else None
    print(f"{'半导体 ETF 2026 基准':<26}{'':>10}{'':>8}{'':>9}{'':>8}"
          f"{r_semi_26:>8.1%}" if r_semi_26 is not None else "")

    # ---- 检查各方案是否买过半导体、什么价位
    print("\n" + "=" * 96)
    print("各方案在 2026 年买入半导体的记录")
    print("=" * 96)
    for tag, kw in cfg:
        nav, trades, expo = daily_backtest(dates, codes, close, amount, sector_of, **kw)
        if trades.empty:
            print(f"  {tag:<26} 无交易")
            continue
        t26 = trades[(trades["date"] >= "2026-01-01") & (trades["sector"] == "半导体")]
        if t26.empty:
            print(f"  {tag:<26} 2026 年未买入半导体")
            continue
        buys = t26[t26["side"] == "buy"]
        info = []
        for r in buys.itertuples():
            # 该笔后续 20 日收益
            k = int(dates.get_indexer([pd.Timestamp(r.date)])[0])
            fwd = (semi[k + 20] / semi[k] - 1) if k + 20 < len(semi) and np.isfinite(semi[k + 20]) else np.nan
            info.append(f"{r.date} @{r.price:.3f}（20日后 {fwd*100:+.1f}%）")
        print(f"  {tag:<26} 买入 {len(buys)} 次：" + "；".join(info))
        res[tag]["semi_buys_2026"] = info

    res["semi_benchmark_2026"] = r_semi_26
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
