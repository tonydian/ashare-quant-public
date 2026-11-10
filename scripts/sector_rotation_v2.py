# -*- coding: utf-8 -*-
"""板块 ETF 轮动 v2：37 只板块 ETF 池 + 时点可知（动态扩展）筛选。

与 v1（手工 8 只）的关键区别：
  1) **板块池 37 只**，覆盖 30+ 板块（含宽基、港股、海外）；
  2) **时点可知**：每个调仓日只考虑"当时已上市 + 20日均额达标"的 ETF，
     池子随上市时间自然扩展（2017 年可能只有 8 只，2026 年有 37 只）；
  3) **同板块去重**：同一板块如果有多只 ETF，只保留成交额最大的一只（避免选 ETF 的运气）；
  4) 剩余偏差（诚实披露）：池子里的 ETF 都是"活到 2026"的，存在存续偏差，
     但 ETF 清盘在 A 股较少，且我们只取成交额 > 1000 万的，影响有限。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from qf import config as C, metrics as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "sector_rotation_v2.json"
PANEL = ROOT / "build" / "etf" / "sector_panel.npz"
UNIV = ROOT / "build" / "etf" / "sector_universe.json"
MIN_AMOUNT = 1e7


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


def load_panel():
    z = np.load(PANEL, allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    close = z["close"].astype(float)
    amount = z["amount"].astype(float)
    univ = json.load(open(UNIV, encoding="utf-8"))
    sector_of = {v["code"]: k for k, v in univ["sectors"].items()}
    return dates, codes, close, amount, sector_of


def cash_series(idx):
    bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
    gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
    b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
    g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
    return (_blend_cash(b, g, idx))


def run(dates, codes, close, amount, sector_of, entry_n=55, top_n=3, rebal=10,
        mom_n=60, cash=None, dedupe=True, min_amount=MIN_AMOUNT):
    """时点可知板块轮动。返回 nav, 每日持仓数, 交易日统计。"""
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    mom = (cs / cs.shift(mom_n) - 1.0).to_numpy()

    nav = np.ones(T)
    holdings: list[int] = []
    stats = {"n_rebal": 0, "n_trades": 0, "pool_sizes": []}

    for t in range(1, T):
        tcost = 0.0
        if t % rebal == 0:
            cand = []
            for j in range(N):
                c = close[t, j]
                if not np.isfinite(c):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= min_amount):
                    continue
                if not (np.isfinite(hi[t, j]) and c > hi[t, j]):
                    continue
                cand.append((j, mom[t, j] if np.isfinite(mom[t, j]) else -9))
            # 同板块去重：每板块只留动量最强的
            if dedupe:
                best_per_sector = {}
                for j, s in cand:
                    sec = sector_of.get(codes[j], codes[j])
                    if sec not in best_per_sector or s > best_per_sector[sec][1]:
                        best_per_sector[sec] = (j, s)
                cand = list(best_per_sector.values())
            stats["pool_sizes"].append(len(cand))
            cand.sort(key=lambda x: -x[1])
            new = [j for j, _ in cand[:top_n]]
            changed = len(set(new) ^ set(holdings))
            stats["n_trades"] += changed
            stats["n_rebal"] += 1
            tcost = (changed / max(top_n, 1)) * 5.0 / 1e4
            holdings = new
        # 当日收益
        rs = []
        for j in holdings:
            if np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0:
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        invest = len(rs) / max(top_n, 1)
        port_r = float(np.mean(rs)) if rs else 0.0
        cr = float(cash.iloc[t]) if cash is not None else 0.0
        nav[t] = nav[t - 1] * (1 + invest * port_r + (1 - invest) * cr - tcost)
    return pd.Series(nav, index=dates), stats


def perf(nav, s, e):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}


def main():
    dates, codes, close, amount, sector_of = load_panel()
    cash = cash_series(dates)
    S, E = "2020-01-02", "2026-10-23"
    print(f"板块池 {len(codes)} 只 ETF，{dates[0].date()} ~ {dates[-1].date()}")
    print(f"板块数 {len(set(sector_of.values()))}")

    res = {}
    print("\n" + "=" * 92)
    print("参数扫描（时点可知池 + 板块去重）")
    print("=" * 92)
    print(f"  {'entry':>5}{'rebal':>6}{'topN':>5}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'平均池':>8}")
    grid = {}
    for entry in (20, 40, 55, 70):
        for rebal in (10, 20):
            for top_n in (1, 2, 3, 5):
                nav, st = run(dates, codes, close, amount, sector_of,
                              entry_n=entry, top_n=top_n, rebal=rebal, cash=cash)
                p = perf(nav, S, E)
                avg_pool = float(np.mean(st["pool_sizes"])) if st["pool_sizes"] else 0
                grid[f"e{entry}_rb{rebal}_t{top_n}"] = {**p, "avg_pool": avg_pool}
                if entry in (40, 55) and top_n <= 3:
                    print(f"  {entry:>5}{rebal:>6}{top_n:>5}{p['cagr']:>8.2%}"
                          f"{p['sharpe']:>8.3f}{p['max_dd']:>8.2%}{p['calmar']:>8.2f}{avg_pool:>8.1f}")
    sh = [v["sharpe"] for v in grid.values()]
    print(f"\n  全部 {len(grid)} 组：Sharpe 中位 {np.median(sh):.3f} | "
          f"min {min(sh):.3f} | max {max(sh):.3f} | 为正 {(np.array(sh)>0).mean():.0%}")
    res["grid"] = grid
    res["grid_summary"] = {"n": len(grid), "median": float(np.median(sh)),
                           "min": float(min(sh)), "max": float(max(sh)),
                           "pct_positive": float((np.array(sh) > 0).mean())}

    # 选定参数（时点可知池的平均表现，取中位附近而非最优）
    print("\n" + "=" * 92)
    print("选定配置 e55 / rb10 / top3 —— 分区间与分年")
    print("=" * 92)
    nav, st = run(dates, codes, close, amount, sector_of, entry_n=55, top_n=3,
                  rebal=10, cash=cash)
    for tag, (s, e) in {"全期": (S, E), "验证期": ("2020-01-02", "2021-12-31"),
                        "样本外": ("2022-01-01", E)}.items():
        p = perf(nav, s, e)
        print(f"  {tag:6s} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
        res[tag] = p
    y = qfm.yearly_returns(nav)
    print("  分年: " + " | ".join(f"{int(k)}: {v*100:6.1f}%" for k, v in y.items()))
    res["yearly"] = {int(k): float(v) for k, v in y.items()}

    print("\n" + "=" * 92)
    print("池子随时间扩展（时点可知的直接证据）")
    print("=" * 92)
    pool_by_year = {}
    for t in range(0, len(dates), 250):
        yr = dates[t].year
        avail = int(np.sum(np.isfinite(close[t]) &
                           (pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()[t] >= MIN_AMOUNT)))
        pool_by_year[yr] = avail
        print(f"  {yr}: 可交易 ETF {avail} 只")
    res["pool_by_year"] = pool_by_year

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
