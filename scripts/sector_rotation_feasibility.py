# -*- coding: utf-8 -*-
"""板块 ETF 轮动 可行性预验证（在正式建管道前）。

关键问题：
  1) 板块 ETF 趋势跟踪（Donchian55）单独表现如何？ vs 宽基 ETF 的 10.57%/0.91
  2) 板块轮动（同时持有多只、按动量排序取前 N）表现如何？
  3) 是否存在"选板块"的过拟合空间（参数高原 vs 尖峰）？
  4) 与宽基 ETF 趋势、主策略的相关性 → 是否有组合价值？

风险预警：板块 ETF 只有 8 只、且 2018 起，样本极短；轮动策略对"选哪几只板块"高度敏感，
容易过拟合。本脚本先给结论，再决定是否值得做成模块。
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
from scripts.build_etf_data import fetch  # noqa: E402

OUT = ROOT / "build" / "eval" / "sector_rotation_feasibility.json"
CACHE = ROOT / "build" / "etf" / "sector_cache.npz"

SECTORS = {
    "512880": "证券", "512480": "半导体", "512660": "军工", "512800": "银行",
    "512010": "医药", "159928": "消费", "512690": "酒", "512400": "有色",
}
COST_BP = 5.0


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


def load_sectors():
    """抓取并缓存板块 ETF 前复权日线。"""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        dates = pd.DatetimeIndex(z["dates"])
        codes = [str(c) for c in z["codes"]]
        close = z["close"].astype(float)
        amount = z["amount"].astype(float)
        print(f"[cache] {len(dates)} 日 × {len(codes)} 只")
        return dates, codes, close, amount
    frames = {}
    for code in SECTORS:
        df = fetch(code, "qfq", "2018-01-01", "2026-10-23")
        if df.empty:
            print(f"  {code} 无数据，跳过")
            continue
        frames[code] = df.set_index("date")[["close", "amount"]]
        print(f"  {code} {SECTORS[code]}: {len(df)} 日")
    dates = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
    codes = list(frames)
    close = np.full((len(dates), len(codes)), np.nan)
    amount = np.full((len(dates), len(codes)), np.nan)
    for j, c in enumerate(codes):
        f = frames[c].reindex(dates)
        close[:, j] = f["close"].to_numpy()
        amount[:, j] = f["amount"].to_numpy()
    np.savez_compressed(CACHE, dates=dates.values, codes=np.array(codes),
                        close=close.astype(np.float32), amount=amount.astype(np.float32))
    return dates, codes, close, amount


def cash_series(idx):
    bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
    gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
    b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
    g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
    return (_blend_cash(b, g, idx))


def bt_single_or_rotation(dates, codes, close, amount, mode, entry_n=55,
                          top_n=1, rebal=10, mom_n=60, cash_ret=None):
    """mode='donchian'（突破新高才买）或 'mom'（按动量排序取前 N）。"""
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    mom = (cs / cs.shift(mom_n) - 1.0).to_numpy()
    nav = np.ones(T)
    holdings: set[int] = set()
    total_trades = 0
    tcost = 0.0            # 当日交易成本（只在调仓日产生）
    for t in range(1, T):
        # 调仓日：按收盘信号，次日（下一循环）生效——这里简化在 t 日收盘调仓、t+1 生效
        if t % rebal == 0:
            cand = []
            for j in range(N):
                c = close[t, j]
                if not np.isfinite(c):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= 3e7):
                    continue
                if mode == "donchian":
                    if not (np.isfinite(hi[t, j]) and c > hi[t, j]):
                        continue
                score = mom[t, j] if np.isfinite(mom[t, j]) else -9
                cand.append((j, score))
            cand.sort(key=lambda x: -x[1])
            new = set(j for j, _ in cand[:top_n])
            changed = len(new.symmetric_difference(holdings))
            total_trades += changed
            # 当次换手比例（相对持仓数）→ 成本
            tcost = (changed / max(top_n, 1)) * COST_BP / 1e4
            holdings = new
        else:
            tcost = 0.0
        # 当日收益：持有等权，其余现金
        rs = []
        for j in holdings:
            if (np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0):
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        invest = len(rs) / max(top_n, 1)          # 实际投出比例（缺失数据则留现金）
        port_r = float(np.mean(rs)) if rs else 0.0
        cr = float(cash_ret.iloc[t]) if cash_ret is not None else 0.0
        nav[t] = nav[t - 1] * (1 + invest * port_r + (1 - invest) * cr - tcost)
    return pd.Series(nav, index=dates)


def perf(nav, s, e):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


def main():
    dates, codes, close, amount = load_sectors()
    cash = cash_series(dates)
    S, E = "2020-01-02", "2026-10-23"
    res = {}
    print(f"\n回测区间 {S} ~ {E}，{len(codes)} 只板块 ETF\n")

    print("=" * 88)
    print("1) 单只板块 ETF：Donchian55 紧出场（与宽基同规则）")
    print("=" * 88)
    single = {}
    for j, c in enumerate(codes):
        sub_close = close[:, [j]]
        sub_amt = amount[:, [j]]
        nav = bt_single_or_rotation(dates, [c], sub_close, sub_amt, "donchian",
                                    top_n=1, cash_ret=cash)
        p = perf(nav, S, E)
        single[c] = p
        print(f"  {c} {SECTORS[c]:6s} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
    res["single_sector"] = single
    avg = np.mean([v["sharpe"] for v in single.values()])
    print(f"  → 8 只平均 Sharpe {avg:.3f}")

    print()
    print("=" * 88)
    print("2) 板块轮动：按动量排序取前 N（不做突破过滤）")
    print("=" * 88)
    rot = {}
    for top_n in (1, 2, 3):
        for rebal in (10, 20):
            nav = bt_single_or_rotation(dates, codes, close, amount, "mom",
                                        top_n=top_n, rebal=rebal, cash_ret=cash)
            p = perf(nav, S, E)
            rot[f"mom_top{top_n}_rb{rebal}"] = p
            print(f"  mom top{top_n} rb{rebal:2d}: CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
                  f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
    res["rotation"] = rot

    print()
    print("=" * 88)
    print("3) 板块突破轮动：突破 55 日高 + 按动量取前 N")
    print("=" * 88)
    brk = {}
    for top_n in (1, 2, 3):
        nav = bt_single_or_rotation(dates, codes, close, amount, "donchian",
                                    top_n=top_n, rebal=10, cash_ret=cash)
        p = perf(nav, S, E)
        brk[f"donchian_top{top_n}"] = p
        print(f"  donchian top{top_n}: CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
    res["breakout_rotation"] = brk

    print()
    print("=" * 88)
    print("4) 参照：宽基 ETF 趋势（3 只）")
    print("=" * 88)
    print("  CAGR 10.57% / Sharpe 0.912 / MaxDD -10.74% / Calmar 0.98")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
