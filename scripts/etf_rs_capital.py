# -*- coding: utf-8 -*-
"""探测：如何提高 ETF 右侧的资金利用率（当前仅 14.6% 仓位）。

方向：
  E1 增加标的数量（宽基 3 只 → 加入行业 ETF，扩大机会集）
  E2 多标的并行 top_k（已测：top3 改善 Sharpe）
  E3 分层仓位（趋势强度决定投入比例）
  E4 空仓期的现金增强（已在用 80/20 国债黄金，可测其他配比）

重点：E1 —— 把 25 个行业板块 ETF 也纳入右侧交易机会集
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

OUT = ROOT / "build" / "eval" / "etf_rs_capital.json"

# ---- 载入宽基 + 板块 ETF（板块用修复后的数据）
zb = np.load(ROOT / "build" / "etf" / "etf_panel.npz", allow_pickle=True)
d_wide = pd.DatetimeIndex(zb["dates"])
c_wide = [str(c) for c in zb["codes"]]
cl_wide = zb["close"].astype(float)
am_wide = zb["amount"].astype(float)

zs = np.load(ROOT / "build" / "etf" / "sector_panel_fixed.npz", allow_pickle=True)
d_sec = pd.DatetimeIndex(zs["dates"])
c_sec = [str(c) for c in zs["codes"]]
cl_sec = zs["close"].astype(float)
am_sec = zs["amount"].astype(float)
univ = json.load(open(ROOT / "build" / "etf" / "sector_universe.json", encoding="utf-8"))
sector_of = {v["code"]: k for k, v in univ["sectors"].items()}

# 排除海外/宽基（避免与宽基重叠）
EXCLUDE = {"纳指", "标普", "日经", "港股", "恒生科技", "沪深300",
           "中证500", "中证1000", "创业板", "科创50", "红利", "金融"}
keep = [j for j, c in enumerate(c_sec) if sector_of.get(c) not in EXCLUDE]
c_sec = [c_sec[j] for j in keep]
cl_sec = cl_sec[:, keep]
am_sec = am_sec[:, keep]

# 合并到统一日期轴
dates = pd.DatetimeIndex(sorted(set(d_wide) | set(d_sec)))
N_w, N_s = len(c_wide), len(c_sec)
close = np.full((len(dates), N_w + N_s), np.nan)
amount = np.full((len(dates), N_w + N_s), np.nan)
close[:] = np.nan

wi = dates.get_indexer(d_wide)
close[wi, :N_w] = cl_wide
amount[wi, :N_w] = am_wide
si = dates.get_indexer(d_sec)
close[si, N_w:] = cl_sec
amount[si, N_w:] = am_sec
codes = c_wide + c_sec
labels = ["宽基" if i < N_w else sector_of.get(codes[i], "") for i in range(len(codes))]

T, N = close.shape
print(f"合并面板 {T} 日 × {N} 只（宽基 {N_w} + 板块 {N_s}）")
print(f"区间 {dates[0].date()} ~ {dates[-1].date()}")

bond = pd.read_parquet(ROOT.parent / "data_extract" / "was_stock_data" / "bond_000012.parquet")
gold = pd.read_parquet(ROOT.parent / "data_extract" / "was_stock_data" / "gold_518880.parquet")
b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
cash = _blend_cash(b, g, dates)


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


def bt(entry_n=55, top_k=1, use_wide=True, use_sector=False,
       min_amt=5e7, start_row=250):
    """通用回测：可选宽基/板块池。"""
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    mom60 = (cs / cs.shift(60) - 1.0).to_numpy()

    pool = list(range(N)) if (use_wide and use_sector) else (
        list(range(N_w)) if use_wide else list(range(N_w, N)))

    nav = np.ones(T)
    holdings = {}
    pending = None
    trades = []
    expo = np.zeros(T)

    for t in range(1, T):
        tcost = 0.0
        if pending is not None:
            old, new = set(holdings), set(pending)
            ch = len(old ^ new)
            if ch:
                tcost = (ch / max(top_k, 1)) * 5.0 / 1e4
            for j in old - new:
                del holdings[j]
                trades.append(1)
            for j in new - old:
                holdings[j] = t
                trades.append(1)
            pending = None

        rs = []
        for j in holdings:
            if np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0:
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        invest = len(rs) / max(top_k, 1)
        pr = float(np.mean(rs)) if rs else 0.0
        cr = float(cash.iloc[t]) if np.isfinite(cash.iloc[t]) else 0.0
        nav[t] = nav[t - 1] * (1 + invest * pr + (1 - invest) * cr - tcost)
        expo[t] = invest

        if t >= start_row:
            keep = [j for j in holdings
                    if np.isfinite(close[t, j]) and np.isfinite(hi[t, j]) and close[t, j] > hi[t, j]]
            cand = []
            for j in pool:
                if j in keep:
                    continue
                c = close[t, j]
                if not np.isfinite(c):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= min_amt):
                    continue
                if not (np.isfinite(hi[t, j]) and c > hi[t, j]):
                    continue
                m = float(mom60[t, j]) if np.isfinite(mom60[t, j]) else -9.0
                cand.append((j, m))
            cand.sort(key=lambda x: -x[1])
            slots = top_k - len(keep)
            new = keep + [j for j, _ in cand[:slots]]
            if set(new) != set(holdings):
                pending = {j: -1 for j in new}

    return pd.Series(nav, index=dates), trades, pd.Series(expo, index=dates)


def perf(nav, s="2020-01-02", e="2026-10-23"):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}


res = {}
print("\n" + "=" * 102)
print("资金利用率探测")
print("=" * 102)
print(f"{'方案':<38}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'仓位':>8}{'交易':>7}")
print("-" * 102)
cases = [
    ("基准：宽基3只 top1", dict(use_wide=True, use_sector=False, top_k=1)),
    ("宽基3只 top2", dict(use_wide=True, use_sector=False, top_k=2)),
    ("宽基3只 top3", dict(use_wide=True, use_sector=False, top_k=3)),
    ("宽基+板块 top1", dict(use_wide=True, use_sector=True, top_k=1)),
    ("宽基+板块 top2", dict(use_wide=True, use_sector=True, top_k=2)),
    ("宽基+板块 top3", dict(use_wide=True, use_sector=True, top_k=3)),
    ("宽基+板块 top5", dict(use_wide=True, use_sector=True, top_k=5)),
    ("仅板块 top2", dict(use_wide=False, use_sector=True, top_k=2)),
    ("仅板块 top3", dict(use_wide=False, use_sector=True, top_k=3)),
]
for tag, kw in cases:
    nav, tr, ex = bt(**kw)
    p = perf(nav)
    res[tag] = {**p, "avg_expo": float(ex[ex.index >= "2020-01-02"].mean()),
                "n_trades": len(tr)}
    print(f"{tag:<38}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}{p['max_dd']:>8.2%}"
          f"{p['calmar']:>8.2f}{res[tag]['avg_expo']:>7.1%}{len(tr):>7}")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
