# -*- coding: utf-8 -*-
"""ETF 箱体突破策略（用户策略的 ETF 版）。

设计与个股版的差异（基于实测结论：个股箱体突破无 alpha，ETF 有效）：
  · 标的：38 只 ETF（宽基 + 行业 + 海外）
  · 其余规则沿用用户设计

严格防前视（backtest-guard 合规）：
  ★ 信号用周线（周五收盘后可得），成交在**下一交易日开盘**
  ★ 箱体上沿只用 t-1 及之前的数据
  ★ 股息率/换手率排序用 t 时点已知值
  ★ 现金按国债 ETF 计息
  ★ 成本 25bp 往返（ETF 免印花税，佣金+滑点）

DEV/OOS 分离：
  · DEV 期（参数只看这段）：2017-2021
  · OOS 期（从不参与调参）：2022-2026
  · 参数只调一次，OOS 只看一次
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

from qf import backtest as qfbt, config as C, metrics as qfm  # noqa: E402

# ================================================================ 参数（固定，不调优）
PARAMS = {
    "ma_weeks": 52,            # MA52（年线）
    "box_band": 0.10,          # MA52 ±10%（用户原值）
    "box_min_weeks": 10,       # 箱体 ≥10 周（用户原值）
    "vol_win": 10,             # 放量对比窗口（周）
    "vol_mult": 1.5,           # 放量倍数
    "max_pos": 10,             # 最多 10 只（ETF 池只有 38 个）
    "turn_win": 20,            # 换手率窗口（日）
    # 卖出四层
    "sell_surge": 0.07,        # ① 单日涨 >7%
    "sell_fade_high": 0.05,    # ② 盘中最高 >5%
    "sell_fade_close": 0.02,   #    收盘 <2%
    "sell_fade_drop": 0.03,    #    回落 >3pp
    "sell_fri_profit": 0.03,   # ③ 周五盈利 >3%
    # 成本
    "cost_bp": 25.0,           # ETF 无印花税：佣金 2.5bp×2 + 滑点 10bp×2 ≈ 25bp
}

DEV_END = "2021-12-31"
OOS_START = "2022-01-01"
BT_START = "2017-01-03"       # 37 只行业 ETF 大多 2017 年后上市


# ================================================================ 面板
def load_panel():
    z = np.load(ROOT / "build" / "etf" / "etf_ohlc_panel.npz", allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    meta = json.load(open(ROOT / "build" / "etf" / "etf_ohlc_meta.json",
                          encoding="utf-8"))
    P = {
        "dates": dates, "codes": codes, "meta": meta,
        "open": z["open"].astype(np.float64),
        "close": z["close"].astype(np.float64),
        "high": z["high"].astype(np.float64),
        "low": z["low"].astype(np.float64),
        "volume": z["volume"].astype(np.float64),
        "amount": z["amount"].astype(np.float64),
        "T": len(dates), "N": len(codes),
    }
    print(f"[data] {P['T']} 日 × {P['N']} 只，"
          f"{dates[0].date()} ~ {dates[-1].date()}")
    return P


def weekly_grid(P):
    """每周最后一个交易日（作为周线收盘点）。"""
    dts = P["dates"]
    iso = dts.isocalendar()
    wk = np.asarray(iso["year"]) * 100 + np.asarray(iso["week"])
    last = []
    for i in range(1, len(dts)):
        if wk[i] != wk[i - 1]:
            last.append(i - 1)
    if not last or last[-1] != len(dts) - 1:
        last.append(len(dts) - 1)
    return np.array(last, dtype=np.int64)


def build_features(P):
    """周线特征 + 日线换手率。"""
    close, high, low = P["close"], P["high"], P["low"]
    amt, vol = P["amount"], P["volume"]
    wl = weekly_grid(P)
    W = len(wl)
    N = P["N"]

    # 周收盘 / 周均量
    wc = close[wl]                                   # (W, N)
    wv = np.array([np.nanmean(vol[max(0, wl[i] - 4):wl[i] + 1], axis=0)
                   for i in range(W)])               # 周均量

    # MA52（周线，52 周均）
    ma = pd.DataFrame(wc).rolling(PARAMS["ma_weeks"], min_periods=40).mean().to_numpy()

    # 日线 20 日均成交额（作为流动性/换手代理；ETF 无市值）
    amt20 = pd.DataFrame(amt).rolling(PARAMS["turn_win"], min_periods=10).mean().to_numpy()

    return {"wl": wl, "W": W, "wc": wc, "wv": wv, "ma": ma, "amt20": amt20}


# ================================================================ 信号
def scan_box(P, F):
    """扫描箱体突破。返回 {执行日索引: [(j, box_top), ...]}。

    严格无前视：
      · 箱体与均线只用 wi-1 及之前
      · 突破判定用 wi（该周五收盘后可得）
      · 确认用 wi+1（再下一周收盘后可得）
      · 执行 = wl[wi+1] 的下一交易日开盘
    """
    wc, wv, ma = F["wc"], F["wv"], F["ma"]
    wl, W = F["wl"], F["W"]
    T = P["T"]
    band = PARAMS["box_band"]
    bw = PARAMS["box_min_weeks"]
    vm = PARAMS["vol_mult"]

    sig: dict[int, list] = {}
    n_box = n_brk = n_conf = 0

    for wi in range(bw + 2, W - 1):
        # ---- 箱体（用 wi-1 及之前，MA 也用 wi-1）
        base = ma[wi - 1]
        ok = np.isfinite(base) & (base > 0) & np.isfinite(wc[wi - 1])
        if ok.sum() < 3:
            continue
        lo, hi = base * (1 - band), base * (1 + band)

        cands = []
        for j in np.flatnonzero(ok):
            k = 0
            for back in range(1, min(45, wi) + 1):
                c = wc[wi - back, j]
                if not np.isfinite(c) or not (lo[j] <= c <= hi[j]):
                    break
                k += 1
            if k < bw:
                continue
            n_box += 1
            top = float(np.nanmax(wc[wi - k:wi, j]))   # 箱体上沿（不含本周）
            if not np.isfinite(top):
                continue
            # ---- 突破周
            if not (np.isfinite(wc[wi, j]) and wc[wi, j] > top):
                continue
            avg = np.nanmean(wv[wi - PARAMS["vol_win"]:wi, j])
            if not (np.isfinite(avg) and avg > 0):
                continue
            if not (np.isfinite(wv[wi, j]) and wv[wi, j] > avg * vm):
                continue
            n_brk += 1
            cands.append((j, top))

        if not cands:
            continue

        # ---- 确认周（wi+1）
        conf = [(j, top) for j, top in cands
                if np.isfinite(wc[wi + 1, j]) and wc[wi + 1, j] > wc[wi, j]]
        if not conf:
            continue
        n_conf += len(conf)

        exec_day = int(wl[wi + 1]) + 1
        if exec_day >= T:
            continue
        sig.setdefault(exec_day, []).extend(conf)

    print(f"[sig] 检查 {W - bw - 3} 周 | 箱体 {n_box:,} | 突破 {n_brk:,} "
          f"| 确认 {n_conf:,} | 信号日 {len(sig)}")
    return sig


# ================================================================ 排序
def rank_candidates(P, F, t: int, cands: list):
    """候选排序：成交额(流动性) + 突破强度，各 50%。

    ETF 没有股息率，用"近 20 日日均成交额"（流动性）替代
    换手率用"近 20 日成交额 / 近 250 日均成交额"（相对活跃度）
    """
    if len(cands) <= 1:
        return cands
    amt20 = F["amt20"][t]
    amt = P["amount"]
    t0 = max(0, t - 250)
    amt250 = np.nanmean(amt[t0:t + 1], axis=0)

    rows = []
    for j, top in cands:
        liq = float(amt20[j]) if np.isfinite(amt20[j]) else 0.0
        base = float(amt250[j]) if np.isfinite(amt250[j]) and amt250[j] > 0 else 1.0
        rel = liq / base
        rows.append((j, top, liq, rel))
    liq_r = pd.Series([r[2] for r in rows]).rank(ascending=True).to_numpy()
    rel_r = pd.Series([r[3] for r in rows]).rank(ascending=True).to_numpy()
    score = liq_r * 0.5 + rel_r * 0.5
    order = np.argsort(-score)
    return [(rows[i][0], rows[i][1]) for i in order]


# ================================================================ 回测
def run(P, F, sig, start: str, end: str, tag: str):
    dates = P["dates"]
    close, openp, high = P["close"], P["open"], P["high"]
    T = P["T"]
    wl_set = set(int(x) for x in F["wl"])
    cost = PARAMS["cost_bp"] / 1e4

    t_start = int(dates.searchsorted(pd.Timestamp(start)))
    t_end = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    t_end = min(t_end, T - 1)

    # 现金收益（国债 ETF）
    cash_ret = _cash_returns(P)["r"]

    nav = np.full(T, np.nan)
    hold: dict[int, dict] = {}
    cash = 1.0
    nav[t_start] = 1.0
    pend_buy = None
    pend_sell = None
    n_tr = 0
    reasons = {"surge": 0, "fade": 0, "friday": 0, "stop": 0}
    expo = np.zeros(T)

    for t in range(t_start + 1, t_end + 1):
        # ---- 执行卖（开盘）
        if pend_sell:
            for j in list(pend_sell):
                if j in hold:
                    px = openp[t, j]
                    if np.isfinite(px) and px > 0:
                        cash += hold[j]["sh"] * px * (1 - cost / 2)
                        del hold[j]
                        n_tr += 1
            pend_sell = None

        # ---- 执行买（开盘）
        if pend_buy:
            slots = PARAMS["max_pos"] - len(hold)
            if slots > 0:
                tot = cash + sum(hold[j2]["sh"] * openp[t, j2]
                                 for j2 in hold
                                 if np.isfinite(openp[t, j2]))
                per = tot / PARAMS["max_pos"]
                for j, top in pend_buy[:slots]:
                    px = openp[t, j]
                    if not (np.isfinite(px) and px > 0):
                        continue
                    budget = min(per, cash)
                    if budget <= 1e-9:
                        break
                    sh = budget / (px * (1 + cost / 2))
                    cash -= sh * px * (1 + cost / 2)
                    hold[j] = {"sh": sh, "cost": px, "top": top, "day": t}
                    n_tr += 1
            pend_buy = None

        # ---- 现金计息 + 净值
        cr = float(cash_ret[t]) if np.isfinite(cash_ret[t]) else 0.0
        cash *= (1 + cr)
        val = cash
        for j, h in hold.items():
            px = close[t, j]
            if np.isfinite(px):
                val += h["sh"] * px
        nav[t] = val
        expo[t] = 1.0 - cash / val if val > 0 else 0.0

        # ---- 收盘前决策（14:20 检查的近似）
        sells = set()
        for j, h in hold.items():
            c, pc, hi = close[t, j], close[t - 1, j], high[t, j]
            if not (np.isfinite(c) and np.isfinite(pc) and pc > 0):
                continue
            chg = c / pc - 1
            hic = (hi / pc - 1) if np.isfinite(hi) else chg
            # ① 极端涨幅
            if chg > PARAMS["sell_surge"]:
                sells.add(j); reasons["surge"] += 1; continue
            # ② 冲高回落（持仓满 3 日）
            if (hic > PARAMS["sell_fade_high"] and chg < PARAMS["sell_fade_close"]
                    and (hic - chg) > PARAMS["sell_fade_drop"]
                    and (t - h["day"]) >= 3):
                sells.add(j); reasons["fade"] += 1; continue
            # ④ 跌破箱体上沿
            if np.isfinite(h["top"]) and c < h["top"]:
                sells.add(j); reasons["stop"] += 1; continue
            # ③ 周五止盈（当日收盘卖）
            if t in wl_set and c > h["cost"] * (1 + PARAMS["sell_fri_profit"]):
                sells.add(j); reasons["friday"] += 1

        for j in list(sells):
            if t in wl_set and j in hold:
                cash += hold[j]["sh"] * close[t, j] * (1 - cost / 2)
                del hold[j]
                n_tr += 1
                sells.discard(j)
        if sells:
            pend_sell = sells

        # ---- 新信号（次日开盘买）
        if t in sig and len(hold) < PARAMS["max_pos"]:
            cands = [(j, top) for j, top in sig[t] if j not in hold]
            if cands:
                pend_buy = rank_candidates(P, F, t, cands)

    s = pd.Series(nav, index=dates).dropna()
    s = s[s.index <= pd.Timestamp(end)]
    if len(s) < 60:
        print(f"  [{tag}] 数据不足")
        return None, {}
    perf = qfm.performance(s, rf=C.CASH_YIELD)
    print(f"\n  [{tag}] {s.index[0].date()} ~ {s.index[-1].date()}"
          f"（{len(s)} 天）")
    print(f"    CAGR {perf['cagr']:7.2%} | Sharpe {perf['sharpe']:6.3f} | "
          f"MaxDD {perf['max_dd']:7.2%} | Calmar {perf['calmar']:5.2f}")
    print(f"    交易 {n_tr} 笔 | 平均仓位 {np.nanmean(expo[t_start:t_end+1]):.1%}")
    print(f"    卖出: {reasons}")
    return s, {k: float(perf[k]) for k in
               ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}


def _cash_returns(P):
    """现金收益：用国债 ETF（511010 或 bond_000012）。"""
    p = ROOT.parent / "data_extract" / "was_stock_data_ext" / "bond_000012.parquet"
    dates = P["dates"]
    if p.exists():
        d = pd.read_parquet(p)
        col = "close" if "close" in d.columns else d.columns[0]
        s = pd.Series(d[col].to_numpy(), index=pd.to_datetime(d.index))
        s = s.reindex(dates).ffill()
        r = s.pct_change().fillna(0.0).to_numpy()
        return {"r": np.nan_to_num(r, nan=0.0)}
    return {"r": np.full(len(dates), C.CASH_YIELD / 252)}


# ================================================================ 主流程
def main():
    print("=" * 92)
    print("ETF 箱体突破策略")
    print("=" * 92)
    for k, v in PARAMS.items():
        print(f"  {k:<20} {v}")

    P = load_panel()
    F = build_features(P)
    sig = scan_box(P, F)

    print("\n" + "=" * 92)
    print("DEV 期（2017-2021，仅用于确认策略是否值得做）")
    print("=" * 92)
    dev_nav, dev_perf = run(P, F, sig, BT_START, DEV_END, "DEV")

    print("\n" + "=" * 92)
    print("OOS 期（2022-2026，从不参与任何决策）")
    print("=" * 92)
    oos_nav, oos_perf = run(P, F, sig, OOS_START, "2026-12-31", "OOS")

    print("\n" + "=" * 92)
    print("全期（2017-2026）")
    print("=" * 92)
    full_nav, full_perf = run(P, F, sig, BT_START, "2026-12-31", "FULL")

    # ---- 基准对比
    print("\n" + "=" * 92)
    print("基准对比（同期买入持有）")
    print("=" * 92)
    bench = {}
    for i, c in enumerate(P["codes"]):
        if P["meta"].get(c) in ("沪深300", "中证500", "中证1000"):
            s = pd.Series(P["close"][:, i], index=P["dates"]).dropna()
            s = s[(s.index >= pd.Timestamp(BT_START))]
            if len(s) < 100:
                continue
            p2 = qfm.performance(s, rf=C.CASH_YIELD)
            bench[P["meta"][c]] = {k: float(p2[k]) for k in
                                   ("cagr", "sharpe", "max_dd")}
            print(f"  {P['meta'][c]:<10} CAGR {p2['cagr']:7.2%} "
                  f"Sharpe {p2['sharpe']:6.3f} MaxDD {p2['max_dd']:7.2%}")

    # ---- 保存
    out = ROOT / "build" / "runs" / "etf_box"
    out.mkdir(parents=True, exist_ok=True)
    if full_nav is not None:
        pd.DataFrame({"nav": full_nav}).to_parquet(out / "nav.parquet")
    res = {"params": PARAMS,
           "dev": dev_perf, "oos": oos_perf, "full": full_perf,
           "bench": bench}
    (out / "stats.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    print(f"\n→ {out}")

    # ---- 结论
    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    if oos_perf and dev_perf:
        d = oos_perf["cagr"] - dev_perf["cagr"]
        print(f"  DEV CAGR {dev_perf['cagr']:.2%} → OOS CAGR {oos_perf['cagr']:.2%}"
              f"（{d*100:+.2f}pp）")
        print(f"  DEV Sharpe {dev_perf['sharpe']:.3f} → "
              f"OOS Sharpe {oos_perf['sharpe']:.3f}")


if __name__ == "__main__":
    main()
