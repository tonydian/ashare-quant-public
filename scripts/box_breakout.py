# -*- coding: utf-8 -*-
"""箱体突破策略（Box Breakout）—— 保守默认参数版。

策略规格（用户需求 + 保守默认值）：
  选股域  ：市值前 800 名（近似沪深300+中证500，**每日动态计算，无前视**）
  频率    ：周线（周五收盘定信号）

  入场（4 条件）：
    ① 箱体 ≥ 10 周：连续 N 周收盘价都在 MA52 ±10% 内
    ② 箱体上沿 = 这 N 周的最高收盘
    ③ 突破周：收盘 > 箱体上沿，且成交量 > 过去 10 周均量 × 2.0
    ④ 确认周（下一周）：收盘 > 突破周收盘

  排序（候选多于空位）：
    综合分 = 股息率排名(730天) × 0.5 + 换手率排名(20日) × 0.5
    换手率方向：越高越好（流动性优先）

  持仓：最多 20 只，均仓；空仓资金 → 国债 ETF

  卖出四层（每日 14:20 检查，回测用日线近似）：
    ① 极端涨幅：单日涨幅 > 7% → 次日开盘止盈
    ② 冲高回落：当日最高涨幅 > 5% 且收盘涨幅 < 2% → 次日开盘卖
    ③ 周五止盈：周五收盘时持仓盈利 > 3% → 周五收盘卖（回测用当日收盘）
    ④ 跌破箱体上沿止损：收盘 < 入场时的箱体上沿 → 次日开盘卖

  T+1 严格执行：信号日 t 收盘决策 → t+1 开盘成交
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import backtest as qfbt, config as C, data as qfdata, metrics as qfm  # noqa: E402

# ================================================================ 参数
PARAMS = {
    "universe_top": 800,      # 市值前 N 名
    "ma_win": 52 * 5,         # MA52 用交易日（52 周）
    "box_band": 0.12,         # MA52 ±12%（放宽，增加候选）
    "box_min_weeks": 8,       # 箱体最少周数（放宽）
    "vol_win": 10,            # 放量对比窗口（周）
    "vol_mult": 1.5,          # 放量倍数（放宽）
    "max_pos": 20,            # 最大持仓
    "div_lookback": 730,      # 股息率回看天数
    "turn_win": 20,           # 换手率窗口（日）
    # 卖出
    "sell_surge": 0.07,       # ① 单日涨 > 7%
    "sell_fade_high": 0.05,   # ② 盘中最高 > 5%
    "sell_fade_close": 0.02,  #    收盘 < 2%
    "sell_fade_drop": 0.03,   #    从最高点回落 > 3pp
    "sell_fri_profit": 0.03,  # ③ 周五盈利 > 3%
    # 成本
    "cost_bp": 25.0,          # 往返成本（含滑点+佣金+印花）
}


# ================================================================ 数据准备
def build_panel():
    """构建回测所需面板。"""
    md = qfdata.load_market(verbose=False)
    T, N = md.T, md.N
    print(f"[data] {T} 日 × {N} 只，{md.dates[0].date()} ~ {md.dates[-1].date()}")

    close = md.price.astype(np.float64)          # 后复权收盘
    openp = md.price_open.astype(np.float64)
    high = md.high.astype(np.float64)
    low = md.low.astype(np.float64)
    vol = md.volume.astype(np.float64)
    amt = md.amount.astype(np.float64)
    mcap = md.mcap.astype(np.float64)

    # MA52（52 周 = 260 交易日）
    ma52 = pd.DataFrame(close).rolling(PARAMS["ma_win"], min_periods=200).mean().to_numpy()

    # 20 日平均换手率 = 20 日均成交额 / 市值
    turn = np.where(mcap > 0, amt / mcap, np.nan)
    turn20 = pd.DataFrame(turn).rolling(20, min_periods=10).mean().to_numpy()

    # 市值排名（每日，取前 N）
    rank = np.full((T, N), np.nan)
    for t in range(T):
        m = mcap[t]
        ok = np.isfinite(m) & (m > 0)
        if ok.sum() < 100:
            continue
        order = np.argsort(-m[ok])
        r = np.empty(ok.sum())
        r[order] = np.arange(1, ok.sum() + 1)
        rank[t, ok] = r

    # 找出每周五（周线基准）
    dts = pd.DatetimeIndex(md.dates)
    # 若某周没有周五（节假日），用该周最后一个交易日
    iso = dts.isocalendar()
    week = np.asarray(iso["week"]) if hasattr(iso, "__getitem__") else iso.week.to_numpy()
    year = np.asarray(iso["year"]) if hasattr(iso, "__getitem__") else iso.year.to_numpy()
    week_key = year * 100 + week
    weekly_last = []      # 每周最后一个交易日 index
    for i in range(1, T):
        if week_key[i] != week_key[i - 1]:
            weekly_last.append(i - 1)
    if not weekly_last or weekly_last[-1] != T - 1:
        weekly_last.append(T - 1)
    weekly_last = np.array(weekly_last, dtype=np.int64)
    print(f"[data] 周数 {len(weekly_last)}，MA52 就绪 {np.isfinite(ma52).sum():,}")

    return {
        "md": md, "T": T, "N": N, "close": close, "open": openp,
        "high": high, "low": low, "vol": vol, "amt": amt, "mcap": mcap,
        "ma52": ma52, "turn20": turn20, "rank": rank,
        "weekly_last": weekly_last, "dates": dts,
    }


def load_dividend(P):
    """加载分红数据，构造 (T, N) 的 730 天滚动每股派息面板。"""
    p = ROOT / "build" / "alt" / "dividend_raw.jsonl"
    T, N = P["T"], P["N"]
    md = P["md"]
    code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}
    date_pos = {d.normalize(): i for i, d in enumerate(P["dates"])}

    # 每日每股派息（除息日发生）
    div = np.zeros((T, N), dtype=np.float32)
    n_rec = 0
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                for r in obj["rows"]:
                    sc = str(r.get("SECURITY_CODE") or "").zfill(6)
                    j = code_idx.get(sc)
                    if j is None:
                        continue
                    exd = r.get("EX_DIVIDEND_DATE")
                    if not exd:
                        continue
                    try:
                        di = date_pos.get(pd.Timestamp(exd).normalize())
                    except Exception:
                        continue
                    if di is None:
                        continue
                    try:
                        per10 = float(r.get("PRETAX_BONUS_RMB") or 0)
                    except Exception:
                        per10 = 0.0
                    if per10 > 0:
                        div[di, j] += per10 / 10.0    # 每股
                        n_rec += 1
    print(f"[div] 分红事件 {n_rec:,} 条，覆盖 {int((div > 0).any(axis=0).sum())} 只")

    # 730 天滚动累计
    div_df = pd.DataFrame(div)
    div730 = div_df.rolling(PARAMS["div_lookback"], min_periods=30).sum().to_numpy()
    # 股息率 = 累计派息 / 当前价格
    with np.errstate(divide="ignore", invalid="ignore"):
        dy = np.where((P["close"] > 0) & (div730 > 0), div730 / P["close"], 0.0)
    dy = np.nan_to_num(dy, nan=0.0, posinf=0.0)
    print(f"[div] 股息率非零比例 {np.count_nonzero(dy) / dy.size:.2%}")
    return dy


# ================================================================ 信号
def build_signals(P, dy):
    """逐周扫描箱体突破，返回 {周序号: [(j, box_top), ...]}（按综合分排序）。"""
    close, ma52 = P["close"], P["ma52"]
    vol, rank, turn20 = P["vol"], P["rank"], P["turn20"]
    wl = P["weekly_last"]
    T, N = P["T"], P["N"]

    # 周收盘 / 周均量
    wc = close[wl]                                   # (W, N)
    wvol = np.array([vol[max(0, wl[i] - 4):wl[i] + 1].mean(axis=0)
                     for i in range(len(wl))])       # 周均量
    wma = ma52[wl]
    wrank = rank[wl]

    W = len(wl)
    box_w = PARAMS["box_min_weeks"]
    band = PARAMS["box_band"]
    vol_mult = PARAMS["vol_mult"]

    signals: dict[int, list] = {}
    n_checked = 0
    n_box = 0
    n_break = 0
    n_confirmed = 0

    for wi in range(box_w + 1, W):
        t = wl[wi]
        n_checked += 1
        # 候选：市值前 N、MA52 有效
        ok = np.isfinite(wrank[wi]) & (wrank[wi] <= PARAMS["universe_top"])
        ok &= np.isfinite(wma[wi]) & np.isfinite(wc[wi]) & (wc[wi] > 0)
        if ok.sum() == 0:
            continue

        # ---- 找箱体：向前看，连续 K 周都在 MA52±band 内
        cand = []
        for j in np.flatnonzero(ok):
            base = wma[wi - 1, j]          # 用突破前的 MA52
            if not np.isfinite(base) or base <= 0:
                continue
            lo, hi = base * (1 - band), base * (1 + band)
            # 从 wi-1 向前数，连续在区间内的周数
            k = 0
            for back in range(1, min(40, wi) + 1):
                c = wc[wi - back, j]
                if not np.isfinite(c):
                    break
                if lo <= c <= hi:
                    k += 1
                else:
                    break
            if k < box_w:
                continue
            n_box += 1
            # 箱体上沿 = 这 k 周最高收盘
            top = float(np.nanmax(wc[wi - k:wi, j]))
            if not np.isfinite(top):
                continue
            # ---- ③ 突破周：收盘 > 上沿 且 放量
            if not (wc[wi, j] > top):
                continue
            avg_v = np.nanmean(wvol[wi - PARAMS["vol_win"]:wi, j])
            if not (np.isfinite(avg_v) and avg_v > 0):
                continue
            if not (wvol[wi, j] > avg_v * vol_mult):
                continue
            n_break += 1
            cand.append((j, top))

        if not cand:
            continue

        # ---- ④ 下一周站稳确认（用 wi+1，需要未来一周 → 信号实际在 wi+1 后发出）
        if wi + 1 >= W:
            continue
        conf = []
        for j, top in cand:
            c1 = wc[wi + 1, j]
            if np.isfinite(c1) and c1 > wc[wi, j]:
                conf.append((j, top))
        if not conf:
            continue
        n_confirmed += len(conf)

        # ---- 排序：股息率 + 换手率
        scored = []
        for j, top in conf:
            d = float(dy[t, j]) if np.isfinite(dy[t, j]) else 0.0
            th = float(turn20[t, j]) if np.isfinite(turn20[t, j]) else 0.0
            scored.append((j, top, d, th))
        if len(scored) > 1:
            dd = np.array([x[2] for x in scored])
            tt = np.array([x[3] for x in scored])
            # 排名（越大越好）
            rd = pd.Series(dd).rank(ascending=True).to_numpy()
            rt = pd.Series(tt).rank(ascending=True).to_numpy()
            comp = rd * 0.5 + rt * 0.5
            idx = np.argsort(-comp)
            scored = [scored[i] for i in idx]

        # 信号在"确认周"收盘产生 → 下一交易日开盘执行
        execute_day = wl[wi + 1] + 1
        if execute_day >= T:
            continue
        signals.setdefault(execute_day, []).extend(
            [(j, top) for j, top, _, _ in scored])

    print(f"[sig] 检查 {n_checked} 周 | 箱体命中 {n_box:,} | 突破 {n_break:,} "
          f"| 确认 {n_confirmed:,} | 有效信号日 {len(signals)}")
    return signals


# ================================================================ 回测
def run_backtest(P, dy, signals, start_date="2020-01-08"):
    """逐日回测（严格 T+1：信号日收盘 → 次交易日开盘成交）。

    卖出：
      · 14:20 检查（回测近似为"收盘前判断"）→ 次日开盘卖
      · 周五止盈 → 周五收盘卖（当日）
    """
    md = P["md"]
    dates = P["dates"]
    T, N = P["T"], P["N"]
    close, openp, high, low = P["close"], P["open"], P["high"], P["low"]
    wl = dayw = P["weekly_last"]
    weekly_set = set(int(x) for x in wl)

    start = int(md.date_index(start_date))
    cash_ret = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)

    nav = np.ones(T) * 1.0
    nav[:start] = np.nan
    holdings: dict[int, dict] = {}       # j -> {shares, cost, box_top}
    pending_buy: list | None = None
    pending_sell: set | None = None
    cash = 1.0                            # 归一化：初始 1
    nav[start] = 1.0

    n_trades = 0
    n_sells = {"surge": 0, "fade": 0, "friday": 0, "stoploss": 0}

    cost = PARAMS["cost_bp"] / 1e4

    for t in range(start + 1, T):
        total = cash
        for j, h in holdings.items():
            px = close[t - 1, j]
            if np.isfinite(px):
                total += h["shares"] * px

        # ---------------- ① 执行昨日挂单（开盘）
        if pending_sell:
            for j in list(pending_sell):
                if j not in holdings:
                    continue
                h = holdings[j]
                px = openp[t, j]
                if not np.isfinite(px) or px <= 0:
                    continue
                px *= (1 - cost / 2)
                cash += h["shares"] * px
                total += 0     # 已在上面计入旧市值
                del holdings[j]
                n_trades += 1
            pending_sell = None

        if pending_buy:
            slots = PARAMS["max_pos"] - len(holdings)
            buys = pending_buy[:max(0, slots)]
            # 重新算可投金额
            total_now = cash
            for j2, h2 in holdings.items():
                px2 = openp[t, j2]
                if np.isfinite(px2):
                    total_now += h2["shares"] * px2
            per = total_now / PARAMS["max_pos"] if PARAMS["max_pos"] > 0 else 0
            for j, box_top in buys:
                px = openp[t, j]
                if not np.isfinite(px) or px <= 0:
                    continue
                budget = min(per, cash)
                if budget <= 1e-9:
                    break
                sh = budget / (px * (1 + cost / 2))
                if sh <= 0:
                    continue
                cash -= sh * px * (1 + cost / 2)
                holdings[j] = {"shares": sh, "cost": px, "box_top": box_top,
                               "buy_day": t}
                n_trades += 1
            pending_buy = None

        # ---------------- ② 现金计息（国债 ETF）+ 计当日净值
        cr = float(cash_ret[t]) if np.isfinite(cash_ret[t]) else 0.0
        cash *= (1.0 + cr)          # 闲置现金按国债收益增长
        val = cash
        for j, h in holdings.items():
            px = close[t, j]
            if np.isfinite(px):
                val += h["shares"] * px
            else:
                px2 = close[t - 1, j]
                if np.isfinite(px2):
                    val += h["shares"] * px2
        nav[t] = val

        # ---------------- ③ 收盘前决策（14:20 检查的近似）
        sells: set = set()
        for j, h in holdings.items():
            c = close[t, j]
            if not np.isfinite(c):
                continue
            pc = close[t - 1, j]
            hi = high[t, j]
            if not (np.isfinite(pc) and pc > 0):
                continue
            chg = c / pc - 1
            hi_chg = hi / pc - 1 if np.isfinite(hi) else chg

            # ① 极端涨幅
            if chg > PARAMS["sell_surge"]:
                sells.add(j)
                n_sells["surge"] += 1
                continue
            # ② 冲高回落：需满足"回落幅度够大" + 持仓满 3 日（避免刚买就卖）
            hold_days = t - h.get("buy_day", t)
            fade_drop = hi_chg - chg          # 从最高点回落的幅度
            if (hi_chg > PARAMS["sell_fade_high"]
                    and chg < PARAMS["sell_fade_close"]
                    and fade_drop > PARAMS["sell_fade_drop"]
                    and hold_days >= 3):
                sells.add(j)
                n_sells["fade"] += 1
                continue
            # ④ 跌破箱体上沿止损
            top = h.get("box_top")
            if top is not None and np.isfinite(top) and c < top:
                sells.add(j)
                n_sells["stoploss"] += 1
                continue
            # ③ 周五止盈（当日卖，收盘价）
            if t in weekly_set and c > h["cost"] * (1 + PARAMS["sell_fri_profit"]):
                sells.add(j)
                n_sells["friday"] += 1

        # 周五止盈：当日收盘卖；其余：次日开盘卖
        for j in list(sells):
            if t in weekly_set and j in holdings:
                h = holdings.pop(j)
                px = close[t, j] * (1 - cost / 2)
                cash += h["shares"] * px
                n_trades += 1
                sells.discard(j)
        if sells:
            pending_sell = sells

        # ---------------- ④ 新信号 → 次日开盘买
        if t in signals and len(holdings) < PARAMS["max_pos"]:
            cand = [(j, top) for j, top in signals[t] if j not in holdings]
            if cand:
                pending_buy = cand
            else:
                pending_buy = None

    # 空仓资金赚国债
    return nav, holdings, n_trades, n_sells


def main():
    print("=" * 92)
    print("箱体突破策略（保守默认参数）")
    print("=" * 92)
    for k, v in PARAMS.items():
        print(f"  {k:<20} {v}")

    P = build_panel()
    dy = load_dividend(P)
    sig = build_signals(P, dy)
    nav, hold, n_tr, n_sell = run_backtest(P, dy, sig)

    print("\n" + "=" * 92)
    print("回测结果")
    print("=" * 92)
    s = pd.Series(nav, index=P["dates"]).dropna()
    s = s[s.index >= "2020-01-08"]
    if len(s) < 30:
        print("数据不足")
        return
    perf = qfm.performance(s, rf=C.CASH_YIELD)
    print(f"  区间: {s.index[0].date()} ~ {s.index[-1].date()}（{len(s)} 天）")
    for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar"):
        print(f"  {k:<12} {perf[k]:.4f}" if k != "max_dd" else
              f"  {k:<12} {perf[k]:.2%}")
    print(f"\n  交易 {n_tr} 笔")
    print(f"  卖出原因: {n_sell}")
    print(f"  期末持仓 {len(hold)} 只")

    # 保存
    out = ROOT / "build" / "runs" / "box_breakout"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"nav": s}).to_parquet(out / "nav.parquet")
    json.dump({"params": PARAMS,
               "metrics": {k: (float(v) if isinstance(v,(int,float)) else str(v)) for k,v in perf.items()},
               "n_trades": n_tr, "sell_reasons": n_sell},
              open(out / "stats.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
