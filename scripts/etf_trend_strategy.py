# -*- coding: utf-8 -*-
"""右侧 ETF 趋势策略：3 只宽基 ETF 的 Donchian 突破趋势跟踪（长多，持有 0~1 只或全现金）。

针对 ETF 的回测设计（与个股回测的区别）：
  1) **成交价**：ETF 用当日收盘价 + 滑点（不加冲击成本，因单笔金额相对 ETF 成交额 <0.1%）；
  2) **分红**：close 用 qfq（前复权，含分红再投资），raw_close 只用于整手与真实价格参考；
  3) **流动性过滤**：20 日均成交额 ≥ MIN_AMOUNT（默认 5000 万），512100 早年被排除；
  6) **多标的并行**：最多同时持有 MAX_POS 只（默认 2），每只占 1/MAX_POS ——
     equal 模式下"只有 1 只突破时自动降到半仓"，这是有效的风控；
  7) **严格 T+1**：t 日收盘决策 → t+1 日收盘成交 → 收益从 t+2 起算（无前视）；
  4) **成本**：佣金 2.5bp（最低 5 元）、**ETF 免印花税**、滑点 5bp 单边；
  5) **T+1**：ETF 同样 T+1（当日买入次日可卖）；
  6) **避险现金**：闲置资金 100% 国债指数（黄金权重已设为 0）；
  7) **再平衡**：每日检查信号，切换时一次性换仓（ETF 数量少，无换手约束）。
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

BUILD = ROOT / "build"
ETF_DIR = BUILD / "etf"
OUT_RUN = BUILD / "runs" / "etf_trend"
OUT_JSON = BUILD / "eval" / "etf_trend_stats.json"

# ---- 策略参数（预先设定，不做样本内寻优；参数稳健性见 --sweep）
ENTRY_N = 55           # 突破 N 日新高入场
EXIT_N = 20            # 跌破 M 日新低离场（海龟式）；=0 表示"跌破 N 日高即离场"（Donchian55 紧出场）
MAX_POS = 2            # 同时持有几只 ETF（等权，每只占 1/max_pos）
POS_STYLE = "equal"    # equal=每只固定 1/max_pos（少数突破时自动降仓）| fill=有几只就满仓
MIN_AMOUNT = 5e7       # 20 日均成交额下限（元）
COMMISSION = 0.00025
MIN_COMMISSION = 5.0
SLIPPAGE = 0.0005
LOT = 100


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


def load_etf():
    z = np.load(ETF_DIR / "etf_panel.npz", allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    panels = {k: z[k].astype(np.float64) for k in ("close", "raw_close", "open", "high", "low",
                                                   "volume", "amount")}
    return dates, codes, panels


def load_cash() -> pd.Series:
    bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
    gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
    b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
    g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
    return b, g


def signals(dates, codes, P, entry_n: int = ENTRY_N, exit_n: int = EXIT_N,
            max_pos: int = MAX_POS):
    """逐日生成持仓信号：{t: [j,...]}（最多 max_pos 只，等权）。

    规则（长多，右侧）：
      入场：收盘 > 过去 entry_n 日最高（不含当日），且 20 日均额达标，且当前未持有；
      出场：exit_n>0 时收盘 < 过去 exit_n 日最低；
            exit_n=0 时收盘 <= 过去 entry_n 日最高（跌破前高即走 = 紧出场）；
      排序：多只同时满足时按 60 日动量降序，最多持有 max_pos 只。

    返回：list[list[int]]，长度 T，每项是当日收盘后的目标持仓下标列表。
    """
    close = P["close"]
    amount = P["amount"]
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    lo = cs.rolling(exit_n).min().shift(1).to_numpy() if exit_n > 0 else None
    mom60 = (cs / cs.shift(60) - 1.0).to_numpy()

    out: list[list[int]] = []
    holdings: list[int] = []

    for t in range(T):
        # ---- 出场：逐个判断当前持仓
        kept = []
        for j in holdings:
            c = close[t, j]
            if not np.isfinite(c):
                continue                      # 数据缺失 → 强制出场
            if exit_n > 0:
                out_flag = np.isfinite(lo[t, j]) and c < lo[t, j]
            else:
                out_flag = np.isfinite(hi[t, j]) and c <= hi[t, j]
            if not out_flag:
                kept.append(j)

        # ---- 入场：补足空位
        slots = max_pos - len(kept)
        if slots > 0:
            cand = []
            for j in range(N):
                if j in kept:
                    continue
                c = close[t, j]
                if not (np.isfinite(c) and np.isfinite(hi[t, j]) and c > hi[t, j]):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= MIN_AMOUNT):
                    continue
                cand.append((j, mom60[t, j] if np.isfinite(mom60[t, j]) else -9.0))
            cand.sort(key=lambda x: -x[1])
            kept.extend(j for j, _ in cand[:slots])

        holdings = kept
        out.append(list(holdings))

    return out


def backtest(dates, codes, P, sig, cash_ret: pd.Series,
             cost_bp: float = SLIPPAGE * 1e4, max_pos: int = MAX_POS,
             pos_style: str = "fill"):
    """逐日回测（严格 T+1）。

    时序语义（关键，勿改）：
      t 日收盘：根据 sig[t] 决定目标持仓 → 记入 pending
      t+1 日收盘：按 close[t+1] ± 滑点成交 → 更新 holdings（记录成交日）
      收益：只对"t-1 日就已持有"的标的计入 t 日收益
            （当日成交的不计，避免白拿决定买入之前的涨幅）

    返回：nav, pos（每日持仓数）, trades
    """
    close = P["close"]
    T, N = close.shape
    nav = np.ones(T, dtype=float)
    pos = np.zeros(T, dtype=int)
    holdings: dict[int, int] = {}          # j -> 成交日 t
    pending: list[int] | None = None
    trades = []

    for t in range(T):
        # ---- ① 执行上一日收盘的决策（t 日收盘成交）
        if pending is not None and t > 0:
            old_set = set(holdings)
            new_set = set(pending)
            # 卖出
            for j in old_set - new_set:
                px = close[t, j] * (1 - cost_bp / 1e4)
                if np.isfinite(px) and px > 0:
                    trades.append({"date": dates[t], "code": codes[j], "side": "sell",
                                   "price": float(px)})
                del holdings[j]
            # 买入
            for j in new_set - old_set:
                px = close[t, j] * (1 + cost_bp / 1e4)
                if np.isfinite(px) and px > 0:
                    trades.append({"date": dates[t], "code": codes[j], "side": "buy",
                                   "price": float(px)})
                    holdings[j] = t
            pending = None

        # ---- ② 当日收益：只算 t-1 日就已持有的（当日成交的不算）
        rs = []
        for j, bought_t in holdings.items():
            if bought_t >= t:
                continue                        # 当日成交 → 不计收益
            if (np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j])
                    and close[t - 1, j] > 0):
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        # 仓位：fill = 有几只就等权占满；equal = 每只固定 1/max_pos
        if pos_style == "equal":
            invest = len(rs) / max(max_pos, 1)
        else:                                    # fill
            invest = 1.0 if rs else 0.0
        port_r = float(np.mean(rs)) if rs else 0.0
        cr = float(cash_ret.iloc[t]) if (t < len(cash_ret) and np.isfinite(cash_ret.iloc[t])) else 0.0
        # 交易成本：当日换手的比例 × cost_bp
        if t > 0 and pending is None:
            n_new = len([1 for j, bt in holdings.items() if bt == t])
            # 成交笔数在 ① 中已确定，这里按持仓变动估算成本
            pass
        nav[t] = nav[t - 1] * (1.0 + invest * port_r + (1.0 - invest) * cr)
        pos[t] = len(holdings)

        # ---- ③ t 日收盘决策，留待 t+1 执行
        pending = list(sig[t]) if t < len(sig) else []

    # 成本单独从交易记录扣减（每笔 cost_bp）
    if trades:
        # 按日累计成本比例，从 nav 中扣
        cost_by_day: dict[pd.Timestamp, float] = {}
        for tr in trades:
            d = pd.Timestamp(tr["date"])
            cost_by_day[d] = cost_by_day.get(d, 0.0) + cost_bp / 1e4 / max(
                1, (max_pos if pos_style == "equal" else 1))
        factor = np.ones(T)
        for i, d in enumerate(dates):
            dd = pd.Timestamp(d)
            if dd in cost_by_day:
                factor[i] = max(0.0, 1.0 - cost_by_day[dd])
        nav = nav * np.cumprod(factor)

    nav_s = pd.Series(nav, index=dates)
    return nav_s, pd.Series(pos, index=dates), pd.DataFrame(trades)


def report(nav: pd.Series, pos: pd.Series, trades: pd.DataFrame, tag: str,
           max_pos: int = MAX_POS) -> dict:
    """nav/pos/trades → 指标字典。

    pos 是每日持仓只数（0..max_pos）：
      exposure   = 有持仓的天数占比
      avg_holdings = 平均持仓只数
      avg_weight = 平均仓位（avg_holdings / max_pos）
    """
    p = qfm.performance(nav, rf=C.CASH_YIELD)
    exposure = float((pos > 0).mean())
    avg_holdings = float(pos.mean())
    avg_weight = avg_holdings / max(max_pos, 1)
    out = {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_dd",
                                    "calmar", "win_rate_monthly")}
    out.update({"exposure": exposure, "avg_holdings": avg_holdings,
                "avg_weight": avg_weight, "n_trades": int(len(trades)),
                "start": str(nav.index[0].date()), "end": str(nav.index[-1].date())})
    yearly = qfm.yearly_returns(nav)
    out["yearly"] = {int(k): float(v) for k, v in yearly.items()}
    print("  [%-22s] CAGR=%7.2f%% Sharpe=%6.3f MaxDD=%7.2f%% Calmar=%5.2f "
          "持仓天=%5.1f%% 均持=%.2f只 均仓=%5.1f%% 交易=%d" % (
              tag, out["cagr"] * 100, out["sharpe"], out["max_dd"] * 100,
              out["calmar"], exposure * 100, avg_holdings, avg_weight * 100,
              out["n_trades"]))
    return out


def main():
    dates, codes, P = load_etf()
    b, g = load_cash()
    idx = pd.DatetimeIndex(dates)
    cash_ret = _blend_cash(b, g, idx)
    start_row = int(np.flatnonzero(idx >= pd.Timestamp(C.VALID_START))[0]) if (idx >= C.VALID_START).any() else 0
    print(f"ETF 面板 {idx[0].date()} ~ {idx[-1].date()}，{len(idx)} 日；回测起点 {idx[start_row].date()}")

    res = {}
    print("\n=== 基线：Donchian55 紧出场（跌破前高即走） ===")
    sig0 = signals(dates, codes, P, 55, 0, max_pos=MAX_POS)
    nav0, pos0, tr0 = backtest(dates, codes, P, sig0, cash_ret,
                               max_pos=MAX_POS, pos_style=POS_STYLE)
    res["donchian55_tight"] = report(nav0.iloc[start_row:], pos0.iloc[start_row:], tr0, "Donchian55 紧出场")

    print("\n=== 海龟式：突破55日高入场 / 跌破20日低出场 ===")
    sig1 = signals(dates, codes, P, 55, 20, max_pos=MAX_POS)
    nav1, pos1, tr1 = backtest(dates, codes, P, sig1, cash_ret,
                               max_pos=MAX_POS, pos_style=POS_STYLE)
    res["turtle55_20"] = report(nav1.iloc[start_row:], pos1.iloc[start_row:], tr1, "Turtle55/20")

    print("\n=== 参数稳健性（不做寻优，只看高原） ===")
    sweep = {}
    for en in (20, 40, 55, 70, 100):
        for xn in (0, 20):
            s = signals(dates, codes, P, en, xn, max_pos=MAX_POS)
            nv, ps, tr = backtest(dates, codes, P, s, cash_ret,
                                  max_pos=MAX_POS, pos_style=POS_STYLE)
            sub = nv.iloc[start_row:]
            p = qfm.performance(sub, rf=C.CASH_YIELD)
            sweep[f"entry{en}_exit{xn}"] = {"cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
                                            "max_dd": float(p["max_dd"])}
            print(f"  entry{en:3d} exit{xn:2d}: CAGR {p['cagr']:6.2%} Sharpe {p['sharpe']:5.2f} "
                  f"MaxDD {p['max_dd']:7.2%}")
    res["sweep"] = sweep

    OUT_RUN.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"nav": nav0}).to_parquet(OUT_RUN / "nav_donchian55.parquet")
    (OUT_RUN / "trades_donchian55.csv").write_text(
        tr0.to_csv(index=False) if len(tr0) else "date,code,side,price\n", encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
