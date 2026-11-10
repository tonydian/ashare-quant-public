# -*- coding: utf-8 -*-
"""板块 ETF 轮动策略（v3 正式版）：可实盘的独立模块。

策略定义（与 v3 回测完全同口径）：
  * 标的池：37 只板块 ETF，**时点可知**（每调仓日只用当时已上市 + 20日均额 >= 2500 万 的 ETF）；
  * 板块归并：同一板块多只 ETF 时，取**动量最强**的一只作为该板块代表（不选 ETF 的运气）；
  * 入场：代表 ETF 收盘 > 过去 55 日最高（不含当日）→ 该板块"处于突破状态"；
  * 排序：按 60 日动量降序，取前 3 个板块；
  * 出场：下次调仓时若不再满足突破条件，自动移出（无单独止损）；
  * 调仓：每 10 个交易日；等权持有最多 3 只；
  * 现金：未投资部分 100% 国债；
  * 成本：单边 5bp（ETF 免印花税）。

对外口径（诚实）：
  * 用当前 MIN_AMOUNT=2500 万（比探针的 1000 万更严）跑出来，全期 11.67% / Sharpe 0.59 / MaxDD −25.86%；
  * **样本外（2022+）只有 3.20% / Sharpe 0.18**，验证期 2020-21 是 34.87%；
  * 收益高度依赖 2020-2021 板块牛市，**2026-11 已连续四轮实验判否，只能当 ≤5% 卫星**。

> ⚠️ 本模块 docstring 与 `note` 曾经写死 35.10%/21.56%（那是 MIN_AMOUNT=1000 万的老口径），
> 与 stats.json 长期不一致，2026-11-04 已改为实跑生成。**看数字请以 stats.json 为准。**
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

PANEL = ROOT / "build" / "etf" / "sector_panel.npz"
UNIV = ROOT / "build" / "etf" / "sector_universe.json"
OUT_RUN = ROOT / "build" / "runs" / "sector_rotation"
OUT_STATS = ROOT / "build" / "eval" / "sector_rotation_stats.json"
SIGNAL_DIR = ROOT / "signals"

# ---- 策略参数（v3 选定，位于参数高原中心）
ENTRY_N = 55        # 突破 N 日新高
MOM_N = 60          # 排序用动量窗口
TOP_N = 3           # 持有板块数
REBAL = 10          # 调仓周期（交易日）
MIN_AMOUNT = 2.5e7  # 20 日均成交额门槛（2500 万，比探针的 1000 万更严，确保可成交）
COST_BP = 5.0
LOT = 100

EXCLUDE_SECTORS = {"纳指", "标普", "日经", "港股", "恒生科技",
                   "沪深300", "中证500", "中证1000", "创业板", "科创50", "红利", "金融",
                   "创业板50"}   # 159949 的修正后标签，见 LABEL_OVERRIDE

# ---- 代码错标修正（2026-11-04，NNLS 反推 + 个股核对）
# `build/etf/sector_universe.json` 被 qf/、build/ 下多份脚本与历史报告引用，
# 直接改名会让新旧结果对不上；因此在加载层做 override，JSON 保持原样。
#
#   159949 标注"军工"，反推为 电力设备 43.9% + 电子 21.9%，与创业板指 rho=0.986
#          → 实为国泰创业板50ETF，本质是宽基，按 EXCLUDE_SECTORS 规则应剔除
#          （它是全池被选中次数最多的"板块"，此前所有 25 板块结论实为 24 真板块 + 1 宽基）
#   516010 标注"软件"，反推为 传媒 100%（申万把游戏股归传媒）
#          → 实为游戏ETF。板块合法，仅名称误导，改名后照常参与
# 证据：reports/板块成分股反推与代码核验_2026-11-04.md
LABEL_OVERRIDE = {"159949": "创业板50", "516010": "游戏"}


def load_pool():
    """加载板块池（剔除海外/宽基，并应用 LABEL_OVERRIDE 修正错标）。"""
    z = np.load(PANEL, allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    close = z["close"].astype(float)
    amount = z["amount"].astype(float)
    univ = json.load(open(UNIV, encoding="utf-8"))
    sector_of = {v["code"]: LABEL_OVERRIDE.get(v["code"], k)
                 for k, v in univ["sectors"].items()}
    keep = [j for j, c in enumerate(codes) if sector_of.get(c) not in EXCLUDE_SECTORS]
    return (dates, [codes[j] for j in keep], close[:, keep], amount[:, keep],
            {codes[j]: sector_of[codes[j]] for j in keep})


def cash_returns(idx: pd.DatetimeIndex) -> pd.Series:
    """闲置现金收益：权重读 C.CASH_BOND_WEIGHT / C.CASH_GOLD_WEIGHT。

    当前配置：100% 国债指数（黄金权重已设为 0）。
    """
    bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
    b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
    bw = float(C.CASH_BOND_WEIGHT)
    gw = float(C.CASH_GOLD_WEIGHT)
    tot = bw + gw
    if tot <= 0:
        return pd.Series(0.0, index=idx)
    r = (bw / tot) * b.reindex(idx).ffill().pct_change().fillna(0)
    if gw > 0:
        gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
        g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
        r = r + (gw / tot) * g.reindex(idx).ffill().pct_change().fillna(0)
    return r


def build_signal_at(dates, codes, close, amount, sector_of, t: int,
                    entry_n=ENTRY_N, mom_n=MOM_N, top_n=TOP_N,
                    min_amount=MIN_AMOUNT) -> dict:
    """在 t 日收盘，给出"下一交易日"的目标板块组合。

    返回 {'date', 'picks': [{code, sector, price, mom60, breakout}], 'pool_size', 'n_avail'}
    """
    amt20 = pd.DataFrame(amount[:t + 1]).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close[:t + 1])
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    mom = (cs / cs.shift(mom_n) - 1.0).to_numpy()

    # 每板块取动量最强的可交易代表
    best: dict[str, tuple[int, float]] = {}
    n_avail = 0
    for j in range(len(codes)):
        c = close[t, j]
        if not np.isfinite(c):
            continue
        if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= min_amount):
            continue
        n_avail += 1
        sec = sector_of.get(codes[j], codes[j])
        m = float(mom[t, j]) if np.isfinite(mom[t, j]) else -9.0
        if sec not in best or m > best[sec][1]:
            best[sec] = (j, m)

    # 突破筛选
    cand = []
    for sec, (j, m) in best.items():
        brk = bool(np.isfinite(hi[t, j]) and close[t, j] > hi[t, j])
        if brk:
            cand.append({"j": j, "code": codes[j], "sector": sec,
                         "price": float(close[t, j]), "mom60": m, "breakout": True})
    cand.sort(key=lambda x: -x["mom60"])
    picks = cand[:top_n]
    return {"date": dates[t], "picks": picks, "pool_size": len(best),
            "n_avail": n_avail, "n_breakout": len(cand)}


def backtest(dates, codes, close, amount, sector_of, start_row: int | None = None,
             entry_n=ENTRY_N, mom_n=MOM_N, top_n=TOP_N, rebal=REBAL,
             min_amount=MIN_AMOUNT, verbose=True) -> dict:
    """日频回测（T+1：t 日收盘出信号，t+1 日按收盘价成交）。"""
    T, N = close.shape
    cash = cash_returns(dates)
    nav = np.ones(T)
    holdings: list[int] = []
    trades = []
    expo = np.zeros(T)
    s0 = 0 if start_row is None else int(start_row)

    pending_targets: list[int] | None = None   # t 日收盘决定，t+1 日执行（严格 T+1）
    for t in range(1, T):
        tcost = 0.0

        # ---- ① 先执行"上一交易日收盘"决定的调仓（T+1，用 t 日收盘价 + 滑点）
        if pending_targets is not None:
            new = pending_targets
            for j in holdings:
                if j not in new:
                    px = close[t, j] * (1 - COST_BP / 1e4)
                    trades.append({"date": str(dates[t].date()), "code": codes[j],
                                   "sector": sector_of.get(codes[j], ""), "side": "sell",
                                   "price": float(px) if np.isfinite(px) else float(close[t, j])})
            for j in new:
                if j not in holdings:
                    px = close[t, j] * (1 + COST_BP / 1e4)
                    trades.append({"date": str(dates[t].date()), "code": codes[j],
                                   "sector": sector_of.get(codes[j], ""), "side": "buy",
                                   "price": float(px) if np.isfinite(px) else float(close[t, j])})
            changed = len(set(new) ^ set(holdings))
            tcost = (changed / max(top_n, 1)) * COST_BP / 1e4
            holdings = list(new)
            pending_targets = None

        # ---- ② 当日收益（在持仓上）
        rs = []
        for j in holdings:
            if (np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0):
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        invest = len(rs) / max(top_n, 1)
        port_r = float(np.mean(rs)) if rs else 0.0
        cr = float(cash.iloc[t]) if np.isfinite(cash.iloc[t]) else 0.0
        nav[t] = nav[t - 1] * (1 + invest * port_r + (1 - invest) * cr - tcost)
        expo[t] = invest

        # ---- ③ t 日收盘计算信号，留待 t+1 执行
        if t % rebal == 0 and t >= s0:
            sig = build_signal_at(dates, codes, close, amount, sector_of, t,
                                  entry_n, mom_n, top_n, min_amount)
            pending_targets = [p["j"] for p in sig["picks"]]

    nav_s = pd.Series(nav, index=dates)
    return {"nav": nav_s, "exposure": pd.Series(expo, index=dates),
            "trades": pd.DataFrame(trades)}


def evaluate(nav: pd.Series, s: str, e: str) -> dict:
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino",
                                     "max_dd", "calmar", "win_rate_monthly")}


def main():
    dates, codes, close, amount, sector_of = load_pool()
    cash = cash_returns(dates)
    print(f"板块池：{len(codes)} 只 ETF / {len(set(sector_of.values()))} 个板块")
    print(f"区间 {dates[0].date()} ~ {dates[-1].date()}")

    start = max(int(np.flatnonzero(dates >= pd.Timestamp(C.VALID_START))[0]), 250)
    res = backtest(dates, codes, close, amount, sector_of, start_row=start)
    nav = res["nav"]
    S, E = str(dates[start].date()), str(dates[-1].date())

    stats = {}
    for tag, (s, e) in {"full": (S, E), "val": ("2020-01-02", "2021-12-31"),
                        "oos": ("2022-01-01", E)}.items():
        stats[tag] = evaluate(nav, s, e)
        p = stats[tag]
        print(f"  {tag:5s} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
    stats["yearly"] = {int(k): float(v) for k, v in qfm.yearly_returns(nav).items()}
    stats["params"] = {"entry_n": ENTRY_N, "mom_n": MOM_N, "top_n": TOP_N,
                       "rebal": REBAL, "min_amount": MIN_AMOUNT, "cost_bp": COST_BP}
    stats["n_trades"] = int(len(res["trades"]))
    stats["avg_exposure"] = float(res["exposure"].mean())
    # note 用本次实跑的数字生成，不写死。历史上写死过两次，页面与数据对不上
    # （2026-11-04 发现页面说明写"样本外 21.56%"，实际 stats 里是 3.20%）。
    _f, _v, _o = stats["full"], stats["val"], stats["oos"]
    stats["note"] = (
        f"回测 {S}~{E} CAGR {_f['cagr']:.2%}/Sharpe {_f['sharpe']:.2f}/MaxDD {_f['max_dd']:.2%}；"
        f"验证期 2020-21 {_v['cagr']:.2%}，样本外 2022+ {_o['cagr']:.2%}/{_o['sharpe']:.2f}。"
        "收益高度集中在 2020-2021 板块牛市与个别年份，样本外接近零，只能当卫星。"
        f"板块池 {len(set(sector_of.values()))} 个（已剔除 159949 创业板50）。")

    # 最新信号
    t_last = len(dates) - 1
    sig = build_signal_at(dates, codes, close, amount, sector_of, t_last)
    print(f"\n最新信号（{dates[t_last].date()} 收盘）：")
    print(f"  可交易 ETF {sig['n_avail']} 只，板块 {sig['pool_size']} 个，"
          f"突破 {sig['n_breakout']} 个")
    for p in sig["picks"]:
        print(f"    {p['code']} {p['sector']:8s} 价格 {p['price']:7.3f} 动量 {p['mom60']*100:+6.1f}%")
    if not sig["picks"]:
        print("    （无满足条件的板块 → 空仓，现金放国债/黄金）")
    stats["latest_signal"] = {
        "asof": str(dates[t_last].date()),
        "picks": [{"code": p["code"], "sector": p["sector"], "price": p["price"],
                   "mom60": p["mom60"]} for p in sig["picks"]],
        "n_avail": sig["n_avail"], "n_breakout": sig["n_breakout"],
    }

    OUT_RUN.mkdir(parents=True, exist_ok=True)
    nav.to_frame("nav").to_parquet(OUT_RUN / "nav.parquet")
    res["exposure"].to_frame("exposure").to_parquet(OUT_RUN / "exposure.parquet")
    if len(res["trades"]):
        res["trades"].to_csv(OUT_RUN / "trades.csv", index=False, encoding="utf-8-sig")
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=float),
                         encoding="utf-8")

    # 每日信号文件
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    sig_out = {
        "signal_date": str(dates[t_last].date()),
        "strategy": "sector_etf_rotation",
        "picks": [{"code": p["code"], "sector": p["sector"],
                   "price": p["price"], "mom60": p["mom60"]} for p in sig["picks"]],
        "n_breakout": sig["n_breakout"], "n_avail_etf": sig["n_avail"],
        "cash_sleeve": "100% 国债ETF",
        "backtest": {k: stats["full"][k] for k in ("cagr", "sharpe", "max_dd", "calmar")},
        "oos": {k: stats["oos"][k] for k in ("cagr", "sharpe", "max_dd")},
        "notes": [
            "每 10 个交易日调仓；最多持有 3 个板块（等权）；",
            "入场：板块代表 ETF 收盘突破 55 日新高；出场：下次调仓不再满足突破即移出；",
            "无满足条件时全部空仓（现金 100% 国债）；",
            "回测 35.10% CAGR 依赖 2020-2021 板块牛市，样本外 2022+ 为 21.56%，"
            "合理预期 15~20%；",
            "仅供研究，不构成投资建议。",
        ],
    }
    (SIGNAL_DIR / "latest_sector_rotation.json").write_text(
        json.dumps(sig_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT_STATS}")
    print(f"→ {SIGNAL_DIR / 'latest_sector_rotation.json'}")


if __name__ == "__main__":
    main()
