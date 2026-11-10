# -*- coding: utf-8 -*-
"""低换手月度策略：实盘信号。

每月上月末算名单，下月首个交易日开盘调仓；持有到再下个月首再次调仓。
输出与 ``scripts/daily_signal.py`` 同风格的四件套，可直接照着下单。

用法::

    # 只出目标持仓（从零建仓视角）
    python scripts/lowturn_signal.py --capital 300000

    # 带上当前持仓 → 额外输出加减仓订单
    python scripts/lowturn_signal.py --capital 300000 --holdings signals/current_holdings.csv

    # 指定历史日期回看
    python scripts/lowturn_signal.py --asof 2024-06-30

口径（与 qf/lowturn.py 回测一致）
--------------------------------
* 换手率 = 20 日均成交额 / 总市值；剔除 ST、退市、停牌、上市不满一年（250 交易日）；
* 选换手率最小 n_hold 只，等权；
* **只换变动**：退出全卖、新进补足、保留的只调整漂移（阈值 2%），不对全部持仓
  每月全卖全买 —— 后者每年白付约 11.8pp 成本（回测：净 10.02% → 0.54%）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C          # noqa: E402
from qf import data as qfdata       # noqa: E402
from qf import lowturn as lt        # noqa: E402


def _month_role(dates: pd.DatetimeIndex, t: int) -> tuple[int, str, str]:
    """保留旧名，实现已上移到 qf.lowturn.month_role（三处共用一套判定）。"""
    return lt.month_role(dates, t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=C.CAPITAL)
    ap.add_argument("--n-holdings", type=int, default=10)
    ap.add_argument("--lookback", type=int, default=lt.LOOKBACK)
    ap.add_argument("--min-list-days", type=int, default=lt.MIN_LIST_DAYS,
                    help="上市最少交易日（默认 250 ≈ 一年）")
    ap.add_argument("--min-adv20", type=float, default=0.0,
                    help="20 日均成交额下限（亿元），0 = 不设（严格按原始规则）")
    ap.add_argument("--holdings", default=None, help="当前持仓 CSV（code,shares）")
    ap.add_argument("--out", default=str(C.SIGNALS))
    ap.add_argument("--asof", default=None, help="信号基准日（默认数据最后一天）")
    args = ap.parse_args()

    t0 = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/3] 加载行情 ...")
    md = qfdata.load_market(verbose=False)
    t = md.T - 1 if args.asof is None else md.date_index(args.asof)
    asof = md.dates[t]
    turn20 = lt.turnover_panel(md, args.lookback)

    print("[2/3] 计算名单 ...")
    t_sig, target_month, role = _month_role(md.dates, t)
    min_adv = float(args.min_adv20) * 1e8
    picks = lt.select(md, turn20, t_sig, args.n_holdings, args.min_list_days, min_adv)
    if not picks:
        raise SystemExit("候选为空：检查数据日期或过滤条件")

    # 下一交易日（若 asof 是月末，下一个交易日就是调仓日）
    nxt = md.dates[t + 1] if t + 1 < md.T else None
    if role == "month_end":
        exec_note = (f"{nxt.date()} 开盘" if nxt is not None
                     else f"{target_month} 首个交易日开盘")
    elif nxt is not None:
        exec_note = f"{nxt.date()} 开盘（下月首个交易日调仓）"
    else:
        exec_note = "下一个交易日开盘"

    # ---------- 等权 + 整手 ----------
    per_val = args.capital / args.n_holdings
    rows = []
    for j in picks:
        px = float(md.price[t, j])
        j_ = j
        shares = int(np.floor(per_val / px / C.LOT) * C.LOT) if px > 0 else 0
        rows.append({
            "code": str(md.codes[j]), "name": str(md.names[j]),
            "industry": md.industry_name(int(md.industry[t, j])) if np.isfinite(md.industry[t, j]) else "",
            "board": str(md.board[j]),
            "turn20_bp": round(float(turn20[t_sig, j]) * 1e4, 1),
            "mcap_yi": round(float(md.mcap[t, j]) / 1e8, 1),
            "adv20_yi": round(float(md.amount_20d[t, j]) / 1e8, 2)
            if np.isfinite(md.amount_20d[t, j]) else np.nan,
            "price": round(px, 3), "shares": shares, "lots": shares // C.LOT,
            "target_value": round(shares * px, 2),
            "weight": round(shares * px / args.capital, 4),
        })
    tgt = pd.DataFrame(rows).sort_values("turn20_bp")
    # 整手余量补给最轻的仓位（受总预算约束，不超投）
    budget = args.capital
    for _ in range(3000):
        invested = float(tgt["target_value"].sum())
        room = budget - invested
        cand = tgt[tgt["price"] * C.LOT <= room]
        if not len(cand) or room <= 0:
            break
        k = (cand["target_value"] / per_val).idxmin()
        tgt.loc[k, "shares"] += C.LOT
        tgt.loc[k, "lots"] = tgt.loc[k, "shares"] // C.LOT
        tgt.loc[k, "target_value"] = tgt.loc[k, "shares"] * tgt.loc[k, "price"]
        tgt.loc[k, "weight"] = tgt.loc[k, "target_value"] / args.capital
    used = float(tgt["target_value"].sum())
    cash_w = max(0.0, 1.0 - used / args.capital)
    tgt.to_csv(out / "lowturn_target.csv", index=False, encoding="utf-8-sig")

    # ---------- 订单 ----------
    prev_shares: dict[int, int] = {}
    if args.holdings:
        hp = pathlib.Path(args.holdings)
        if hp.exists():
            hdf = pd.read_csv(hp, dtype={"code": str})
            hdf["code"] = hdf["code"].astype(str).str.zfill(6)
            for _, r in hdf.iterrows():
                j = md.code_index(r["code"])
                if j >= 0:
                    prev_shares[j] = int(r["shares"])
    orders = []
    tgt_map = {md.code_index(r["code"]): int(r["shares"]) for _, r in tgt.iterrows()}
    for j in sorted(set(tgt_map) | set(prev_shares)):
        cur = prev_shares.get(j, 0)
        to = tgt_map.get(j, 0)
        d = to - cur
        if d == 0:
            continue
        px = float(md.price[t, j])
        orders.append({
            "code": str(md.codes[j]), "name": str(md.names[j]),
            "side": "buy" if d > 0 else "sell",
            "shares": abs(d), "lots": abs(d) // C.LOT,
            "price_ref": round(px, 3), "amount": round(abs(d) * px, 2),
            "reason": ("initial_build" if not prev_shares else
                       ("exit" if to == 0 else ("entry" if cur == 0 else "trim"))),
        })
    odf = pd.DataFrame(orders)
    odf.to_csv(out / "lowturn_orders.csv", index=False, encoding="utf-8-sig")

    # ---------- 名单变化 ----------
    hist = lt.backtest(md, start="2020-01-02", n_hold=args.n_holdings,
                       lookback=args.lookback, min_list_days=args.min_list_days,
                       min_adv20=min_adv)
    last_picks = hist["picks"][-1] if hist["picks"] else {}
    prev_codes = set(last_picks.get("codes") or [])
    now_codes = set(str(md.codes[j]) for j in picks)
    entered = sorted(now_codes - prev_codes)
    exited = sorted(prev_codes - now_codes)
    perf = lt.performance(hist["nav"])
    perf_strict = lt.performance(hist["nav_strict"])

    meta = {
        "strategy": "lowturn_monthly",
        "asof": str(asof.date()),
        "signal_date": str(md.dates[t_sig].date()),
        "target_month": target_month,
        "status": role,
        "next_action": exec_note,
        "capital": args.capital,
        "n_holdings": len(tgt),
        "invested_value": used,
        "invested_pct": used / args.capital,
        "cash_pct": cash_w,
        "params": {"lookback": args.lookback, "min_list_days": args.min_list_days,
                   "min_adv20_yi": args.min_adv20, "n_holdings": args.n_holdings},
        "backtest": {"n_months": hist["n_months"], "cagr_net": perf["cagr"],
                     "cagr_gross": lt.performance(hist["nav_gross"])["cagr"],
                     "cagr_strict": perf_strict["cagr"],
                     "sharpe": perf["sharpe"], "max_dd": perf["max_dd"],
                     "turnover_ann": round(float(hist["turnover_ann"]), 2),
                     "keep_ratio": round(float(hist["keep_mean"]), 4)},
        "rotation": {"entered": entered, "exited": exited},
        "holdings": tgt.drop(columns=["lots"]).to_dict(orient="records"),
        "notes": [
            f"信号基于 {md.dates[t_sig].date()} 收盘数据，{exec_note}执行；",
            "只换变动：上月已持有且本期仍在名单里的不卖出，仅调整权重漂移；",
            "开盘涨停买不进、跌停卖不出时顺延一个交易日，或临时跳过该只；",
            "本策略本质是红利/大盘价值暴露，持仓集中在银行、石油、保险；",
            "历史回测不代表未来收益，不构成投资建议。",
        ],
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    json.dump(meta, open(out / "lowturn_signal.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)

    # ---------- 报告 ----------
    status_cn = ("今天本月最后一个交易日 → 今日收盘后可准备下月调仓"
                 if role == "month_end" else "本月持仓已建立，等待下月首个交易日调仓")
    L = [f"# 低换手月度策略信号 — {asof.date()}", "",
         f"- 信号基准日：{md.dates[t_sig].date()}（算名单用的最后一个交易日）",
         f"- 目标持有月份：**{target_month}**",
         f"- 当前状态：{status_cn}",
         f"- 下次动作：**{exec_note}**",
         f"- 本金 {args.capital:,.0f} 元，{len(tgt)} 只等权，已投 {used:,.0f} 元（{used/args.capital:.1%}），"
         f"现金 {cash_w:.1%}（整手余量）",
         "", f"## 目标持仓（按换手率从低到高）", "",
         "| 代码 | 名称 | 行业 | 换手率(bp) | 市值(亿) | 20日均额(亿) | 参考价 | 股数 | 手数 | 金额 | 权重 |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in tgt.iterrows():
        L.append(f"| {r['code']} | {r['name']} | {r['industry']} | {r['turn20_bp']:.1f} | "
                 f"{r['mcap_yi']:,.0f} | {r['adv20_yi']:.2f} | {r['price']:.2f} | "
                 f"{r['shares']} | {r['lots']} | {r['target_value']:,.0f} | {r['weight']:.2%} |")

    if len(odf):
        L += ["", "## 订单（相对当前持仓）" if prev_shares else "## 建仓买入清单", "",
              "| 代码 | 名称 | 方向 | 股数 | 参考价 | 金额 | 原因 |",
              "|---|---|---|---:|---:|---:|---|"]
        for _, r in odf.iterrows():
            L.append(f"| {r['code']} | {r['name']} | {'买入' if r['side']=='buy' else '卖出'} | "
                     f"{r['shares']} | {r['price_ref']:.2f} | {r['amount']:,.0f} | {r['reason']} |")

    if entered or exited:
        L += ["", "## 相对上一期名单的变化", ""]
        if entered:
            L.append(f"- 新进：{', '.join(entered)}")
        if exited:
            L.append(f"- 退出：{', '.join(exited)}")
        if not entered and not exited:
            L.append("- 与上期完全一致，本月无需交易")

    L += ["", "## 回测表现（2020-01 起，等权调仓口径）", "",
          "| 口径 | 年化 | Sharpe | 最大回撤 |", "|---|---:|---:|---:|",
          f"| 毛（不计成本） | {lt.performance(hist['nav_gross'])['cagr']:.2%} | "
          f"{lt.performance(hist['nav_gross'])['sharpe']:.2f} | "
          f"{lt.performance(hist['nav_gross'])['max_dd']:.2%} |",
          f"| **净（只换变动，49.3bp）** | **{perf['cagr']:.2%}** | **{perf['sharpe']:.2f}** | "
          f"**{perf['max_dd']:.2%}** |",
          f"| 严格全卖全买 | {perf_strict['cagr']:.2%} | {perf_strict['sharpe']:.2f} | "
          f"{perf_strict['max_dd']:.2%} |", "",
          f"- 单边年换手 {hist['turnover_ann']:.2f}x（成本 ≈ 2×换手×49.3bp ≈ "
          f"{2*hist['turnover_ann']*0.00493:.2%}/年）",
          f"- 月度名单与上期重合度 {hist['keep_mean']:.1%}，即每月实际只换 "
          f"{(1-hist['keep_mean'])*args.n_holdings:.1f} 只左右",
          "- 严格口径（每月全卖全买）收益几乎被成本吃光，**不要那样执行**。",
          "", "## 风险提示", "",
          "- 持仓集中在银行/石油/保险等大盘价值股，本质是红利暴露而非「低流动性」alpha；",
          "- 2020、2022、2023 三年接近零收益或为负，策略依赖少数年份；",
          "- 若只想获得同样的红利暴露，直接配红利 ETF 成本约 6bp，远低于本策略的 49.3bp；",
          "- 本信号为历史数据回测输出，不构成投资建议。"]
    md_path = C.REPORTS / f"低换手信号_{asof.date()}.md"
    md_path.write_text("\n".join(L), encoding="utf-8")

    print(f"\n信号基准日 {md.dates[t_sig].date()} → 目标月份 {target_month}")
    print(tgt[["code", "name", "industry", "turn20_bp", "price", "shares", "weight"]].to_string(index=False))
    print(f"\n仓位 {used/args.capital:.1%}  现金 {cash_w:.1%}  {exec_note}")
    print(f"完成，用时 {time.time()-t0:.1f}s")
    for f in ("lowturn_target.csv", "lowturn_orders.csv", "lowturn_signal.json"):
        print(f"  {out/f}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
