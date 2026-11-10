# -*- coding: utf-8 -*-
"""实盘信号脚本：基于最新交易日数据生成下一交易日目标持仓与整手订单（本金 30 万）。

用法:
    python scripts/daily_signal.py --model build/models/wf_v1 --capital 300000 \
        --holdings signals/current_holdings.csv --out signals

输出:
    signals/latest_target.csv   目标持仓（代码/名称/行业/板块/权重/目标金额/价格/股数/手数）
    signals/latest_orders.csv   相对当前持仓的买卖订单（若提供 current_holdings.csv）
    signals/latest_signal.json  信号元数据（日期/模型 refit/市场状态/现金比例/风险提示）
    reports/实盘信号_<日期>.md   人类可读信号说明
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C
from qf import data as qfdata
from qf import factors as qff
from qf import lowturn as lt
from qf import model as qfm
from qf import portfolio as qfport  # backtest-engine 提供


def market_regime(md, t: int) -> tuple[float, dict]:
    """趋势 regime 系数：csi_all > MA120 且 hs300 > MA60 才满仓。"""
    def above_ma(name: str, win: int) -> bool:
        s = md.bench_prices[name].iloc[:t + 1]
        if len(s) < win:
            return True
        return bool(s.iloc[-1] > s.iloc[-win:].mean())

    c1 = above_ma("csi_all", 120)
    c2 = above_ma("hs300", 60)
    coeff = (1.0 if c1 else 0.6) * (1.0 if c2 else 0.75)
    detail = {"csi_all_above_MA120": c1, "hs300_above_MA60": c2,
              "regime_coeff": coeff,
              "csi_all": float(md.bench_prices["csi_all"].iloc[t]),
              "hs300": float(md.bench_prices["hs300"].iloc[t])}
    return coeff, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(C.BUILD_MODELS / "wf_h10"))
    ap.add_argument("--capital", type=float, default=C.CAPITAL)
    ap.add_argument("--n-holdings", type=int, default=40)
    ap.add_argument("--holdings", default=None, help="当前持仓 CSV（code,shares）")
    ap.add_argument("--out", default=str(C.SIGNALS))
    ap.add_argument("--asof", default=None, help="信号日期（默认数据最后一天）")
    ap.add_argument("--no-risk-overlay", action="store_true")
    ap.add_argument("--cash-bond", type=float, default=C.CASH_BOND_WEIGHT,
                    help="闲置现金中 国债指数 权重（默认 0.8）")
    ap.add_argument("--combo", action="store_true",
                    help="额外输出 50/50 组合（主策略 + ETF 趋势）方案")
    ap.add_argument("--three", action="store_true",
                    help="额外输出三策略组合（50% 主策略 + 30% 宽基ETF + 20% 板块轮动）")
    ap.add_argument("--four", action="store_true",
                    help="额外输出四策略组合（40%% 主策略 + 30%% 低换手 + 20%% 宽基ETF + 10%% 板块轮动）；"
                         "今日推荐页优先读这个")
    ap.add_argument("--cash-gold", type=float, default=C.CASH_GOLD_WEIGHT,
                    help="闲置现金中 黄金ETF 权重（默认 0.2；给 0 即纯国债）")
    args = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载数据 ...")
    md = qfdata.load_market(verbose=True)
    t = md.T - 1 if args.asof is None else md.date_index(args.asof)
    asof_date = md.dates[t]
    print(f"    信号日: {asof_date.date()}")

    print("[2/4] 加载模型并计算最新因子 ...")
    res = qfm.load_result(args.model)
    inp = qff.FactorInputs(md)
    scores, weights, refit_date = qfm.live_scores(md, inp, res)
    print(f"    模型 refit {pd.Timestamp(refit_date).date()}，池内因子 {len(res.pool)} 个")

    # 上一期持仓权重（用于缓冲/换手控制）
    prev_weights: dict[int, float] = {}
    prev_shares: dict[int, int] = {}
    if args.holdings:
        hp = pathlib.Path(args.holdings)
        if hp.exists():
            hdf = pd.read_csv(hp, dtype={"code": str})
            hdf["code"] = hdf["code"].str.zfill(6)
            for _, row in hdf.iterrows():
                j = md.code_index(row["code"])
                if j >= 0:
                    prev_shares[j] = int(row["shares"])
                    prev_weights[j] = float(row["shares"]) * float(md.price[t, j]) / args.capital
            print(f"    读取当前持仓 {len(prev_weights)} 只")

    print("[3/4] 组合构建 ...")
    # 有当前持仓时限制每期新进 6 只；从零建仓时不做新进上限（否则只能买 6 只）
    max_new = 6 if prev_weights else None
    target = qfport.build_target(md, t, scores, prev_weights,
                                 n_holdings=args.n_holdings, max_new_entries=max_new)
    coeff, regime = (1.0, {"regime_coeff": 1.0, "note": "risk overlay disabled"})
    if not args.no_risk_overlay:
        coeff, regime = market_regime(md, t)
    target = {j: w * coeff for j, w in target.items()}
    cash_w = max(0.0, 1.0 - sum(target.values()))

    print("[4/4] 生成整手订单与报告 ...")
    rows = []
    for j, w in sorted(target.items(), key=lambda x: -x[1]):
        px = float(md.price[t, j])
        tgt_val = w * args.capital
        shares = int(np.floor(tgt_val / px / C.LOT) * C.LOT)
        if shares <= 0:
            continue
        rows.append({
            "code": md.codes[j], "name": md.names[j],
            "industry": md.industry_name(int(md.industry[t, j])),
            "board": md.board[j], "weight": w,
            "target_value": tgt_val, "price": px,
            "shares": shares, "lots": shares // C.LOT,
            "adv20_yi": float(md.amount_20d[t, j]) / 1e8 if np.isfinite(md.amount_20d[t, j]) else np.nan,
            "j": j,
        })
    # 目标仓位对应的金额（regime 系数已乘入 target），整手取整后把余量补到最轻仓位
    budget = float(sum(target.values())) * args.capital
    invested = sum(r["shares"] * r["price"] for r in rows)
    for _ in range(2000):
        under = [r for r in rows if invested + r["price"] * C.LOT <= budget * 1.001
                 and (r["shares"] + C.LOT) * r["price"] <= budget * C.MAX_WEIGHT]
        if not under:
            break
        # 每次给"当前权重（按目标值归一）最低"的补 1 手；同等时按金额从小到大
        r = min(under, key=lambda x: (x["shares"] * x["price"] / max(x["target_value"], 1e-9),
                                      x["shares"] * x["price"]))
        r["shares"] += C.LOT
        r["lots"] = r["shares"] // C.LOT
        invested += r["price"] * C.LOT
    for r in rows:
        r["target_value"] = r["shares"] * r["price"]
        r["weight"] = r["target_value"] / args.capital
    tgt = pd.DataFrame(rows)
    cash_w = max(0.0, 1.0 - tgt["target_value"].sum() / args.capital) if len(tgt) else 1.0
    tgt["cash_weight"] = cash_w
    tgt = tgt.drop(columns=["j"])
    tgt.to_csv(out / "latest_target.csv", index=False, encoding="utf-8-sig")
    used = tgt["target_value"].sum() if len(tgt) else 0.0

    # 订单（若给当前持仓）
    if prev_shares:
        orders = []
        target_shares = {md.code_index(r["code"]): int(r["shares"]) for _, r in tgt.iterrows()}
        all_codes = set(target_shares) | set(prev_shares)
        for j in all_codes:
            cur = prev_shares.get(j, 0)
            tgt_s = target_shares.get(j, 0)
            delta = tgt_s - cur
            if delta == 0:
                continue
            px = float(md.price[t, j])
            orders.append({"code": md.codes[j], "name": md.names[j],
                           "side": "buy" if delta > 0 else "sell",
                           "shares": abs(delta), "lots": abs(delta) // C.LOT,
                           "price_ref": px, "amount": abs(delta) * px,
                           "reason": "rebalance" if j in target_shares and cur else
                                     ("liquidate" if j not in target_shares else "rebalance")})
        odf = pd.DataFrame(orders)
        odf.to_csv(out / "latest_orders.csv", index=False, encoding="utf-8-sig")
    else:
        # 未提供当前持仓：输出"从零建仓"的买入清单，便于用户对照现有持仓自行调整
        odf = pd.DataFrame([{
            "code": r["code"], "name": r["name"], "side": "buy",
            "shares": int(r["shares"]), "lots": int(r["lots"]),
            "price_ref": float(r["price"]), "amount": float(r["target_value"]),
            "reason": "initial_build",
        } for _, r in tgt.iterrows()])
        odf.to_csv(out / "latest_orders.csv", index=False, encoding="utf-8-sig")

    # 闲置现金的替代资产配置（与回测口径一致：现金内部 100% 国债指数）
    cash_value = max(0.0, float(args.capital) - float(used))
    bw, gw = float(args.cash_bond), float(args.cash_gold)
    tot = bw + gw
    cash_bond_value = cash_value * (bw / tot) if tot > 0 else cash_value
    cash_gold_value = cash_value * (gw / tot) if tot > 0 else 0.0

    meta = {
        "signal_date": str(asof_date.date()), "model_refit": str(pd.Timestamp(refit_date).date()),
        "capital": args.capital, "n_holdings": len(tgt),
        "invested_value": float(used), "invested_pct": float(used / args.capital),
        "cash_pct": float(cash_w), "regime": regime,
        "cash_sleeve": {
            "bond_weight": bw / tot if tot > 0 else 0.0,
            "gold_weight": gw / tot if tot > 0 else 0.0,
            "cash_value": float(cash_value),
            "bond_value": float(cash_bond_value),
            "gold_value": float(cash_gold_value),
            "instruments": ["国债ETF(如 511010/511260)"] if tot > 0 else [],
        },
        "pool": res.pool, "weights": {k: float(v) for k, v in zip(res.pool, weights)},
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "notes": [
            "信号基于数据最后交易日收盘计算，建议下一交易日开盘执行；",
            "开盘涨停的标的买不进、跌停的卖不出，可顺延到下一交易日；",
            "单票权重上限 8%、行业上限 25%、双创合计上限 50%；",
            f"闲置现金建议 {bw/tot:.0%} 国债ETF；"
            if tot > 0 else "闲置现金保持货币基金/逆回购；",
            "本信号仅为量化研究结果，不构成投资建议。",
        ],
    }
    json.dump(meta, open(out / "latest_signal.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)

    # ---------------- 组合（50/50 / 三策略 / 四策略）
    # 四策略权重依据（对齐区间 2020-01-08~2026-10-23 的固定权重回放）：
    #   原 50/30/20 全期 8.26%/0.824，但 2023-2026 只有 5.65%/0.495；
    #   换成 40/30/20/10 → 全期 8.62%/0.923，2023-2026 8.36%/0.861（+2.7pp 年化）；
    #   低换手拿 30%（净 10.02%，与主策略相关 0.36），主策略降到 40% 仍是最大单一仓位。
    #   2026-11-05：第四腿从「板块 ETF」换成「可转债」—— 板块 OOS 仅 3.20%（衰减 91%），
    #   转债与三腿相关 0.03~0.07、净 SR 1.30。权重仍是 10%，只换腿不换比例。
    combo = None
    if args.four:
        combo = _build_combo(md, asof_date, args.capital, meta, out,
                             weights={"main": 0.40, "lowturn": 0.30, "etf": 0.20, "cb": 0.10})
    elif args.three:
        combo = _build_combo(md, asof_date, args.capital, meta, out,
                             weights={"main": 0.5, "etf": 0.3, "sector": 0.2})
    elif args.combo:
        combo = _build_combo(md, asof_date, args.capital, meta, out,
                             weights={"main": 0.5, "etf": 0.5})

    # markdown
    lines = [f"# 实盘信号 — {asof_date.date()}", "",
             f"- 模型 refit：{pd.Timestamp(refit_date).date()}；因子池 {len(res.pool)} 个",
             f"- 市场状态系数：{regime.get('regime_coeff', 1.0):.2f}"
             f"（csi_all>MA120: {regime.get('csi_all_above_MA120')}, hs300>MA60: {regime.get('hs300_above_MA60')}）",
             f"- 目标持仓：{len(tgt)} 只；目标仓位 {used/args.capital:.1%}；现金 {cash_w:.1%}",
             "", "## 目标持仓（本金 {:,.0f} 元，整手）".format(args.capital), "",
             "| 代码 | 名称 | 行业 | 板块 | 权重 | 参考价 | 股数 | 手数 | 金额 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in tgt.iterrows():
        lines.append(f"| {r['code']} | {r['name']} | {r['industry']} | {r['board']} | "
                     f"{r['weight']:.2%} | {r['price']:.2f} | {r['shares']} | {r['lots']} | "
                     f"{r['target_value']:,.0f} |")
    if len(odf):
        lines += ["", "## 相对当前持仓的订单", "",
                  "| 代码 | 名称 | 方向 | 股数 | 参考价 | 金额 |", "|---|---|---|---|---|---|"]
        for _, r in odf.iterrows():
            lines.append(f"| {r['code']} | {r['name']} | {'买入' if r['side']=='buy' else '卖出'} | "
                         f"{r['shares']} | {r['price_ref']:.2f} | {r['amount']:,.0f} |")
    lines += ["", "## 现金部分配置建议（可选）", "",
              f"- 现金合计 {cash_value:,.0f} 元（{cash_w:.1%}）：建议 "
              f"{bw/tot:.0%} 国债ETF（约 {cash_bond_value:,.0f} 元）+ {gw/tot:.0%} 黄金ETF"
              f"（约 {cash_gold_value:,.0f} 元）" if tot > 0 else
              f"- 现金合计 {cash_value:,.0f} 元（{cash_w:.1%}）：保持货币基金/逆回购即可",
              "- 现金替代资产只是把\"闲置资金\"换成低相关资产，不改变股票仓位与选股；",
              "- 黄金波动大（年化 ~18%），稳健起见可只配国债ETF，或把黄金权重降到 0~20%；",
              "", "## 风险提示", "",
              "- 本信号为历史数据回测模型输出，不构成投资建议；实盘请结合自身风险承受能力。",
              "- 执行价以次日开盘为准，开盘涨跌停/停牌可能无法成交。",
              "- 建议单笔滑点按 15bp、佣金 2.5bp（最低 5 元）、卖出印花税 5bp 估算成本。"]
    md_path = C.REPORTS / f"实盘信号_{asof_date.date()}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n目标持仓 {len(tgt)} 只，仓位 {used/args.capital:.1%}，现金 {cash_w:.1%}")
    print(tgt[["code", "name", "weight", "price", "shares", "lots"]].to_string(index=False))
    print(f"\n完成，用时 {time.time()-t0:.1f}s")
    print(f"  {out/'latest_target.csv'}")
    if len(odf):
        print(f"  {out/'latest_orders.csv'}")
    print(f"  {md_path}")




def _etf_trend_signal() -> dict:
    """读取 ETF 面板，按 Donchian55 紧出场规则给出最新信号（与回测同口径）。"""
    import json as _json

    import numpy as _np

    panel = C.BUILD / "etf" / "etf_panel.npz"
    if not panel.exists():
        return {"error": "ETF 数据不存在，请先运行 scripts/build_etf_data.py"}
    z = _np.load(panel, allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    close = z["close"].astype(float)
    amount = z["amount"].astype(float)
    meta_p = C.BUILD / "etf" / "etf_meta.json"
    names = {}
    if meta_p.exists():
        m = _json.load(open(meta_p, encoding="utf-8"))
        names = {k: v.get("name", k) for k, v in (m.get("etfs") or {}).items()}

    n = min(len(dates), 400)
    cs = pd.DataFrame(close[-n:])
    amt20 = pd.DataFrame(amount[-n:]).rolling(20, min_periods=10).mean().to_numpy()
    hi = cs.rolling(55).max().shift(1).to_numpy()
    mom60 = (cs / cs.shift(60) - 1.0).to_numpy()
    t = len(cs) - 1
    cand = []
    for j, code in enumerate(codes):
        c = cs.iloc[t, j]
        if not (_np.isfinite(c) and _np.isfinite(hi[t, j])):
            continue
        entry = bool(c > hi[t, j]) and bool(_np.isfinite(amt20[t, j]) and amt20[t, j] >= 5e7)
        cand.append({"code": code, "name": names.get(code, code),
                     "price": float(c), "entry": entry,
                     "mom60": float(mom60[t, j]) if _np.isfinite(mom60[t, j]) else None,
                     "amount_20d": float(amt20[t, j]) if _np.isfinite(amt20[t, j]) else None,
                     "dist_to_high": float(c / hi[t, j] - 1.0) if _np.isfinite(hi[t, j]) else None})
    cand.sort(key=lambda x: -(x["mom60"] if x["mom60"] is not None else -9))
    # 与回测同口径（scripts/etf_trend_strategy.py：MAX_POS=2、POS_STYLE="equal"）：
    # 最多同时持 2 只、每只固定占该腿 1/2；只有 1 只突破时自动降到半仓。
    picks = [c for c in cand if c["entry"]][:2]
    return {"asof": str(dates[-1].date()), "universe": cand,
            "pick": picks[0] if picks else None, "picks": picks,
            "rule": "收盘突破 55 日新高入场；跌破前高即离场；"
                    "最多同时持 2 只、每只占该腿 1/2（仅 1 只突破时自动半仓）"}


def _sector_rotation_signal() -> dict:
    """读取板块轮动最新信号（qf/sector_rotation.py 产出）。"""
    import json as _json

    p = C.BUILD / "eval" / "sector_rotation_stats.json"
    sig_p = C.SIGNALS / "latest_sector_rotation.json"
    if not p.exists() or not sig_p.exists():
        return {"error": "板块轮动未运行，请先执行 python qf/sector_rotation.py"}
    st = _json.load(open(p, encoding="utf-8"))
    sig = _json.load(open(sig_p, encoding="utf-8"))
    return {"asof": sig.get("signal_date"), "picks": sig.get("picks") or [],
            "n_breakout": sig.get("n_breakout"), "n_avail": sig.get("n_avail_etf"),
            "backtest": sig.get("backtest"), "oos": sig.get("oos"),
            "rule": "板块代表 ETF 突破 55 日新高 + 60 日动量前 3，每 10 日调仓"}


def _lowturn_signal(md, n_hold: int = 10) -> dict:
    """低换手月度策略最新名单（与 qf/lowturn.py、scripts/lowturn_signal.py 同一套选股）。

    月末最后一个交易日 → 出下月名单（次日开盘建仓）；
    月中 → 出本月名单（已在持有，等下月首个交易日调仓）。
    """
    t = md.T - 1
    turn20 = lt.turnover_panel(md, lt.LOOKBACK)
    t_sig, target_month, role = lt.month_role(md.dates, t)
    picks = lt.select(md, turn20, t_sig, n_hold, lt.MIN_LIST_DAYS, 0.0)
    rows = []
    for j in picks:
        px = float(md.price[t, j]) if np.isfinite(md.price[t, j]) else None
        rows.append({
            "code": str(md.codes[j]), "name": str(md.names[j]), "price": px,
            "sector": (md.industry_name(int(md.industry[t, j]))
                       if np.isfinite(md.industry[t, j]) else ""),
            "turn20_bp": float(turn20[t_sig, j] * 1e4) if np.isfinite(turn20[t_sig, j]) else None,
            "mcap_yi": float(md.mcap[t_sig, j] / 1e8) if np.isfinite(md.mcap[t_sig, j]) else None,
        })
    rows.sort(key=lambda r: -(r["turn20_bp"] or 0))
    return {
        "asof": str(pd.Timestamp(md.dates[t]).date()),
        "signal_date": str(pd.Timestamp(md.dates[t_sig]).date()),
        "target_month": target_month, "status": role, "picks": rows,
        "n_picks": len(rows),
        "rule": ("月末算过去 20 日平均换手率 → 剔 ST/退市/停牌/上市不满一年 → "
                 "取换手率最低 10 只等权，月频调仓，只换变动"),
    }


def _size_equal_legs(picks: list, budget: float, overshoot: float = 0.10) -> tuple[list, float]:
    """等额分预算 + 整手取整 + 余额贪心补回。

    2026-11-05：原来直接 int(budget/n/lot)*lot，10 只 × 9,000 元在 100 股整手下
    每只都浪费几百到三千，合计 8.2%（2.5% 本金）躺在现金里。这里把余量补回
    缺口最大的标的，但单只最多超配 overshoot（默认 10%），避免为填坑把某只买成 1.2 倍。
    """
    n = len(picks)
    if n == 0 or budget <= 0:
        return [], 0.0
    each = budget / n
    plan = []
    for pk in picks:
        px = float(pk.get("price") or 0.0)
        lot_px = px * C.LOT
        sh = int(each / lot_px) * C.LOT if lot_px > 0 else 0
        plan.append({"pk": pk, "px": px, "lot_px": lot_px, "shares": sh})
    used = sum(p["shares"] * p["px"] for p in plan)
    cap_each = each * (1.0 + overshoot)
    while True:
        cand = [p for p in plan
                if p["lot_px"] > 0 and used + p["lot_px"] <= budget
                and (p["shares"] + C.LOT) * p["px"] <= cap_each]
        if not cand:
            break
        p = max(cand, key=lambda x: each - x["shares"] * x["px"])
        p["shares"] += C.LOT
        used += p["lot_px"]
    legs = [{**p["pk"], "shares": p["shares"], "lots": p["shares"] // C.LOT,
             "target_value": p["shares"] * p["px"]} for p in plan]
    return legs, used


def _size_etf_leg(picks: list, budget: float, max_pos: int = 2) -> tuple[list, float]:
    """ETF 趋势腿：每只**固定**占 1/max_pos 预算，不是"有几只就均分满仓"。

    与回测 `scripts/etf_trend_strategy.py`（MAX_POS=2, POS_STYLE="equal"）同口径：
    只有 1 只突破时该腿自动半仓。2026-11-05 前实盘只取 1 只且投满整腿，比回测激进一倍。
    """
    if not picks or budget <= 0:
        return [], 0.0
    each = budget / max(max_pos, 1)
    legs, used = [], 0.0
    for pk in picks[:max_pos]:
        px = float(pk.get("price") or 0.0)
        lot_px = px * C.LOT
        sh = int(each / lot_px) * C.LOT if lot_px > 0 else 0
        if sh <= 0:
            continue
        legs.append({**pk, "shares": sh, "lots": sh // C.LOT, "target_value": sh * px})
        used += sh * px
    return legs, used


CB_MIN_LOT = 10                      # 转债最小申报 10 张（1 手），与股票 100 股不同


def _cb_signal(max_picks: int = 10) -> dict:
    """读取可转债信号（`scripts/cb_signal.py` 产出）。

    转债榜单本身就是「YTM 前 3N 建池 → 池内短久期优先取 N」，所以取前 max_picks 只
    等价于把 N 缩小（久期更短），不需要重跑选券。腿预算只有 10%，20 只均分后每只
    1,500 元，在「1 手 = 10 张 ≈ 1,100 元」的整手下会白扔 25% 以上，故默认取 10 只。
    """
    import json as _json
    p = C.SIGNALS / "latest_cb_signal.json"
    if not p.exists():
        return {"error": "转债信号未运行，请先执行 python scripts/cb_signal.py"}
    try:
        d = _json.load(open(p, encoding="utf-8"))
    except Exception as e:                                  # noqa: BLE001
        return {"error": f"转债信号读取失败：{e}"}
    hold = d.get("holdings") or []
    picks = [{"code": str(x.get("code")), "name": x.get("name"),
              "price": float(x.get("price") or 0.0),
              "ytm": x.get("ytm"), "years_left": x.get("years_left"),
              "rating": x.get("rating")}
             for x in hold[:max_picks]]
    env = d.get("environment") or {}
    return {"asof": d.get("signal_date"), "picks": picks, "n_total": len(hold),
            "advice": d.get("advice"), "ytm_median": env.get("ytm_median"),
            "rule": "YTM 高 → 短久期优先；价格≥90 + 规模≥2亿 + 20日均额≥500万；10 日调仓"}


def _size_cb_leg(picks: list, budget: float, overshoot: float = 0.25) -> tuple[list, float]:
    """转债腿：等额分预算 + **10 张整手** + 余量补回。

    与 `_size_equal_legs` 逻辑一致，唯一差别是一手 = 10 张而不是 100 股。
    容差给到 25%（股票腿是 10%）：10% 腿预算 30,000 分 10 只 → 单只 3,000 元，
    而一手约 1,100 元 = 单只预算的 37%，10% 容差（300 元）连一手都跨不过去，
    结果每只卡在 2 手、整腿只投 90%。25% 后才允许 2 手/3 手混排、投到 ~99%。
    """
    n = len(picks)
    if n == 0 or budget <= 0:
        return [], 0.0
    each = budget / n
    plan = []
    for pk in picks:
        px = float(pk.get("price") or 0.0)
        lot_px = px * CB_MIN_LOT
        sh = int(each / lot_px) * CB_MIN_LOT if lot_px > 0 else 0
        plan.append({"pk": pk, "px": px, "lot_px": lot_px, "shares": sh})
    used = sum(p["shares"] * p["px"] for p in plan)
    cap_each = each * (1.0 + overshoot)
    while True:
        cand = [p for p in plan
                if p["lot_px"] > 0 and used + p["lot_px"] <= budget
                and (p["shares"] + CB_MIN_LOT) * p["px"] <= cap_each]
        if not cand:
            break
        p = max(cand, key=lambda x: each - x["shares"] * x["px"])
        p["shares"] += CB_MIN_LOT
        used += p["lot_px"]
    legs = [{**p["pk"], "shares": p["shares"], "lots": p["shares"] // CB_MIN_LOT,
             "target_value": p["shares"] * p["px"]} for p in plan]
    return legs, used


def _build_combo(md, asof_date, capital: float, main_meta: dict, out,
                 weights: dict | None = None) -> dict:
    """多策略组合：主策略 + （宽基）ETF 趋势 +（可选）低换手月度 +（可选）板块/转债。

    weights 形如 {"main":0.40,"lowturn":0.30,"etf":0.20,"cb":0.10}；缺省为 50/50。
    第四腿可以是板块 ETF（`sector`）或可转债（`cb`），2026-11-05 起生产用转债。
    """
    import json as _json

    w = dict(weights or {"main": 0.5, "etf": 0.5})
    tot = sum(w.values())
    w = {k: v / tot for k, v in w.items()}
    is_sec = w.get("sector", 0) > 1e-9
    is_cb = w.get("cb", 0) > 1e-9
    is_low = w.get("lowturn", 0) > 1e-9

    # ---------------- 主策略侧
    main_alloc = float(capital) * w["main"]
    # 主策略腿必须保留它**自己的**投入比例：latest_target.csv 里的金额已经含
    #   ① regime 系数（破位时 0.45）② 主策略自身的现金 ③ 整手余量。
    # 2026-11-05 修：此前 scale = main_alloc / base 把该腿拉满预算，
    # 等于把 regime 系数整体抵消 —— 实盘 50% 股票 vs 回测 22.5%，
    # 比回测激进一倍多，−6.93% 的回撤是打了折的。
    # 现在 scale = main_cap / base，main_cap = 腿预算 × 自身投入比例。
    inv_ratio = float(main_meta.get("invested_pct", 1.0))
    regime_coeff = float((main_meta.get("regime") or {}).get("regime_coeff", 1.0))
    main_cap = main_alloc * inv_ratio
    main_targets = []
    tgt_csv = out / "latest_target.csv"
    if tgt_csv.exists():
        df = pd.read_csv(tgt_csv, dtype={"code": str})
        base = float(df["target_value"].sum())
        scale = (main_cap / base) if base > 0 else 0.0
        # 先全部向下取整到整手，再把剩余零头补回去：
        # 原来的 round() 会向上取整，39 只整体超预算 2.9%（120,000 → 123,523），
        # 违反"每侧不得超预算"的约定，也让页面出现「已投 > 预算」。
        plan = []
        for r in df.itertuples():
            px = float(r.price)
            lot_px = px * C.LOT
            raw = float(r.shares) * scale
            sh = int(raw // C.LOT) * C.LOT if lot_px > 0 else 0
            plan.append({"r": r, "price": px, "lot_px": lot_px,
                         "shares": sh, "drop": (raw - sh) * px})
        used = sum(p["shares"] * p["price"] for p in plan)
        for p in sorted(plan, key=lambda x: -x["drop"]):
            if p["lot_px"] <= 0 or used + p["lot_px"] > main_cap:
                continue
            p["shares"] += C.LOT
            used += p["lot_px"]
        for p in plan:
            r, sh = p["r"], p["shares"]
            main_targets.append({
                "code": str(r.code), "name": str(r.name), "industry": str(r.industry),
                "board": str(r.board),
                # 权重相对**总本金**（不是相对腿预算），否则页面加总对不上
                "weight": (sh * p["price"]) / float(capital),
                "target_value": sh * p["price"], "shares": sh, "price": p["price"],
            })
    used_main = sum(x["target_value"] for x in main_targets)

    # ---------------- 宽基 ETF 趋势侧
    etf = _etf_trend_signal()
    etf_budget = float(capital) * w.get("etf", 0.0)
    etf_legs: list = []
    used_etf = 0.0
    etf_picks = list(etf.get("picks") or [])
    if not etf_picks and etf.get("pick"):
        etf_picks = [etf["pick"]]
    if etf_budget > 0:
        etf_legs, used_etf = _size_etf_leg(etf_picks, etf_budget)
    etf_leg = etf_legs[0] if etf_legs else None

    # ---------------- 板块轮动侧
    sec = _sector_rotation_signal() if is_sec else {}
    sec_budget = float(capital) * w.get("sector", 0.0)
    sec_legs: list = []
    used_sec = 0.0
    if is_sec and sec.get("picks") and sec_budget > 0:
        sec_legs, used_sec = _size_equal_legs(sec["picks"], sec_budget)

    # ---------------- 可转债侧（2026-11-05 起替代板块腿）
    # 换腿依据：转债与三腿（主/低换手/ETF）相关仅 0.03~0.07、净 SR 1.30；
    # 板块 OOS 只有 3.20%、衰减 91%（黑名单 #24 那一族已在证伪链里封口）。
    cb = _cb_signal() if is_cb else {}
    cb_budget = float(capital) * w.get("cb", 0.0)
    cb_legs: list = []
    used_cb = 0.0
    if is_cb and cb.get("picks") and cb_budget > 0:
        cb_legs, used_cb = _size_cb_leg(cb["picks"], cb_budget)
    _cb_adv = cb.get("advice") or {}
    _sug = _cb_adv.get("suggest_pct")
    cb_advice_txt = (f"{_sug[0]:.0%}~{_sug[1]:.0%}" if isinstance(_sug, (list, tuple)) and len(_sug) == 2
                     else "—")

    # ---------------- 低换手月度侧
    low = _lowturn_signal(md) if is_low else {}
    low_budget = float(capital) * w.get("lowturn", 0.0)
    low_legs: list = []
    used_low = 0.0
    if is_low and low.get("picks") and low_budget > 0:
        low_legs, used_low = _size_equal_legs(low["picks"], low_budget)

    cash = float(capital) - used_main - used_etf - used_sec - used_cb - used_low
    seq = pd.Timestamp(asof_date)
    is_q_end = (seq + pd.Timedelta(days=1)).month != seq.month

    labels = {"main": "主策略（个股）", "lowturn": "低换手月度（个股）",
              "etf": "宽基 ETF 趋势", "sector": "板块 ETF 轮动", "cb": "可转债"}
    parts = " + ".join(f"{w[k]:.0%} {labels[k]}" for k in labels if w.get(k, 0) > 0)
    n_legs = 1 + int(is_low) + int(w.get("etf", 0) > 1e-9) + int(is_sec) + int(is_cb)
    _mode = {2: "two", 3: "three"}.get(n_legs, "four")

    combo = {
        "signal_date": str(seq.date()),
        "capital": float(capital),
        "mode": _mode,
        "weights": {k: float(v) for k, v in w.items()},
        "parts": parts,
        "main": {"allocated": used_main, "allocated_pct": used_main / float(capital),
                 "budget_pct": w["main"], "target_alloc": main_cap,
                 "invested_ratio": inv_ratio, "regime_coeff": regime_coeff,
                 "leg_cash_pct": (1.0 - used_main / main_alloc) if main_alloc > 0 else 0.0,
                 "targets": main_targets},
        "etf": {"allocated": used_etf, "allocated_pct": used_etf / float(capital),
                "budget_pct": w.get("etf", 0.0), "leg": etf_leg, "legs": etf_legs,
                "universe": etf.get("universe"), "asof": etf.get("asof"),
                "rule": etf.get("rule")},
        "lowturn": {"allocated": used_low, "allocated_pct": used_low / float(capital),
                    "budget_pct": w.get("lowturn", 0.0), "legs": low_legs,
                    "asof": low.get("asof"), "signal_date": low.get("signal_date"),
                    "target_month": low.get("target_month"), "status": low.get("status"),
                    "n_picks": low.get("n_picks"), "rule": low.get("rule")}
                   if is_low else None,
        "sector": {"allocated": used_sec, "allocated_pct": used_sec / float(capital),
                   "budget_pct": w.get("sector", 0.0), "legs": sec_legs,
                   "picks": sec.get("picks"), "asof": sec.get("asof"),
                   "n_breakout": sec.get("n_breakout"), "rule": sec.get("rule")}
                  if is_sec else None,
        "cb": {"allocated": used_cb, "allocated_pct": used_cb / float(capital),
               "budget_pct": w.get("cb", 0.0), "legs": cb_legs,
               "asof": cb.get("asof"), "advice": cb.get("advice"),
               "advice_txt": cb_advice_txt,
               "n_total": cb.get("n_total"), "ytm_median": cb.get("ytm_median"),
               "rule": cb.get("rule"), "error": cb.get("error")}
              if is_cb else None,
        "cash": {"value": cash, "pct": cash / float(capital),
                 "sleeve": "100% 国债ETF"},
        "rebalance_hint": (
            f"本月底前为季度再平衡窗口：把各侧资金调回目标权重（{parts}）"
            if is_q_end else
            ("低换手侧每月首个交易日调仓，其余各侧仅在信号变化时调仓"
             if is_low else "非再平衡窗口，仅在信号变化时调仓")),
        "notes": [
            f"组合构成：{parts}；每季度末再平衡；",
            "某一侧无入场信号时，该侧预算全部放国债/黄金；",
            (f"主策略侧 regime 系数 {regime_coeff:.2f}：该腿只投预算的 "
             f"{inv_ratio:.0%}（{used_main:,.0f} 元），其余 "
             f"{main_cap - used_main:,.0f} 元转现金——与回测口径一致，不是漏买；"
             if regime_coeff < 0.999 else ""),
            "主策略开盘涨停买不进、跌停卖不出，可顺延；",
            ("低换手侧月频调仓、只换变动：相邻月份名单约 70% 重合，"
             "不要按“月末清仓、次月初全买回”执行，那样年成本约 11.8%、净收益归零；"
             if is_low else ""),
            (f"低换手侧回测（2020-01 起）毛 14.03%、扣费后 10.02%，"
             f"与主策略日收益相关仅 0.31，是这次加它的主因；组合里占 "
             f"{w.get('lowturn', 0):.0%} 权重。" if is_low else ""),
            (f"板块轮动全期 11.67% 但样本外仅 3.20%（衰减 91%），"
             f"组合里只占 {w.get('sector', 0):.0%} 权重，合理预期贡献约 "
             f"{w.get('sector', 0) * 0.032 * 100:.1f}pp；" if is_sec else ""),
            (f"可转债侧占 {w.get('cb', 0):.0%}：与主/低换手/ETF 三腿相关仅 0.03~0.07、"
             f"净 SR 1.30，是本次替换板块腿的原因；当前全市场 YTM 中位 "
             f"{(cb.get('ytm_median') or 0):.2%}（"
             f"{(cb.get('advice') or {}).get('level', '—')}），"
             f"官方建议仓位 {cb_advice_txt}，"
             f"我们只配 {w.get('cb', 0):.0%}，处于建议区间下沿；"
             f"高 YTM 策略本质是做多信用利差修复，榜单里 YTM 两位数以上的个券"
             f"是市场已在定价违约 —— 只能靠 10 只分散兜住，不要单只加码；"
             f"转债 T+0、无涨跌停，止损与 A 股不同，强赎需人工核对公告；" if is_cb else ""),
            "本方案仅供研究，不构成投资建议。",
        ],
    }
    combo["notes"] = [n for n in combo["notes"] if n]
    _json.dump(combo, open(out / "latest_combo.json", "w", encoding="utf-8"),
               ensure_ascii=False, indent=1, default=float)
    _write_combo_md(combo)
    print(f"[combo] {parts}")
    print(f"        主策略 {used_main:,.0f} | 低换手 {used_low:,.0f} | 宽基ETF {used_etf:,.0f} | "
          f"{'可转债' if is_cb else '板块轮动'} {used_cb + used_sec:,.0f} | "
          f"现金 {cash:,.0f}（共 {capital:,.0f} 元）")
    return combo


def _data_stamp() -> str:
    """面板数据指纹：落盘时间 + 规模。

    记录这份名单是基于哪一版数据算出来的 —— 2026-11-10 的复权面板事故就是
    「名单已经写了、数据后来被改了」却无从追溯，从此每份实盘报告都留一个戳。
    """
    try:
        p = C.DATA_RAW / "panel_ohlcv.npz"
        st = p.stat()
        when = pd.Timestamp.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        z = np.load(p, allow_pickle=True)
        return (f"`{p.name}` {when} · {len(z['codes'])} 只 · "
                f"{len(z['dates'])} 日 · 末 {str(z['dates'][-1])[:10]}")
    except Exception as exc:  # 数据缺失不该阻塞出信号
        return f"（无法读取：{exc}）"


def _write_combo_md(combo: dict) -> None:
    """输出组合实盘报告 markdown（两/三/四策略自适应，章节号按实际侧数编号）。"""
    d = pd.Timestamp(combo["signal_date"])
    mode = combo.get("mode") or "two"
    is_sec = bool(combo.get("sector"))
    is_cb = bool(combo.get("cb"))
    is_low = bool(combo.get("lowturn"))
    title = {"four": "实盘组合信号（四策略：主策略+低换手+宽基ETF+"
                     + ("可转债" if is_cb else "板块") + "）",
             "three": "实盘组合信号（三策略 50/30/20）"}.get(mode, "实盘组合信号（50/50）")
    _CN = ["一", "二", "三", "四", "五", "六", "七"]
    _seq = [0]

    def nx() -> str:
        _seq[0] += 1
        return _CN[min(_seq[0], len(_CN)) - 1]

    lines = [f"# {title} — {d.date()}", "",
             f"- 组合构成：**{combo['parts']}**",
             f"- 本金：{combo['capital']:,.0f} 元",
             f"- 再平衡：{combo['rebalance_hint']}",
             f"- 数据口径：{_data_stamp()}", "",
             f"## {nx()}、资金分配", "",
             "| 部分 | 目标权重 | 实际金额（元） | 实际占比 | 说明 |",
             "|---|---:|---:|---:|---|",
             f"| 主策略（个股） | {combo['main']['budget_pct']:.0%} | "
             f"{combo['main']['allocated']:,.0f} | {combo['main']['allocated_pct']:.2%} | "
             f"{len(combo['main']['targets'])} 只个股"
             + (f" · regime {combo['main'].get('regime_coeff', 1.0):.2f}，"
                f"该腿余 {combo['main'].get('leg_cash_pct', 0):.0%} 现金"
                if combo['main'].get('leg_cash_pct', 0) > 0.005 else "")
             + " |"]
    if is_low:
        lt_d = combo["lowturn"]
        lines.append(
            f"| 低换手月度（个股） | {lt_d['budget_pct']:.0%} | "
            f"{lt_d['allocated']:,.0f} | {lt_d['allocated_pct']:.2%} | "
            f"{lt_d.get('n_picks') or 0} 只 · 目标月 {lt_d.get('target_month')} |")
    etf_codes = ", ".join(x["code"] for x in (combo["etf"].get("legs") or [])) \
        or "当前无入场信号"
    lines += [
             f"| 宽基 ETF 趋势 | {combo['etf']['budget_pct']:.0%} | "
             f"{combo['etf']['allocated']:,.0f} | {combo['etf']['allocated_pct']:.2%} | "
             f"{etf_codes} |"]
    if is_sec:
        legs = combo["sector"]["legs"] or []
        lines.append(
            f"| 板块 ETF 轮动 | {combo['sector']['budget_pct']:.0%} | "
            f"{combo['sector']['allocated']:,.0f} | {combo['sector']['allocated_pct']:.2%} | "
            + ("、".join(f"{x['code']}({x['sector']})" for x in legs)
               if legs else "当前无板块突破") + " |")
    if is_cb:
        _cl = combo["cb"]["legs"] or []
        lines.append(
            f"| 可转债 | {combo['cb']['budget_pct']:.0%} | "
            f"{combo['cb']['allocated']:,.0f} | {combo['cb']['allocated_pct']:.2%} | "
            + (f"{len(_cl)} 只高 YTM 短久期" if _cl else "无信号，转现金") + " |")
    lines += [f"| 现金（国债+黄金） | — | {combo['cash']['value']:,.0f} | "
              f"{combo['cash']['pct']:.2%} | 100% 国债ETF |", "",
              "> **实际占比 ≠ 目标权重**的原因：① 整手取整（股票 100 股、ETF 100 份、转债 10 张）；"
              "② 无入场信号的策略其预算转入现金；③ 主策略腿保留自身的 regime 系数"
              "（破位时只投预算的 45%~60%）与整手余量，其余转现金，与回测口径一致。"
              "**只要某侧有信号，其实际金额不超过预算**。", ""]

    # ---- 宽基 ETF 侧
    no_etf = nx()
    _etf_legs = combo["etf"].get("legs") or []
    if _etf_legs:
        lines += [f"## {no_etf}、宽基 ETF 趋势侧目标", "",
                  "| 代码 | 名称 | 参考价 | 份数 | 手数 | 金额 |",
                  "|---|---|---:|---:|---:|---:|"]
        for leg in _etf_legs:
            lines.append(f"| {leg['code']} | {leg.get('name','')} | {leg['price']:.3f} | "
                         f"{leg['shares']} | {leg['lots']} | {leg['target_value']:,.0f} |")
        lines += ["", f"该腿预算 {combo['etf']['budget_pct']:.0%} 按「每只固定 1/2」分配"
                  "（与回测 POS_STYLE=equal 一致，只有 1 只突破时自动半仓），"
                  "未投部分放国债。", ""]
    else:
        lines += [f"## {no_etf}、宽基 ETF 趋势侧", "",
                  "**当前无 ETF 满足入场条件**（未突破 55 日新高或成交额不足），"
                  "该部分预算放国债/黄金。", ""]
    lines += ["### 宽基 ETF 候选状态", "",
              "| 代码 | 名称 | 最新价 | 60日动量 | 距55日高 | 20日均额 | 入场 |",
              "|---|---|---:|---:|---:|---:|---|"]
    for u in combo["etf"].get("universe") or []:
        lines.append(f"| {u['code']} | {u.get('name','')} | {u['price']:.3f} | "
                     f"{u['mom60']*100:+.1f}% | {u['dist_to_high']*100:+.1f}% | "
                     f"{u['amount_20d']/1e8:.2f} 亿 | {'✅' if u['entry'] else '观察'} |")

    # ---- 低换手月度侧
    if is_low:
        lt_d = combo["lowturn"]
        st_cn = {"month_end": "月末已出下月名单，次日开盘建仓",
                 "mid_month": "本月名单已生效，等下月首个交易日调仓"}.get(
                     lt_d.get("status"), "")
        lines += ["", f"## {nx()}、低换手月度侧目标", "",
                  f"- 目标月份：**{lt_d.get('target_month')}**（信号基准日 "
                  f"{lt_d.get('signal_date')}）",
                  f"- 执行时点：{st_cn}",
                  f"- 规则：{lt_d.get('rule', '')}", ""]
        legs = lt_d.get("legs") or []
        if legs:
            lines += ["| 代码 | 名称 | 行业 | 参考价 | 股数 | 金额 | 20日均换手 | 市值(亿) |",
                      "|---|---|---|---:|---:|---:|---:|---:|"]
            for x in legs:
                _t = x.get("turn20_bp")
                _m = x.get("mcap_yi")
                lines.append(
                    f"| {x['code']} | {x['name']} | {x.get('sector','')} | "
                    f"{float(x.get('price') or 0):.3f} | {x['shares']} | "
                    f"{x['target_value']:,.0f} | "
                    f"{(f'{_t / 100:.2f}%' if _t is not None else '—')} | "
                    f"{(f'{_m:,.0f}' if _m is not None else '—')} |")
            lines += ["", "> 与上月名单约 70% 重合，**只换变动**；"
                       "保留的标的仅在偏离目标权重超过 2% 时才调整。", ""]
        else:
            lines += ["**无可选标的**（全部被 ST/退市/停牌/上市不满一年过滤），"
                      "该部分预算放国债。", ""]

    # ---- 板块轮动侧
    if is_sec:
        no_sec = nx()
        lines += ["", f"## {no_sec}、板块 ETF 轮动侧目标", ""]
        legs = combo["sector"]["legs"] or []
        if legs:
            lines += ["| 代码 | 板块 | 参考价 | 份额 | 手数 | 金额 | 60日动量 |",
                      "|---|---|---:|---:|---:|---:|---:|"]
            for x in legs:
                lines.append(f"| {x['code']} | {x['sector']} | {x['price']:.3f} | "
                             f"{x['shares']} | {x['lots']} | {x['target_value']:,.0f} | "
                             f"{x['mom60']*100:+.1f}% |")
        else:
            lines += [f"**当前无板块满足突破条件**（{combo['sector'].get('n_breakout', 0)} 个突破），"
                      "该 20% 预算放国债/黄金。", ""]
        lines += ["", f"规则：{combo['sector'].get('rule', '')}", ""]

    # ---- 可转债侧
    if is_cb:
        no_cb = nx()
        _cd = combo["cb"]
        _ad = _cd.get("advice") or {}
        lines += [f"## {no_cb}、可转债侧目标", ""]
        if _cd.get("error"):
            lines += [f"**{_cd['error']}**，该 {_cd['budget_pct']:.0%} 预算放国债。", ""]
        elif _cd["legs"]:
            lines += ["| 代码 | 名称 | 参考价 | YTM | 剩余年限 | 张数 | 手数 | 金额 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"]
            for x in _cd["legs"]:
                _y = x.get("ytm")
                _yr = x.get("years_left")
                lines.append(
                    f"| {x['code']} | {x.get('name','')} | {x['price']:.2f} | "
                    f"{(f'{_y:.2%}' if _y is not None else '—')} | "
                    f"{(f'{_yr:.2f}' if _yr is not None else '—')} | "
                    f"{x['shares']} | {x['lots']} | {x['target_value']:,.0f} |")
            lines += ["", f"该腿预算 {_cd['budget_pct']:.0%} 等额分配，1 手 = 10 张"
                      "（与股票 100 股不同），余量补回缺口最大的标的、单只超配 ≤25%"
                      "（整手粒度是单只预算的 37%，10% 容差连一手都跨不过去）。", ""]
            if _ad:
                lines += [f"> 市场环境：全市场 YTM 中位 {(combo['cb'].get('ytm_median') or 0):.2%}"
                          f"（YTM 越低 = 越贵），判定「{_ad.get('level', '—')}」，"
                          f"官方建议仓位 {_cd.get('advice_txt', '—')}。"
                          f"我们固定配 {_cd['budget_pct']:.0%}，"
                          "处于建议区间下沿 —— 贵的时候不追，但没有降仓（回测降仓全部变差）。", ""]
        else:
            lines += [f"**当前无可选转债**（信号日 {_cd.get('asof') or '—'}），"
                      "该部分预算放国债。", ""]
        lines += [f"规则：{_cd.get('rule', '')}", ""]

    # ---- 主策略侧
    sec_no = nx()
    lines += [f"## {sec_no}、主策略侧目标", "",
              "| 代码 | 名称 | 行业 | 板块 | 参考价 | 股数 | 金额 |",
              "|---|---|---|---|---:|---:|---:|"]
    for t in combo["main"]["targets"][:40]:
        lines.append(f"| {t['code']} | {t['name']} | {t['industry']} | {t['board']} | "
                     f"{t.get('price', 0):.2f} | {t['shares']} | {t['target_value']:,.0f} |")

    tail_no = nx()
    lines += ["", f"## {tail_no}、注意事项", ""]
    lines += [f"- {n}" for n in combo.get("notes", [])]
    out = C.REPORTS / f"实盘组合_{d.date()}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[combo] 报告 → {out}")


if __name__ == "__main__":
    main()
