# -*- coding: utf-8 -*-
"""可转债每日信号：按高 YTM 选出目标持仓，给出整手订单与仓位建议。

用法::

    python scripts/cb_signal.py                          # 默认 20 只、3 万/只（30万本金）
    python scripts/cb_signal.py --n-holdings 20 --capital 300000
    python scripts/cb_signal.py --holdings signals/cb_holdings.csv   # 生成调仓订单
    python scripts/cb_signal.py --refresh-history        # 重算全市场 YTM 历史分位缓存

输出::

    signals/latest_cb_target.csv     目标持仓（含 YTM / 剩余年限 / 账本/张数/金额）
    signals/latest_cb_orders.csv     相对当前持仓的买卖订单（提供 --holdings 时）
    signals/latest_cb_signal.json    信号元数据 + 市场环境 + 仓位建议
    reports/可转债信号_<日期>.md      人读报告

转债交易单位：1 张 = 100 元面值；报价为「每百元面值价格」；申报最小 10 张
（沪市 1 手 = 10 张，深市同为 10 张），故张数一律取 10 的倍数。
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

from qf import config as C            # noqa: E402
from qf.cb import CBData              # noqa: E402

MIN_LOT = 10                          # 转债最小申报 10 张
HIST = C.BUILD / "cb" / "ytm_history.parquet"


# ---------------------------------------------------------------- 历史分位
def refresh_history(cb: CBData, start: str = "2019-01-01") -> pd.DataFrame:
    """逐日计算全市场 YTM 中位数 / 分位，用于判断当前估值在历史的什么位置。"""
    t0 = int(np.searchsorted(cb.cal.values, np.datetime64(start)))
    rows = []
    for t in range(t0, cb.n_days):
        px = cb.close[t].astype(np.float64)
        ytm = cb.ytm_on(t)
        live = (np.isfinite(px) & (px > 0) & (t >= cb.list_pos + 10)
                & (t < cb.delist_pos) & np.isfinite(ytm))
        if live.sum() < 20:
            continue
        y = ytm[live]
        rows.append({"date": cb.cal[t], "n_live": int(live.sum()),
                     "ytm_median": float(np.median(y)),
                     "ytm_p25": float(np.percentile(y, 25)),
                     "ytm_p75": float(np.percentile(y, 75)),
                     "price_median": float(np.median(px[live])),
                     "pct_below_100": float((px[live] < 100).mean())})
        if t % 200 == 0:
            print(f"      {cb.cal[t].date()} …")
    df = pd.DataFrame(rows)
    HIST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(HIST, index=False)
    print(f"      历史分位缓存 {len(df)} 天 → {HIST}")
    return df


def ytm_percentile(cur: float, hist: pd.DataFrame) -> float | None:
    if hist is None or not len(hist):
        return None
    return float((hist["ytm_median"] <= cur).mean())


# ---------------------------------------------------------------- 仓位建议
def position_advice(med_ytm: float, pct: float | None, hist: pd.DataFrame | None) -> dict:
    """基于 YTM 环境给出建议仓位区间。

    方向约定（容易写反，务必记住）：
        YTM **高** = 债底厚、到期回归空间大 = **便宜** → 高仓位
        YTM **低** = 价格已充分反映转股期权 = **贵** → 低仓位
    所以分位（YTM 在历史中的百分位）越高 → 越该配。

    口径来自实证（reports/可转债策略实测_2026-10-29.md）：
    前瞻分组单调——组合 YTM 最低 20% 组未来 1 年 +3.3%，最高 20% 组 +33.9%；
    但「按 YTM 降仓」回测全部变差（基线 18.32% 最好），因为低价转债有凸性、
    降仓到现金亏机会成本。故这里只作**风险提示**，不自动降仓。
    """
    if pct is None:
        return {"level": "未知", "suggest_pct": None, "reason": "无历史分位缓存"}
    if pct >= 0.75:
        lvl, lo, hi = "便宜", 0.7, 1.0
    elif pct >= 0.50:
        lvl, lo, hi = "中性偏便宜", 0.5, 0.7
    elif pct >= 0.25:
        lvl, lo, hi = "中性偏贵", 0.3, 0.5
    else:
        lvl, lo, hi = "贵", 0.2, 0.4
    fwd = _fwd_proxy(med_ytm, hist)
    return {"level": lvl, "suggest_pct": [lo, hi],
            "ytm_percentile": pct, "fwd_proxy": fwd,
            "reason": f"全市场 YTM 中位 {med_ytm:.2%}，位于 2019 以来 {pct:.0%} 分位 → "
                      f"{lvl}；建议转债仓位 {lo:.0%}~{hi:.0%}"}


def _fwd_proxy(med_ytm: float, hist: pd.DataFrame | None) -> dict | None:
    """历史同环境下的前瞻。优先用真实策略回测口径（fwd_ytm.csv），否则降级为价格代理。"""
    real = C.BUILD / "runs" / "cb" / "fwd_ytm.csv"
    if real.exists():
        try:
            fw = pd.read_csv(real, parse_dates=["date"])
            below = fw[fw["avg_ytm"] <= med_ytm]["fwd_1y"].dropna()
            if len(below) >= 10:
                return {"source": "策略真实回测（组合 YTM → 未来 1 年净值）",
                        "n": int(len(below)), "median": float(below.median()),
                        "p10": float(below.quantile(0.10)),
                        "min": float(below.min())}
        except Exception:
            pass
    if hist is None or len(hist) < 300:
        return None
    h = hist.set_index("date")
    vals = []
    for d, row in h.iterrows():
        nxt = h.index[h.index >= d + pd.Timedelta(days=365)]
        if not len(nxt) or row["ytm_median"] > med_ytm:
            continue
        f0, f1 = row["price_median"], h["price_median"].loc[nxt[0]]
        if f0 > 0:
            vals.append(f1 / f0 - 1)
    if len(vals) < 10:
        return None
    return {"source": "全市场中位价格变动（粗代理，非策略收益）",
            "n": len(vals), "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)), "min": float(np.min(vals))}


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=C.CAPITAL)
    ap.add_argument("--n-holdings", type=int, default=20)
    ap.add_argument("--asof", default=None)
    ap.add_argument("--holdings", default=None, help="当前转债持仓 CSV（code,sheets）")
    ap.add_argument("--out", default=str(C.SIGNALS))
    ap.add_argument("--min-price", type=float, default=90.0)
    ap.add_argument("--min-issue", type=float, default=2.0, help="发行规模下限（亿元）")
    ap.add_argument("--min-amount", type=float, default=5e6, help="20日均成交额下限（元）")
    ap.add_argument("--min-years", type=float, default=0.30,
                    help="剩余期限下限（年）。<0.3 年临近到期，实盘已停止交易/兑付冻结")
    ap.add_argument("--cb-weight", type=float, default=1.0,
                    help="转债在总资产中的配置比例（用于计算每只实际金额）")
    ap.add_argument("--rank", default="ytm_duration", choices=["ytm", "ytm_duration"],
                    help="选券口径：ytm=纯YTM；ytm_duration=YTM建池后取短久期（生产默认）")
    ap.add_argument("--pool-mult", type=float, default=3.0,
                    help="建池倍数：先取 YTM 前 n_hold*pool_mult，再按剩余年限升序取 n_hold")
    ap.add_argument("--refresh-history", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载可转债数据 ...")
    cb = CBData(verbose=True)
    if args.refresh_history or not HIST.exists():
        print("[2/4] 计算全市场 YTM 历史分位（首次较慢，约 40s）...")
        hist = refresh_history(cb)
    else:
        print("[2/4] 读取历史分位缓存 ...")
        hist = pd.read_parquet(HIST)

    t = cb.n_days - 1 if args.asof is None else cb.date_index(args.asof)
    asof = cb.cal[t]
    print(f"      信号日 {asof.date()}")

    print("[3/4] 选出目标持仓 ...")
    tgt = cb.select(t, n_hold=args.n_holdings, min_price=args.min_price,
                    min_issue=args.min_issue, min_amount=args.min_amount,
                    min_years_left=args.min_years, rank=args.rank,
                    pool_mult=args.pool_mult)
    env = cb.universe_stats(t)
    med = float(np.median([x for x in [env.get("ytm_median")] if x is not None] or [np.nan]))
    pct = ytm_percentile(med, hist) if np.isfinite(med) else None
    advice = position_advice(med, pct, hist)

    budget = args.capital * args.cb_weight
    per = budget / max(len(tgt), 1)
    rows = []
    for r in tgt.itertuples():
        px = float(r.price)
        sheets = int(np.floor(per / px / MIN_LOT) * MIN_LOT) if px > 0 else 0
        rows.append({
            "code": r.code, "name": r.name, "price": round(px, 3),
            "ytm": round(float(r.ytm), 5), "years_left": round(float(r.years_left), 2),
            "issue_size_yi": round(float(r.issue_size), 2),
            "amount_20d_yi": round(float(r.amount_20d) / 1e8, 2),
            "rating": r.rating, "maturity": str(pd.Timestamp(r.maturity).date()),
            "sheets": sheets, "lots": sheets // MIN_LOT,
            "amount": round(sheets * px, 2),
            "weight": round(sheets * px / args.capital, 4),
        })
    tdf = pd.DataFrame(rows)
    used = float(tdf["amount"].sum()) if len(tdf) else 0.0
    tdf["cash_left"] = args.capital - used
    tdf.to_csv(out / "latest_cb_target.csv", index=False, encoding="utf-8-sig")

    # ---- 订单
    odf = pd.DataFrame()
    if args.holdings:
        hp = pathlib.Path(args.holdings)
        if hp.exists():
            hdf = pd.read_csv(hp, dtype={"code": str})
            hdf["code"] = hdf["code"].str.zfill(6)
            col = "sheets" if "sheets" in hdf.columns else hdf.columns[-1]
            cur = dict(zip(hdf["code"], hdf[col].astype(int)))
            tgtm = dict(zip(tdf["code"], tdf["sheets"].astype(int)))
            pmap = dict(zip(tdf["code"], tdf["price"]))
            orders = []
            for code in sorted(set(cur) | set(tgtm)):
                have, want = cur.get(code, 0), tgtm.get(code, 0)
                d = want - have
                if d == 0:
                    continue
                orders.append({"code": code, "side": "buy" if d > 0 else "sell",
                               "sheets": abs(d), "lots": abs(d) // MIN_LOT,
                               "price_ref": pmap.get(code),
                               "reason": "rebalance" if have else "new",
                               **({} if d > 0 or code in tgtm else {"reason": "exit"})})
            odf = pd.DataFrame(orders)
            if len(odf):
                odf.to_csv(out / "latest_cb_orders.csv", index=False, encoding="utf-8-sig")

    meta = {
        "signal_date": str(asof.date()),
        "capital": args.capital, "cb_weight": args.cb_weight,
        "n_holdings": int(len(tdf)), "invested": used,
        "rank": args.rank, "pool_mult": args.pool_mult,
        "invested_pct": used / args.capital if args.capital else None,
        "environment": {**env, "ytm_percentile": pct},
        "advice": advice,
        "holdings": tdf.to_dict("records"),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "notes": [
            "信号基于 t 日收盘，建议 t+1 开盘执行；开盘涨停买不进可顺延",
            "已公告强赎的转债不在数据中标记，实盘前需人工核对公告（强赎后价格向 100 收敛）",
            "信用风险控制：价格≥90 + 发行规模≥2亿 + 20日均额≥500万，不用评级（非 PIT）",
            "选券口径：YTM 前 3N 建池 → 池内按剩余年限升序取 N（短久期优先）",
            "剩余期限下限 0.30 年：到期前约 5 个交易日停止交易，更短的债实盘无法执行",
            "本信号为量化研究结果，不构成投资建议",
        ],
    }
    json.dump(meta, open(out / "latest_cb_signal.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)

    # ---- markdown
    lines = [f"# 可转债信号 — {asof.date()}", "",
             f"- 策略：{'高 YTM 建池 → 短久期优先' if args.rank == 'ytm_duration' else '高 YTM'}"
             f"选券，{args.n_holdings} 只等权，10 日调仓"
             + (f"（建池 {args.pool_mult:g}×）" if args.rank == "ytm_duration" else ""),
             f"- 本金 {args.capital:,.0f} 元，转债配置比例 {args.cb_weight:.0%}",
             f"- 目标持仓 {len(tdf)} 只，金额 {used:,.0f} 元（{used/args.capital:.1%}）",
             "", "## 市场环境", "",
             f"| 指标 | 值 |", "|---|---:|",
             f"| 在市转债数 | {env.get('n_live', 0)} |",
             f"| YTM 中位 | {env.get('ytm_median', float('nan')):.2%} |",
             f"| YTM 25/75 分位 | {env.get('ytm_p25', float('nan')):.2%} / {env.get('ytm_p75', float('nan')):.2%} |",
             f"| YTM 历史分位 | {pct:.0%} |" if pct is not None else "| YTM 历史分位 | — |",
             f"| 价格中位 | {env.get('price_median', float('nan')):.2f} 元 |",
             f"| 价格 <100 占比 | {env.get('price_below_100', float('nan')):.1%} |",
             "", f"**仓位建议：{advice['level']}** — {advice['reason']}", ""]
    if advice.get("fwd_proxy"):
        fp = advice["fwd_proxy"]
        lines += [f"历史同类环境（YTM 不高于当前值）样本 {fp['n']} 个，其后一年收益："
                  f"中位 {fp['median']:.2%}、p10 {fp['p10']:.2%}、最差 {fp['min']:.2%}"
                  f"（口径：{fp.get('source', '—')}）", ""]
    lines += ["## 目标持仓", "",
              "| 代码 | 名称 | 价格 | YTM | 剩余年限 | 规模(亿) | 日额(亿) | 评级 | 张数 | 金额 |",
              "|---|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in tdf.itertuples():
        lines.append(f"| {r.code} | {r.name} | {r.price:.2f} | {r.ytm:.2%} | "
                     f"{r.years_left:.2f} | {r.issue_size_yi:.1f} | {r.amount_20d_yi:.2f} | "
                     f"{r.rating} | {r.sheets} | {r.amount:,.0f} |")
    if len(odf):
        lines += ["", "## 调仓订单", "", "| 代码 | 方向 | 张数 | 参考价 |", "|---|---|---:|---:|"]
        for r in odf.itertuples():
            lines.append(f"| {r.code} | {'买入' if r.side == 'buy' else '卖出'} | "
                         f"{r.sheets} | {r.price_ref if r.price_ref == r.price_ref else '—'} |")
    lines += ["", "## 风险提示", ""] + [f"- {n}" for n in meta["notes"]]
    md_path = C.REPORTS / f"可转债信号_{asof.date()}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n目标持仓 {len(tdf)} 只，金额 {used:,.0f} 元（{used/args.capital:.1%}）")
    if len(tdf):
        print(tdf[["code", "name", "price", "ytm", "years_left", "sheets", "amount"]]
              .to_string(index=False))
    print(f"\n仓位建议：{advice['level']}（{advice['suggest_pct']}）")
    print(f"完成，用时 {time.time()-t_start:.1f}s")
    for p in (out / "latest_cb_target.csv", md_path):
        print(f"  {p}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    main()
