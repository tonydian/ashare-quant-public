# -*- coding: utf-8 -*-
"""实盘跟踪台账：记录每日信号、成交、净值，用于长期验证。

设计目标（回答三个问题）：
  1. 策略是否按规则执行了？（信号 vs 实际成交的偏差）
  2. 实盘表现是否与回测一致？（跟踪误差）
  3. 策略是否在失效？（滚动 Sharpe、回撤、与回测的偏离）

数据文件（都在 build/tracking/）：
  paper_trades.jsonl    每笔成交记录（你手动填入）
  daily_nav.jsonl       每日净值（自动或手动）
  signal_log.jsonl      每日信号快照（自动）
  tracking_state.json   当前状态（持仓、成本、累计）

用法：
  1. 每天收盘后运行：
       python scripts/track_record.py signal          # 记录今日信号
  2. 实际成交后补录：
       python scripts/track_record.py fill --code 600000 --side buy \
           --price 12.34 --shares 800 [--date 2026-10-28]
  3. 更新净值：
       python scripts/track_record.py nav             # 用收盘价自动算
  4. 生成报告：
       python scripts/track_record.py report
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.pop("QF_DATA_RAW", None)
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, metrics as qfm  # noqa: E402

TRACK = ROOT / "build" / "tracking"
TRACK.mkdir(parents=True, exist_ok=True)

F_SIGNAL = TRACK / "signal_log.jsonl"
F_TRADE = TRACK / "paper_trades.jsonl"
F_NAV = TRACK / "daily_nav.jsonl"
F_STATE = TRACK / "tracking_state.json"
F_BACKTEST = TRACK / "backtest_reference.json"


# ================================================================ 工具
def append_jsonl(path: pathlib.Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_state() -> dict:
    if F_STATE.exists():
        return json.load(open(F_STATE, encoding="utf-8"))
    return {"inception": None, "capital": 300000.0,
            "holdings": {}, "cash": 300000.0, "last_update": None}


def save_state(st: dict) -> None:
    F_STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1,
                                  default=str), encoding="utf-8")


# ================================================================ 1) 记录信号
def cmd_signal(args):
    """记录今日信号快照（与 latest_combo.json 对齐）。"""
    p = ROOT / "signals" / "latest_combo.json"
    if not p.exists():
        print("❌ 找不到 signals/latest_combo.json，先跑 daily_signal.py")
        return 1
    d = json.load(open(p, encoding="utf-8"))
    sd = d.get("signal_date")
    if not sd:
        print("❌ 信号文件缺 signal_date")
        return 1

    # 去重：同一 signal_date 只记一次
    existing = read_jsonl(F_SIGNAL)
    if any(x.get("signal_date") == sd for x in existing):
        print(f"  信号 {sd} 已记录，跳过")
    else:
        snap = {
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "signal_date": sd,
            "capital": d.get("capital"),
            "parts": d.get("parts"),
            "weights": d.get("weights"),
            "cash": d.get("cash"),
            "n_main_targets": len((d.get("main") or {}).get("targets") or []),
            "etf_leg": ((d.get("etf") or {}).get("leg")),
            "rebalance_hint": d.get("rebalance_hint"),
            # 主策略目标明细（用于后续对比）
            "main_targets": [
                {"code": t.get("code"), "name": t.get("name"),
                 "weight": t.get("weight"), "industry": t.get("industry")}
                for t in ((d.get("main") or {}).get("targets") or [])
            ],
        }
        append_jsonl(F_SIGNAL, snap)
        print(f"✅ 已记录信号 {sd}（主策略 {snap['n_main_targets']} 只）")

    st = load_state()
    if not st.get("inception"):
        st["inception"] = sd
        st["capital"] = float(d.get("capital") or 300000.0)
        st["cash"] = float(d.get("capital") or 300000.0)
        save_state(st)
        print(f"  台账起始日设为 {sd}，初始资金 {st['capital']:,.0f}")
    return 0


# ================================================================ 2) 补录成交
def cmd_fill(args):
    """补录实际成交（手动）。"""
    if not args.code or not args.side or args.price is None:
        print("❌ 需要 --code --side --price")
        return 1
    if args.shares is None:
        print("❌ 需要 --shares")
        return 1

    rec = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "date": args.date or datetime.now().strftime("%Y-%m-%d"),
        "code": str(args.code).zfill(6),
        "side": args.side,          # buy / sell
        "price": float(args.price),
        "shares": int(args.shares),
        "amount": float(args.price) * int(args.shares),
        "fee": float(args.fee or 0.0),
        "reason": args.reason or "",
        "signal_date": args.signal_date or "",
    }
    append_jsonl(F_TRADE, rec)

    # 更新状态
    st = load_state()
    code = rec["code"]
    pos = st["holdings"].get(code) or {"shares": 0, "cost": 0.0}
    if rec["side"] == "buy":
        total_cost = pos["cost"] * pos["shares"] + rec["amount"] + rec["fee"]
        pos["shares"] += rec["shares"]
        pos["cost"] = total_cost / pos["shares"] if pos["shares"] else 0.0
        st["cash"] -= (rec["amount"] + rec["fee"])
    else:
        pos["shares"] -= rec["shares"]
        st["cash"] += (rec["amount"] - rec["fee"])
        if pos["shares"] <= 0:
            st["holdings"].pop(code, None)
        else:
            st["holdings"][code] = pos
    if rec["side"] == "buy":
        st["holdings"][code] = pos
    st["last_update"] = rec["date"]
    save_state(st)

    print(f"✅ {rec['date']} {rec['side']} {code} "
          f"{rec['shares']} @ {rec['price']:.3f} = {rec['amount']:,.0f}")
    print(f"   现金余额: {st['cash']:,.0f}，持仓 {len(st['holdings'])} 只")
    return 0


# ================================================================ 3) 更新净值
def cmd_nav(args):
    """按当前持仓 + 收盘价算 NAV。"""
    st = load_state()
    if not st["holdings"]:
        # 无持仓：净值 = 现金
        nav = st["cash"]
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        append_jsonl(F_NAV, {
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "date": date, "nav": nav, "cash": st["cash"],
            "n_holdings": 0, "market_value": 0.0})
        print(f"✅ {date} NAV = {nav:,.0f}（空仓）")
        return 0

    md = qfdata.load_market(verbose=False)
    dates = pd.DatetimeIndex(md.dates)
    code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}
    t = md.T - 1 if not args.date else int(dates.searchsorted(
        pd.Timestamp(args.date), side="right")) - 1
    t = max(0, min(t, md.T - 1))

    mv = 0.0
    detail = []
    for code, pos in st["holdings"].items():
        j = code_idx.get(code)
        if j is None:
            continue
        px = md.price[t, j]
        if not np.isfinite(px):
            # 停牌：用成本价
            px = pos["cost"]
        val = pos["shares"] * px
        mv += val
        detail.append({"code": code, "shares": pos["shares"],
                       "price": float(px), "value": float(val),
                       "cost": pos["cost"],
                       "pnl_pct": float(px / pos["cost"] - 1)
                       if pos["cost"] > 0 else 0.0})

    nav = st["cash"] + mv
    date = str(dates[t].date())
    append_jsonl(F_NAV, {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "date": date, "nav": nav, "cash": st["cash"],
        "n_holdings": len(detail), "market_value": mv,
        "detail": detail})

    pnl = nav / st["capital"] - 1 if st["capital"] > 0 else 0.0
    print(f"✅ {date} NAV = {nav:,.0f}（{pnl:+.2%}）")
    print(f"   现金 {st['cash']:,.0f} + 持仓 {mv:,.0f}"
          f"（{len(detail)} 只）")
    for d in detail[:5]:
        print(f"     {d['code']}: {d['shares']} 股 @ {d['price']:.2f} "
              f"= {d['value']:,.0f}（{d['pnl_pct']:+.1%}）")
    return 0


# ================================================================ 4) 报告
def cmd_report(args):
    """生成跟踪报告：实盘 vs 回测的偏离。"""
    navs = read_jsonl(F_NAV)
    trades = read_jsonl(F_TRADE)
    sigs = read_jsonl(F_SIGNAL)
    st = load_state()

    print("=" * 88)
    print("实盘跟踪报告")
    print("=" * 88)
    if not navs:
        print("\n⚠️ 还没有净值记录")
        print("   用法：python scripts/track_record.py nav")
        return 0

    df = pd.DataFrame(navs).drop_duplicates("date", keep="last")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    s = df["nav"]

    print(f"\n跟踪区间: {s.index[0].date()} ~ {s.index[-1].date()}"
          f"（{len(s)} 个记录）")
    print(f"起始资金: {st['capital']:,.0f}")
    print(f"当前净值: {s.iloc[-1]:,.0f}（{s.iloc[-1]/st['capital']-1:+.2%}）")

    if len(s) >= 20:
        p = qfm.performance(s, rf=C.CASH_YIELD)
        print(f"\n  累计收益: {(s.iloc[-1]/s.iloc[0]-1):+.2%}")
        print(f"  年化:     {p['cagr']:+.2%}")
        print(f"  波动:     {p['ann_vol']:.2%}")
        print(f"  Sharpe:   {p['sharpe']:.3f}")
        print(f"  MaxDD:    {p['max_dd']:.2%}")

    # 交易统计
    if trades:
        td = pd.DataFrame(trades)
        td["date"] = pd.to_datetime(td["date"])
        buys = td[td["side"] == "buy"]
        sells = td[td["side"] == "sell"]
        print(f"\n  成交: {len(td)} 笔（买 {len(buys)} / 卖 {len(sells)}）")
        print(f"  总买入: {buys['amount'].sum():,.0f}")
        print(f"  总卖出: {sells['amount'].sum():,.0f}")
        print(f"  总费用: {td['fee'].sum():,.0f}")
    else:
        print("\n  成交: 无记录")

    # 信号一致性
    if sigs:
        print(f"\n  信号记录: {len(sigs)} 条")
        last = sigs[-1]
        print(f"  最近信号: {last['signal_date']}"
              f"（主策略 {last['n_main_targets']} 只）")

        # 对比：信号里的目标 vs 实际持仓
        if last.get("main_targets") and st["holdings"]:
            target_codes = set(t["code"] for t in last["main_targets"]
                               if t.get("code"))
            held = set(st["holdings"].keys())
            if target_codes and held:
                overlap = target_codes & held
                print(f"\n  信号-持仓一致性:")
                print(f"    信号目标: {len(target_codes)} 只")
                print(f"    实际持仓: {len(held)} 只")
                print(f"    重合:     {len(overlap)} 只"
                      f"（{len(overlap)/max(len(target_codes),1)*100:.0f}%）")
                missing = target_codes - held
                extra = held - target_codes
                if missing:
                    print(f"    ⚠️ 未买入 {len(missing)} 只: "
                          f"{sorted(missing)[:8]}")
                if extra:
                    print(f"    ⚠️ 超出信号 {len(extra)} 只: "
                          f"{sorted(extra)[:8]}")

    # 与回测对比
    if F_BACKTEST.exists():
        ref = json.load(open(F_BACKTEST, encoding="utf-8"))
        print(f"\n  回测参考（同期，理论值）:")
        for k in ("cagr", "sharpe", "max_dd"):
            if k in ref:
                print(f"    {k}: {ref[k]:.4f}"
                      if isinstance(ref[k], float) else f"    {k}: {ref[k]}")

    print("\n" + "=" * 88)
    print("下一步：")
    print("  每天收盘后： python scripts/track_record.py nav")
    print("  有成交时：   python scripts/track_record.py fill --code X "
          "--side buy --price Y --shares Z")
    print("=" * 88)
    return 0


# ================================================================ 5) 生成回测参考
def cmd_reference(args):
    """把当前回测结果存为参考基准，供对比。"""
    p = ROOT / "build" / "runs" / "main" / "metrics.json"
    if not p.exists():
        print("❌ 找不到主回测结果")
        return 1
    m = json.load(open(p, encoding="utf-8"))
    ref = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "wf_h10",
        "rebalance_days": 40,
        "n_holdings": 40,
        "slip_bp": float(getattr(C, "SLIPPAGE_BP", 40)),
        "cagr": float(m.get("cagr", 0)),
        "sharpe": float(m.get("sharpe", 0)),
        "max_dd": float(m.get("max_dd", 0)),
        "calmar": float(m.get("calmar", 0)),
        "turnover_ann": float(m.get("turnover_ann", 0)),
    }
    F_BACKTEST.write_text(json.dumps(ref, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print(f"✅ 回测参考已保存：CAGR {ref['cagr']:.2%}, "
          f"Sharpe {ref['sharpe']:.3f}, MaxDD {ref['max_dd']:.2%}")
    return 0


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="实盘跟踪台账")
    sub = ap.add_subparsers(dest="cmd")

    p1 = sub.add_parser("signal", help="记录今日信号")
    p2 = sub.add_parser("fill", help="补录成交")
    p2.add_argument("--code")
    p2.add_argument("--side", choices=["buy", "sell"])
    p2.add_argument("--price", type=float)
    p2.add_argument("--shares", type=int)
    p2.add_argument("--fee", type=float, default=0.0)
    p2.add_argument("--date")
    p2.add_argument("--reason", default="")
    p2.add_argument("--signal-date", default="")

    p3 = sub.add_parser("nav", help="更新净值")
    p3.add_argument("--date")

    sub.add_parser("report", help="生成报告")
    sub.add_parser("reference", help="保存回测参考")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return 0

    return {"signal": cmd_signal, "fill": cmd_fill, "nav": cmd_nav,
            "report": cmd_report, "reference": cmd_reference}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
