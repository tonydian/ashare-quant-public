# -*- coding: utf-8 -*-
"""滑点校准：成交记录模板 → 实测滑点 → 反推年化成本。

## 为什么需要这个

`qf/config.py` 里 `SLIPPAGE_BP = 40` 是 **2026-10-28 拍的一个估算值**，
`build/exp_slippage.py` 开头自己写着"这个数是拍的，没有任何成交数据支撑"。
而它在成本里占 78.6%（单笔 50.4bp 中 40bp 是滑点），年化 0.60pp ——
比任何还没试过的因子都值钱。

日线数据**测不出真实滑点**（没有买卖盘口、没有成交明细）。唯一可靠办法是
每个调仓日把「实际成交均价」记一列，累积 20~30 笔就能反推。
本脚本做两件事：

1. `new`   —— 从当日信号生成待填的成交模板（自动带信号日收盘价、T+1 开盘价）
2. `report`—— 读已填模板，算实测滑点 + bootstrap 置信区间 + 折算年化成本
3. `proxy` —— 用日线数据给出**代理估计**（tick 折算 / 日内价格漂移），
              明确它只能给量级、不能替代实测

## 口径定义

- **滑点（成本口径，正数 = 吃亏）**
  - 买入：`(成交价 / T+1 开盘价 − 1) × 1e4`
  - 卖出：`(T+1 开盘价 − 成交价) / T+1 开盘价 × 1e4`
- **执行总成本**：`(成交价 / 信号日收盘价)` 口径，含隔夜跳空 —— 这个数是
  回测**已经**吃掉的（回测按 T+1 开盘成交），所以只有"滑点"部分才是回测缺的。
- **年化换算**：`年化 = 单笔滑点(bp) / 1e4 × 年双边换手`。
  年双边换手默认取 `build/runs/main/metrics.json` 的 `turnover_ann_double`（1.514），
  可用 `--turnover` 覆盖。

## 跑法

    python scripts/fill_log.py status                 # 先看哪笔还没做
    python scripts/fill_log.py new                    # 生成今日模板（四条腿）
    python scripts/fill_log.py new --legs lowturn,cb  # 只生成这两条（分批建仓）
    python scripts/fill_log.py list --date 2026-11-09 # 打印可下单清单（含建议限价）
    python scripts/fill_log.py fill --date 2026-11-09 --mode open  # 懒人回填（不进校准）
    python scripts/fill_log.py report                 # 累积 40 笔真实成交后再跑
    python scripts/fill_log.py proxy                  # 现在就能跑，给量级

完整流程见 `docs/EXECUTION_SOP.md`。
"""
from __future__ import annotations

import sys
import json
import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qf import config as C

FILL_DIR = ROOT / "journal" / "fills"
SIG = ROOT / "signals" / "latest_combo.json"
METRICS = ROOT / "build" / "runs" / "main" / "metrics.json"
COLS = ["signal_date", "trade_date", "code", "name", "kind", "leg", "side",
        "plan_shares", "unit", "ref_close", "open_px", "filled_px", "filled_shares", "note"]


# ---------------------------------------------------------------- 行情
def load_panel():
    z = np.load(pathlib.Path(C.DATA_RAW) / "panel_ohlcv.npz", allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c).zfill(6) for c in z["codes"]]
    return dates, codes, {k: z[k] for k in ("open", "high", "low", "close", "volume", "amount")}


def cell(panel, field, date, code, dates, codes):
    i = dates.get_indexer([pd.Timestamp(date)])[0]
    if i < 0:
        return np.nan
    j = codes.index(code) if code in codes else -1
    if j < 0:
        return np.nan
    return float(panel[field][i, j])


def next_trading_day(dates, d) -> pd.Timestamp | None:
    later = dates[dates > pd.Timestamp(d)]
    return later[0] if len(later) else None


# ---------------------------------------------------------------- 下单清单
LEG_ORDER = (("main", "stock", "股"), ("lowturn", "stock", "股"),
             ("etf", "etf", "份"), ("cb", "cb", "张"))


def order_list(legs: list[str] | None = None) -> tuple[dict, list[dict]]:
    """从 latest_combo.json 展开成逐笔清单（四腿都可能为空）。

    legs 给定则只保留这些腿，用于「分批建仓」——30 万本金一天填 60 笔不现实，
    先跑笔数少、费率低的那几条腿，是第一笔真实成交最快的路径。
    """
    c = json.loads(SIG.read_text(encoding="utf-8"))
    rows = []
    for key, kind, unit in LEG_ORDER:
        if legs and key not in legs:
            continue
        seg = c.get(key) or {}
        arr = seg.get("targets") or seg.get("legs") or []
        for x in arr:
            sh = int(x.get("shares") or 0)
            if sh <= 0:
                continue
            rows.append({"code": str(x["code"]).zfill(6), "name": x.get("name", ""),
                         "kind": kind, "leg": key, "unit": unit, "plan_shares": sh,
                         "price": float(x.get("price") or 0.0)})
    return c, rows


def cmd_new(args) -> None:
    dates, codes, panel = load_panel()
    legs_raw = getattr(args, "legs", None)
    legs = [s.strip() for s in legs_raw.split(",")] if legs_raw else None
    if legs:
        bad = [x for x in legs if x not in dict((k, 1) for k, _, _ in LEG_ORDER)]
        if bad:
            raise SystemExit(f"未知腿 {bad}，可选：{[k for k, _, _ in LEG_ORDER]}")
    combo, rows = order_list(legs)
    if not rows:
        raise SystemExit("该腿组当日无标的（可能空仓中），换一组或去掉 --legs")
    sig_date = pd.Timestamp(combo["signal_date"])
    if args.date:
        trade_date = pd.Timestamp(args.date)
    else:
        trade_date = next_trading_day(dates, sig_date)
        if trade_date is None:
            # 行情面板还没更新到信号日之后：按下一工作日预填，开盘价留空待补。
            trade_date = (sig_date + pd.offsets.BDay(1)).normalize()
            print(f"[提示] 行情面板止于 {dates[-1].date()}，模板按下一工作日 "
                  f"{trade_date.date()} 预填；若当天休市，用 --date 指定实际交易日")

    FILL_DIR.mkdir(parents=True, exist_ok=True)
    tag = ("_" + "-".join(legs)) if legs else ""
    out = FILL_DIR / f"fills_{trade_date.date()}{tag}.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} 已存在；要覆盖加 --force")

    recs = []
    for r in rows:
        # 行情面板是**后复权**价，信号里的 price 是**真实可下单**价。
        # 用信号日的「后复权收盘 / 信号价」求复权因子，把 T+1 开盘换算回真实价格空间，
        # 否则模板里会出现 51.44 而实际只要 5.32 这种没法核对的情况。
        adj_close = cell(panel, "close", sig_date, r["code"], dates, codes)
        ref = r["price"]
        ratio = (adj_close / ref) if (np.isfinite(adj_close) and ref > 0) else None
        op = cell(panel, "open", trade_date, r["code"], dates, codes)
        if ratio:
            op = op / ratio
            if np.isfinite(adj_close):
                ref = adj_close / ratio
        recs.append({"signal_date": str(sig_date.date()), "trade_date": str(trade_date.date()),
                     "code": r["code"], "name": r["name"], "kind": r["kind"],
                     "leg": r["leg"], "side": "买入",
                     "plan_shares": r["plan_shares"], "unit": r["unit"],
                     "ref_close": round(ref, 4) if np.isfinite(ref) and ref > 0 else "",
                     "open_px": round(op, 4) if np.isfinite(op) and op > 0 else "",
                     "filled_px": "", "filled_shares": "",
                     "note": ""})
    df = pd.DataFrame(recs, columns=COLS)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[模板] {out}   {len(df)} 笔  信号日 {sig_date.date()} → 交易日 {trade_date.date()}")
    miss = sum(1 for x in recs if x["open_px"] == "")
    if miss:
        print(f"       {miss} 笔的 T+1 开盘价还不可得（交易日未到），成交后补 filled_px 即可")
    print("       填完 filled_px / filled_shares 后跑 `python scripts/fill_log.py report`")


# ---------------------------------------------------------------- 批量回填 / 进度
ASSUMED_TAG = "批量回填"


def _find_files(date_str: str) -> list[pathlib.Path]:
    if not FILL_DIR.exists():
        return []
    return sorted(f for f in FILL_DIR.glob(f"fills_{date_str}*.csv"))


def cmd_fill(args) -> None:
    """批量回填：把某日模板一次填满，只用于「账本对齐」，不产生真实滑点。

    ⚠ 按开盘价回填算出的滑点恒为 0，那不是实测值。`report` 会把这类行
    单独归入「假设组」，不会污染实测滑点。真正要校准滑点，必须逐笔填真实成交价。
    """
    files = _find_files(args.date)
    if not files:
        raise SystemExit(f"没找到 fills_{args.date}*.csv，先跑 `new`")
    col = "open_px" if args.mode == "open" else "ref_close"
    label = "T+1 开盘价" if args.mode == "open" else "信号日收盘价"

    for f in files:
        df = pd.read_csv(f, dtype={"code": str})
        for c in ("filled_px", "filled_shares", "note"):
            # 空列会被 pandas 读成 float64，直接赋字符串会 TypeError；统一转 object
            if c not in df.columns:
                df[c] = pd.Series([""] * len(df), dtype=object)
            else:
                df[c] = df[c].astype(object)
        have = pd.to_numeric(df["filled_px"], errors="coerce").fillna(0) > 0
        need = pd.Series(True, index=df.index) if args.force else ~have
        src = pd.to_numeric(df[col], errors="coerce")
        ok = need & src.gt(0)
        df.loc[ok, "filled_px"] = src[ok].round(4)
        df.loc[ok, "filled_shares"] = df.loc[ok, "plan_shares"]
        df.loc[ok, "note"] = f"{ASSUMED_TAG}:{args.mode}（{label}，待逐笔核对）"
        df.to_csv(f, index=False, encoding="utf-8-sig")
        print(f"[回填] {f.name}  填入 {int(ok.sum())} 笔（{label}）"
              f"，跳过已有 {int((need & ~ok).sum())} 笔（{col} 不可得）")
    print(f"\n⚠ 这批是「假设成交」，滑点恒为 0。要校准成本，逐笔改成券商回报的真实成交价，"
          f"note 留空即可自动计入实测组。")


def cmd_list(args) -> None:
    """把模板打印成能照着下单的清单：代码/股数/参考价/金额/建议限价。"""
    files = _find_files(args.date) if args.date else sorted(
        f for f in FILL_DIR.glob("fills_*.csv") if f.name != "_summary.csv")
    if not files:
        raise SystemExit(f"没找到 fills_{args.date or '*'}.csv，先跑 `new`")

    for f in files:
        df = pd.read_csv(f, dtype={"code": str})
        px = pd.to_numeric(df["ref_close"], errors="coerce")
        sh = pd.to_numeric(df["plan_shares"], errors="coerce").fillna(0)
        amt = px * sh
        dec = 3 if ("kind" in df and df["kind"].eq("cb").any()) else 2
        lim = (px * (1 + args.limit_bp / 1e4)).round(dec)
        td = str(df["trade_date"].iloc[0]) if len(df) else "?"
        legs = "+".join(sorted(df["leg"].unique())) if "leg" in df else "?"
        print("=" * 82)
        print(f"{td} · {legs} · {len(df)} 笔")
        print("=" * 82)
        print(f"{'腿':<9}{'代码':<8}{'名称':<10}{'数量':>8}{'参考价':>9}"
              f"{'金额':>11}{'建议限价':>10}")
        print("-" * 82)
        for i, (_, r) in enumerate(df.iterrows()):
            print(f"{r['leg']:<9}{r['code']:<8}{str(r['name'])[:8]:<10}"
                  f"{int(r['plan_shares']):>8}{px.iloc[i]:>9.3f}"
                  f"{amt.iloc[i]:>11,.0f}{lim.iloc[i]:>10.2f}")
        n = len(df)
        fee = n * args.min_fee
        print("-" * 82)
        print(f"{'合计':<9}{'':<8}{'':<10}{n:>8} 笔{'':>4}"
              f"{amt.sum():>11,.0f}{'费率':>10}")
        print(f"  佣金按 {args.min_fee:.0f} 元/笔（最低佣金）估 {fee:,.0f} 元"
              f" = {fee / max(amt.sum(), 1) * 1e4:.1f}bp")
        print(f"  建议限价 = 参考价 × (1 + {args.limit_bp}bp)。"
              f"开盘价高于限价的放弃该笔，在 note 写「未成交:高开」。")
        if "kind" in df and (df["kind"] == "cb").any():
            print("  转债按张计（1 张 = 100 元面值），沪市 10 张起、深市 10 张起。")


def cmd_status(args) -> None:
    """看每个模板的执行进度，解决「我到底做没做」这个问题。"""
    files = sorted(FILL_DIR.glob("fills_*.csv")) if FILL_DIR.exists() else []
    files = [f for f in files if f.name != "_summary.csv"]
    if not files:
        raise SystemExit(f"{FILL_DIR} 下没有 fills_*.csv")
    try:
        dates = load_panel()[0]
        last = dates[-1]
    except Exception:
        last, dates = None, None

    print(f"{'交易日':<12}{'文件':<34}{'笔数':>5}{'实测':>6}{'假设':>6}  状态")
    print("-" * 84)
    n_todo = 0
    todo_days: set = set()
    for f in files:
        df = pd.read_csv(f, dtype={"code": str})
        n = len(df)
        px = pd.to_numeric(df.get("filled_px"), errors="coerce").fillna(0) > 0
        note = df.get("note", pd.Series([""] * n)).fillna("").astype(str)
        assumed = px & note.str.contains(ASSUMED_TAG)
        measured = px & ~note.str.contains(ASSUMED_TAG)
        try:
            td = pd.Timestamp(df["trade_date"].iloc[0])
        except Exception:
            td = pd.NaT
        if measured.sum() == n:
            state = "已完成"
        elif px.sum() == 0:
            age = int(np.sum(dates > td)) if (dates is not None and pd.notna(td)) else 0
            state = (f"未执行（落后 {age} 个交易日）" if age > 0
                     else "未执行（最近交易日）")
            n_todo += 1
            if pd.notna(td):
                todo_days.add(td)
        else:
            state = f"部分完成，还差 {n - int(px.sum())} 笔"
            n_todo += 1
            if pd.notna(td):
                todo_days.add(td)
        print(f"{str(td.date()) if pd.notna(td) else '?':<12}{f.name:<34}"
              f"{n:>5}{int(measured.sum()):>6}{int(assumed.sum()):>6}  {state}")
    if n_todo:
        print(f"\n→ {len(todo_days)} 个交易日 / {n_todo} 个文件未完成。执行后跑 "
              f"`python scripts/fill_log.py fill --date <交易日> --mode open` 先对齐账本，"
              f"再把真实成交价逐笔覆盖。")


# ---------------------------------------------------------------- 实测报告
def _slip(row) -> float:
    f, o = float(row["filled_px"]), float(row["open_px"])
    if not (np.isfinite(f) and np.isfinite(o)) or f <= 0 or o <= 0:
        return np.nan
    return (f / o - 1.0) * 1e4 if row["side"] == "买入" else (o - f) / o * 1e4


def _total(row) -> float:
    f, c0 = float(row["filled_px"]), float(row["ref_close"])
    if not (np.isfinite(f) and np.isfinite(c0)) or f <= 0 or c0 <= 0:
        return np.nan
    return (f / c0 - 1.0) * 1e4 if row["side"] == "买入" else (c0 - f) / c0 * 1e4


def cmd_report(args) -> None:
    files = sorted(FILL_DIR.glob("fills_*.csv")) if FILL_DIR.exists() else []
    if not files:
        raise SystemExit(f"{FILL_DIR} 下没有 fills_*.csv，先跑 `new`")
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files], ignore_index=True)
    df = df.dropna(subset=["filled_px"])
    df = df[pd.to_numeric(df["filled_px"], errors="coerce").fillna(0) > 0].copy()
    if len(df) == 0:
        raise SystemExit(f"读了 {len(files)} 个模板，但没有一笔填了 filled_px —— 还无法校准")

    df["slip_bp"] = df.apply(_slip, axis=1)
    df["total_bp"] = df.apply(_total, axis=1)

    note = df["note"].fillna("").astype(str) if "note" in df.columns else pd.Series([""] * len(df))
    is_assumed = note.str.contains(ASSUMED_TAG)
    px_s = pd.to_numeric(df["filled_px"], errors="coerce")
    sh_s = pd.to_numeric(df["filled_shares"], errors="coerce").fillna(0)
    amt = float((px_s * sh_s).sum())
    print("=" * 78)
    print(f"记账规模：{len(files)} 个交易日、{len(df)} 笔已成交、合计 {amt:,.0f} 元")
    print(f"  其中实测 {int((~is_assumed).sum())} 笔、假设成交 {int(is_assumed.sum())} 笔"
          f"（假设组滑点恒为 0，不计入下面的校准）")
    print("=" * 78)

    df_meas = df[~is_assumed]
    if len(df_meas) == 0:
        print("\n还没有一笔真实成交价 —— 滑点无法校准。")
        print("把券商回报的成交均价填进 filled_px（note 留空），再跑一次本命令。")
        print("当前只能给方向：假设组按开盘价成交，所以真实滑点只会比 0 更差。")
        df.to_csv(FILL_DIR / "_summary.csv", index=False, encoding="utf-8-sig")
        return
    df = df_meas

    ok = df.dropna(subset=["slip_bp"])
    n = len(ok)
    x = ok["slip_bp"].to_numpy()

    turnover = args.turnover
    if turnover is None:
        turnover = float(json.loads(METRICS.read_text(encoding="utf-8"))["turnover_ann_double"])

    print(f"\n实测滑点（{n} 笔有真实成交价）")
    print("-" * 78)
    print(f"  均值        {x.mean():+7.2f} bp")
    print(f"  中位        {np.median(x):+7.2f} bp")
    print(f"  标准差      {x.std(ddof=1) if n > 1 else 0:7.2f} bp")
    if n >= 5:
        rng = np.random.default_rng(42)
        bs = np.array([rng.choice(x, n, replace=True).mean() for _ in range(10000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  95% CI 均值 [{lo:+.2f}, {hi:+.2f}] bp   （n={n}，bootstrap 10000）")
    # 金额加权（大单更重要）
    w = pd.to_numeric(ok["filled_px"], errors="coerce") * \
        pd.to_numeric(ok["filled_shares"], errors="coerce").fillna(0)
    if w.sum() > 0:
        print(f"  金额加权    {(x * w / w.sum()).sum():+7.2f} bp")
    print(f"  相对信号日收盘（含隔夜跳空）均值 {ok['total_bp'].mean():+.2f} bp"
          f"  ← 这部分回测已经吃掉，不算缺失")

    print("\n" + "=" * 78)
    print(f"反推年化成本（年双边换手 {turnover:.3f}x）")
    print("=" * 78)
    print(f"  {'假设滑点':>10}{'年化成本':>12}{'vs 40bp 假设':>16}")
    base = 40.0
    for b in (x.mean(), base):
        ann = b / 1e4 * turnover
        print(f"  {b:>9.1f}bp{ann*100:>11.3f}%{-(ann - base/1e4*turnover)*100:>+15.3f}pp")
    delta = (base - x.mean()) / 1e4 * turnover          # 正数 = 真实成本比模型低
    verdict = "偏高（可上调收益）" if x.mean() < base else "偏低（要下调收益）"
    print(f"\n  → 实测均值 {x.mean():.1f}bp vs 配置 40bp：{verdict}")
    vol = float(json.loads(METRICS.read_text(encoding="utf-8"))["ann_vol"])
    print(f"     若把 SLIPPAGE_BP 改成实测值：年化收益 {delta*100:+.3f}pp、"
          f"Sharpe {delta/vol:+.4f}")
    if n < 20:
        print(f"\n  ⚠ n={n} < 20，单笔噪声可达 ±50bp，此结论只能当方向，别改配置")
    df.to_csv(FILL_DIR / "_summary.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- 代理估计
def cmd_proxy(args) -> None:
    dates, codes, panel = load_panel()
    combo, rows = order_list()
    px = {r["code"]: r["price"] for r in rows}
    target = list(px)[:args.n]

    print("=" * 78)
    print(f"代理估计（{len(target)} 只当前持仓，最近 {args.days} 个交易日日线）")
    print("=" * 78)
    i_end = len(dates) - 1
    i_beg = max(1, i_end - args.days)
    tick_bp, drift, rngs = [], [], []
    for code in target:
        if code not in codes:
            continue
        j = codes.index(code)
        p = px[code] or float(np.nanmedian(panel["close"][i_beg:i_end, j]))
        if np.isfinite(p) and p > 0:
            tick_bp.append(0.01 / p * 1e4)          # 1 个最小变动价位 = 多少钱
        for i in range(i_beg, i_end):
            o, c = panel["open"][i, j], panel["close"][i, j]
            hi, lo = panel["high"][i, j], panel["low"][i, j]
            if not (np.isfinite(o) and np.isfinite(c)) or o <= 0 or c <= 0:
                continue
            d = (c / o - 1.0) * 1e4
            if abs(d) < 2000:                        # 剔除复权跳空等异常
                drift.append(d)
            if np.isfinite(hi) and np.isfinite(lo) and (hi + lo) > 0:
                rngs.append((hi - lo) / ((hi + lo) / 2) * 1e4)

    tb = np.array(tick_bp)
    dr = np.array(drift)
    rg = np.array(rngs)
    print(f"\n  最低变动价位（1 tick = 0.01 元）折算：")
    print(f"    中位 {np.median(tb):6.2f} bp   均值 {tb.mean():6.2f} bp   "
          f"最贵 {tb.max():6.2f} bp  最便宜 {tb.min():6.2f} bp")
    print(f"    → 40bp 折合 {40/np.median(tb):.2f} 个 tick（中位股）")
    print(f"    → 若价差 = 1 tick，只付半价差，则「必须付」的部分约 "
          f"{np.median(tb)*0.5:.1f} bp")

    print(f"\n  日内价格漂移（{len(dr)} 个样本，后复权同空间）：")
    print(f"    开盘→收盘  中位 {np.median(dr):+6.2f} bp   均值 {dr.mean():+6.2f} bp   "
          f"标准差 {dr.std():6.2f} bp")
    print(f"    |开盘→收盘| 中位 {np.median(np.abs(dr)):.2f} bp   "
          f"75 分位 {np.percentile(np.abs(dr), 75):.2f} bp")
    print(f"    日内振幅 (high−low)/中价 中位 {np.median(rg):.1f} bp")
    print(f"    开盘后 30 分钟的典型漂移 ≈ 全天 × sqrt(30/240) = "
          f"{np.median(np.abs(dr))*np.sqrt(30/240):.2f} bp")
    print("    正数 = 收盘高于开盘。这一行衡量「坚持在开盘价成交」要承担的价格暴露，")
    print("    不是滑点 —— 但它给滑点设了个物理量级。")

    print("\n" + "=" * 78)
    print("怎么读这张表")
    print("=" * 78)
    print("  40bp 拆成三块：半价差 + 成交价劣化 + 缓冲。")
    print(f"  日线只能给前两块的下界：半价差 ≈ {np.median(tb)*0.5:.0f}bp，"
          f"开盘后 30 分钟漂移量级 ≈ {np.median(np.abs(dr))*np.sqrt(30/240):.0f}bp。")
    print(f"  两块相加 ≈ {np.median(tb)*0.5 + np.median(np.abs(dr))*np.sqrt(30/240):.0f}bp，"
          f"与配置的 40bp 相比" +
          ("偏高（40bp 留了充足缓冲）" if 40 > np.median(tb)*0.5 + np.median(np.abs(dr))*np.sqrt(30/240) * 1.5
           else "接近（40bp 基本贴合）"))
    print("  真实滑点仍必须靠 `new` + `report` 实测，这里只回答「40bp 是不是离谱」。")
    if 40 / np.median(tb) > 4:
        print("  → 40bp 超过 4 个 tick，若持仓不是极冷门小盘，这个数偏保守。")


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="滑点校准：模板生成 / 实测反推 / 代理估计")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("new", help="从当日信号生成成交记录模板")
    p1.add_argument("--date", help="交易日 YYYY-MM-DD，缺省为信号日的下一个交易日")
    p1.add_argument("--legs", help="只生成这些腿，逗号分隔（如 lowturn,cb）——分批建仓用")
    p1.add_argument("--force", action="store_true")
    p1.set_defaults(func=cmd_new)

    p4 = sub.add_parser("fill", help="批量回填成交价（只对齐账本，不产生真实滑点）")
    p4.add_argument("--date", required=True, help="交易日 YYYY-MM-DD，可只写日期前缀")
    p4.add_argument("--mode", choices=["open", "close"], default="open",
                    help="open=按 T+1 开盘价；close=按信号日收盘价")
    p4.add_argument("--force", action="store_true", help="覆盖已填的行")
    p4.set_defaults(func=cmd_fill)

    p5 = sub.add_parser("status", help="查看每个模板的执行进度")
    p5.set_defaults(func=cmd_status)

    p6 = sub.add_parser("list", help="打印能照着下单的清单（含建议限价）")
    p6.add_argument("--date", help="交易日，缺省为全部模板")
    p6.add_argument("--limit-bp", type=float, default=200.0, help="限价上浮，默认 200bp")
    p6.add_argument("--min-fee", type=float, default=5.0,
                    help="每笔最低佣金（免5的券商填 0.1），默认 5 元")
    p6.set_defaults(func=cmd_list)

    p2 = sub.add_parser("report", help="读已填模板，算实测滑点")
    p2.add_argument("--turnover", type=float, help="年双边换手，缺省读 metrics.json")
    p2.set_defaults(func=cmd_report)

    p3 = sub.add_parser("proxy", help="用日线做代理估计（不能替代实测）")
    p3.add_argument("--days", type=int, default=120)
    p3.add_argument("--n", type=int, default=40)
    p3.set_defaults(func=cmd_proxy)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
