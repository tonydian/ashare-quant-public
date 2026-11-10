# -*- coding: utf-8 -*-
"""实盘对账：计划单 → 成交 → 持仓 → 净值，三层对账。

## 为什么需要这个

`scripts/fill_log.py` 解决的是**一个参数**：SLIPPAGE_BP=40 到底贵不贵。
它只看「已成交那几行」的成交价，不回答下面三个更要命的问题：

1. **数量对不对** —— 计划 60 笔，实际成交几笔？有漏单吗？部分成交吗？
2. **钱对不对** —— 买进去的钱、佣金、现金余额，和信号算的差多少？
   小单撞上 5 元最低佣金时，实际费率能到 30bp 以上，这部分回测有建模但没人核对过。
3. **净值差多少** —— 按实际成交重建组合，和「按开盘价完美执行」的影子组合比，差多少？

## 进场价口径（写死，不可改）

    **理论进场价 = 成交日开盘价**（与 `qf/execution.py` 一致：回测就是 t+1 开盘成交）

于是所有偏差被拆成互不重叠的三段：

| 指标 | 算法 | 谁的责任 |
|---|---|---|
| `gap_bp` 隔夜跳空 | `open / ref_close − 1` | **回测也吃了**，不算你的滑点 |
| `exec_bp` 执行偏差 | `filled / open − 1` | 你自己的问题（挂单价、下单时点、冲击） |
| `fee_bp` 费用 | 佣金/印花/过户 ÷ 成交额 | 券商费率 + 最低佣金 |

    `filled / ref_close − 1` = gap + exec，两个混在一起，
    所以这里**不报这个数**，只报拆开后的两段 —— 否则会把隔夜跳空算到你头上。

## 与回测假设的对照

各腿回测的成本口径本来就不一样，不能用一个 40bp 套全部：

| 腿 | 回测在成交价里让掉的 | 来源 |
|---|---|---|
| `main` 主策略 | 40bp | `C.SLIPPAGE_BP`（`execution.py` 买价上浮） |
| `lowturn` 低换手 | 49.3bp | `qf/lowturn.py:COST_BP`（含佣金 6.67/印花 2.5/过户 0.1） |
| `etf` 宽基 | 5bp | `qf/sector_rotation.py:COST_BP` |
| `cb` 可转债 | 5bp | `qf/cb.py:cost_bp` |

所以「实盘贵了多少」= `exec_bp + fee_bp − 该腿的假设`。为负 = 实盘比模型便宜。

## 跑法

    python scripts/live_recon.py plan                 # 今天的单子体检（无需成交，现在就能跑）
    python scripts/live_recon.py recon                # 逐笔对账（需要填 filled_px/filled_shares）
    python scripts/live_recon.py nav                  # 实盘净值 vs 影子净值
    python scripts/live_recon.py plan --date 2026-11-06
    python scripts/live_recon.py nav --capital 300000 --holdings 初始持仓.csv

## 已知限制（必须随结论一起引用）

1. **行情数据只到 `md.dates[-1]`**。成交日晚于这个日期时，开盘价拿不到，
   脚本会把该笔标 `px_avail=False`，只做数量/预算检查，价格类指标跳过 ——
   **不会拿信号日收盘价冒充开盘价**（那会把跳空算成滑点）。
2. **转债价格取 `cb_daily.parquet` 原始价**（元/张），不做复权；股票用 `md.price`
   （常数缩放后复权，≈实际价，与 `ref_close` 同源，已核对）。
3. **初始持仓默认空**。若账户里本来就有票，用 `--holdings` 传一份
   `code,shares` 的 CSV；否则「卖出未持有标的」会被标成异常 —— 这正是想要的效果，
   因为那通常意味着成交表填错了。
4. `nav` 模式对闲置现金按国债指数计息（与生产 `cash_sleeve` 一致，可在 `--cash-yield` 关掉），
   这样实盘与影子的**绝对**收益才对得上，而两者之差不受影响。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qf import config as C

FILL_DIR = ROOT / "journal" / "fills"
RECON_DIR = ROOT / "journal" / "recon"
COMBO = ROOT / "signals" / "latest_combo.json"
CB_DAILY = C.BUILD / "altdata" / "cb_daily.parquet"

# 各腿回测在**成交价**里让掉的部分（bp，单边）。改这里必须同步改对应模块。
LEG_PRICE_BP = {"main": 40.0, "lowturn": 49.3, "etf": 5.0, "cb": 5.0}
LEG_CN = {"main": "主策略", "lowturn": "低换手", "etf": "宽基ETF", "cb": "可转债"}


# ============================================================ 纯函数（可单测）
def _rel(p: pathlib.Path) -> str:
    """相对路径优先（屏幕短），不在仓库内就返回绝对路径（测试会指到临时目录）。"""
    try:
        return str(pathlib.Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def leg_price_bp(leg: str) -> float:
    return float(LEG_PRICE_BP.get(leg, C.SLIPPAGE_BP))


def exec_bp(filled: float, open_px: float, side: str) -> float:
    """执行偏差：正数 = 吃亏。买入贵了为正，卖出贱了为正。"""
    if not (np.isfinite(filled) and np.isfinite(open_px)) or filled <= 0 or open_px <= 0:
        return np.nan
    if side.startswith("买"):
        return (filled / open_px - 1.0) * 1e4
    return (open_px - filled) / open_px * 1e4


def gap_bp(open_px: float, ref_close: float) -> float:
    """隔夜跳空（买入口径）：正数 = 开盘比信号日收盘高，回测同样承担。"""
    if not (np.isfinite(open_px) and np.isfinite(ref_close)) or open_px <= 0 or ref_close <= 0:
        return np.nan
    return (open_px / ref_close - 1.0) * 1e4


def model_fill(open_px: float, leg: str, side: str) -> float:
    """回测口径的成交价：开盘价上让掉该腿的假设成本。"""
    b = leg_price_bp(leg) / 1e4
    return open_px * (1.0 + b) if side.startswith("买") else open_px * (1.0 - b)


def stock_fee(amount: float, side: str) -> float:
    """个股实盘费用：佣金（含 5 元最低）+ 过户费 + 卖出印花税。"""
    if amount <= 0:
        return 0.0
    fee = max(C.MIN_COMMISSION, C.COMMISSION_RATE * amount) + C.TRANSFER_FEE * amount
    if not side.startswith("买"):
        fee += C.STAMP_TAX_SELL * amount
    return float(fee)


def cb_fee(amount: float) -> float:
    """可转债费用：按 5bp 单边估（与 qf/cb.py 一致）。

    真实转债佣金一般万 0.5~1、且**没有 5 元最低**，所以这里偏保守。
    转债无印花税、无过户费。
    """
    return float(max(0.0, amount) * 5e-4)


def trade_fee(kind: str, amount: float, side: str) -> float:
    return cb_fee(amount) if kind == "cb" else stock_fee(amount, side)


def classify(plan_shares, filled_shares) -> str:
    """成交状态分类。"""
    p = float(plan_shares) if pd.notna(plan_shares) else 0.0
    f = float(filled_shares) if pd.notna(filled_shares) else 0.0
    if f <= 0:
        return "未成交"
    if f < p - 0.5:
        return "部分成交"
    if f > p + 0.5:
        return "超量"
    return "足额"


def wavg(x: pd.Series, w: pd.Series) -> float:
    """金额加权均值；权重全零或含 NaN 时退回等权。"""
    xx = pd.to_numeric(x, errors="coerce")
    ww = pd.to_numeric(w, errors="coerce").fillna(0.0)
    m = xx.notna() & (ww > 0)
    if not m.any():
        return float(xx.mean()) if xx.notna().any() else float("nan")
    return float((xx[m] * ww[m]).sum() / ww[m].sum())


# ============================================================ 数据装载
def load_combo() -> dict:
    return json.loads(COMBO.read_text(encoding="utf-8")) if COMBO.exists() else {}


def leg_map(combo: dict) -> dict[str, str]:
    """code → 腿名。成交表老文件没有 leg 列时靠它补。"""
    out: dict[str, str] = {}
    for key, leg in (("main", "main"), ("lowturn", "lowturn"), ("etf", "etf"), ("cb", "cb")):
        seg = combo.get(key) or {}
        for x in (seg.get("targets") or seg.get("legs") or []):
            code = str(x.get("code", "")).zfill(6)
            if code.strip("0"):
                out.setdefault(code, leg)
    return out


def load_fills(date: str | None = None) -> tuple[pd.DataFrame, list[pathlib.Path]]:
    files = sorted(FILL_DIR.glob("fills_*.csv")) if FILL_DIR.exists() else []
    if date:
        # 前缀匹配：分批模板名为 fills_<日期>_<腿组>.csv
        files = [f for f in files if f.stem.startswith(f"fills_{date}")]
    if not files:
        raise SystemExit(f"{FILL_DIR} 下没有匹配的 fills_*.csv，先跑 "
                         f"`python scripts/fill_log.py new`")
    df = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in files], ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "leg" not in df.columns:
        lm = leg_map(load_combo())
        df["leg"] = df.apply(
            lambda r: r.get("leg") if isinstance(r.get("leg"), str) else lm.get(r["code"], "main"),
            axis=1)
    df["leg"] = df["leg"].fillna("main")
    return df, files


class Prices:
    """价格提供者：股票走 md（常数缩放后复权 ≈ 实际价），转债走 cb_daily 原始价。"""

    def __init__(self, verbose: bool = False):
        from qf import data as qfdata
        self.md = qfdata.load_market(verbose=verbose)
        self.dates = self.md.dates
        self.last = self.dates[-1]
        self._d2i = {d: i for i, d in enumerate(self.dates)}
        self._cb = None

    def cb(self) -> pd.DataFrame:
        if self._cb is None:
            if not CB_DAILY.exists():
                self._cb = pd.DataFrame(columns=["date", "code", "open", "close"])
            else:
                d = pd.read_parquet(CB_DAILY, columns=["date", "code", "open", "close"])
                d["date"] = pd.to_datetime(d["date"])
                d["code"] = d["code"].astype(str).str.zfill(6)
                self._cb = d.set_index(["date", "code"]).sort_index()
        return self._cb

    def available(self, date) -> bool:
        return pd.Timestamp(date) <= self.last

    def open_close(self, code: str, kind: str, date) -> tuple[float, float]:
        """返回 (开盘价, 收盘价)，拿不到就是 NaN。"""
        d = pd.Timestamp(date)
        if not self.available(d):
            return np.nan, np.nan
        if kind == "cb":
            try:
                row = self.cb().loc[(d, code)]
                return float(row["open"]), float(row["close"])
            except KeyError:
                return np.nan, np.nan
        j = self.md.code_index(code)
        if j < 0:
            return np.nan, np.nan
        i = self._d2i.get(d)
        if i is None:
            return np.nan, np.nan
        return float(self.md.price_open[i, j]), float(self.md.price[i, j])

    def close(self, code: str, kind: str, date) -> float:
        return self.open_close(code, kind, date)[1]


# ============================================================ plan：计划单体检
def cmd_plan(args) -> None:
    df, files = load_fills(args.date)
    px = Prices() if args.prices else None
    cap = args.capital or float(load_combo().get("capital") or 0) or np.nan

    rows = []
    for _, r in df.iterrows():
        open_px, close_px = (np.nan, np.nan)
        if px is not None:
            open_px, close_px = px.open_close(r["code"], r["kind"], r["trade_date"])
        ref = float(r["ref_close"]) if pd.notna(r["ref_close"]) and r["ref_close"] != "" else np.nan
        # 计划金额：优先用开盘价（真实要付的钱），拿不到退回信号日收盘
        est_px = open_px if np.isfinite(open_px) else ref
        amt = float(r["plan_shares"]) * est_px if np.isfinite(est_px) else np.nan
        fee = trade_fee(r["kind"], amt, r["side"]) if np.isfinite(amt) else np.nan
        rows.append({"code": r["code"], "name": r["name"], "leg": r["leg"], "kind": r["kind"],
                     "side": r["side"], "shares": float(r["plan_shares"]),
                     "ref_close": ref, "open_px": open_px,
                     "px_avail": bool(np.isfinite(open_px)),
                     "est_px": est_px, "amount": amt, "fee": fee,
                     "fee_bp": fee / amt * 1e4 if np.isfinite(amt) and amt > 0 else np.nan,
                     "not_round_lot": bool(r["kind"] != "cb" and int(r["plan_shares"]) % 100)})
    d = pd.DataFrame(rows)

    print("=" * 86)
    head = f"计划单体检  {len(files)} 个交易日 · {len(d)} 笔"
    if np.isfinite(cap):
        head += f" · 基准资金 {cap:,.0f}"
    print(head)
    print("=" * 86)

    g = d.groupby("leg").agg(笔数=("code", "size"), 金额=("amount", "sum"),
                             费用=("fee", "sum")).reindex(
        ["main", "lowturn", "etf", "cb"]).dropna(how="all")
    g["腿名"] = [LEG_CN.get(k, k) for k in g.index]
    tot = g["金额"].sum()
    g["占比"] = g["金额"] / tot if tot else np.nan
    g["费率bp"] = g["费用"] / g["金额"] * 1e4
    print(g[["腿名", "笔数", "金额", "占比", "费用", "费率bp"]].to_string(
        float_format=lambda x: f"{x:,.2f}"))
    print(f"\n  合计买入 {tot:,.0f} 元，费用 {g['费用'].sum():,.2f} 元，"
          f"加权费率 {g['费用'].sum() / tot * 1e4:.2f}bp")
    if np.isfinite(cap):
        print(f"  占总资产 {tot / cap:.2%}，执行后现金 {cap - tot - g['费用'].sum():,.0f} 元")

    # 最低佣金侵蚀：这是小资金最容易忽略的一块
    small = d[(d["leg"].isin(["main", "lowturn"])) & (d["amount"] < 3000)]
    if len(small):
        naive = small["amount"].sum() * C.COMMISSION_RATE
        real = small["fee"].sum() - small["amount"].sum() * C.TRANSFER_FEE
        print(f"\n  ⚠ 主策略/低换手有 {len(small)} 笔金额 < 3000 元（均值 "
              f"{small['amount'].mean():,.0f} 元）：佣金按 5 元最低收，"
              f"实付 {real:,.0f} 元 vs 费率制 {naive:,.0f} 元，多付 {real - naive:,.0f} 元"
              f"（{small['fee_bp'].mean():.1f}bp/笔）")

    # 数据滞后提醒
    stale = d[~d["px_avail"]]
    if len(stale):
        if px is None:
            why = "未加载行情（加 --prices 可检查开盘价是否已可得）"
        else:
            why = f"行情止于 {px.last.date()}"
        print(f"\n  ⚠ {len(stale)} 笔开盘价不可得（{why}）：计划金额用信号日收盘价估算，"
              f"**不是**开盘价；成交后填 filled_px 再跑 recon")

    bad = d[d["not_round_lot"]]
    if len(bad):
        print(f"  ⚠ {len(bad)} 笔股数不是 100 的整数倍：{bad['code'].tolist()[:8]}")
    tiny = d[(d["amount"] < 500)]
    if len(tiny):
        print(f"  ⚠ {len(tiny)} 笔金额 < 500 元，佣金占比 > {tiny['fee_bp'].min():.0f}bp，"
              f"建议跳过或与同腿合并：{tiny['code'].tolist()[:8]}")

    RECON_DIR.mkdir(parents=True, exist_ok=True)
    tag = (pd.Timestamp(df["trade_date"].iloc[0]).strftime("%Y-%m-%d")
           if len(df) else files[-1].stem.replace("fills_", ""))
    legs_tag = "-".join(sorted(set(df["leg"].astype(str)))) if "leg" in df.columns else ""
    out = RECON_DIR / f"plan_{tag}{('_' + legs_tag) if legs_tag else ''}.json"
    out.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "files": [f.name for f in files], "capital": cap,
        "total_amount": float(tot), "total_fee": float(g["费用"].sum()),
        "by_leg": {k: {"n": int(v["笔数"]), "amount": float(v["金额"]), "fee": float(v["费用"])}
                   for k, v in g.iterrows()},
        "orders": d.astype(object).where(pd.notna(d), None).to_dict("records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  → {_rel(out)}")


# ============================================================ recon：逐笔对账
def cmd_recon(args) -> None:
    df, files = load_fills(args.date)
    px = Prices()
    df["filled_px"] = pd.to_numeric(df["filled_px"], errors="coerce")
    df["filled_shares"] = pd.to_numeric(df["filled_shares"], errors="coerce")
    df["open_px_f"] = [px.open_close(c, k, d)[0] for c, k, d in
                       zip(df["code"], df["kind"], df["trade_date"])]
    # 模板里的 open_px 若已填，优先用它（那是下单当天看到的价）
    tp = pd.to_numeric(df.get("open_px"), errors="coerce")
    df["open_px_f"] = tp.where(np.isfinite(tp) & (tp > 0), df["open_px_f"])
    df["ref_close"] = pd.to_numeric(df["ref_close"], errors="coerce")

    df["status"] = [classify(p, f) for p, f in zip(df["plan_shares"], df["filled_shares"])]
    df["exec_bp"] = [exec_bp(f, o, s) for f, o, s in zip(df["filled_px"], df["open_px_f"], df["side"])]
    df["gap_bp"] = [gap_bp(o, c) for o, c in zip(df["open_px_f"], df["ref_close"])]
    df["model_px"] = [model_fill(o, l, s) for o, l, s in
                      zip(df["open_px_f"], df["leg"], df["side"])]
    df["model_bp"] = [leg_price_bp(l) for l in df["leg"]]
    df["amount"] = df["filled_px"] * df["filled_shares"]
    df["fee"] = [trade_fee(k, a, s) if np.isfinite(a) else np.nan for k, a, s in
                 zip(df["kind"], df["amount"], df["side"])]
    df["fee_bp"] = df["fee"] / df["amount"] * 1e4
    df["real_bp"] = df["exec_bp"] + df["fee_bp"]
    df["excess_bp"] = df["real_bp"] - df["model_bp"]

    print("=" * 86)
    print(f"逐笔对账  {len(files)} 个交易日 · 计划 {len(df)} 笔")
    print("=" * 86)
    st = df["status"].value_counts()
    filled = df[df["status"] != "未成交"]
    print("  成交状态：" + "  ".join(f"{k} {v}" for k, v in st.items())
          + f"   成交率 {len(filled) / len(df):.1%}"
          + f"   股数完成率 {df['filled_shares'].fillna(0).sum() / df['plan_shares'].sum():.1%}")
    if len(filled) == 0:
        print("\n  还没有任何成交记录，价格类指标无法计算。")
        print("  填完 journal/fills/*.csv 的 filled_px / filled_shares 后再跑本命令。")
        return

    print(f"\n  {'腿':<10}{'笔数':>5}{'执行偏差':>11}{'费用':>9}{'实盘合计':>11}"
          f"{'回测假设':>11}{'超额':>9}")
    for leg, sub in filled.groupby("leg"):
        w = sub["amount"]
        e, f_ = wavg(sub["exec_bp"], w), wavg(sub["fee_bp"], w)
        r, m = e + f_, leg_price_bp(leg)
        print(f"  {LEG_CN.get(leg, leg):<10}{len(sub):>5}{e:>+11.1f}{f_:>9.1f}"
              f"{r:>+11.1f}{m:>11.1f}{r - m:>+9.1f}")
    e, f_ = wavg(filled["exec_bp"], filled["amount"]), wavg(filled["fee_bp"], filled["amount"])
    m = wavg(filled["model_bp"], filled["amount"])
    print(f"  {'加权合计':<10}{len(filled):>5}{e:>+11.1f}{f_:>9.1f}{e + f_:>+11.1f}"
          f"{m:>11.1f}{e + f_ - m:>+9.1f}")
    print("\n  执行偏差 = 成交价 vs **开盘价**（正=吃亏）；费用含佣金/印花/过户；"
          "回测假设 = 各腿成交价让价")

    g = filled["gap_bp"].dropna()
    if len(g):
        print(f"  隔夜跳空 中位 {g.median():+.0f}bp / 均值 {g.mean():+.0f}bp "
              f"← 回测同样承担，不计入上表")

    n = len(filled)
    turnover = args.turnover
    if turnover is None:
        turnover = 1.514
    excess = e + f_ - m
    print(f"\n  反推年化：加权超额 {excess:+.1f}bp/边 × 年双边换手 {turnover:.3f}"
          f" → {excess / 1e4 * turnover * 100:+.3f}pp/年")
    if n < 20:
        print(f"  ⚠ n={n} < 20，单笔噪声可达 ±50bp，只当方向，别改配置")

    anom = df[(~df["side"].str.startswith("买")) & (df["status"] == "未成交")]
    if len(anom):
        print(f"  ⚠ {len(anom)} 笔卖单没成交：{anom['code'].tolist()[:8]} —— "
              f"该卖的没卖，仓位会偏离目标")
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    out = RECON_DIR / "recon.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  → {_rel(out)}")


# ============================================================ nav：组合对账
def _seed_holdings(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    h = pd.read_csv(path, dtype={"code": str})
    h["code"] = h["code"].astype(str).str.zfill(6)
    return {c: float(s) for c, s in zip(h["code"], h["shares"]) if float(s) > 0}


def cmd_nav(args) -> None:
    df, files = load_fills(args.date)
    px = Prices()
    df["filled_px"] = pd.to_numeric(df["filled_px"], errors="coerce")
    df["filled_shares"] = pd.to_numeric(df["filled_shares"], errors="coerce")
    # open_px：模板里已填就用模板的（下单当天看到的价），否则回行情取
    tp = pd.to_numeric(df.get("open_px"), errors="coerce")
    open_map: dict[int, float] = {}
    kind_of: dict[str, str] = {}
    for i, r in df.iterrows():
        kind_of.setdefault(r["code"], r["kind"])
        o = tp.get(i, np.nan)
        if not (np.isfinite(o) and o > 0):
            o = px.open_close(r["code"], r["kind"], r["trade_date"])[0]
        open_map[i] = o

    days = [d for d in sorted(df["trade_date"].unique()) if px.available(d)]
    if not days:
        print(f"行情止于 {px.last.date()}，成交日都在这之后，无法估值。等数据更新后再跑。")
        return
    filled = df[(df["filled_shares"].fillna(0) > 0) & (df["filled_px"].fillna(0) > 0)]
    if len(filled) == 0:
        print("没有任何成交记录 → 只能算影子组合（按开盘价完美执行）。")
        print("影子组合就是「如果单子全部按开盘价成交」的净值，可作为实盘上线前的基线。")

    cap = args.capital or float(load_combo().get("capital") or 0)
    if not cap:
        raise SystemExit("给 --capital（账户初始资金），或让 signals/latest_combo.json 带 capital")
    seed = _seed_holdings(args.holdings)

    cal = [d for d in px.dates if pd.Timestamp(d) >= pd.Timestamp(days[0])]

    def run(use_fills: bool) -> tuple[pd.Series, pd.Series]:
        """use_fills=True 用实际成交；False 用计划股数×开盘价（影子）。"""
        pos: dict[str, float] = dict(seed)
        cash = float(cap)
        # 初始持仓按首日开盘价折算占用资金，保证总资产 = cap
        for c, s in list(pos.items()):
            o, _ = px.open_close(c, kind_of.get(c, "stock"), cal[0])
            if np.isfinite(o):
                cash -= s * o
        navs = []
        for d in cal:
            day = df[df["trade_date"] == d]
            for i, r in day.iterrows():
                if use_fills:
                    sh, p = r["filled_shares"], r["filled_px"]
                else:
                    sh, p = float(r["plan_shares"]), open_map.get(i, np.nan)
                if not (np.isfinite(sh) and sh > 0 and np.isfinite(p) and p > 0):
                    continue
                amt = sh * p
                fee = trade_fee(r["kind"], amt, r["side"])
                if r["side"].startswith("买"):
                    cash -= amt + fee
                    pos[r["code"]] = pos.get(r["code"], 0.0) + sh
                else:
                    hold = pos.get(r["code"], 0.0)
                    sh = min(sh, hold)
                    if sh <= 0:
                        continue
                    cash += sh * p - fee
                    pos[r["code"]] = hold - sh
            mv = 0.0
            for c, s in pos.items():
                if s <= 0:
                    continue
                cpx = px.close(c, kind_of.get(c, "stock"), d)
                if np.isfinite(cpx):
                    mv += s * cpx
            navs.append((pd.Timestamp(d), cash + mv, mv, cash))
        out = pd.DataFrame(navs, columns=["date", "nav", "mv", "cash"]).set_index("date")
        return out

    shadow = run(False)
    live = run(True) if len(filled) else None

    print("=" * 86)
    print(f"组合对账  资金 {cap:,.0f} · {shadow.index[0].date()} ~ {shadow.index[-1].date()}"
          f" · {len(shadow)} 个交易日")
    print("=" * 86)
    print(f"  影子（开盘价完美执行）  期末净值 {shadow['nav'].iloc[-1]:,.0f}  "
          f"累计 {shadow['nav'].iloc[-1] / cap - 1:+.2%}  "
          f"平均仓位 {shadow['mv'].mean() / shadow['nav'].mean():.1%}")
    if live is not None:
        n = min(len(live), len(shadow))
        diff = (live["nav"].iloc[:n].to_numpy() / shadow["nav"].iloc[:n].to_numpy() - 1.0) * 1e4
        print(f"  实盘                    期末净值 {live['nav'].iloc[-1]:,.0f}  "
              f"累计 {live['nav'].iloc[-1] / cap - 1:+.2%}  "
              f"平均仓位 {live['mv'].mean() / live['nav'].mean():.1%}")
        print(f"  实盘 − 影子            期末 {diff[-1]:+.0f}bp  "
              f"最大偏离 {np.abs(diff).max():.0f}bp  "
              f"← 这一列就是「执行」值多少钱")
    else:
        print("  实盘                    尚无成交记录，只有影子基线")

    RECON_DIR.mkdir(parents=True, exist_ok=True)
    out = shadow.rename(columns={"nav": "nav_shadow", "mv": "mv_shadow", "cash": "cash_shadow"})
    if live is not None:
        out = out.join(live.rename(columns={"nav": "nav_live", "mv": "mv_live",
                                            "cash": "cash_live"}), how="outer")
        out["diff_bp"] = (out["nav_live"] / out["nav_shadow"] - 1.0) * 1e4
    p = RECON_DIR / "nav_recon.csv"
    out.to_csv(p, encoding="utf-8-sig")
    print(f"\n  → {_rel(p)}")


# ============================================================ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="实盘对账：计划单 / 逐笔成交 / 组合净值")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("plan", help="计划单体检（无需成交）")
    p1.add_argument("--date", help="只看某个交易日 YYYY-MM-DD")
    p1.add_argument("--capital", type=float, help="账户总资金，缺省读 latest_combo.json")
    p1.add_argument("--prices", action="store_true", help="加载行情取开盘价（慢，但更准）")
    p1.set_defaults(func=cmd_plan)

    p2 = sub.add_parser("recon", help="逐笔对账（需要 filled_px / filled_shares）")
    p2.add_argument("--date", help="只看某个交易日")
    p2.add_argument("--turnover", type=float, help="年双边换手，缺省 1.514")
    p2.set_defaults(func=cmd_recon)

    p3 = sub.add_parser("nav", help="实盘 vs 影子净值")
    p3.add_argument("--date", help="只用某个交易日的成交")
    p3.add_argument("--capital", type=float, help="账户初始资金")
    p3.add_argument("--holdings", help="初始持仓 CSV（code,shares）")
    p3.set_defaults(func=cmd_nav)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
