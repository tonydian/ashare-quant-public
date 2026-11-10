# -*- coding: utf-8 -*-
"""板块轮动 v3：剔除海外/宽基/货币，只保留**纯 A 股行业板块**。

v2 发现的问题：37 只池子里混入了 纳指(523%)/标普(259%)/日经/港股/恒生科技/沪深300/中证500/中证1000/创业板/科创50/红利
—— 这些是**宽基或海外品种**，不是"板块轮动"，且纳指+标普贡献了主要收益（美股大牛市 + 汇兑），
属于口径污染。v3 严格只保留**行业/主题板块**。

同时修复一个隐患：v2 中"同板块去重"发生在**候选筛选后**，
如果同板块多只 ETF 只有一个满足突破条件，去重逻辑可能把整个板块漏掉。
v3 改为：先按板块聚合，取该板块**动量最强的代表**，再判断是否满足突破。
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

OUT = ROOT / "build" / "eval" / "sector_rotation_v3.json"
PANEL = ROOT / "build" / "etf" / "sector_panel.npz"
UNIV = ROOT / "build" / "etf" / "sector_universe.json"

# 与生产模块共用同一套口径（2026-11-05）：此前这里各自硬编码
# EXCLUDE_SECTORS（不含"创业板50"）与 MIN_AMOUNT=1e7（生产是 2.5e7），
# 导致本脚本与生产在同一份数据上语义不同，而 tests/test_lookahead_audit.py
# 靠 3pp 容差把这种漂移"兜"过去了。现在改为引用 qf.sector_rotation。
from qf import sector_rotation as SR  # noqa: E402

EXCLUDE_SECTORS = SR.EXCLUDE_SECTORS
MIN_AMOUNT = SR.MIN_AMOUNT


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


def load_panel():
    z = np.load(PANEL, allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    close = z["close"].astype(float)
    amount = z["amount"].astype(float)
    univ = json.load(open(UNIV, encoding="utf-8"))
    # 应用与生产同一份错标修正：159949 实为创业板50（宽基，应剔除）、516010 实为游戏
    sector_of = {v["code"]: SR.LABEL_OVERRIDE.get(v["code"], k)
                 for k, v in univ["sectors"].items()}
    keep = [j for j, c in enumerate(codes) if sector_of.get(c) not in EXCLUDE_SECTORS]
    codes = [codes[j] for j in keep]
    close = close[:, keep]
    amount = amount[:, keep]
    sector_of = {c: sector_of[c] for c in codes}
    return dates, codes, close, amount, sector_of


def cash_series(idx):
    bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
    gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
    b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
    g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
    return (_blend_cash(b, g, idx))


def run(dates, codes, close, amount, sector_of, entry_n=55, top_n=3, rebal=10,
        mom_n=60, cash=None, min_amount=MIN_AMOUNT):
    T, N = close.shape
    amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
    cs = pd.DataFrame(close)
    hi = cs.rolling(entry_n).max().shift(1).to_numpy()
    mom = (cs / cs.shift(mom_n) - 1.0).to_numpy()
    nav = np.ones(T)
    holdings: list[int] = []
    stats = {"n_rebal": 0, "n_trades": 0, "pool_sizes": [], "picks": {}}
    pending: list[int] | None = None      # t 日决策 → t+1 日成交（严格 T+1）

    for t in range(1, T):
        # ① 先执行上一日收盘决定的调仓
        tcost = 0.0
        if pending is not None:
            new = pending
            changed = len(set(new) ^ set(holdings))
            stats["n_trades"] += changed
            stats["n_rebal"] += 1
            if changed:
                tcost = (changed / max(top_n, 1)) * 5.0 / 1e4
            holdings = list(new)
            pending = None

        # ② 当日收益（持有到 t 的组合收益）
        rs = []
        for j in holdings:
            if np.isfinite(close[t, j]) and np.isfinite(close[t - 1, j]) and close[t - 1, j] > 0:
                rs.append(close[t, j] / close[t - 1, j] - 1.0)
        invest = len(rs) / max(top_n, 1)
        port_r = float(np.mean(rs)) if rs else 0.0
        cr = float(cash.iloc[t]) if cash is not None else 0.0
        nav[t] = nav[t - 1] * (1 + invest * port_r + (1 - invest) * cr - tcost)

        # ③ t 日收盘计算信号，留待 t+1 执行
        if t % rebal == 0:
            # 步骤1：按板块聚合，每板块取动量最强的"可交易代表"
            best_per_sector: dict[str, tuple[int, float]] = {}
            for j in range(N):
                c = close[t, j]
                if not np.isfinite(c):
                    continue
                if not (np.isfinite(amt20[t, j]) and amt20[t, j] >= min_amount):
                    continue
                sec = sector_of.get(codes[j], codes[j])
                m = mom[t, j] if np.isfinite(mom[t, j]) else -9.0
                if sec not in best_per_sector or m > best_per_sector[sec][1]:
                    best_per_sector[sec] = (j, m)
            stats["pool_sizes"].append(len(best_per_sector))
            # 步骤2：代表 ETF 中筛"突破 55 日新高"
            cand = [(j, m) for sec, (j, m) in best_per_sector.items()
                    if np.isfinite(hi[t, j]) and close[t, j] > hi[t, j]]
            cand.sort(key=lambda x: -x[1])
            pending = [j for j, _ in cand[:top_n]]
            if t % 250 == 0:
                stats["picks"][str(dates[t].date())] = [codes[j] for j in pending]
    return pd.Series(nav, index=dates), stats


def perf(nav, s, e):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}


def main():
    dates, codes, close, amount, sector_of = load_panel()
    cash = cash_series(dates)
    S, E = "2020-01-02", "2026-10-23"
    print(f"纯 A 股行业板块池：{len(codes)} 只 / {len(set(sector_of.values()))} 个板块")
    print(f"板块：{sorted(set(sector_of.values()))}")
    print(f"区间 {dates[0].date()} ~ {dates[-1].date()}")

    res = {}
    print("\n" + "=" * 94)
    print("参数扫描（纯行业板块）")
    print("=" * 94)
    print(f"  {'entry':>5}{'rebal':>6}{'topN':>5}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'板块数':>8}")
    grid = {}
    for entry in (20, 40, 55, 70):
        for rebal in (10, 20):
            for top_n in (1, 2, 3, 5):
                nav, st = run(dates, codes, close, amount, sector_of,
                              entry_n=entry, top_n=top_n, rebal=rebal, cash=cash)
                p = perf(nav, S, E)
                avgp = float(np.mean(st["pool_sizes"])) if st["pool_sizes"] else 0
                grid[f"e{entry}_rb{rebal}_t{top_n}"] = {**p, "avg_pool": avgp}
                if entry in (40, 55) and top_n <= 3:
                    print(f"  {entry:>5}{rebal:>6}{top_n:>5}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}"
                          f"{p['max_dd']:>8.2%}{p['calmar']:>8.2f}{avgp:>8.1f}")
    sh = [v["sharpe"] for v in grid.values()]
    print(f"\n  全部 {len(grid)} 组：Sharpe 中位 {np.median(sh):.3f} | min {min(sh):.3f} | "
          f"max {max(sh):.3f} | 为正 {(np.array(sh)>0).mean():.0%}")
    res["grid"] = grid
    res["grid_summary"] = {"n": len(grid), "median": float(np.median(sh)),
                           "min": float(min(sh)), "max": float(max(sh)),
                           "pct_positive": float((np.array(sh) > 0).mean())}

    print("\n" + "=" * 94)
    print("选定 e55/rb10/top3 —— 分区间")
    print("=" * 94)
    nav, st = run(dates, codes, close, amount, sector_of, 55, 3, 10, cash=cash)
    for tag, (s, e) in {"全期": (S, E), "验证期": ("2020-01-02", "2021-12-31"),
                        "样本外": ("2022-01-01", E)}.items():
        p = perf(nav, s, e)
        print(f"  {tag:6s} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")
        res[tag] = p
    y = qfm.yearly_returns(nav)
    print("  分年: " + " | ".join(f"{int(k)}: {v*100:6.1f}%" for k, v in y.items()))
    res["yearly"] = {int(k): float(v) for k, v in y.items()}

    print("\n" + "=" * 94)
    print("各板块在池中被选中的年份（抽样）")
    print("=" * 94)
    for d, picks in list(st["picks"].items())[:10]:
        print(f"  {d}: {picks}")
    res["sample_picks"] = {k: v for k, v in list(st["picks"].items())[:15]}
    res["sector_count"] = len(set(sector_of.values()))
    # 口径元信息：让 tests/test_lookahead_audit.py 能校验"与生产同口径"，
    # 而不是靠一个凭经验拍的 3pp 容差去兜住漂移
    res["universe"] = {
        "n_codes": len(codes),
        "sector_count": len(set(sector_of.values())),
        "min_amount": float(MIN_AMOUNT),
        "data_end": str(dates[-1].date()),
        "excluded": sorted(EXCLUDE_SECTORS),
        "label_override": dict(SR.LABEL_OVERRIDE),
    }

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
