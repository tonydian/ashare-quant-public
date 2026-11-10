# -*- coding: utf-8 -*-
"""诊断：ST 前视偏差的货币价值（不需新增数据）

背景（2026-10-30 验证结论）：
  1. 原始 market_cap.npz 对当前 ST 股的市值 100% 为 NaN → 当前 ST 股本来就被
     MIN_MCAP 门槛挡住，EXCLUDE_ST 开关是冗余的（改名称不改变任何东西）。
  2. 真正的偏差在「摘帽股」：2016-10 的 57 只 ST 中 41 只已摘帽（现名不含 ST），
     它们有完整市值数据，贡献 1.005% 的 eligible 单元格。
     → 这些股票在历史上的 ST 期间被错误纳入选股域（ST 5% 涨跌停也未建模）。

本脚本做 bound test：把已知「曾 ST 且已摘帽」的股票全期屏蔽，看基准差多少。
这是上界估计——真实影响一定更小（它们只在 ST 期间该被剔除，摘帽后应当可买）。

用法: python scripts/_diag_st_pit.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C
from qf import data as qfdata
from qf import model as qfm
from qf import backtest as qfbt

# 2016-10-27 仍为 ST、但当前名称已不含 ST/退 的股票（PandaData get_st_list 实测，共 41 只）
CLEARED_ST = """000037 000155 000403 000408 000504 000505 000617 000629 000633 000670
000691 000717 000831 000856 000913 000933 000950 000962 000968 000995
002061 002069 002173 002199 002289 002379 002423 002513 002608 600179
600212 600230 600234 600301 600319 600339 600346 600375 600390 600520
600539""".split()

MODEL = "build/models/wf_g_prod"
START = "2020-01-03"
REBAL = 40
NHOLD = 40


def _setup(scores, md):
    """复刻 run_backtest.py 的起始行/相位对齐逻辑。"""
    valid_rows = np.flatnonzero(np.isfinite(scores).any(axis=1))
    first_score = int(valid_rows[0])
    start_row = max(md.date_index(START), first_score)
    rows_w = None
    try:
        rows_w = qfm.load_result(MODEL).rows_weekly
    except Exception:
        pass
    if rows_w is not None and len(rows_w):
        phase = int(rows_w[0]) % 5
        while start_row % 5 != phase and start_row < md.T - 1:
            start_row += 1
    end_row = md.T - 1
    phase = int(rows_w[0]) % 5 if (rows_w is not None and len(rows_w)) else 0
    rebalance_offset = (phase - start_row) % REBAL if REBAL % 5 == 0 else 0
    return start_row, end_row, rebalance_offset


def run(md, scores, label):
    start_row, end_row, reb_off = _setup(scores, md)
    cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)
    old_nh = C.N_HOLDINGS
    C.N_HOLDINGS = NHOLD
    t0 = time.time()
    try:
        res = qfbt.run_backtest(
            md, scores, start_row, end_row,
            risk_overlay=True, stop_loss=True,
            rebalance_days=REBAL, rebalance_offset=reb_off,
            trade_band_rel=0.30, trade_band_abs=6000.0,
            max_new_entries=6, verbose=False,
            cash_asset_returns=cash,
        )
    finally:
        C.N_HOLDINGS = old_nh
    m = res.metrics
    print(f"[{label}] {time.time()-t0:.0f}s  CAGR={m['cagr']:.2%}  Sharpe={m['sharpe']:.3f}  "
          f"MaxDD={m['max_dd']:.2%}  Calmar={m.get('calmar', float('nan')):.3f}")
    return res, m


def exposure(res, md, codes_block):
    """统计被屏蔽股票在持仓/交易中的实际暴露。"""
    out = {}
    hold = getattr(res, "holdings", None)
    if hold is not None and len(hold):
        col = "code" if "code" in hold.columns else hold.columns[1]
        codes = hold[col].astype(str).str.lstrip("0").str.zfill(6) if False else hold[col].astype(str).str[:6]
        s = set(codes_block)
        hit = codes.isin(s)
        out["holding_rows"] = int(len(hold))
        out["holding_rows_blocked"] = int(hit.sum())
        out["holding_share"] = float(hit.mean())
        # 按股票聚合：出现次数
        vc = codes[hit].value_counts()
        out["top_blocked"] = {str(k): int(v) for k, v in vc.head(12).items()}
    tr = getattr(res, "trades", None)
    if tr is not None and len(tr):
        c2 = tr["code"].astype(str).str[:6] if "code" in tr.columns else None
        if c2 is not None:
            hit2 = c2.isin(set(codes_block))
            out["trade_rows"] = int(len(tr))
            out["trade_rows_blocked"] = int(hit2.sum())
            out["trade_share"] = float(hit2.mean())
            # 买入金额占比
            if "side" in tr.columns and "amount" in tr.columns:
                buys = tr[tr["side"] == "buy"]
                if len(buys):
                    bb = buys.assign(code6=buys["code"].astype(str).str[:6])
                    out["buy_amount_share_blocked"] = float(
                        bb.loc[bb["code6"].isin(set(codes_block)), "amount"].sum()
                        / max(bb["amount"].sum(), 1e-9))
    return out


def main():
    print("[1] 加载行情 ...")
    md = qfdata.load_market(verbose=False)
    print("[2] 加载打分 ...")
    scores = qfm.load_result(MODEL).scores

    res_base, m_base = run(md, scores, "基准(现状)")

    # 定位屏蔽股票的列索引
    idx_map = {c: i for i, c in enumerate(md.codes)}
    block = [idx_map[c] for c in CLEARED_ST if c in idx_map]
    print(f"    待屏蔽股票 {len(block)} 只 / 名单 {len(CLEARED_ST)} 只")

    exp = exposure(res_base, md, CLEARED_ST)
    print("    基准中的摘帽股暴露:", json.dumps(exp, ensure_ascii=False))

    # ---- 上界测试：全期屏蔽
    md2 = md
    elig_bak = md2.eligible.copy()
    buy_bak = md2.buyable.copy()
    md2.eligible[:, block] = False
    md2.buyable[:, block] = False
    res_blk, m_blk = run(md2, scores, "屏蔽摘帽股(全期)")
    md2.eligible = elig_bak
    md2.buyable = buy_bak

    print()
    print("=== 差异 ===")
    for k in ["cagr", "sharpe", "max_dd", "calmar"]:
        a, b = m_base.get(k), m_blk.get(k)
        if a is None or b is None:
            continue
        print(f"  {k:8s} 基准 {a:+.4f}  屏蔽 {b:+.4f}  差 {b-a:+.4f}")

    out = {
        "cleared_st_universe": CLEARED_ST,
        "baseline": {k: m_base.get(k) for k in ["cagr", "sharpe", "max_dd", "calmar", "cost_total", "cost_ratio"]},
        "blocked": {k: m_blk.get(k) for k in ["cagr", "sharpe", "max_dd", "calmar", "cost_total", "cost_ratio"]},
        "baseline_exposure": exp,
    }
    p = ROOT / "build" / "pit" / "st_pit_diag.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"\n写入 {p}")


if __name__ == "__main__":
    main()
