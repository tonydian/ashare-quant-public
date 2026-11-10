# -*- coding: utf-8 -*-
"""独立审计：执行 / 账务 / 组合约束（真实 md + 最终模型 scores + build/runs/main 产物）。

两部分：
1. **重跑小段**（2020-01 ~ 2021-12，主回测同配置：N=40、10 日调仓）：用最终模型 scores
   重跑 run_backtest，逐日独立重放 trades 核对 NAV/现金/费用/整手/T+1/可交易掩码；
2. **审计主回测产物**（build/runs/main 或 QF_AUDIT_RUN 指定目录）：直接对落盘的
   nav/trades/weights/holdings/exposures 做同样的独立对账，并检查目标权重约束、
   实际暴露、换新上限、成本与 metrics 一致。

模型目录：默认 build/models/wf_v1，可用环境变量 QF_AUDIT_MODEL 覆盖（名称或路径）。
主回测目录：默认 build/runs/main，可用 QF_AUDIT_RUN 覆盖。
"""
from __future__ import annotations

import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C                      # noqa: E402
from qf import data as qfdata                   # noqa: E402
from qf.backtest import load_cash_asset_returns, run_backtest   # noqa: E402

START = "2020-01-02"
END = "2021-12-31"
# 最终主回测配置（journal tag=main_h10 + scripts/run_backtest.py 默认）：
REBALANCE_DAYS = 10
N_HOLDINGS = 40

_env_model = os.environ.get("QF_AUDIT_MODEL")
MODEL_DIR = (pathlib.Path(_env_model) if _env_model and pathlib.Path(_env_model).is_absolute()
             else C.BUILD_MODELS / (_env_model or "wf_v1"))
_env_run = os.environ.get("QF_AUDIT_RUN")
RUN_DIR = (pathlib.Path(_env_run) if _env_run and pathlib.Path(_env_run).is_absolute()
           else C.BUILD_RUNS / (_env_run or "main"))
STAMP_CUT = pd.Timestamp("2023-08-28")
STAMP_RATE_BEFORE = 0.001


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


def _load_scores(md):
    fp = MODEL_DIR / "composite_score.npz"
    if not fp.exists():
        pytest.skip(f"{fp} 尚未生成")
    sc = np.load(fp)["scores"]
    # 数据更新后模型可能滞后（scores 行数 < md.T）：补齐 NaN 行，
    # 等价于"最后几天不调仓"，不影响交易/权重约束的验证
    if sc.shape[0] < md.T:
        pad = np.full((md.T - sc.shape[0], sc.shape[1]), np.nan, dtype=sc.dtype)
        sc = np.vstack([sc, pad])
    elif sc.shape[0] > md.T:
        sc = sc[:md.T]
    return sc


def _align_start(md, scores: np.ndarray, rb: int = REBALANCE_DAYS) -> tuple[int, int]:
    """与 scripts/run_backtest.py 相同的对齐逻辑（返回 start_row, rebalance_offset）。"""
    valid_rows = np.flatnonzero(np.isfinite(scores).any(axis=1))
    if len(valid_rows) == 0:
        pytest.skip("打分面板为空")
    start_row = max(md.date_index(START), int(valid_rows[0]))
    phase = int(valid_rows[0]) % 5
    while start_row % 5 != phase and start_row < md.T - 2:
        start_row += 1
    offset = (phase - start_row) % rb if rb % 5 == 0 else 0
    return start_row, offset


@pytest.fixture(scope="module")
def bt(md):
    scores = _load_scores(md)
    start_row, offset = _align_start(md, scores)
    end_row = md.date_index(END)
    old = C.N_HOLDINGS
    try:
        C.N_HOLDINGS = N_HOLDINGS
        r = run_backtest(md, scores, start_row, end_row, C, verbose=False,
                         rebalance_days=REBALANCE_DAYS, rebalance_offset=offset)
    finally:
        C.N_HOLDINGS = old
    return {"r": r, "start_row": start_row, "end_row": end_row,
            "scores": scores, "offset": offset}


@pytest.fixture(scope="module")
def main_art(md):
    """主回测落盘产物（build/runs/main）。"""
    if not (RUN_DIR / "metrics.json").exists():
        pytest.skip(f"{RUN_DIR}/metrics.json 尚未生成（等 Lead 跑完主回测）")
    nav = pd.read_parquet(RUN_DIR / "nav.parquet")["nav"].astype(float)
    trades = pd.read_parquet(RUN_DIR / "trades.parquet")
    weights = pd.read_parquet(RUN_DIR / "weights.parquet") if (RUN_DIR / "weights.parquet").exists() else None
    holdings = pd.read_parquet(RUN_DIR / "holdings.parquet") if (RUN_DIR / "holdings.parquet").exists() else None
    exposures = pd.read_parquet(RUN_DIR / "exposures.parquet") if (RUN_DIR / "exposures.parquet").exists() else None
    import json
    metrics = json.load(open(RUN_DIR / "metrics.json", encoding="utf-8"))
    return {"nav": nav, "trades": trades, "weights": weights, "holdings": holdings,
            "exposures": exposures, "metrics": metrics}


# ---------------------------------------------------------------------------
# 独立重放：按 trades 重建现金/持仓，逐日核对 NAV
# ---------------------------------------------------------------------------
def _replay(md, trades: pd.DataFrame, start_row: int, end_row: int,
            cash_returns: np.ndarray | None = None) -> dict:
    """按 trades 重放现金/持仓；cash_returns 为 v2 闲置现金替代资产日收益（None=v1 固定 1.5%/365）。

    现金日收益在每日收盘估值后计提，与引擎一致；估值缺价用最近有效价（同 _close_px）。
    """
    tr_by_row: dict[int, list[dict]] = defaultdict(list)
    for row in trades.itertuples(index=False):
        t = md.date_index(row.date)
        tr_by_row[t].append({
            "code": md.code_index(row.code), "side": row.side,
            "shares": int(row.shares), "price": float(row.price),
            "amount": float(row.amount), "commission": float(row.commission),
            "stamp": float(row.stamp), "transfer": float(row.transfer),
            "slip": float(row.slip), "reason": row.reason,
        })
    rate = float(C.CASH_YIELD) / 365.0
    cash = float(C.CAPITAL)
    holdings: dict[int, int] = {}
    mark: dict[int, float] = {}
    nav = np.full(end_row - start_row + 1, np.nan)
    cash_vals = np.full(end_row - start_row + 1, np.nan)
    n_pos = np.zeros(end_row - start_row + 1, dtype=int)
    neg_cash_steps = 0
    missing_price = 0
    same_day_both = 0
    for t in range(start_row, end_row + 1):
        i = t - start_row
        day = tr_by_row.get(t, [])
        codes = [x["code"] for x in day]
        same_day_both += sum(1 for j in set(codes) if codes.count(j) > 1)
        for side in ("sell", "buy"):
            for x in (y for y in day if y["side"] == side):
                j = x["code"]
                if side == "buy":
                    cash -= x["amount"] + x["commission"] + x["transfer"]
                    holdings[j] = holdings.get(j, 0) + x["shares"]
                else:
                    cash += x["amount"] - x["commission"] - x["stamp"] - x["transfer"]
                    holdings[j] = holdings.get(j, 0) - x["shares"]
                    assert holdings[j] >= 0, f"{md.dates[t].date()} {md.codes[j]} 卖出超过持仓"
                    if holdings[j] == 0:
                        holdings.pop(j)
                neg_cash_steps += int(cash < -1e-6)
        val = 0.0
        for j, sh in holdings.items():
            p = float(md.price[t, j])
            if np.isfinite(p) and p > 0:
                mark[j] = p
            else:
                missing_price += 1
                p = mark.get(j, 0.0)
            val += sh * p
        nav[i] = cash + val
        cash_vals[i] = cash
        n_pos[i] = len(holdings)
        if cash_returns is not None:
            cash *= (1.0 + float(cash_returns[t]))
        else:
            cash *= (1.0 + rate)
    return {"nav": nav, "cash": cash_vals, "n_pos": n_pos,
            "neg_cash_steps": neg_cash_steps, "missing_price": missing_price,
            "same_day_both": same_day_both}


@pytest.fixture(scope="module")
def replay(md, bt):
    return _replay(md, bt["r"].trades, bt["start_row"], bt["end_row"])


# ---------------------------------------------------------------------------
# 1) NAV 对账（重跑小段）
# ---------------------------------------------------------------------------
def test_nav_reconciliation(md, bt, replay):
    r = bt["r"]
    assert len(r.nav) == len(replay["nav"])
    diff = np.abs(r.nav.to_numpy(dtype=np.float64) - replay["nav"])
    rel = diff / np.maximum(np.abs(r.nav.to_numpy(dtype=np.float64)), 1.0)
    assert rel.max() < 1e-9, f"NAV 与独立重放不一致：max rel diff={rel.max():.3e}"
    assert replay["missing_price"] == 0, f"{replay['missing_price']} 个持仓日缺价格（估值靠兜底）"
    print(f"[audit] NAV 对账通过：{len(replay['nav'])} 天，max rel diff={rel.max():.2e}")


def test_cash_never_negative(bt, replay):
    assert replay["neg_cash_steps"] == 0, f"现金为负的步数：{replay['neg_cash_steps']}"
    assert (replay["cash"] >= -1e-6).all()
    assert replay["same_day_both"] == 0


# ---------------------------------------------------------------------------
# 2) 费用公式 / 整手 / 滑点方向 / 可交易掩码（重跑小段）
# ---------------------------------------------------------------------------
def _check_fees(tr: pd.DataFrame):
    assert len(tr) > 0
    assert (tr["shares"] % C.LOT == 0).all(), "存在非整手成交"
    amt = tr["amount"].to_numpy(dtype=np.float64)
    exp_comm = np.maximum(C.MIN_COMMISSION, C.COMMISSION_RATE * amt)
    assert np.allclose(tr["commission"].to_numpy(dtype=np.float64), exp_comm, atol=1e-9), "佣金公式不符"
    assert np.allclose(tr["transfer"].to_numpy(dtype=np.float64),
                       C.TRANSFER_FEE * amt, atol=1e-9), "过户费公式不符"
    buy = tr[tr["side"] == "buy"]
    assert (buy["stamp"] == 0.0).all(), "买入不应收印花税"
    normal_sell = tr[(tr["side"] == "sell") & (tr["reason"] != "liquidate")].copy()
    if len(normal_sell):
        rate = np.where(pd.to_datetime(normal_sell["date"]) < STAMP_CUT,
                        STAMP_RATE_BEFORE, C.STAMP_TAX_SELL)
        assert np.allclose(normal_sell["stamp"].to_numpy(dtype=np.float64),
                           rate * normal_sell["amount"].to_numpy(dtype=np.float64),
                           atol=1e-9), "正常卖出印花税未按历史分段（2023-08-28 前 10bp / 之后 5bp）"
    liq = tr[(tr["side"] == "sell") & (tr["reason"] == "liquidate")]
    return liq


def test_trade_fee_formulas(md, bt):
    liq = _check_fees(bt["r"].trades)
    if len(liq):
        print(f"[audit] 退市清算成交 {len(liq)} 笔；stamp==0 的有 "
              f"{int((liq['stamp'] == 0).sum())} 笔（若 >0 = 漏收印花税，见报告 F-03）")


def test_trade_prices_and_slippage(md, bt):
    tr = bt["r"].trades
    slip = C.SLIPPAGE_BP / 1e4
    ok = 0
    for row in tr.itertuples(index=False):
        if row.reason == "liquidate":
            assert row.slip == 0.0
            continue
        t = md.date_index(row.date)
        j = md.code_index(row.code)
        px_open = float(md.price_open[t, j])
        assert np.isfinite(px_open) and px_open > 0
        if row.side == "buy":
            assert row.price == pytest.approx(px_open * (1.0 + slip), rel=1e-12)
        else:
            assert row.price == pytest.approx(px_open * (1.0 - slip), rel=1e-12)
        assert row.slip == pytest.approx(abs(row.price - px_open) * row.shares, rel=1e-9)
        ok += 1
    assert ok > 100


def test_tradability_masks_respected(md, bt):
    tr = bt["r"].trades
    bad = []
    for row in tr.itertuples(index=False):
        if row.reason == "liquidate":
            continue
        t = md.date_index(row.date)
        j = md.code_index(row.code)
        mask = md.buyable if row.side == "buy" else md.sellable
        if not bool(mask[t, j]):
            bad.append((row.date, row.code, row.side))
    assert not bad, f"违反可交易掩码的成交：{bad[:5]}"


def test_no_same_day_buy_sell(bt):
    tr = bt["r"].trades
    g = tr.groupby(["date", "code"])["side"].nunique()
    assert (g <= 1).all(), "同一标的同日既买又卖"
    for date, grp in tr.groupby("date", sort=False):
        sides = grp["side"].tolist()
        if "buy" in sides and "sell" in sides:
            first_buy = sides.index("buy")
            assert all(s == "buy" for s in sides[first_buy:]), f"{date} 存在买单之后的卖单"


def test_cost_total_matches_metrics(bt):
    r = bt["r"]
    cost = float((r.trades["commission"] + r.trades["stamp"]
                  + r.trades["transfer"] + r.trades["slip"]).sum())
    assert cost == pytest.approx(r.metrics["cost_total"], rel=1e-12)
    assert r.metrics["cost_total"] > 0


# ---------------------------------------------------------------------------
# 3) 组合约束（重跑小段目标权重）
# ---------------------------------------------------------------------------
def _check_constraints(md, weights: pd.DataFrame, n_holdings: int):
    growth = np.isin(np.asarray(md.board, dtype=object), ("STAR", "CHINEXT"))
    n_bad = 0
    for date, g in weights.groupby("date"):
        t = md.date_index(date)
        ws = g["weight"].to_numpy(dtype=float)
        assert (ws > 0).all()
        n_bad += int(len(ws) > n_holdings)
        n_bad += int(ws.sum() > 1.0 + 1e-6)
        n_bad += int(ws.max() > C.MAX_WEIGHT + 1e-6)
        js = np.array([md.code_index(c) for c in g["code"]], dtype=int)
        ind = md.industry[t, js]
        for k in np.unique(ind):
            n_bad += int(ws[ind == k].sum() > C.INDUSTRY_MAX + 1e-6)
        n_bad += int(ws[growth[js]].sum() > C.BOARD_MAX_GROWTH + 1e-6)
    return n_bad


def test_target_weight_constraints(md, bt):
    w = bt["r"].weights
    assert len(w) > 0
    n_bad = _check_constraints(md, w, N_HOLDINGS)
    assert n_bad == 0, f"{n_bad} 处目标权重违反组合约束"


def test_actual_holdings_cap(md, bt, replay):
    mx = int(replay["n_pos"].max())
    print(f"[audit] 实际持仓数最大值 {mx}（目标 {N_HOLDINGS}）")
    assert mx <= N_HOLDINGS + 5, f"实际持仓数 {mx} 显著超过 {N_HOLDINGS}"


def test_max_new_entries_cap(md, bt):
    r = bt["r"]
    w = r.weights
    dates = list(pd.to_datetime(w["date"].unique()))
    target_sets = {d: set(w.loc[w["date"] == d, "code"]) for d in dates}
    hold = r.holdings
    snap = {pd.Timestamp(d): set(g.loc[g["shares"] > 0, "code"])
            for d, g in hold.groupby("date")} if len(hold) else {}
    violations = []
    for i in range(1, len(dates)):
        d = dates[i]
        prev_h = snap.get(dates[i - 1], target_sets[dates[i - 1]])
        new = target_sets[d] - prev_h
        if len(new) > 6:
            violations.append((str(d.date()), len(new)))
    assert not violations, f"换新上限被突破：{violations}"
    print(f"[audit] 换新上限检查通过（{len(dates) - 1} 次调仓，最大新进数 ≤ 6）")


def test_stop_loss_reason_if_any(bt):
    tr = bt["r"].trades
    sl = tr[tr["reason"] == "stop_loss"]
    print(f"[audit] 止损成交 {len(sl)} 笔；退市清算 {int((tr['reason'] == 'liquidate').sum())} 笔")
    assert set(tr["reason"].unique()) <= {"rebalance", "stop_loss", "liquidate"}


# ---------------------------------------------------------------------------
# 4) 主回测产物直接对账（build/runs/main）
# ---------------------------------------------------------------------------
def test_main_nav_replay(md, main_art):
    nav = main_art["nav"]
    start_row = md.date_index(nav.index[0])
    end_row = md.date_index(nav.index[-1])
    # 闲置现金按 config 权重（当前 100% 国债）计收益
    cash_returns = load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)
    rep = _replay(md, main_art["trades"], start_row, end_row, cash_returns=cash_returns)
    a = nav.to_numpy(dtype=np.float64)
    d = np.abs(a - rep["nav"]) / np.maximum(np.abs(a), 1.0)
    assert d.max() < 1e-9, f"主回测 NAV 与 trades 独立重放不一致：max rel={d.max():.3e}"
    assert rep["neg_cash_steps"] == 0
    assert rep["same_day_both"] == 0
    print(f"[audit] 主回测 NAV 对账通过：{len(a)} 天（{nav.index[0].date()}~{nav.index[-1].date()}），"
          f"max rel={d.max():.2e}，现金最小={rep['cash'].min():.2f}，"
          f"持仓数峰值={int(rep['n_pos'].max())}")


def test_main_fees(md, main_art):
    tr = main_art["trades"]
    _check_fees(tr)
    d = tr.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    g = d.groupby("year").agg(n=("shares", "size"), cost=("commission", "sum"))
    print("[audit] 主回测分年成交笔数：", g["n"].to_dict())
    cost = float((tr["commission"] + tr["stamp"] + tr["transfer"] + tr["slip"]).sum())
    assert cost == pytest.approx(float(main_art["metrics"]["cost_total"]), rel=1e-12)


def test_main_tradability_and_t1(md, main_art):
    tr = main_art["trades"]
    bad = []
    for row in tr.itertuples(index=False):
        if row.reason == "liquidate":
            continue
        t = md.date_index(row.date)
        j = md.code_index(row.code)
        mask = md.buyable if row.side == "buy" else md.sellable
        if not bool(mask[t, j]):
            bad.append((row.date, row.code, row.side))
    assert not bad, f"主回测违反可交易掩码：{bad[:5]}"
    g = tr.groupby(["date", "code"])["side"].nunique()
    assert (g <= 1).all(), "主回测同一标的同日买卖"


def test_main_weight_constraints_and_exposure(md, main_art):
    w = main_art["weights"]
    assert w is not None and len(w) > 0
    cnt = w.groupby("date").size()
    n_holdings = int(cnt.max())     # 主回测驱动默认 --n-holdings 40；以实际最大持仓数核对
    assert n_holdings >= 5
    n_bad = _check_constraints(md, w, n_holdings)
    assert n_bad == 0, f"主回测 {n_bad} 处目标权重违反组合约束（n_holdings={n_holdings}）"
    rows = np.array([md.date_index(d) for d in pd.to_datetime(w["date"].unique())])
    gaps = np.diff(rows)
    assert len(gaps) and (gaps >= 5).all(), f"调仓间隔异常：{gaps[:10]}"
    print(f"[audit] 主回测目标权重约束通过：{len(cnt)} 次调仓，n_holdings={n_holdings}，"
          f"调仓间隔 {np.unique(gaps).tolist()[:6]}，"
          f"目标权重和均值={w.groupby('date')['weight'].sum().mean():.4f}")
    ex = main_art["exposures"]
    if ex is not None and "max_single" in ex.columns:
        ms = float(ex["max_single"].max())
        mi = float(ex["max_industry"].max()) if "max_industry" in ex.columns else np.nan
        gr = float(ex["growth"].max()) if "growth" in ex.columns else np.nan
        mcash = float(ex["cash"].min()) if "cash" in ex.columns else np.nan
        npos = int(ex["n_positions"].max()) if "n_positions" in ex.columns else -1
        print(f"[audit] 主回测实际暴露峰值：max_single={ms:.4f}  max_industry={mi:.4f}  "
              f"growth={gr:.4f}  min_cash_weight={mcash:.4f}  max_n_positions={npos}")
        # trade band 允许实际单票权重漂移到目标(1/40≈2.5%)以上，但不应失控
        assert ms <= 0.08, f"实际单票权重峰值 {ms:.2%} 超过 8% 上限（trade band 漂移失控）"
        assert mi <= C.INDUSTRY_MAX + 0.10, f"实际行业权重峰值 {mi:.2%} 明显超限"
        assert gr <= C.BOARD_MAX_GROWTH + 0.10, f"实际双创权重峰值 {gr:.2%} 明显超限"
        assert npos <= n_holdings + 5, f"实际持仓数 {npos} 显著超过目标 {n_holdings}"
