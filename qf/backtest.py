# -*- coding: utf-8 -*-
"""回测引擎：组合构建 + 执行模拟 + 风控 overlay + 绩效指标。

时序（严格无未来函数）：

* **调仓日 t 收盘**：用 ``scores[t]``（t 日收盘可得）→ ``portfolio.build_target``
  → 乘风控系数（趋势 regime / 回撤节流）→ 目标权重 ``{j: w}``；
* **t+1 开盘**：按目标权重 × **t 日收盘 NAV** 换算目标市值，再按 t+1 开盘价取整手
  得到**冻结的目标股数**，先卖后买执行；涨跌停/停牌/现金不足的残差每天开盘重试，
  直到下一次调仓（目标权重被替换）；
* **每日收盘**：NAV = 现金 + Σ 股数 × ``md.price[t]``；现金按 1.5%/365 日计息；
* **风控 overlay**（risk_overlay=True）：
  a) 趋势：``csi_all > MA120`` 且 ``hs300 > MA60`` 满仓，否则按
     ``(csi_all>MA120 ? 1 : 0.6) × (hs300>MA60 ? 1 : 0.75)`` 缩放目标权重；
  b) 回撤节流（**滚动 252 交易日峰值**，不是 all-time peak）：策略 NAV 相对
     近 252 日峰值的回撤 ``dd_1y`` ≤ -10% → 暴露上限 60%；≤ -15% → 上限 40%；
     ``dd_1y`` 回升到 > -5% 后恢复 100%（滞回；历史高点滚出 252 日窗口后
     回撤自然收敛，避免深套后永久锁死低仓位）；
  c) 个股止损：``md.price[t]/entry_price - 1 <= -15%`` → t+1 开盘强制卖出
     （stop_loss=True），且在当前调仓周期内不再回补。

调仓相位可用 ``rebalance_offset`` 调整（第一个调仓日 = start_row+offset，之后
每 rebalance_days 一次），便于与打分周频行对齐。

成本口径：卖出印花税历史分段（成交日 < 2023-08-28 为 10bp，之后 5bp），
其余见 ``qf/execution.py``。

输出 ``BacktestResult``：nav/daily_ret/weights(长表)/trades(长表)/holdings(长表)/
exposures(宽表)/metrics/config。
"""
from __future__ import annotations

import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as qmetrics
from .execution import ExecutionEngine, Trade
from .portfolio import build_target

__all__ = ["BacktestResult", "run_backtest"]

_GROWTH_BOARDS = ("STAR", "CHINEXT")


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    nav: pd.Series
    daily_ret: pd.Series
    weights: pd.DataFrame        # 调仓日目标权重长表 [date, code, weight, reason]
    trades: pd.DataFrame         # 成交长表 [date, code, side, shares, price, amount,
                                 #          commission, stamp, transfer, slip, reason]
    holdings: pd.DataFrame       # 调仓后实际持仓快照（长表）
    exposures: pd.DataFrame      # 调仓日暴露宽表（index=日期，含 cash + 行业名）
    metrics: dict
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 风控工具
# ---------------------------------------------------------------------------
def _trend_regime(md, cfg=C) -> np.ndarray:
    """逐日趋势 regime 系数：满仓=1；csi_all 破 MA120 ×lo；hs300 破 MA60 ×hi。

    lo/hi 从 cfg.TREND_MULT_LO / TREND_MULT_HI 读取（默认 0.6 / 0.75），便于
    做「更温和 overlay」的敏感性对比。
    """
    lo = float(getattr(cfg, "TREND_MULT_LO", 0.6))
    hi = float(getattr(cfg, "TREND_MULT_HI", 0.75))
    idx_name = str(getattr(cfg, "TREND_INDEX", "csi_all"))
    n = int(md.T)
    try:
        bp = md.bench_prices
        csi = bp[idx_name].astype(float)
        hs = bp["hs300"].astype(float)
    except Exception:  # pragma: no cover - 数据缺失时退化为满仓
        return np.ones(n, dtype=float)
    ma120 = csi.rolling(120, min_periods=120).mean().to_numpy(dtype=float)
    ma60 = hs.rolling(60, min_periods=60).mean().to_numpy(dtype=float)
    csi_arr = csi.to_numpy(dtype=float)
    hs_arr = hs.to_numpy(dtype=float)
    # 均线样本不足时视为"趋势向上"（不因数据 warmup 而空仓）
    up1 = np.where(np.isfinite(ma120), csi_arr > ma120, True)
    up2 = np.where(np.isfinite(ma60), hs_arr > ma60, True)
    return np.where(up1, 1.0, lo) * np.where(up2, 1.0, hi)


def _rolling_peak_dd(nav_vals: np.ndarray, i: int, window: int = 252) -> float:
    """第 i 日相对"滚动 window 日峰值"的回撤（样本不足时用已覆盖区间，min_periods=1）。

    与 all-time peak 不同：回撤窗口会滚动，历史高点在 window 个交易日后自动
    "滚出"峰值窗口，避免策略深套后永久锁死低仓位。
    """
    seg = nav_vals[max(0, i - window + 1): i + 1]
    peak = float(np.max(seg))
    return float(nav_vals[i] / peak - 1.0) if peak > 0 else 0.0


def _update_dd_state(dd: float, cfg, state: int) -> tuple[int, float]:
    """回撤节流状态机（滚动 252 日回撤 + 滞回），返回 (state, exposure_cap)。

    * ``state=0``：正常，暴露上限 100%；
    * ``dd <= DD_THROTTLE_1(-10%)`` → state=1，暴露上限 ``DD_LEVEL_1``(60%)；
    * ``dd <= DD_THROTTLE_2(-15%)`` → state=2，暴露上限 ``DD_LEVEL_2``(40%)；
    * 已被节流后，只要 ``dd`` 回升到 > ``DD_THROTTLE_1/2``(-5%) 即恢复 100%；
      在 (-10%, -5%] 区间维持 60%（滞回，避免频繁切换）。
    """
    if state == 0:
        if dd <= cfg.DD_THROTTLE_2:
            state = 2
        elif dd <= cfg.DD_THROTTLE_1:
            state = 1
    else:
        if dd > cfg.DD_THROTTLE_1 / 2.0:      # 回撤收复到一半以内 → 恢复
            state = 0
        elif dd <= cfg.DD_THROTTLE_2:
            state = 2
        else:
            state = 1
    cap = 1.0 if state == 0 else (cfg.DD_LEVEL_2 if state == 2 else cfg.DD_LEVEL_1)
    return int(state), float(cap)


def vol_target_scale(nav_vals: np.ndarray, i: int, cfg) -> tuple[float, float]:
    """波动率目标仓位系数：``target / 已实现波动``，截断在 [VOL_MIN_EXP, VOL_MAX_EXP]。

    * 用截至当日收盘的净值算已实现波动（日收益 std × √252），无前视——
      与趋势 regime / 回撤节流同一信息时点（t 收盘定仓，t+1 开盘执行）。
    * 回看 ``VOL_LOOKBACK`` 个交易日；不足时用现有样本（至少 5 个），再少则不干预。
    * 已实现波动 ≤ 0 或非有限 → 不干预（scale=1）。
    * ``VOL_TARGET <= 0`` → 关闭，返回 (1.0, nan)。

    Returns
    -------
    (scale, realized_vol) : 仓位系数与当年化已实现波动（仅供暴露表记录）。
    """
    vt = float(getattr(cfg, "VOL_TARGET", 0.0) or 0.0)
    if vt <= 0.0 or i < 1:
        return 1.0, float("nan")
    lb = max(int(getattr(cfg, "VOL_LOOKBACK", 20) or 20), 5)
    lo = max(0, i - lb)
    seg = np.asarray(nav_vals[lo:i + 1], dtype=float)
    seg = seg[np.isfinite(seg) & (seg > 0)]
    if seg.size < 6:
        return 1.0, float("nan")
    r = np.diff(seg) / seg[:-1]
    sd = float(np.std(r, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return 1.0, float("nan")
    vol = sd * float(np.sqrt(252.0))
    lo_exp = float(getattr(cfg, "VOL_MIN_EXP", 0.5))
    hi_exp = float(getattr(cfg, "VOL_MAX_EXP", 1.0))
    scale = float(np.clip(vt / vol, lo_exp, hi_exp))
    return scale, vol


# ---------------------------------------------------------------------------
# 快照工具
# ---------------------------------------------------------------------------
def _holdings_snapshot(md, engine: ExecutionEngine, t: int) -> list[dict]:
    nav = engine.nav(t)
    rows = []
    for j, pos in engine.positions.items():
        p = float(md.price[t, j])
        if not np.isfinite(p):
            p = float(pos.entry_price)
        value = pos.shares * p
        rows.append({
            "date": md.dates[t],
            "code": md.codes[j],
            "shares": int(pos.shares),
            "price": p,
            "value": float(value),
            "weight": float(value / nav) if nav > 0 else 0.0,
            "entry_price": float(pos.entry_price),
            "pnl_pct": float(p / pos.entry_price - 1.0) if pos.entry_price > 0 else np.nan,
            "hold_days": int(t - pos.entry_date),
        })
    return rows


def _industry_columns(md) -> list[str]:
    cols: list[str] = []
    for k in range(len(getattr(md, "industry_cats", []))):
        nm = md.industry_name(k)
        cols.append(nm if nm not in cols else f"{nm}_{k}")
    if "未知" not in cols:
        cols.append("未知")
    return cols


def _exposure_row(md, engine: ExecutionEngine, t: int, nav: float,
                  target_weights: dict[int, float], regime_scale: float,
                  dd_1y: float, dd_state: int, exposure_cap: float,
                  ind_cols: list[str], vol_scale: float = 1.0,
                  realized_vol: float = float("nan")) -> dict:
    w = engine.current_weights(t)
    row: dict = {
        "date": md.dates[t],
        "cash": float(engine.cash / nav) if nav > 0 else 0.0,
        "stock": float(sum(w.values())),
        "n_positions": int(len(engine.positions)),
        "max_single": float(max(w.values())) if w else 0.0,
        "growth": 0.0,
        "max_industry": 0.0,
        "target_exposure": float(sum(target_weights.values())),
        "regime_scale": float(regime_scale),
        "dd_1y": float(dd_1y),
        "dd_state": int(dd_state),
        "exposure_cap": float(exposure_cap),
        "dd_cap": float(exposure_cap),   # 兼容旧列名：等于 exposure_cap
        "vol_scale": float(vol_scale),
        "realized_vol": float(realized_vol),
    }
    ind_w: dict[str, float] = defaultdict(float)
    growth_mask = np.isin(np.asarray(md.board, dtype=object), _GROWTH_BOARDS)
    for j, wj in w.items():
        if bool(growth_mask[j]):
            row["growth"] += wj
        ind = int(md.industry[t, j])
        name = md.industry_name(ind) if ind >= 0 else "未知"
        if name not in ind_cols:
            name = f"{name}_{ind}"
        ind_w[name] += wj
    for name in ind_cols:
        row[name] = float(ind_w.get(name, 0.0))
    row["max_industry"] = float(max(ind_w.values())) if ind_w else 0.0
    return row


def _trade_rows(md, trades: list[Trade]) -> list[dict]:
    return [{
        "date": md.dates[tr.date],
        "code": md.codes[tr.code],
        "side": tr.side,
        "shares": int(tr.shares),
        "price": float(tr.price),
        "amount": float(tr.amount),
        "commission": float(tr.commission),
        "stamp": float(tr.stamp),
        "transfer": float(tr.transfer),
        "slip": float(tr.slip),
        "reason": tr.reason,
    } for tr in trades]


# ---------------------------------------------------------------------------
# 主回测
# ---------------------------------------------------------------------------
def run_backtest(md, scores: np.ndarray, start_row: int, end_row: int, cfg=C,
                 risk_overlay: bool = True, rebalance_days: int | None = None,
                 stop_loss: bool = True, verbose: bool = True,
                 liquidate_at_end: bool = False, slippage_bp: float | None = None,
                 trade_band_rel: float = 0.30,
                 trade_band_abs: float = 6000.0,
                 max_new_entries: int | None = 6,
                 rebalance_offset: int = 0,
                 cash_asset_returns: np.ndarray | None = None,
                 drift_band: float | None = None,
                 turnover_budget: float | None = None,
                 n_holdings: int | None = None) -> BacktestResult:
    """跑一次日频回测。

    参数
    ----
    md : MarketData
    scores : (T, N) float32 复合分数面板；``scores[t]`` = t 日收盘可用分数，
        NaN 表示无信号。禁止含未来数据（模型层 walk-forward 保证）。
    start_row, end_row : 回测交易日行号 **闭区间**（end_row 含在净值序列内）。
        第一个信号在 start_row 收盘生成、start_row+1 开盘执行。
    cfg : 配置对象
    risk_overlay : 是否启用趋势/回撤节流 overlay（个股止损另由 stop_loss 控制）
    rebalance_days : 调仓周期（交易日），缺省 ``cfg.REBALANCE_DAYS``
    stop_loss : 是否启用个股 -15% 硬止损
    verbose : 是否打印进度与摘要
    liquidate_at_end : 期末是否按最后收盘价清算（默认否，净值按市价计）
    slippage_bp : 临时覆盖 ``cfg.SLIPPAGE_BP``（成本压力测试用；结束/异常都恢复原值）
    trade_band_rel / trade_band_abs : 交易带宽（防微调），见 ExecutionEngine：
        对"已持有且目标仍持有"的标的，若 |目标市值 - 当前市值| <
        max(rel × 当前市值, abs) 则不交易；新建仓/清仓/止损单不受限。
        缺省 0.30 / 6000 元。
    max_new_entries : 单期最多纳入多少只新标的（换手上限，缺省 6；None=不限制）。
        透传给 ``portfolio.build_target``。
    rebalance_offset : 调仓相位偏移（交易日，缺省 0）。第一个调仓日 =
        ``start_row + (rebalance_offset mod rebalance_days)``，之后每
        ``rebalance_days`` 个交易日一次；用于让调仓日与打分周频行同相位。
    drift_band : 漂移带 / 换手**下限**（缺省 ``None`` = 关闭）。调仓日收盘先算
        单向换手 ``TO = 0.5·Σ_j |目标权重_j − 当前实际权重_j|``（当前权重含
        30 个交易日内的价格漂移）；若 ``TO`` 小于该值，**整次调仓跳过**
        （持仓不动、止损照旧），用于抑制"为了几个 bp 的偏离去付一整轮成本"。
        空仓（首次建仓）与存在止损单时不跳过。
    turnover_budget : 换手**上限**（缺省 ``None`` = 关闭）。若 ``TO`` 超过该值，
        则所有偏离按 ``budget / TO`` 等比收缩，做**部分调仓**。止损/退市单不受约束。
    n_holdings : 目标持仓数（缺省 ``None`` = 用 ``cfg.N_HOLDINGS``）。透传给
        ``portfolio.build_target``；仅用于"持仓数 × 最小交易额"成本曲线研究，
        生产路径不传此参数，故行为逐位不变（见 tests/test_n_holdings_passthrough.py）。

    ``drift_band`` / ``turnover_budget`` 同时关闭时，回测结果与历史口径
    **逐位一致**（见 tests/test_rebalance_policy.py）。
    """
    # 兼容 docs/任务书两种位置参数顺序：(..., risk_overlay, stop_loss, rebalance_days)
    if isinstance(rebalance_days, bool):
        stop_loss, rebalance_days = bool(rebalance_days), None
    # 滑点临时覆盖（run_robustness 成本压力测试用；正常结束 / 异常都恢复）
    old_slip = None
    if slippage_bp is not None:
        old_slip = getattr(cfg, "SLIPPAGE_BP", None)
        cfg.SLIPPAGE_BP = float(slippage_bp)
    try:
        return _run_backtest_impl(
            md, scores, start_row, end_row, cfg=cfg, risk_overlay=risk_overlay,
            rebalance_days=rebalance_days, stop_loss=stop_loss, verbose=verbose,
            liquidate_at_end=liquidate_at_end, trade_band_rel=trade_band_rel,
            trade_band_abs=trade_band_abs, max_new_entries=max_new_entries,
            rebalance_offset=rebalance_offset,
            cash_asset_returns=cash_asset_returns,
            drift_band=drift_band, turnover_budget=turnover_budget,
            n_holdings=n_holdings)
    finally:
        if old_slip is not None:
            cfg.SLIPPAGE_BP = old_slip


def load_cash_asset_returns(md, bond_w: float = 0.8, gold_w: float = 0.2,
                            cfg=C) -> np.ndarray:
    """闲置现金的替代资产日收益（国债指数 + 黄金 ETF 的固定权重组合）。

    用 ``cfg.BOND_FILE`` / ``cfg.GOLD_FILE``（数据包自带的债券指数与黄金 ETF 日线），
    对齐到行情交易日并 ffill；缺失日用 ``cfg.CASH_YIELD/365`` 兜底。
    返回 (T,) float64 数组，供 ``run_backtest(cash_asset_returns=...)`` 使用。
    """
    out = np.full(md.T, float(cfg.CASH_YIELD) / 365.0, dtype=np.float64)
    series = []
    for w, f in ((bond_w, cfg.BOND_FILE), (gold_w, cfg.GOLD_FILE)):
        if w <= 0 or not pathlib.Path(f).exists():
            continue
        df = pd.read_parquet(f)
        dcol = "date" if "date" in df.columns else df.columns[0]
        ccol = "close" if "close" in df.columns else df.columns[-1]
        s = pd.Series(pd.to_numeric(df[ccol], errors="coerce").to_numpy(),
                      index=pd.DatetimeIndex(pd.to_datetime(df[dcol]))).sort_index()
        s = s[~s.index.duplicated(keep="last")].reindex(md.dates).ffill()
        ret = s.pct_change().fillna(0.0).to_numpy(dtype=np.float64)
        series.append((w, ret))
    if not series:
        return out
    tot = sum(w for w, _ in series)
    blend = np.zeros(md.T, dtype=np.float64)
    for w, ret in series:
        blend += (w / tot) * ret
    if blend.shape != (md.T,):
        raise ValueError(f"现金替代收益长度 {blend.shape} != (T,)={md.T}")
    mask = np.isfinite(blend)
    out[mask] = blend[mask]
    return out


def _run_backtest_impl(md, scores, start_row, end_row, cfg=C, risk_overlay=True,
                       rebalance_days=None, stop_loss=True, verbose=True,
                       liquidate_at_end=False, trade_band_rel=0.30,
                       trade_band_abs=6000.0, max_new_entries: int | None = 6,
                       rebalance_offset: int = 0,
                       cash_asset_returns: np.ndarray | None = None,
                       drift_band: float | None = None,
                       turnover_budget: float | None = None,
                       n_holdings: int | None = None) -> BacktestResult:
    t0 = time.time()

    big_t = int(md.T)
    start_row = int(start_row)
    end_row = int(min(int(end_row), big_t - 1))
    if end_row <= start_row:
        raise ValueError(f"end_row({end_row}) 必须大于 start_row({start_row})+1")
    rb = int(rebalance_days or cfg.REBALANCE_DAYS)
    rb = max(1, rb)
    offset = int(rebalance_offset) % rb

    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape != (big_t, int(md.N)):
        raise ValueError(f"scores 形状应为 ({big_t}, {md.N})，得到 {scores.shape}")

    regime = _trend_regime(md, cfg) if risk_overlay else np.ones(big_t, dtype=float)
    engine = ExecutionEngine(md, cfg, trade_band_rel=trade_band_rel,
                             trade_band_abs=trade_band_abs)
    dates = md.dates[start_row:end_row + 1]
    n_rows = end_row - start_row + 1
    ind_cols = _industry_columns(md)

    nav_vals = np.full(n_rows, np.nan, dtype=float)
    n_pos_daily = np.zeros(n_rows, dtype=np.int32)
    trades: list[Trade] = []
    weight_rows: list[dict] = []
    holding_rows: list[dict] = []
    exposure_rows: list[dict] = []

    pending_weights: dict[int, float] | None = None
    pending_nav: float | None = None
    pending_shares: dict[int, int | None] | None = None
    pending_first_exec = False     # 下一开盘是否为该调仓周期的首次执行
    forced: dict[int, str] = {}    # 待强制卖出 {j: reason}
    dd_state = 0                   # 回撤节流状态：0=正常 / 1=60% / 2=40%
    exposure_cap = 1.0
    n_rebalances = 0
    n_holdings_sum = 0
    n_drift_skipped = 0            # 漂移带跳过的调仓次数
    n_budget_capped = 0            # 换手预算收缩的调仓次数

    for i, t in enumerate(range(start_row, end_row + 1)):
        # ---------------------------------------------------------- 1) 开盘执行
        if pending_weights is not None or forced:
            if pending_weights is not None:
                exec_weights = dict(pending_weights)
                exec_nav = pending_nav
                if pending_shares is None:
                    pending_shares = engine.plan_target_shares(t, exec_weights, exec_nav)
                else:
                    missing = [j for j, v in pending_shares.items() if v is None]
                    if missing:
                        fix = engine.plan_target_shares(
                            t, {j: exec_weights.get(j, 0.0) for j in missing}, exec_nav)
                        for j, v in fix.items():
                            if v is not None:
                                pending_shares[j] = v
                exec_shares = dict(pending_shares)
            else:
                # 只有止损单（无待执行目标权重）：其余持仓保持不动
                exec_weights = {}
                exec_nav = None
                exec_shares = {j: int(pos.shares) for j, pos in engine.positions.items()}
            for j in forced:
                exec_shares[j] = 0
                exec_weights[j] = 0.0
            reasons = {j: r for j, r in forced.items()}
            new_trades = engine.execute(t, exec_weights, reasons=reasons,
                                        target_nav=exec_nav, target_shares=exec_shares)
            trades.extend(new_trades)
            # 已卖出的强制卖出标的：清除标记，并从当前调仓周期目标中剔除（不回补）
            for j in list(forced.keys()):
                if j not in engine.positions:
                    forced.pop(j, None)
                    if pending_weights is not None:
                        pending_weights[j] = 0.0
                        if pending_shares is not None:
                            pending_shares[j] = 0
            # 调仓后持仓快照（每个调仓周期首次执行日）
            if pending_weights is not None and pending_first_exec:
                holding_rows.extend(_holdings_snapshot(md, engine, t))
                pending_first_exec = False
        elif pending_weights is not None:
            pending_first_exec = False

        # ---------------------------------------------------------- 2) 收盘估值
        nav = engine.nav(t)
        nav_vals[i] = nav
        n_pos_daily[i] = len(engine.positions)
        n_holdings_sum += len(engine.positions)
        if cash_asset_returns is not None:
            engine.cash *= (1.0 + float(cash_asset_returns[t]))
        else:
            engine.cash *= (1.0 + float(cfg.CASH_YIELD) / 365.0)
        # 回撤节流：滚动 252 日峰值回撤（不是 all-time peak，避免永久锁死）
        dd_1y = _rolling_peak_dd(nav_vals, i, window=252)
        if risk_overlay:
            dd_state, exposure_cap = _update_dd_state(dd_1y, cfg, dd_state)
        else:
            dd_state, exposure_cap = 0, 1.0
        # 波动率目标仓位（独立于 risk_overlay 开关；VOL_TARGET<=0 时为 1.0，零改动）
        vt_scale, realized_vol = vol_target_scale(nav_vals, i, cfg)

        # ---------------------------------------------------------- 3) 收盘信号
        is_rebal = ((t - start_row - offset) % rb == 0) and (t < end_row)
        if is_rebal:
            cur_weights = engine.current_weights(t)
            raw = build_target(md, t, scores, cur_weights, cfg=cfg,
                               max_new_entries=max_new_entries,
                               n_holdings=n_holdings)
            if raw:  # 无候选（全 NaN/无选股域）时不发新信号，避免误清仓
                reg_scale = float(regime[t]) if risk_overlay else 1.0
                scale = reg_scale * float(exposure_cap) * vt_scale
                # 仓位下限（EXPOSURE_FLOOR，默认 0.0 = 不生效、行为与原版完全一致）：
                # 趋势 overlay 与回撤节流的乘积低于该值时抬到下限，用于研究"少择时"的
                # 风险—收益权衡。见 reports/仓位下限敏感性_*.md
                floor = float(getattr(cfg, "EXPOSURE_FLOOR", 0.0) or 0.0)
                if floor > 0.0:
                    scale = max(scale, min(floor, 1.0))
                target = {j: float(w) * scale for j, w in raw.items()}
                # 个股止损：t 收盘触发 → t+1 开盘强制卖出，本周期不回补
                if stop_loss:
                    for j in engine.stop_loss_sells(t, cfg):
                        forced[j] = "stop_loss"
                        target[j] = 0.0
                        if pending_weights is not None:
                            pending_weights[j] = 0.0

                # ---- 再平衡政策：漂移带 / 换手预算（默认全 None = 关闭，逐位不变）
                # 参考 QuantJourney 的 drift_threshold / turnover_budget，但换算成
                # **组合层面的单向换手**：本策略 34 只持仓，单只权重仅 ~2.9%，
                # QJ 那种"单只偏离 < 5% 就不动"的绝对带宽比任何一只仓位都大，
                # 会把首次建仓一起跳过（冒烟实测：全程空仓、Sharpe 假高 3.49）。
                # 因此口径统一为 TO = 0.5·Σ_j |target_j − current_j|：
                #   drift_band     = 换手**下限**，本次要动的量太小 → 整次跳过；
                #   turnover_budget= 换手**上限**，超了 → 偏离按比例收缩（部分调仓）。
                # current 用含价格漂移的实际权重；空仓（首次建仓）永不跳过。
                # 止损/退市属风险出口，不被漂移带跳过、也不被预算缩回。
                skip_rebal = False
                to_one = float("nan")
                if drift_band is not None or turnover_budget is not None:
                    keys = set(target) | set(cur_weights)
                    diffs = {j: float(target.get(j, 0.0)) - float(cur_weights.get(j, 0.0))
                             for j in keys}
                    to_one = 0.5 * sum(abs(v) for v in diffs.values())
                    if (drift_band is not None and to_one < float(drift_band)
                            and cur_weights and not forced):
                        skip_rebal = True
                    elif turnover_budget is not None and to_one > float(turnover_budget):
                        k = float(turnover_budget) / to_one
                        target = {j: float(cur_weights.get(j, 0.0)) + dv * k
                                  for j, dv in diffs.items()}
                        for j in forced:      # 风险出口不参与收缩
                            if j in target:
                                target[j] = 0.0
                        target = {j: w for j, w in target.items() if w > 1e-12}
                        n_budget_capped += 1
                        if verbose:
                            print(f"    [budget] {dates[i].date()} 单向换手 "
                                  f"{to_one:.1%} > {float(turnover_budget):.0%} → "
                                  f"收缩系数 {k:.2f}")

                if skip_rebal:
                    n_drift_skipped += 1
                    if verbose:
                        print(f"    [drift] {dates[i].date()} 单向换手仅 "
                              f"{to_one:.1%} < {float(drift_band):.0%} → 跳过本次调仓")
                    continue

                pending_weights = target
                pending_nav = float(nav)
                pending_shares = None
                pending_first_exec = True
                n_rebalances += 1
                for j, w in target.items():
                    if w > 0:
                        weight_rows.append({"date": dates[i], "code": md.codes[j],
                                            "weight": float(w), "reason": "rebalance"})
                exposure_rows.append(_exposure_row(
                    md, engine, t, nav, target, reg_scale, dd_1y, dd_state,
                    exposure_cap, ind_cols, vol_scale=vt_scale,
                    realized_vol=realized_vol))

    # ---------------------------------------------------------- 4) 期末清算（可选）
    last_t = end_row
    if liquidate_at_end and engine.positions:
        px = np.array([engine._close_px(last_t, j) for j in engine.positions], dtype=float)
        cost_total = 0.0
        for j, p in zip(list(engine.positions.keys()), px):
            pos = engine.positions[j]
            if not np.isfinite(p):
                p = pos.entry_price
            amount = pos.shares * p
            comm = max(float(cfg.MIN_COMMISSION), float(cfg.COMMISSION_RATE) * amount)
            stamp = engine.stamp_rate(last_t) * amount
            transfer = float(cfg.TRANSFER_FEE) * amount
            trades.append(Trade(date=last_t, code=j, side="sell", shares=pos.shares,
                                price=float(p), amount=float(amount), commission=comm,
                                stamp=stamp, transfer=transfer, slip=0.0, reason="liquidate"))
            cost_total += comm + stamp + transfer
            engine.cash += amount - comm - stamp - transfer
            engine.positions.pop(j)
        nav_vals[-1] = engine.nav(last_t)  # 修正末值（扣清算费用）

    # ---------------------------------------------------------- 5) 汇总
    nav_series = pd.Series(nav_vals, index=dates, name="nav")
    daily_ret = nav_series.pct_change()
    if len(daily_ret):
        daily_ret.iloc[0] = nav_series.iloc[0] / float(cfg.CAPITAL) - 1.0
    daily_ret = daily_ret.fillna(0.0)

    weights_df = pd.DataFrame(weight_rows, columns=["date", "code", "weight", "reason"])
    trades_df = pd.DataFrame(_trade_rows(md, trades),
                             columns=["date", "code", "side", "shares", "price", "amount",
                                      "commission", "stamp", "transfer", "slip", "reason"])
    holdings_df = pd.DataFrame(holding_rows)
    if exposure_rows:
        exposures_df = pd.DataFrame(exposure_rows).set_index("date")
    else:
        exposures_df = pd.DataFrame()

    # ---------------------------------------------------------- 6) 指标
    bench = None
    try:
        bench = md.bench.reindex(dates)
    except Exception:  # pragma: no cover
        bench = None
    perf = qmetrics.performance(nav_series, bench_returns=bench,
                                rf=float(getattr(cfg, "CASH_YIELD", 0.0)))

    years = max(len(nav_series) - 1, 1) / 252.0
    avg_nav = float(nav_series.mean()) if len(nav_series) else float(cfg.CAPITAL)
    if len(trades_df):
        buy_amt = float(trades_df.loc[trades_df["side"] == "buy", "amount"].sum())
        sell_amt = float(trades_df.loc[trades_df["side"] == "sell", "amount"].sum())
        cost_total = float((trades_df["commission"] + trades_df["stamp"]
                            + trades_df["transfer"] + trades_df["slip"]).sum())
        n_buy = int((trades_df["side"] == "buy").sum())
        n_sell = int((trades_df["side"] == "sell").sum())
    else:
        buy_amt = sell_amt = cost_total = 0.0
        n_buy = n_sell = 0
    turnover_single = (buy_amt + sell_amt) / 2.0 / avg_nav / years if avg_nav > 0 else np.nan
    turnover_double = (buy_amt + sell_amt) / avg_nav / years if avg_nav > 0 else np.nan
    cost_ratio = cost_total / avg_nav / years if avg_nav > 0 else np.nan
    perf.update({
        "n_trades": int(n_buy + n_sell),
        "n_buy_trades": n_buy,
        "n_sell_trades": n_sell,
        "n_rebalances": int(n_rebalances),
        "avg_holdings": float(n_holdings_sum / n_rows) if n_rows else np.nan,
        "trade_amount_buy": buy_amt,
        "trade_amount_sell": sell_amt,
        # turnover_ann = 年化**单边**换手（(买+卖)/2/均值净值/年）；
        # turnover_ann_double = 年化双边换手（(买+卖)/均值净值/年）
        "turnover_ann": float(turnover_single),
        "turnover_ann_single": float(turnover_single),
        "turnover_ann_double": float(turnover_double),
        "cost_total": cost_total,
        "cost_ratio": float(cost_ratio),
        "cost_bp_per_trade": float(cost_total / (buy_amt + sell_amt) * 1e4)
        if (buy_amt + sell_amt) > 0 else np.nan,
        "n_band_skipped": int(engine.n_band_skipped),
        # 再平衡政策计数（默认关闭时两者恒为 0，不影响任何历史口径）
        "n_drift_skipped": int(n_drift_skipped),
        "n_budget_capped": int(n_budget_capped),
        "avg_trade_amount": float((buy_amt + sell_amt) / (n_buy + n_sell))
        if (n_buy + n_sell) > 0 else np.nan,
        "years": float(years),
        "elapsed_sec": float(time.time() - t0),
    })

    config = {
        "start_row": int(start_row), "end_row": int(end_row),
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "capital": float(cfg.CAPITAL),
        "n_holdings": int(n_holdings or cfg.N_HOLDINGS),
        "max_weight": float(cfg.MAX_WEIGHT), "min_weight": float(cfg.MIN_WEIGHT),
        "industry_max": float(cfg.INDUSTRY_MAX),
        "board_max_growth": float(cfg.BOARD_MAX_GROWTH),
        "rebalance_days": int(rb), "turnover_buffer": float(cfg.TURNOVER_BUFFER),
        "drift_band": None if drift_band is None else float(drift_band),
        "turnover_budget": None if turnover_budget is None else float(turnover_budget),
        "rebalance_offset": int(offset),
        "first_signal_row": int(start_row + offset),
        "commission_rate": float(cfg.COMMISSION_RATE),
        "min_commission": float(cfg.MIN_COMMISSION),
        "stamp_tax_sell": float(cfg.STAMP_TAX_SELL),
        "stamp_tax_sell_old": float(engine.stamp_tax_old),
        "stamp_tax_cut_date": str(engine.stamp_cut_date.date()),
        "transfer_fee": float(cfg.TRANSFER_FEE),
        "slippage_bp": float(cfg.SLIPPAGE_BP),
        "trade_band_rel": float(trade_band_rel),
        "trade_band_abs": float(trade_band_abs),
        "max_new_entries": None if max_new_entries is None else int(max_new_entries),
        "risk_overlay": bool(risk_overlay), "stop_loss": bool(stop_loss),
        "vol_target": float(getattr(cfg, "VOL_TARGET", 0.0) or 0.0),
        "vol_lookback": int(getattr(cfg, "VOL_LOOKBACK", 20) or 20),
        "vol_min_exp": float(getattr(cfg, "VOL_MIN_EXP", 0.5)),
        "vol_max_exp": float(getattr(cfg, "VOL_MAX_EXP", 1.0)),
        "stop_loss_thr": float(cfg.STOP_LOSS),
        "dd_throttle_1": float(cfg.DD_THROTTLE_1), "dd_level_1": float(cfg.DD_LEVEL_1),
        "dd_throttle_2": float(cfg.DD_THROTTLE_2), "dd_level_2": float(cfg.DD_LEVEL_2),
        "dd_window": 252, "dd_recovery": float(cfg.DD_THROTTLE_1 / 2.0),
        "cash_yield": float(cfg.CASH_YIELD),
        "liquidate_at_end": bool(liquidate_at_end),
    }

    if verbose:
        print(f"[backtest] {config['start']} ~ {config['end']}  "
              f"CAGR={perf.get('cagr', np.nan):.2%}  Sharpe={perf.get('sharpe', np.nan):.2f}  "
              f"MaxDD={perf.get('max_dd', np.nan):.2%}  trades={perf['n_trades']}  "
              f"耗时={perf['elapsed_sec']:.1f}s")

    return BacktestResult(nav=nav_series, daily_ret=daily_ret, weights=weights_df,
                          trades=trades_df, holdings=holdings_df, exposures=exposures_df,
                          metrics=perf, config=config)
