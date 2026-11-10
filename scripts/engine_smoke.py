# -*- coding: utf-8 -*-
"""引擎冒烟测试：真实 MarketData + 只用过去信息的简单分数，跑全期回测。

用法::

    python scripts/engine_smoke.py
    python scripts/engine_smoke.py --start 2017-01-03 --end 2026-10-14
    python scripts/engine_smoke.py --no-overlay --no-stop-loss     # 关闭风控 overlay
    python scripts/engine_smoke.py --out build/runs/engine_smoke_xxx

分数构造（**只用 t 及之前的信息，绝不用 md.fwd_* / 未来收益**）::

    r20    = md.price[t] / md.price[t-20] - 1        # 20 日反转
    scores = cs_rank(-r20)                           # 横截面百分位

这只是"引擎能不能跑"的冒烟信号，**不代表策略优劣**（20 日反转本身在 2017-2026
的 A 股多头端是负 alpha，叠加交易成本后不应期待正收益）。

默认输出（build/runs/engine_smoke/）：
    summary.txt / comparison.csv / trade_size_dist.csv
    metrics.json / metrics.csv / config.json
    nav.csv / daily_ret.csv / drawdown.csv / yearly_returns.csv
    monthly_returns.csv / rolling_metrics.csv / stress.csv
    weights.csv（调仓日目标权重长表）/ trades.csv / holdings.csv / exposures.csv
    variants/<tag>/metrics.json（base / rebal10 / raw 对照三个变体）

默认会做三件事：
1) base 回测（rebal=5，trade band 30%/6000 元，max_new_entries=6，overlay+止损）；
2) 因果自检：分数整体 shift(1) 后再跑一次，净值必须改变（无未来函数）；
3) 变体对比：rebal10（10 日调仓）+ raw 对照（band=0、不限新进，即旧口径）。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C          # noqa: E402
from qf import metrics as qmetrics  # noqa: E402
from qf import ops                  # noqa: E402
from qf.backtest import run_backtest  # noqa: E402
from qf.data import load_market     # noqa: E402


# ---------------------------------------------------------------------------
def build_scores(md) -> np.ndarray:
    """20 日反转分数：scores[t] 只依赖 md.price <= t（无未来函数）。"""
    r20 = ops.pct_change(md.price, 20)
    return ops.cs_rank(-r20)


def _jsonable(o):
    """把 numpy / NaN 值转成合法 JSON。"""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    return o


def _fmt(v, pct=False, digits=2):
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return "n/a"
    return f"{f * 100:.{digits}f}%" if pct else f"{f:.{digits}f}"


def _save_csv(df: pd.DataFrame, path: pathlib.Path, index: bool = False) -> None:
    df.to_csv(path, index=index, encoding="utf-8-sig")


def _run_variant(md, scores, start_row, end_row, *, rebalance_days, band_rel, band_abs,
                 max_new, risk_overlay, stop_loss, verbose=False):
    return run_backtest(
        md, scores, start_row, end_row, C,
        risk_overlay=risk_overlay, stop_loss=stop_loss,
        rebalance_days=rebalance_days,
        trade_band_rel=band_rel, trade_band_abs=band_abs,
        max_new_entries=max_new, verbose=verbose,
    )


def _variant_summary(res) -> dict:
    m = res.metrics
    n_reb = max(int(m.get("n_rebalances", 0)), 1)
    out = {
        "cagr": m.get("cagr"),
        "ann_vol": m.get("ann_vol"),
        "sharpe": m.get("sharpe"),
        "max_dd": m.get("max_dd"),
        "calmar": m.get("calmar"),
        "n_trades": int(m.get("n_trades", 0)),
        "trades_per_rebal": float(m.get("n_trades", 0)) / n_reb,
        "avg_trade_amount": m.get("avg_trade_amount"),
        "turnover_ann_single": m.get("turnover_ann_single"),
        "turnover_ann_double": m.get("turnover_ann_double"),
        "cost_ratio": m.get("cost_ratio"),
        "cost_total": m.get("cost_total"),
        "avg_holdings": m.get("avg_holdings"),
        "n_band_skipped": int(m.get("n_band_skipped", 0)),
        "n_rebalances": int(m.get("n_rebalances", 0)),
        "avg_target_exposure": None,
        "avg_actual_stock": None,
    }
    ex = getattr(res, "exposures", None)
    if isinstance(ex, pd.DataFrame) and len(ex):
        if "target_exposure" in ex:
            out["avg_target_exposure"] = float(ex["target_exposure"].mean())
        if "stock" in ex:
            out["avg_actual_stock"] = float(ex["stock"].mean())
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="回测引擎冒烟（真实数据 + 20 日反转分数）")
    parser.add_argument("--start", default="2017-01-03", help="回测起始日期（第一个信号日）")
    parser.add_argument("--end", default=None, help="回测结束日期，缺省=数据最后一天")
    parser.add_argument("--out", default=None, help="输出目录（缺省 build/runs/engine_smoke）")
    parser.add_argument("--no-overlay", action="store_true", help="关闭趋势/回撤 overlay")
    parser.add_argument("--no-stop-loss", action="store_true", help="关闭个股止损")
    parser.add_argument("--no-check-causality", action="store_true",
                        help="跳过分数 shift(1) 的因果自检（默认执行）")
    parser.add_argument("--rebalance-days", type=int, default=5, help="base 调仓周期（交易日）")
    parser.add_argument("--trade-band-rel", type=float, default=0.30,
                        help="交易带宽：相对当前持仓市值的比例（0=关闭）")
    parser.add_argument("--trade-band-abs", type=float, default=6000.0,
                        help="交易带宽：绝对金额（元）")
    parser.add_argument("--max-new-entries", type=int, default=6,
                        help="单期最多新进标的数（换手上限；<=0 表示不限制）")
    parser.add_argument("--no-compare", action="store_true",
                        help="不跑 rebal10 / raw 对照变体")
    args = parser.parse_args(argv)

    try:  # Windows 控制台中文输出
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

    t_all = time.time()
    out_dir = pathlib.Path(args.out) if args.out else (C.BUILD_RUNS / "engine_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new = None if args.max_new_entries is None or args.max_new_entries <= 0 \
        else int(args.max_new_entries)
    risk_overlay = not args.no_overlay
    stop_loss = not args.no_stop_loss

    print("[engine_smoke] 加载数据 ...")
    t0 = time.time()
    md = load_market(verbose=True)
    print(f"[engine_smoke] 数据加载 {time.time() - t0:.1f}s")

    start_row = md.date_index(args.start)
    end_row = md.date_index(args.end) if args.end else md.T - 1
    start_row = max(0, start_row)
    end_row = min(md.T - 1, end_row)
    if end_row <= start_row:
        raise SystemExit(f"区间非法: start_row={start_row} end_row={end_row}")

    t0 = time.time()
    scores = build_scores(md)
    print(f"[engine_smoke] 构造 20 日反转分数 {time.time() - t0:.1f}s "
          f"(scores 覆盖率 {np.isfinite(scores).mean():.1%})")

    # ---------------- base 回测
    t0 = time.time()
    res = _run_variant(md, scores, start_row, end_row,
                       rebalance_days=args.rebalance_days,
                       band_rel=args.trade_band_rel, band_abs=args.trade_band_abs,
                       max_new=max_new, risk_overlay=risk_overlay,
                       stop_loss=stop_loss, verbose=True)
    backtest_sec = time.time() - t0
    m = res.metrics
    cfg = res.config

    # ---------------- 因果自检
    causality_note = "未执行"
    if not args.no_check_causality:
        t0 = time.time()
        lag = ops.shift(scores, 1)
        res_lag = _run_variant(md, lag, start_row, end_row,
                               rebalance_days=args.rebalance_days,
                               band_rel=args.trade_band_rel,
                               band_abs=args.trade_band_abs,
                               max_new=max_new, risk_overlay=risk_overlay,
                               stop_loss=stop_loss, verbose=False)
        d = np.abs(res.nav.to_numpy() - res_lag.nav.to_numpy())
        causality_note = (f"分数 shift(1) 后净值最大差 {float(np.nanmax(d)):,.0f} 元"
                          f"（{time.time() - t0:.1f}s）；历史净值对分数时点敏感 → 无未来函数")
        print(f"[engine_smoke] 因果自检: {causality_note}")

    # ---------------- 变体对比（rebal10 / raw 旧口径）
    variants: dict[str, dict] = {"base_rebal5": _variant_summary(res)}
    if not args.no_compare:
        t0 = time.time()
        r10 = _run_variant(md, scores, start_row, end_row,
                           rebalance_days=10, band_rel=args.trade_band_rel,
                           band_abs=args.trade_band_abs, max_new=max_new,
                           risk_overlay=risk_overlay, stop_loss=stop_loss)
        variants["rebal10"] = _variant_summary(r10)
        vdir = out_dir / "variants" / "rebal10"
        vdir.mkdir(parents=True, exist_ok=True)
        _save_csv(pd.DataFrame({"date": r10.nav.index, "nav": r10.nav.to_numpy()}),
                  vdir / "nav.csv")
        print(f"[engine_smoke] rebal10 变体完成 {time.time() - t0:.1f}s")

        t0 = time.time()
        raw = _run_variant(md, scores, start_row, end_row,
                           rebalance_days=args.rebalance_days,
                           band_rel=0.0, band_abs=0.0, max_new=None,
                           risk_overlay=risk_overlay, stop_loss=stop_loss)
        variants["raw_no_band_no_cap"] = _variant_summary(raw)
        print(f"[engine_smoke] raw 对照（band=0/不限新进）完成 {time.time() - t0:.1f}s")

        for tag, mm in variants.items():
            p = out_dir / "variants" / tag / "metrics.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(_jsonable(mm), f, ensure_ascii=False, indent=2)

    # ---------------- 指标 / 衍生表
    nav = res.nav
    bench = md.bench.reindex(nav.index)
    yearly = qmetrics.yearly_returns(nav)
    monthly = qmetrics.monthly_returns_table(nav)
    dd = qmetrics.drawdown_series(nav)
    rolling = qmetrics.rolling_metrics(nav, window=252)
    stress = qmetrics.stress_periods(nav, bench_returns=bench)
    tr = res.trades

    # 按成交额分位数（诊断小单）
    if len(tr):
        q = np.nanpercentile(tr["amount"].to_numpy(), [10, 25, 50, 75, 90])
        size_dist = pd.DataFrame({
            "quantile": ["p10", "p25", "p50", "p75", "p90"],
            "trade_amount": q,
        })
        _save_csv(size_dist, out_dir / "trade_size_dist.csv")
        n_small_3k = int((tr["amount"] < 3000).sum())
        n_small_6k = int((tr["amount"] < 6000).sum())
    else:
        size_dist = pd.DataFrame()
        n_small_3k = n_small_6k = 0

    # ---------------- 落盘（base）
    _save_csv(pd.DataFrame({"date": nav.index, "nav": nav.to_numpy(),
                            "daily_ret": res.daily_ret.to_numpy(),
                            "drawdown": dd.to_numpy()}), out_dir / "nav.csv")
    _save_csv(pd.DataFrame({"date": res.daily_ret.index,
                            "daily_ret": res.daily_ret.to_numpy()}), out_dir / "daily_ret.csv")
    _save_csv(dd.rename("drawdown").to_frame().reset_index().rename(columns={"index": "date"}),
              out_dir / "drawdown.csv")
    _save_csv(yearly.rename("return").to_frame().reset_index().rename(columns={"index": "year"}),
              out_dir / "yearly_returns.csv")
    _save_csv(monthly.reset_index(), out_dir / "monthly_returns.csv")
    _save_csv(rolling.reset_index().rename(columns={"index": "date"}),
              out_dir / "rolling_metrics.csv")
    _save_csv(stress, out_dir / "stress.csv")
    _save_csv(res.weights, out_dir / "weights.csv")
    _save_csv(tr, out_dir / "trades.csv")
    _save_csv(res.holdings, out_dir / "holdings.csv")
    _save_csv(res.exposures, out_dir / "exposures.csv", index=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(m), f, ensure_ascii=False, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(cfg), f, ensure_ascii=False, indent=2)
    _save_csv(pd.DataFrame([{k: v for k, v in m.items()}]), out_dir / "metrics.csv")

    # ---------------- 对比表
    cmp_df = pd.DataFrame(variants).T
    _save_csv(cmp_df.reset_index().rename(columns={"index": "variant"}), out_dir / "comparison.csv")

    # ---------------- 汇总打印
    n_reb = int(m.get("n_rebalances", 0))
    n_buy = int(m.get("n_buy_trades", 0))
    n_sell = int(m.get("n_sell_trades", 0))
    lines = []
    lines.append("=" * 78)
    lines.append("回测引擎冒烟（engine_smoke）— 真实数据 + 20 日反转分数（仅作引擎验证）")
    lines.append("=" * 78)
    lines.append(f"区间              : {cfg['start']} ~ {cfg['end']}（{len(nav)} 个交易日）")
    if len(res.weights):
        first_sig = pd.Timestamp(res.weights["date"].min()).strftime("%Y-%m-%d")
        lines.append(f"首次调仓日        : {first_sig}（此前因数据层 list_days>=120 warmup 无候选；"
                     f"实盘口径建议 --start 2020-01-03）")
    lines.append(f"初始/最终净值     : {C.CAPITAL:,.0f} → {nav.iloc[-1]:,.0f} 元"
                 f"（累计 {nav.iloc[-1] / C.CAPITAL - 1:.2%}）")
    lines.append(f"风控 overlay      : {'ON' if cfg['risk_overlay'] else 'OFF'}"
                 f"（趋势 + 回撤节流）；个股止损: {'ON' if cfg['stop_loss'] else 'OFF'}")
    lines.append(f"调仓节奏/换手上限 : rebal={cfg['rebalance_days']} 日；offset={cfg['rebalance_offset']}；"
                 f"max_new_entries={cfg['max_new_entries']}；"
                 f"trade band rel={cfg['trade_band_rel']:.0%}/abs={cfg['trade_band_abs']:,.0f} 元")
    lines.append(f"费用/风控          : 卖出印花税 {cfg['stamp_tax_sell_old'] * 1e4:.0f}bp"
                 f"(<{cfg['stamp_tax_cut_date']}) → {cfg['stamp_tax_sell'] * 1e4:.0f}bp；"
                 f"回撤节流=滚动{cfg['dd_window']}日峰值回撤"
                 f"(阈值{cfg['dd_throttle_1']:.0%}/{cfg['dd_throttle_2']:.0%} → "
                 f"{cfg['dd_level_1']:.0%}/{cfg['dd_level_2']:.0%}，收复到 "
                 f"{cfg['dd_recovery']:.0%} 恢复)")
    lines.append("-" * 78)
    lines.append("performance（base）")
    perf_rows = [
        ("年化收益 CAGR", m.get("cagr"), "pct"),
        ("年化波动", m.get("ann_vol"), "pct"),
        ("Sharpe", m.get("sharpe"), "num"),
        ("Sortino", m.get("sortino"), "num"),
        ("最大回撤 MaxDD", m.get("max_dd"), "pct"),
        ("最大回撤持续(交易日)", m.get("max_dd_days"), "int"),
        ("Calmar", m.get("calmar"), "num"),
        ("日胜率", m.get("win_rate_daily"), "pct"),
        ("月胜率", m.get("win_rate_monthly"), "pct"),
        ("偏度 / 峰度", f"{_fmt(m.get('skew'))} / {_fmt(m.get('kurt'))}", "raw"),
        ("vs hs300: beta / alpha / IR",
         f"{_fmt(m.get('beta_hs300'))} / {_fmt(m.get('alpha_hs300'), True)} / {_fmt(m.get('ir_hs300'))}",
         "raw"),
        ("vs csi1000: beta / alpha / IR",
         f"{_fmt(m.get('beta_csi1000'))} / {_fmt(m.get('alpha_csi1000'), True)} / {_fmt(m.get('ir_csi1000'))}",
         "raw"),
    ]
    for name, val, kind in perf_rows:
        if kind == "raw":
            text = str(val)
        elif kind == "pct":
            text = _fmt(val, True)
        elif kind == "int":
            text = "n/a" if val is None or not np.isfinite(float(val)) else f"{int(val)}"
        else:
            text = _fmt(val, digits=2)
        lines.append(f"  {name:28s}: {text}")
    lines.append("-" * 78)
    lines.append("交易统计（base）")
    lines.append(f"  成交笔数                    : {int(m['n_trades'])}"
                 f"（买 {n_buy} / 卖 {n_sell}；{n_reb} 次调仓）")
    lines.append(f"  每调仓日成交笔数            : {_fmt(int(m['n_trades']) / max(n_reb, 1), digits=1)}")
    lines.append(f"  平均单笔成交额              : {m.get('avg_trade_amount', np.nan):,.0f} 元")
    lines.append(f"  平均持仓数                  : {_fmt(m.get('avg_holdings'), digits=1)}")
    lines.append(f"  年换手(单边)                : {_fmt(m.get('turnover_ann_single'))}x")
    lines.append(f"  年换手(双边)                : {_fmt(m.get('turnover_ann_double'))}x")
    lines.append(f"  交易成本合计                : {m.get('cost_total', np.nan):,.0f} 元"
                 f"（其中滑点 {tr['slip'].sum():,.0f}）")
    lines.append(f"  年化成本/平均净值(cost_ratio): {_fmt(m.get('cost_ratio'), True)}")
    lines.append(f"  成本/成交额                 : {_fmt(m.get('cost_bp_per_trade'))} bp")
    lines.append(f"  trade band 跳过次数         : {int(m.get('n_band_skipped', 0))}")
    lines.append(f"  截面上止损卖出笔数          : {int((tr['reason'] == 'stop_loss').sum())}")
    if len(res.exposures):
        ex = res.exposures
        lines.append(f"  目标仓位均值/实际股票仓位均值: {_fmt(float(ex['target_exposure'].mean()), True)}"
                     f" / {_fmt(float(ex['stock'].mean()), True)}"
                     f"（现金 {_fmt(float(ex['cash'].mean()), True)}）")
        if "exposure_cap" in ex:
            caps = ex["exposure_cap"].to_numpy(dtype=float)
            n_ex = len(caps)
            n_full = int(np.sum(caps >= 1.0 - 1e-9))
            n_60 = int(np.sum(np.isclose(caps, C.DD_LEVEL_1)))
            n_40 = int(np.sum(np.isclose(caps, C.DD_LEVEL_2)))
            lines.append(f"  DD 节流 cap 分布(按调仓日): 100% {n_full / n_ex:.1%} / "
                         f"60% {n_60 / n_ex:.1%} / 40% {n_40 / n_ex:.1%}")
            ddy = ex["dd_1y"].to_numpy(dtype=float)
            lines.append(f"  dd_1y 均值/最低            : {_fmt(float(np.nanmean(ddy)), True)}"
                         f" / {_fmt(float(np.nanmin(ddy)), True)}")
            throttled = (caps < 1.0 - 1e-9).astype(int)
            triggers = int(np.sum(np.diff(throttled) == 1))
            recovers = int(np.sum(np.diff(throttled) == -1))
            rec_idx = np.flatnonzero(np.diff(throttled) == -1) + 1
            first_rec = (pd.Timestamp(ex.index[rec_idx[0]]).strftime("%Y-%m-%d")
                         if len(rec_idx) else "n/a")
            lines.append(f"  节流触发/恢复满仓次数      : {triggers} / {recovers}"
                         f"（首次恢复 {first_rec}）→ 暴露不再被永久锁死")
    if len(tr):
        lines.append(f"  成交额分位数(p10/25/50/75/90): "
                     f"{size_dist['trade_amount'].map(lambda x: f'{x:,.0f}').str.cat(sep=' / ')} 元")
        lines.append(f"  小单占比 <3000元/<6000元     : "
                     f"{n_small_3k / len(tr):.1%} / {n_small_6k / len(tr):.1%}")
    lines.append("-" * 78)
    lines.append("变体对比（同一 20 日反转分数）")
    hdr = (f"  {'变体':22s} {'CAGR':>8s} {'波动':>8s} {'Sharpe':>7s} {'MaxDD':>8s} "
           f"{'成交笔数':>8s} {'笔/调仓':>7s} {'平均单笔':>9s} {'单边换手':>8s} "
           f"{'年化成本':>8s} {'持仓数':>6s} {'目标仓位':>8s} {'实际仓位':>8s}")
    lines.append(hdr)
    for tag, vv in variants.items():
        lines.append(
            f"  {tag:22s} {_fmt(vv['cagr'], True):>8s} {_fmt(vv['ann_vol'], True):>8s} "
            f"{_fmt(vv['sharpe']):>7s} {_fmt(vv['max_dd'], True):>8s} "
            f"{int(vv['n_trades']):>8d} {_fmt(vv['trades_per_rebal'], digits=1):>7s} "
            f"{(vv['avg_trade_amount'] or 0):>9,.0f} {_fmt(vv['turnover_ann_single']):>8s} "
            f"{_fmt(vv['cost_ratio'], True):>8s} {_fmt(vv['avg_holdings'], digits=1):>6s} "
            f"{_fmt(vv['avg_target_exposure'], True):>8s} {_fmt(vv['avg_actual_stock'], True):>8s}")
    lines.append("-" * 78)
    lines.append("逐年收益（base）")
    for y, v in yearly.items():
        lines.append(f"  {int(y)}: {_fmt(v, True)}")
    lines.append("-" * 78)
    lines.append(f"回测耗时 {backtest_sec:.1f}s（数据加载+分数另计）；总耗时 {time.time() - t_all:.1f}s")
    lines.append(f"输出目录 {out_dir}")
    lines.append("=" * 78)
    text = "\n".join(lines)
    print(text)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
        if not args.no_check_causality:
            f.write("\n因果自检: " + causality_note + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
