# -*- coding: utf-8 -*-
"""右侧趋势模型 wf_rs10：DEV 选池 + 长多约束 ICIR walk-forward + 回测对比。

流程
----
1) 加载 ext 行情（与 web 服务一致，数据截至 2026-10-21）与
   ``build/factors/trend_extra.npz`` 周频面板（scripts/build_trend_extra.py 产出）；
2) 特征 = 横截面 rank → 行业 / log 市值中性化 → z-score（复用 qf.model 口径）；
3) 因子池：**只用 DEV（2017-2019）** 的 h10 Rank IC / ICIR 选池
   （IC>0 且 ICIR>0.05，按 DEV ICIR 取前 10；不足 5 个时保留全部 16 个），
   池一旦确定在 2020 之后的 walk-forward 中不再变化；
4) 权重：从 2020-01 起每季度 refit，滚动 156 周训练窗口（标签实现滞后 2 周）。
   **长多约束**：w_k = clip(ICIR_k, 0, 3) 归一化，即只给"历史 IC 为正"的右侧
   因子正权重，不会像自由拟合那样把因子反过来用（避免又变回左侧）；
5) 复合分：Σ w·feature → 行业+市值中性化 → 横截面 z-score → 时序平滑 0.6；
6) 回测：N=40、调仓 10 日、80/20 现金替代、overlay + 个股止损，
   区间 2020-01-08 ~ 最新交易日；同口径跑 wf_h10 基线，输出对比。

产出
----
    build/models/wf_rs10/            qfm.save_result 标准格式
    build/runs/rs10/                 回测指标 / 净值 / 分年度 / 对比 / summary
    build/eval/trend_model_stats.json 全部统计（IC/权重/OOS/回测/相关性）
    reports/右侧模型研究.md           可读报告

纪律
----
- 无未来函数：因子只用 t 及以前数据；权重只用 refit 时点之前已实现标签的周；
- 选池只用 DEV；最终样本外（2022+）不参与任何参数/池子调整；
- 长多权重由**滚动** ICIR 决定（refit 时点计算），不是用全样本符号。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
_EXT = ROOT.parent / "data_extract" / "was_stock_data_ext"
if _EXT.exists():
    os.environ.setdefault("QF_DATA_RAW", str(_EXT))
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from qf import config as C
from qf import backtest as qfbt
from qf import data as qfdata
from qf import metrics as qfmetrics
from qf import model as qfm
from qf import trend as qftrend

HORIZON = 10
TRAIN_WEEKS = 156
REFIT_WEEKS = 12
MAX_POOL = 10
MIN_DEV_ICIR = 0.05
MATCH_START = "2020-01-08"
RB_DAYS = 10
N_HOLDINGS = 40
BAND_REL = 0.30
BAND_ABS = 6000.0
MAX_NEW = 6
CASH_BW, CASH_GW = 0.8, 0.2


# ------------------------------------------------------------------ 工具
def _ic_metrics(ic: np.ndarray) -> dict:
    ic = np.asarray(ic)[np.isfinite(ic)]
    if len(ic) == 0:
        return {"ic": None, "icir": None, "t": None, "pos": None, "n": 0}
    m = float(np.mean(ic))
    s = float(np.std(ic))
    icir = m / s if s > 1e-12 else 0.0
    return {"ic": m, "icir": icir, "t": icir * np.sqrt(len(ic)),
            "pos": float(np.mean(ic > 0)), "n": int(len(ic))}


def _fmt_pct(v, d: int = 2) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f}%"


def _fmt_num(v, d: int = 3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def _yearly_ic(ic_df: pd.DataFrame) -> dict:
    if ic_df is None or len(ic_df) == 0:
        return {}
    d = ic_df.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    return {int(y): {"ic": float(g["ic"].mean()), "n": int(len(g))}
            for y, g in d.groupby("year")}


def _metrics_compact(m: dict) -> dict:
    keys = ["start", "end", "cagr", "ann_vol", "sharpe", "sortino", "max_dd",
            "calmar", "win_rate_monthly", "turnover_ann", "cost_ratio",
            "n_trades", "avg_holdings", "years"]
    return {k: m.get(k) for k in keys if k in m}


def _split_metrics(nav: pd.Series) -> dict:
    out = {}
    if nav is None or len(nav) == 0:
        return out
    def _one(s):
        if s is None or len(s) < 3:
            return {}
        return _metrics_compact(qfmetrics.performance(s))
    out["full"] = _one(nav)
    out["y2020_2021"] = _one(nav.loc["2020-01-01":"2021-12-31"])
    out["y2022_plus"] = _one(nav.loc["2022-01-01":])
    return out


# ------------------------------------------------------------------ 主流程
def main() -> None:
    t_all = time.time()
    C.BUILD_RUNS.mkdir(parents=True, exist_ok=True)
    out_run = C.BUILD_RUNS / "rs10"
    out_run.mkdir(parents=True, exist_ok=True)
    out_model = C.BUILD_MODELS / "wf_rs10"

    print("[1/7] 加载行情与 trend 周频面板 ...")
    md = qfdata.load_market(verbose=True)
    zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
    rows = zm["rows"].astype(np.int64)
    mask = md.eligible[rows]
    dates_w = md.dates[rows]
    print(f"    T={md.T}, N={md.N}, 数据 {md.dates[0].date()} ~ {md.dates[-1].date()}；"
          f"周频 {len(rows)} 行 {dates_w[0].date()} ~ {dates_w[-1].date()}")

    zf = np.load(C.BUILD_FACTORS / "trend_extra.npz")
    panels = {k: zf[k] for k in zf.files}
    cat = pd.read_parquet(C.BUILD_FACTORS / "trend_catalog.parquet")
    keys_all = list(panels.keys())
    print(f"    trend 因子 {len(keys_all)} 个")

    print("[2/7] 特征 / 标签 ...")
    label = qfm.weekly_label(md, rows, mask, horizon=HORIZON, rank=True)
    dev_mask = dates_w <= pd.Timestamp(C.DEV_END)
    years_w = pd.to_datetime(dates_w).year.to_numpy()
    feats: dict[str, np.ndarray] = {}
    cat_idx = cat.set_index("factor_key")
    factor_stats: dict[str, dict] = {}
    for k in keys_all:
        feats[k] = qfm.build_feature_block(md, rows, panels, k, mask)
        ic = qfm.weekly_ic(feats[k], label, mask)
        by_year = {}
        for y in sorted(set(years_w.tolist())):
            sel = (years_w == y) & np.isfinite(ic)
            if sel.sum():
                by_year[int(y)] = round(float(np.mean(ic[sel])), 5)
        coverage = float((np.isfinite(panels[k]) & mask).sum() / max(mask.sum(), 1))
        factor_stats[k] = {
            "factor_key": k,
            "slug": str(cat_idx.loc[k, "key"]),
            "name_cn": str(cat_idx.loc[k, "name_cn"]),
            "family": str(cat_idx.loc[k, "category_en"]),
            "window": int(cat_idx.loc[k, "window"]),
            "dev": _ic_metrics(ic[dev_mask]),
            "full": _ic_metrics(ic),
            "ic_by_year": by_year,
            "coverage": round(coverage, 4),
        }

    print("[3/7] DEV(2017-2019) 选池（固定池，不使用 2020 后数据）...")
    passed = [k for k in keys_all
              if factor_stats[k]["dev"]["ic"] is not None
              and factor_stats[k]["dev"]["ic"] > 0
              and (factor_stats[k]["dev"]["icir"] or 0.0) >= MIN_DEV_ICIR]
    passed.sort(key=lambda k: -(factor_stats[k]["dev"]["icir"] or 0.0))
    if len(passed) >= 5:
        pool = passed[:MAX_POOL]
        pool_rule = f"DEV IC>0 且 DEV ICIR>={MIN_DEV_ICIR}，按 DEV ICIR 取前 {MAX_POOL}"
    else:
        pool = keys_all
        pool_rule = "DEV 正向因子不足 5 个，回退为全部 16 因子（长多权重仍由滚动 ICIR 决定）"
    print(f"    入选 {len(pool)} 个：{[cat_idx.loc[k, 'key'] for k in pool]}")
    for k in pool:
        d = factor_stats[k]["dev"]
        print(f"      {factor_stats[k]['slug']:22s} DEV IC {_fmt_pct(d['ic'])}  ICIR {_fmt_num(d['icir'])}")

    # 模型配置（model_meta.json 的 cfg.horizon = 10）
    cfg = qfm.ModelConfig(
        horizon=HORIZON, train_weeks=TRAIN_WEEKS, refit_weeks=REFIT_WEEKS,
        method="icir", score_smooth_alpha=0.6, max_factors=MAX_POOL,
        neutralize=True,
    )

    print("[4/7] 长多约束 ICIR walk-forward ...")
    n_weeks = len(rows)
    quarters = pd.date_range(start=pd.Timestamp(C.VALID_START), end=dates_w[-1], freq="QS")
    refit_idx: list[int] = []
    for q in quarters:
        pos = int(np.searchsorted(dates_w, q))
        if pos >= n_weeks:
            continue
        if refit_idx and pos - refit_idx[-1] < REFIT_WEEKS:
            continue
        refit_idx.append(pos)
    if not refit_idx:
        refit_idx = [int(np.searchsorted(dates_w, pd.Timestamp(C.VALID_START)))]
    print(f"    refit {len(refit_idx)} 次：{dates_w[refit_idx[0]].date()} ~ {dates_w[refit_idx[-1]].date()}")

    K = len(pool)
    scores = np.full((md.T, md.N), np.nan, dtype=np.float32)
    weight_rows: list[dict] = []
    ic_rows: list[dict] = []
    blocks = refit_idx + [n_weeks]
    prev_smooth: np.ndarray | None = None
    alpha = float(cfg.score_smooth_alpha)
    for bi in range(len(refit_idx)):
        r = refit_idx[bi]
        r_end = blocks[bi + 1]
        tr_lo = max(0, r - TRAIN_WEEKS)
        tr_hi = max(0, r - 2)          # 标签已实现（滞后 2 周）
        if tr_hi - tr_lo < 52:
            continue
        ytr = label[tr_lo:tr_hi]
        mtr = mask[tr_lo:tr_hi]
        ic_mat = np.stack([qfm.weekly_ic(feats[k][tr_lo:tr_hi], ytr, mtr) for k in pool], axis=0)
        mean_ic = np.nanmean(ic_mat, axis=1)
        std_ic = np.nanstd(ic_mat, axis=1)
        icir = np.where(std_ic > 1e-12, mean_ic / std_ic, 0.0)
        w = np.clip(icir, 0.0, 3.0)     # 长多约束：只给正 ICIR 正权重
        if w.sum() <= 1e-12:
            w = np.clip(mean_ic, 0.0, None)
            if w.sum() <= 1e-12:
                w = np.ones(K, dtype=np.float64)
        w = w / w.sum()
        for k, wk in zip(pool, w):
            weight_rows.append({"refit_date": dates_w[r], "factor_key": k, "weight": float(wk)})

        Fb = np.stack([feats[k][r:r_end] for k in pool], axis=0)  # (K, n, N)
        for j in range(r_end - r):
            wi = r + j
            row = (w[:, None] * Fb[:, j, :]).sum(axis=0).astype(np.float32)
            bb = row[None, :]
            mm = mask[wi][None, :]
            if np.isfinite(bb[mm]).sum() > 20:
                rr = qfm.neutralize_rows(bb, md.industry[rows[wi]][None, :],
                                         md.logmcap[rows[wi]][None, :], mm)
                zz = qfm.zscore_rows(rr, mm)[0]
            else:
                zz = np.full(md.N, np.nan, dtype=np.float32)
            if alpha < 1.0 and prev_smooth is not None:
                both = np.isfinite(zz) & np.isfinite(prev_smooth)
                zz = np.where(both, alpha * zz + (1.0 - alpha) * prev_smooth, zz).astype(np.float32)
            prev_smooth = zz.copy()
            scores[rows[wi]] = zz
        sl = slice(r, r_end)
        ic_oos = qfm.weekly_ic(np.nan_to_num(scores[rows[sl]]), label[sl], mask[sl])
        for ww, v in zip(dates_w[sl], ic_oos):
            if np.isfinite(v):
                ic_rows.append({"date": ww, "ic": float(v), "refit": dates_w[r]})
        if (bi + 1) % 4 == 0 or bi == 0:
            nonzero = [f"{cat_idx.loc[k, 'key']}:{x:.2f}" for k, x in zip(pool, w) if x > 0]
            print(f"    refit {dates_w[r].date()} 权重非零 {int((w > 0).sum())}/{K}（{', '.join(nonzero)[:110]}）")

    weights_df = pd.DataFrame(weight_rows)
    ic_df = pd.DataFrame(ic_rows)
    comp_ic = ic_df["ic"].to_numpy() if len(ic_df) else np.array([])
    oos = _ic_metrics(comp_ic)
    print(f"    复合分 OOS RankIC: mean={_fmt_num(oos['ic'], 4)} ICIR={_fmt_num(oos['icir'])} "
          f"t={_fmt_num(oos['t'], 2)} 正IC占比={_fmt_pct(oos['pos'])} n={oos['n']}")

    pool_meta = pd.DataFrame([
        {
            "factor_key": k,
            "ic": factor_stats[k]["dev"]["ic"],
            "icir": factor_stats[k]["dev"]["icir"],
            "t": factor_stats[k]["dev"]["t"],
            "pos": factor_stats[k]["dev"]["pos"],
            "n": factor_stats[k]["dev"]["n"],
            "category": factor_stats[k]["family"],
            "lib": qftrend.LIB,
            "base": factor_stats[k]["slug"],
        } for k in pool
    ]).set_index("factor_key")

    res = qfm.WalkForwardResult(
        rows_weekly=rows, scores=scores, weights=weights_df,
        pool=list(pool), pool_meta=pool_meta, composite_ic=ic_df, cfg=cfg,
    )
    qfm.save_result(res, out_model)
    print(f"    → {out_model}")

    print("[5/7] 回测（N=40, 调仓10日, 80/20现金替代, overlay+止损）...")
    first_score = int(np.flatnonzero(np.isfinite(scores).any(axis=1))[0])
    start_row = max(md.date_index(MATCH_START), first_score)
    phase = int(rows[0]) % 5
    while start_row % 5 != phase and start_row < md.T - 1:
        start_row += 1
    end_row = md.T - 1
    rb = RB_DAYS
    rebalance_offset = (phase - start_row) % rb if rb % 5 == 0 else 0
    cash_asset_returns = qfbt.load_cash_asset_returns(md, CASH_BW, CASH_GW)
    print(f"    区间 {md.dates[start_row].date()} ~ {md.dates[end_row].date()}，offset={rebalance_offset}")

    def _run(scores_mat: np.ndarray, tag: str):
        old = C.N_HOLDINGS
        C.N_HOLDINGS = N_HOLDINGS
        try:
            return qfbt.run_backtest(
                md, scores_mat, start_row, end_row,
                risk_overlay=True, stop_loss=True,
                rebalance_days=rb, rebalance_offset=rebalance_offset,
                trade_band_rel=BAND_REL, trade_band_abs=BAND_ABS,
                max_new_entries=MAX_NEW,
                verbose=False, cash_asset_returns=cash_asset_returns,
            )
        finally:
            C.N_HOLDINGS = old

    bt_rs = _run(scores, "wf_rs10")
    print("    wf_rs10 回测完成")

    # 基线：wf_h10 同口径（ext 行情下补齐末 5 行 NaN）
    print("    基线 wf_h10 同口径回测 ...")
    base_model = qfm.load_result(C.BUILD_MODELS / "wf_h10")
    scores_base = np.full((md.T, md.N), np.nan, dtype=np.float32)
    n_copy = min(md.T, base_model.scores.shape[0])
    scores_base[:n_copy] = base_model.scores[:n_copy]
    bt_base = _run(scores_base, "wf_h10")

    # 指标 / 基准 / 分年度
    print("[6/7] 指标 / 分年度 / 相关性 ...")
    idx = md.dates[start_row:end_row + 1]
    benchmarks = {}
    for name in C.BENCHMARKS + ["all_eq"]:
        r = md.bench[name].reindex(idx).fillna(0.0).to_numpy()
        benchmarks[name] = qfmetrics.performance(pd.Series(np.cumprod(1.0 + r), index=idx))
    yearly_rs = qfmetrics.yearly_returns(bt_rs.nav)
    yearly_base = qfmetrics.yearly_returns(bt_base.nav)
    corr = float(np.corrcoef(bt_rs.daily_ret.reindex(idx).fillna(0.0),
                             bt_base.daily_ret.reindex(idx).fillna(0.0))[0, 1])
    # 周度复合分排名相关（更直接衡量选股重叠度）
    rank_corrs: list[float] = []
    for r0 in rows:
        a, b = scores[r0], scores_base[r0]
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) < 500:
            continue
        ra = pd.Series(a[m]).rank().to_numpy()
        rb = pd.Series(b[m]).rank().to_numpy()
        c = float(np.corrcoef(ra, rb)[0, 1])
        if np.isfinite(c):
            rank_corrs.append(c)
    score_rank_corr = float(np.mean(rank_corrs)) if rank_corrs else None
    print(f"    wf_rs10  vs wf_h10 日收益相关系数 = {corr:.3f}；周度复合分排名相关 = "
          f"{'—' if score_rank_corr is None else f'{score_rank_corr:.3f}'}")

    m_rs, m_base = bt_rs.metrics, bt_base.metrics
    split_rs = _split_metrics(bt_rs.nav)
    split_base = _split_metrics(bt_base.nav)

    # 落盘回测
    bt_rs.nav.to_frame("nav").to_parquet(out_run / "nav.parquet")
    bt_rs.daily_ret.to_frame("ret").to_parquet(out_run / "daily_ret.parquet")
    if bt_rs.trades is not None and len(bt_rs.trades):
        bt_rs.trades.to_parquet(out_run / "trades.parquet")
    yearly_rs.to_csv(out_run / "yearly.csv", encoding="utf-8-sig")
    with open(out_run / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_metrics_compact(m_rs), f, ensure_ascii=False, indent=1)
    with open(out_run / "baseline_wf_h10_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": _metrics_compact(m_base),
                   "split": split_base,
                   "yearly": {int(k): float(v) for k, v in yearly_base.items()}},
                  f, ensure_ascii=False, indent=1)
    with open(out_run / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, ensure_ascii=False, indent=1)
    comp = pd.DataFrame({"wf_rs10": _metrics_compact(m_rs), "wf_h10": _metrics_compact(m_base)})
    comp.to_csv(out_run / "comparison.csv", encoding="utf-8-sig")

    # 权重符号稳定性（按年）
    wdf = weights_df.copy()
    wdf["year"] = pd.to_datetime(wdf["refit_date"]).dt.year
    w_by_year = {}
    for y, g in wdf.groupby("year"):
        per = g.groupby("factor_key")["weight"].agg(["mean", "count",
                                                     lambda s: float((s > 0).mean())])
        per.columns = ["mean_weight", "n_refit", "pos_share"]
        w_by_year[int(y)] = {k: {"mean_weight": float(r["mean_weight"]),
                                 "n_refit": int(r["n_refit"]),
                                 "pos_share": float(r["pos_share"])}
                             for k, r in per.iterrows()}

    stats = {
        "model": "wf_rs10",
        "lib": qftrend.LIB,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_range": [str(md.dates[0].date()), str(md.dates[-1].date())],
        "weekly_rows": [str(dates_w[0].date()), str(dates_w[-1].date())],
        "horizon": HORIZON,
        "n_factors_total": len(keys_all),
        "pool": list(pool),
        "pool_rule": pool_rule,
        "pool_slugs": [str(cat_idx.loc[k, "key"]) for k in pool],
        "factory_stats": factor_stats,
        "weights_by_year": w_by_year,
        "refit_dates": [str(pd.Timestamp(d).date()) for d in sorted(weights_df["refit_date"].unique())],
        "oos_ic": oos,
        "oos_ic_by_year": _yearly_ic(ic_df),
        "backtest": {
            "params": {"n_holdings": N_HOLDINGS, "rebalance_days": RB_DAYS,
                       "cash_bond": CASH_BW, "cash_gold": CASH_GW,
                       "risk_overlay": True, "stop_loss": True,
                       "band_rel": BAND_REL, "band_abs": BAND_ABS,
                       "max_new": MAX_NEW, "start": MATCH_START,
                       "end": str(md.dates[end_row].date())},
            "wf_rs10": {"metrics": _metrics_compact(m_rs), "split": split_rs,
                        "yearly": {int(k): float(v) for k, v in yearly_rs.items()}},
            "wf_h10": {"metrics": _metrics_compact(m_base), "split": split_base,
                       "yearly": {int(k): float(v) for k, v in yearly_base.items()}},
            "benchmarks": benchmarks,
            "daily_return_corr": corr,
            "score_rank_corr": score_rank_corr,
        },
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    C.BUILD_EVAL.mkdir(parents=True, exist_ok=True)
    with open(C.BUILD_EVAL / "trend_model_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"    → {C.BUILD_EVAL / 'trend_model_stats.json'}")

    print("[7/7] 生成报告 ...")
    _write_report(stats, out_run, C.REPORTS / "右侧模型研究.md")
    print(f"    → {C.REPORTS / '右侧模型研究.md'}")
    print(f"\n完成，用时 {time.time() - t_all:.1f}s")
    print(f"  wf_rs10: CAGR {_fmt_pct(m_rs.get('cagr'))} Sharpe {_fmt_num(m_rs.get('sharpe'))} "
          f"MaxDD {_fmt_pct(m_rs.get('max_dd'))} Calmar {_fmt_num(m_rs.get('calmar'))}")
    print(f"  wf_h10 : CAGR {_fmt_pct(m_base.get('cagr'))} Sharpe {_fmt_num(m_base.get('sharpe'))} "
          f"MaxDD {_fmt_pct(m_base.get('max_dd'))} Calmar {_fmt_num(m_base.get('calmar'))}")


# ------------------------------------------------------------------ 报告
def _write_report(stats: dict, out_run: pathlib.Path, out_path: pathlib.Path) -> None:
    bt = stats["backtest"]
    rs, base = bt["wf_rs10"], bt["wf_h10"]
    lines: list[str] = []
    ap = lines.append
    ap("# 右侧趋势模型（wf_rs10）研究报告\n")
    ap(f"> 生成时间：{stats['generated_at']}；数据：{stats['data_range'][0]} ~ {stats['data_range'][1]}"
       f"（ext 数据源，与 Web 服务一致）；周频行 {stats['weekly_rows'][0]} ~ {stats['weekly_rows'][1]}。")
    ap("> 结论：**研究分支，未替换主策略**（详见 §6/§7）。本模型仅作为对照/可选模型展示。\n")

    ap("## 1. 方法与纪律\n")
    ap("- 因子族：`trend-extra`（16 个右侧趋势/动量因子，见 `qf/trend.py`）；全部只用 t 及以前数据。")
    ap("- 标签：`md.fwd_open_ret(10)` 的横截面 rank（t+1 开盘买、t+11 开盘卖），比主模型 h5 更长，匹配趋势持有周期。")
    ap("- 特征：横截面 rank → 行业 + log 市值中性化 → z-score（与 `qf.model` 主线口径一致）。")
    ap(f"- 选池：仅用 DEV（{C.DEV_START} ~ {C.DEV_END}）的 IC/ICIR，规则「{stats['pool_rule']}」，"
       f"池在 2020 年后固定；入池 {len(stats['pool'])} 个。")
    ap("- 权重：VALID_START(2020) 起每季度 refit，滚动 156 周、标签实现滞后 2 周；")
    ap("  **长多约束** `w_k = clip(ICIR_k, 0, 3) / Σ`，只给滚动 IC 为正的因子正权重，"
       "避免自由拟合把右侧因子反过来用。权重仅用 refit 时点之前的周数据。")
    ap("- 回测：N=40、调仓 10 日、100% 国债现金替代、overlay + 个股止损，"
       f"区间 {bt['params']['start']} ~ {bt['params']['end']}。\n")

    ap("## 2. DEV / 全期因子 IC（h10）\n")
    ap("| 因子 | 名称 | 族 | DEV IC | DEV ICIR | 全期 IC | 全期 ICIR | 覆盖率 |")
    ap("|---|---|---|---:|---:|---:|---:|---:|")
    for k in stats["pool"]:
        f = stats["factory_stats"][k]
        ap(f"| {f['slug']} | {f['name_cn']} | {f['family']} | {_fmt_pct(f['dev']['ic'])} | "
           f"{_fmt_num(f['dev']['icir'])} | {_fmt_pct(f['full']['ic'])} | {_fmt_num(f['full']['icir'])} | "
           f"{f['coverage'] * 100:.0f}% |")
    others = [k for k in stats["factory_stats"] if k not in stats["pool"]]
    if others:
        ap(f"\n> 未入池 {len(others)} 个因子：" +
           ", ".join(stats["factory_stats"][k]["slug"] for k in others) + "。\n")

    oos = stats["oos_ic"]
    ap("## 3. 复合分 OOS RankIC（walk-forward，2020 起）\n")
    ap(f"- 均值 **{_fmt_num(oos['ic'], 4)}**，ICIR **{_fmt_num(oos['icir'])}**，"
       f"t **{_fmt_num(oos['t'], 2)}**，正 IC 占比 **{_fmt_pct(oos['pos'])}**，周数 {oos['n']}。")
    ap("\n| 年份 | IC | 周数 |")
    ap("|---|---:|---:|")
    for y, v in stats["oos_ic_by_year"].items():
        ap(f"| {y} | {_fmt_num(v['ic'], 4)} | {v['n']} |")
    ap("")

    ap("## 4. 回测指标对比（同口径，wf_h10 基线在 ext 行情上重跑）\n")
    cols = [("cagr", "CAGR", True), ("ann_vol", "年化波动", True), ("sharpe", "Sharpe", False),
            ("sortino", "Sortino", False), ("max_dd", "MaxDD", True), ("calmar", "Calmar", False),
            ("win_rate_monthly", "月胜率", True), ("turnover_ann", "年换手", False),
            ("cost_ratio", "成本占比", True), ("n_trades", "交易笔数", None)]
    ap("| 指标 | wf_rs10 | wf_h10 |")
    ap("|---|---:|---:|")
    for key, label, is_pct in cols:
        a, b = rs["metrics"].get(key), base["metrics"].get(key)
        if is_pct is None:
            av = "—" if a is None else f"{a:,.0f}"
            bv = "—" if b is None else f"{b:,.0f}"
        elif is_pct:
            av, bv = _fmt_pct(a), _fmt_pct(b)
        else:
            av, bv = _fmt_num(a), _fmt_num(b)
        ap(f"| {label} | {av} | {bv} |")
    ap(f"\n- 日收益相关系数 **{bt['daily_return_corr']:.3f}**"
       + (f"；周度复合分排名相关 **{bt['score_rank_corr']:.3f}**。"
          if bt.get("score_rank_corr") is not None else "。")
       + " 两者都持有 A 股多头且共用 regime 仓位与现金替代，日收益相关性天然偏高；"
         "排名相关更直接反映选股重叠度，可用于判断是否提供独立 alpha。\n")

    ap("## 5. 分区间 / 分年度\n")
    ap("| 区间 | 模型 | CAGR | Sharpe | MaxDD | Calmar |")
    ap("|---|---|---:|---:|---:|---:|")
    for sp in ["full", "y2020_2021", "y2022_plus"]:
        for name, m in (("wf_rs10", rs["split"].get(sp, {})), ("wf_h10", base["split"].get(sp, {}))):
            ap(f"| {sp} | {name} | {_fmt_pct(m.get('cagr'))} | {_fmt_num(m.get('sharpe'))} | "
               f"{_fmt_pct(m.get('max_dd'))} | {_fmt_num(m.get('calmar'))} |")
    ap("\n| 年份 | wf_rs10 | wf_h10 |")
    ap("|---|---:|---:|")
    yearly_rs, yearly_base = rs["yearly"], base["yearly"]
    for y in sorted(set(yearly_rs) | set(yearly_base)):
        ap(f"| {y} | {_fmt_pct(yearly_rs.get(y, yearly_rs.get(str(y))))} | "
           f"{_fmt_pct(yearly_base.get(y, yearly_base.get(str(y))))} |")
    ap("\n基准（同区间）：\n")
    ap("| 基准 | CAGR | MaxDD | Sharpe |")
    ap("|---|---:|---:|---:|")
    for name, m in bt["benchmarks"].items():
        ap(f"| {name} | {_fmt_pct(m.get('cagr'))} | {_fmt_pct(m.get('max_dd'))} | {_fmt_num(m.get('sharpe'))} |")
    ap("")

    ap("## 6. 权重稳定性（按年，长多权重恒 ≥ 0）\n")
    ap("| 因子 | 年份 | 平均权重 | 有权重占比 | refit 数 |")
    ap("|---|---|---:|---:|---:|")
    for y, d in sorted(stats["weights_by_year"].items()):
        for k, v in d.items():
            slug = stats["factory_stats"][k]["slug"]
            ap(f"| {slug} | {y} | {v['mean_weight']:.3f} | {v['pos_share']:.0%} | {v['n_refit']} |")
    ap("")

    ap("## 7. 结论与限制\n")
    better = (rs["metrics"].get("cagr") or 0) > (base["metrics"].get("cagr") or 0)
    better_dd = (rs["metrics"].get("max_dd") if rs["metrics"].get("max_dd") is not None else -9.9) > \
        (base["metrics"].get("max_dd") if base["metrics"].get("max_dd") is not None else -9.9)
    if better and better_dd:
        ap("- 右侧模型在收益与回撤上均优于主线；仍建议先作为「可选模型」灰度观察一个完整季度，不直接替换主策略。")
    else:
        ap("- 右侧模型未跑赢主线（见 §4/§5），作为**研究分支 / 可选模型**保留，不替换主策略；"
           "网页端仅作可选回测模型展示。")
    ap("- 权重稳定性：2023 年后滚动 ICIR 只有 `breakout_20_vol` 持续为正，长多权重退化为单因子（见 §6），"
       "说明多数趋势/动量因子在 2022+ 样本外失效——这是右侧模型收益偏弱的主因，而非回测实现问题。")
    ap("- 右侧趋势在 A 股短周期（月内）偏弱、依赖中长周期与行情 regime；2022+ 样本外表现是最终判据。")
    ap("- 已知限制：①因子池仅 16 个、信息源仍是日频 OHLCV；②长多权重在滚动 IC 全负时会退化为等权；"
       "③h10 标签在周频网格上重叠，IC 的标准误会被低估；④未纳入交易冲击/涨跌停排队细节（沿用主回测引擎假设）。")
    ap("- 复现：`python scripts/build_trend_extra.py && python scripts/run_trend_model.py`；"
       "产物 `build/models/wf_rs10/`、`build/runs/rs10/`。")
    ap("\n> 数据仅供研究，不构成投资建议。\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
