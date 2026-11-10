# -*- coding: utf-8 -*-
"""重建右侧模型 v2.1：修正 v2 的"伪趋势"问题（600936 类下跌股被误选）。

v2 的问题：
  1) directional-alpha 的 efficiency 因子是**方向无关**的（平滑下跌也会得高分）；
  2) 趋势组用 5 日标签 DEV IC 筛选，部分因子在 h10 标签下 DEV IC 为负却仍等权纳入；
  3) z/rank 变换的 trend_strength 是**时间序列标准化**，会把"跌势放缓"识别为趋势转强。
结果：像 600936（60 日 -18.9%、低于 MA60/MA200）这种下跌股被当成"趋势质量+回调"选中。

v2.1：
  - 趋势组只保留明确向上的因子：上轨突破（breakout）、区间位置（range_position），
    且必须在 h10 DEV 标签下 DEV IC>0；
  - 新增真实右侧选股域过滤：收盘 > MA60（可选 MA60 较 20 日前上行），不满足不参与排名；
  - 保留 Alpha158 向上语义因子组变体（ROC/MA dev/RSV/QTLU/BETA），检验增量；
  - 回测口径不变（N=40、10 日调仓、80/20 现金替代、overlay+止损）。
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, factors as qff, model as qfm, backtest as qfbt, metrics as qfm_metrics

HORIZON = 10
SMOOTH_ALPHA = 0.6


def select_factors(screen: pd.DataFrame) -> tuple[list[str], list[str]]:
    d = screen[screen.lib == "directional-alpha"].copy()
    for col in ("base", "category_en", "transform"):
        d[col] = d[col].astype(str)
    trend = d[d.base.isin(["breakout", "range_position"]) & (d.dev_ic_5d > 0) & (d.coverage >= 0.9)] \
        .sort_values("dev_icir_5d", ascending=False).groupby("base", group_keys=False).head(2)
    pull = d[d.base.isin(["breakdown", "reversal"]) & (d.dev_ic_5d > 0) & (d.coverage >= 0.9)] \
        .sort_values("dev_icir_5d", ascending=False).groupby("base", group_keys=False).head(3)
    return list(trend.factor_key), list(pull.factor_key)


def features(panels, mask, industry, logmcap):
    rows = np.arange(len(mask))
    out = {}
    for k, p in panels.items():
        r = qfm.cs_rank_rows(np.asarray(p, dtype=np.float32), rows, mask=mask)
        r = qfm.neutralize_rows(r, industry, logmcap, mask)
        out[k] = qfm.zscore_rows(r, mask)
    return out


def build_scores(md, feats, weights, mask, industry, logmcap, universe, rows, alpha=SMOOTH_ALPHA):
    keys = list(weights)
    w = np.array([weights[k] for k in keys], dtype=np.float64)
    w = w / max(np.abs(w).sum(), 1e-12)
    F = np.stack([feats[k] for k in keys], axis=0)
    _, n, N = F.shape
    out = np.full((n, N), np.nan, dtype=np.float32)
    prev = None
    for wi in range(n):
        row = (w[:, None] * F[:, wi, :]).sum(axis=0).astype(np.float32)[None, :]
        mm = mask[wi][None, :]
        if np.isfinite(row[mm]).sum() > 20:
            rr = qfm.neutralize_rows(row, industry[wi][None, :], logmcap[wi][None, :], mm)
            zz = qfm.zscore_rows(rr, mm)[0]
        else:
            zz = np.full(N, np.nan, dtype=np.float32)
        if alpha < 1.0 and prev is not None:
            both = np.isfinite(zz) & np.isfinite(prev)
            zz = np.where(both, alpha * zz + (1.0 - alpha) * prev, zz).astype(np.float32)
        prev = zz.copy()
        out[wi] = zz
    if universe:
        out = qfm.apply_universe_filter(md, out, rows, universe)
    return out


def main():
    t0 = time.time()
    out_run = C.BUILD_RUNS / "rs2"
    out_run.mkdir(parents=True, exist_ok=True)

    print("[1/6] 行情 + 周频网格 ...")
    md = qfdata.load_market(verbose=False)
    T = md.T
    rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
    mask = md.eligible[rows]
    industry = np.asarray(md.industry[rows])
    logmcap = np.asarray(md.logmcap[rows])
    n_weeks = len(rows)

    print("[2/6] 选因子（明确向上的右侧语义）...")
    screen = pd.read_parquet(C.BUILD_EVAL / "factor_metrics_screen.parquet")
    cat_all = qff.load_catalog().set_index("factor_key")
    trend_keys, pull_keys = select_factors(screen)
    inp = qff.FactorInputs(md)
    panels = {}
    for k in trend_keys + pull_keys:
        panels[k] = np.asarray(qff.compute_factor(inp, cat_all.loc[k].to_dict())[rows], dtype=np.float32)
    print("  趋势(向上):", [(k, cat_all.loc[k, "name_cn"]) for k in trend_keys])
    print("  回调:", [(k, cat_all.loc[k, "name_cn"]) for k in pull_keys])

    alpha_npz = C.BUILD_FACTORS / "alpha_rs.npz"
    alpha_cat = C.BUILD_FACTORS / "alpha_rs_catalog.parquet"
    alpha_keys: list[str] = []
    if alpha_npz.exists() and alpha_cat.exists():
        za = np.load(alpha_npz)
        ac = pd.read_parquet(alpha_cat).set_index("factor_key")
        up_slugs = {"roc20", "roc60", "ma5_dev", "ma20_dev", "ma60_dev", "rsv20", "qtlu20", "beta20"}
        for k in za.files:
            if za[k].shape[0] == n_weeks and str(ac.loc[k, "slug"]) in up_slugs:
                panels[k] = np.asarray(za[k], dtype=np.float32)
                alpha_keys.append(k)
        print("  Alpha158 向上语义:", [(k, ac.loc[k, "name_cn"]) for k in alpha_keys])

    print("[3/6] 特征 + h10 标签 + h10 DEV IC 复核 ...")
    label = qfm.weekly_label(md, rows, mask, horizon=HORIZON, rank=True)
    feats = features(panels, mask, industry, logmcap)
    dev = np.asarray(pd.to_datetime(md.dates[rows]) <= pd.Timestamp(C.DEV_END))
    stats = {}
    for k in feats:
        ic = qfm.weekly_ic(feats[k], label, mask)
        v = np.isfinite(ic)
        icd = ic[v & dev]
        stats[k] = {"dev_ic": float(np.mean(icd)) if len(icd) else np.nan,
                    "dev_icir": float(np.mean(icd) / max(np.std(icd), 1e-9)) if len(icd) else np.nan,
                    "full_ic": float(np.mean(ic[v])) if v.any() else np.nan}
    trend_keep = [k for k in trend_keys if stats[k]["dev_ic"] > 0]
    alpha_keep = [k for k in alpha_keys if stats[k]["dev_ic"] > 0]
    print("  h10 DEV IC>0 趋势因子:", [(k, round(stats[k]["dev_ic"], 4)) for k in trend_keep])
    print("  h10 DEV IC>0 Alpha158:", [(k, round(stats[k]["dev_ic"], 4)) for k in alpha_keep])

    print("[4/6] 变体（选股域过滤 × 因子权重）...")
    variants: dict[str, tuple[dict, str | None]] = {}
    for univ, utag in [(None, "none"), ("above_ma60", "ma60"), ("above_ma60_slope", "ma60slope")]:
        variants[f"pull_{utag}"] = ({k: 1.0 for k in pull_keys}, univ)
        if trend_keep:
            w = {k: 0.4 / len(trend_keep) for k in trend_keep}
            w.update({k: 0.6 / len(pull_keys) for k in pull_keys})
            variants[f"mix40_{utag}"] = (w, univ)
        if trend_keep and alpha_keep:
            w = {k: 0.25 / len(trend_keep) for k in trend_keep}
            w.update({k: 0.15 / len(alpha_keep) for k in alpha_keep})
            w.update({k: 0.60 / len(pull_keys) for k in pull_keys})
            variants[f"mixalpha_{utag}"] = (w, univ)
        if univ == "above_ma60_slope" and trend_keep:
            variants["trend_ma60slope"] = ({k: 1.0 / len(trend_keep) for k in trend_keep}, univ)

    print("[5/6] 回测 ...")
    phase = int(rows[0]) % 5
    start = max(int(rows[0]), md.date_index(C.VALID_START))
    while start % 5 != phase:
        start += 1
    end = T - 1
    cash_ret = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
    results = {}
    for name, (wdict, univ) in variants.items():
        sc_weekly = build_scores(md, feats, wdict, mask, industry, logmcap, univ, rows)
        sc = np.full((T, md.N), np.nan, dtype=np.float32)
        sc[rows] = sc_weekly
        ic = qfm.weekly_ic(sc[rows], label, mask)
        ic = ic[np.isfinite(ic)]
        ic_mean = float(np.mean(ic)) if len(ic) else np.nan
        ic_ir = ic_mean / max(float(np.std(ic)), 1e-9) if len(ic) else np.nan
        old = C.N_HOLDINGS
        C.N_HOLDINGS = 40
        try:
            r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                                  rebalance_days=10, rebalance_offset=(phase - start) % 10,
                                  cash_asset_returns=cash_ret, verbose=False)
        finally:
            C.N_HOLDINGS = old
        nav = r.nav
        row = {"variant": name, "universe": str(univ), "n_factors": len(wdict),
               "ic": ic_mean, "icir": ic_ir}
        for pn, (s, e) in {"full": (C.VALID_START, "2026-12-31"),
                           "val": ("2020-01-01", "2021-12-31"),
                           "oos": ("2022-01-01", "2026-12-31")}.items():
            sub = nav[(nav.index >= s) & (nav.index <= e)]
            p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
            for k in ("cagr", "sharpe", "max_dd", "calmar"):
                row[f"{pn}_{k}"] = p.get(k)
        row["turnover"] = r.metrics.get("turnover_ann")
        row["avg_holdings"] = r.metrics.get("avg_holdings")
        results[name] = {"row": row, "scores": sc, "weights": wdict, "universe": univ}
        print(f"    [{name:18s}] IC={ic_mean:+.4f} full={row['full_cagr']:7.2%} "
              f"MDD={row['full_max_dd']:7.2%} Cal={row['full_calmar']:5.2f} | "
              f"val={row['val_cagr']:7.2%} Cal={row['val_calmar']:5.2f} | "
              f"oos={row['oos_cagr']:7.2%} MDD={row['oos_max_dd']:7.2%} Cal={row['oos_calmar']:5.2f}")

    df = pd.DataFrame([v["row"] for v in results.values()])
    df.to_csv(out_run / "rs2_v21_compare.csv", index=False, encoding="utf-8-sig")

    print("[6/6] 选型 + 旧模型对照 ...")
    cand = [name for name, res in results.items()
            if res["universe"] in ("above_ma60", "above_ma60_slope")
            and (res["row"].get("val_calmar") or -9) >= 0.5
            and (res["row"].get("val_cagr") or -1) > 0
            and (res["row"].get("full_calmar") or -9) >= 0.4
            and (res["row"].get("full_max_dd") or -1) >= -0.13]
    if not cand:
        cand = [name for name, res in results.items() if res["universe"]]
    chosen = max(cand, key=lambda n: results[n]["row"].get("val_calmar") or -9) if cand else None
    if chosen is None:
        chosen = max(results.items(), key=lambda kv: kv[1]["row"].get("val_calmar") or -9)[0]
    print(f"    选中变体: {chosen}（universe={results[chosen]['universe']}）")

    base_dir = C.BUILD_MODELS / ("wf_rs10_old" if (C.BUILD_MODELS / "wf_rs10_old").exists()
                                 else "wf_rs10")
    r10 = qfm.load_result(base_dir)
    sc10 = np.asarray(r10.scores)
    sc10 = np.vstack([sc10, np.full((T - sc10.shape[0], md.N), np.nan, np.float32)]) if sc10.shape[0] < T else sc10[:T]
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        rb = qfbt.run_backtest(md, sc10, start, end, risk_overlay=True, stop_loss=True,
                               rebalance_days=10, rebalance_offset=(phase - start) % 10,
                               cash_asset_returns=cash_ret, verbose=False)
    finally:
        C.N_HOLDINGS = old
    rowb = {"variant": base_dir.name}
    for pn, (s, e) in {"full": (C.VALID_START, "2026-12-31"),
                       "val": ("2020-01-01", "2021-12-31"),
                       "oos": ("2022-01-01", "2026-12-31")}.items():
        sub = rb.nav[(rb.nav.index >= s) & (rb.nav.index <= e)]
        p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        for k in ("cagr", "sharpe", "max_dd", "calmar"):
            rowb[f"{pn}_{k}"] = p.get(k)
    df = pd.concat([df, pd.DataFrame([rowb])], ignore_index=True)
    df.to_csv(out_run / "rs2_v21_compare.csv", index=False, encoding="utf-8-sig")
    print(f"    [{base_dir.name:18s}] full={rowb['full_cagr']:7.2%} MDD={rowb['full_max_dd']:7.2%} "
          f"Cal={rowb['full_calmar']:5.2f} | val={rowb['val_cagr']:7.2%} | oos={rowb['oos_cagr']:7.2%}")

    sc = results[chosen]["scores"]
    wdict = results[chosen]["weights"]
    univ = results[chosen]["universe"]
    pool = list(wdict)
    weights_df = pd.DataFrame([{"refit_date": pd.Timestamp(md.dates[rows[-1]]),
                                "factor_key": k, "weight": float(wdict[k])} for k in pool])
    pool_meta = pd.DataFrame([{
        "factor_key": k, "ic": stats[k]["dev_ic"], "icir": stats[k]["dev_icir"],
        "t": np.nan, "pos": np.nan, "n": int(dev.sum()),
        "category": "trend-up" if k in trend_keep + alpha_keep else "pullback",
        "lib": ("directional-alpha" if k in trend_keys + pull_keys
                else "alpha-rs" if k.startswith("alpha-rs:") else "?"),
    } for k in pool])
    comp_ic = qfm.weekly_ic(sc[rows], label, mask)
    ic_df = pd.DataFrame([{"date": d, "ic": float(v), "refit": pd.Timestamp(md.dates[rows[-1]])}
                          for d, v in zip(md.dates[rows], comp_ic) if np.isfinite(v)])
    cfg = qfm.ModelConfig(horizon=HORIZON, method="icir", score_smooth_alpha=SMOOTH_ALPHA,
                          max_factors=len(pool), neutralize=True, universe=univ)
    res = qfm.WalkForwardResult(rows_weekly=rows, scores=sc, weights=weights_df, pool=pool,
                                pool_meta=pool_meta, composite_ic=ic_df, cfg=cfg)
    out_model = C.BUILD_MODELS / "wf_rs10"
    if out_model.exists():
        bak = C.BUILD_MODELS / "wf_rs10_prev"
        if bak.exists():
            shutil.rmtree(bak)
        shutil.copytree(out_model, bak)
        shutil.rmtree(out_model)
        print(f"    旧模型已备份 → {bak.name}")
    qfm.save_result(res, out_model)
    json.dump({"chosen_variant": chosen, "universe": univ,
               "rule": "趋势组=上轨突破/区间位置(h10 DEV IC>0)；回调组=下轨跌破/收益反转；"
                       "选股域=收盘>MA60(±MA60上行)；长多线性 40/60",
               "weights": {k: float(v) for k, v in wdict.items()},
               "stats": {k: stats[k] for k in pool}},
              open(out_model / "rs2_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    j = int(np.flatnonzero(md.codes == "600936")[0])
    t = md.T - 1
    ma60 = np.nanmean(md.close[max(0, t - 59):t + 1, j])
    print(f"\n600936 @ {md.dates[t].date()}: 收盘 {md.close[t, j]:.2f} vs MA60 {ma60:.2f} "
          f"→ {'在域内' if md.close[t, j] > ma60 else '已被选股域过滤'}")
    print(f"→ {out_model}（{chosen}，pool={len(pool)}，用时 {time.time()-t0:.1f}s）")


if __name__ == "__main__":
    main()
