# -*- coding: utf-8 -*-
"""候选因子相关性 / 族统计 / 复盘量化输入（factor-review Layer1/2）。

对 ``scripts/build_factors.py`` Stage-2 的 200 个候选因子生成（INTERFACES §2.3）::

    build/eval/factor_corr.npz         候选因子周频 Spearman 相关矩阵 (K×K float32)
    build/eval/family_stats.parquet    按 lib / category_en / base / transform 聚合统计
    build/eval/review_summary.json     n_factors / n_pass / accept_rate / 分布 / top20 /
                                       cluster_representatives 等
    journal/factor_cards/*.json        回填 corr_max_cluster / cluster_id
    journal/INDEX.md                   追加相关性聚类章节
    journal/runs.jsonl                 追加一条实验记录

相关性口径：**逐周横截面 Spearman 相关的 Fisher-z 均值**（pairwise-complete，只统计
当周选股域内两因子均有限的股票；有效股票数 <100 的周跳过）。它衡量"同一周内两个因子
给股票排序的相似度"，是周频换仓组合去重最相关的口径。

用法::

    python scripts/evaluate_factors.py            # 全量
    python scripts/evaluate_factors.py --smoke    # 复用 build/*/smoke 产物
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C          # noqa: E402
from qf import evaluate as qfe      # noqa: E402

MIN_OBS = 100                 # 单周参与相关计算的股票数下限
CORR_CLUSTER = float(C.FACTOR_CORR_MAX)   # 0.75：超过即视为同一相关簇
ICIR_PASS = 0.20              # "通过"的 |ICIR| 阈值（因子卡口径，可调）
COVERAGE_PASS = 0.90          # "通过"的覆盖率阈值
CHUNK_WEEKS = 48              # 分块读周频面板（控制内存 ~400MB）

CAND_FIELDS = ["factor_key", "name_cn", "lib", "category_en", "base", "transform", "window",
               "dev_ic_5d", "dev_icir_5d", "dev_ic_n", "dev_coverage",
               "ic_5d", "icir_5d", "ic_pos_ratio", "coverage", "turnover_top20",
               "ls_mean_5d", "monotonicity"]


# ================================================================ 工具
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _dist(s: pd.Series, qs=(5, 25, 50, 75, 95)) -> dict:
    v = np.asarray(s, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {}
    return {f"p{q:02d}": float(np.percentile(v, q)) for q in qs} | {
        "mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}


def _resolve_paths(smoke: bool) -> dict:
    if smoke:
        eval_dir = C.BUILD_EVAL / "smoke"
        fac_dir = C.BUILD_FACTORS / "smoke"
        journal_dir = C.JOURNAL / "smoke"
    else:
        eval_dir = C.BUILD_EVAL
        fac_dir = C.BUILD_FACTORS
        journal_dir = C.JOURNAL
    cards_dir = journal_dir / "factor_cards"
    for p in (eval_dir, fac_dir, cards_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "eval_dir": eval_dir, "fac_dir": fac_dir, "journal_dir": journal_dir,
        "cards_dir": cards_dir,
        "metrics": eval_dir / "factor_metrics.parquet",
        "screen": eval_dir / "factor_metrics_screen.parquet",
        "weekly_top": fac_dir / "weekly_top.npz",
        "weekly_meta": fac_dir / "weekly_meta.npz",
        "catalog": fac_dir / "catalog.parquet",
        "corr": eval_dir / "factor_corr.npz",
        "family": eval_dir / "family_stats.parquet",
        "summary": eval_dir / "review_summary.json",
        "index": journal_dir / "INDEX.md",
        "runs": journal_dir / "runs.jsonl",
    }


# ================================================================ 相关性矩阵
def compute_corr_matrix(weekly_path: Path, keys: list[str], eligible: np.ndarray,
                        n_cols: int, chunk_weeks: int = CHUNK_WEEKS,
                        verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """逐周横截面 Spearman 相关（pairwise-complete）的 Fisher-z 均值 → (corr, n_obs)。"""
    K = len(keys)
    n_w = eligible.shape[0]
    zsum = np.zeros((K, K), dtype=np.float64)
    cnt = np.zeros((K, K), dtype=np.float64)
    zf = zipfile.ZipFile(weekly_path)
    t0 = time.time()
    n_chunks = (n_w + chunk_weeks - 1) // chunk_weeks
    try:
        for ci, s in enumerate(range(0, n_w, chunk_weeks)):
            e = min(n_w, s + chunk_weeks)
            wlen = e - s
            vals = np.empty((K, wlen, n_cols), dtype=np.float32)
            for i, k in enumerate(keys):
                a = np.load(io.BytesIO(zf.read(k + ".npy")))
                vals[i] = a[s:e]
                del a
            ranks = np.empty_like(vals)
            for i in range(K):
                ranks[i] = qfe.row_rank(vals[i], eligible[s:e])
            del vals
            for wi in range(wlen):
                act = np.flatnonzero(eligible[s + wi])
                if len(act) < MIN_OBS:
                    continue
                X = ranks[:, wi, :][:, act].astype(np.float64)   # (K,n)
                M = np.isfinite(X)
                Xz = np.where(M, X, 0.0)
                Mf = M.astype(np.float64)
                n_ij = Mf @ Mf.T
                sxy = Xz @ Xz.T
                sx = Xz @ Mf.T                    # sx[i,j] = Σ X_i·M_j = 交集内 sum X_i
                sxx = (Xz * Xz) @ Mf.T
                with np.errstate(invalid="ignore", divide="ignore"):
                    n = np.maximum(n_ij, 1.0)
                    cov = sxy - (sx * sx.T) / n
                    vx = sxx - (sx * sx) / n
                    vy = vx.T
                    r = cov / np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
                ok = (n_ij >= MIN_OBS) & (vx > 1e-12) & (vy > 1e-12) & np.isfinite(r)
                rz = np.arctanh(np.clip(r, -0.999, 0.999))
                zsum += np.where(ok, rz, 0.0)
                cnt += ok
            if verbose:
                print(f"[corr] chunk {ci+1:2d}/{n_chunks} weeks {s}-{e-1} "
                      f"| elapsed {time.time()-t0:.0f}s", flush=True)
    finally:
        zf.close()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.tanh(zsum / np.maximum(cnt, 1.0))
    corr = np.where(cnt > 0, corr, 0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32), cnt


# ================================================================ 聚类
def greedy_clusters(keys: list[str], corr: np.ndarray, order: list[str],
                    threshold: float = CORR_CLUSTER) -> tuple[np.ndarray, list[int]]:
    """贪心相关簇：按给定优先级依次成为簇代表；未分配因子若与某代表 |corr|>=阈值则归入该簇。"""
    kpos = {k: i for i, k in enumerate(keys)}
    cid = np.full(len(keys), -1, dtype=np.int32)
    reps: list[int] = []
    for k in order:
        i = kpos[k]
        if cid[i] >= 0:
            continue
        c = len(reps)
        reps.append(i)
        cid[i] = c
        row = np.abs(corr[i])
        assign = (cid == -1) & (row >= threshold)
        cid[assign] = c
    return cid, reps


# ================================================================ 族统计
def build_family_stats(metrics: pd.DataFrame, screen: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    specs = [("lib", "lib"), ("category_en", "category_en"),
             ("base", "base"), ("transform", "transform")]
    for ftype, col in specs:
        if col not in metrics.columns:
            continue
        for val, g in metrics.groupby(col, dropna=False):
            dev_abs = g["dev_icir_5d"].abs()
            best = g.loc[dev_abs.idxmax()]
            n_lib = 0
            if screen is not None and col in screen.columns:
                n_lib = int((screen[col] == val).sum())
            rows.append({
                "family_type": ftype,
                "family": str(val),
                "n_candidates": int(len(g)),
                "n_factors_library": n_lib,
                "ic_mean": float(g["ic_5d"].mean()),
                "ic_median": float(g["ic_5d"].median()),
                "icir_mean": float(g["icir_5d"].mean()),
                "icir_median": float(g["icir_5d"].median()),
                "abs_icir_mean": float(g["icir_5d"].abs().mean()),
                "dev_ic_mean": float(g["dev_ic_5d"].mean()),
                "dev_icir_mean": float(g["dev_icir_5d"].mean()),
                "dev_icir_median": float(g["dev_icir_5d"].median()),
                "dev_abs_icir_mean": float(dev_abs.mean()),
                "coverage_mean": float(g["coverage"].mean()),
                "dev_coverage_mean": float(g["dev_coverage"].mean()),
                "ic_pos_ratio_mean": float(g["ic_pos_ratio"].mean()),
                "turnover_mean": float(g["turnover_top20"].mean()),
                "ls_mean": float(g["ls_mean_5d"].mean()),
                "best_factor": str(best["factor_key"]),
                "best_name_cn": str(best.get("name_cn", "")),
                "best_dev_ic": float(best["dev_ic_5d"]),
                "best_dev_icir": float(best["dev_icir_5d"]),
                "best_full_icir": float(best["icir_5d"]),
            })
    return pd.DataFrame(rows)


# ================================================================ review_summary
def build_review_summary(metrics: pd.DataFrame, screen: pd.DataFrame | None,
                         corr: np.ndarray, cnt: np.ndarray, cid: np.ndarray,
                         reps: list[int], meta: dict, paths: dict,
                         n_weeks_valid: int) -> dict:
    keys = metrics["factor_key"].tolist()
    m = metrics.set_index("factor_key")
    corr_abs = np.abs(corr).astype(np.float64).copy()
    np.fill_diagonal(corr_abs, np.nan)

    def pass_count(df: pd.DataFrame, icir_col: str, cov_col: str) -> int:
        return int(((df[icir_col].abs() > ICIR_PASS) & (df[cov_col] > COVERAGE_PASS)).sum())

    summary: dict = {
        "generated_at": _now(),
        "n_factors": int(len(screen)) if screen is not None else int(len(metrics)),
        "n_candidates": int(len(metrics)),
        "accept_rate": float(len(metrics) / len(screen)) if screen is not None and len(screen) else None,
        "pass_thresholds": {"abs_icir": ICIR_PASS, "coverage": COVERAGE_PASS,
                            "min_obs_weeks": qfe.MIN_IC_STOCKS},
        "selection_rule": ("Stage-2 候选只用 DEV 期(2017-01-03~2019-12-31) 的 |dev_icir_5d| + "
                           "dev_coverage 排序，并保证每个 lib / category 至少一个代表；"
                           "全样本指标仅作为报告字段，不参与筛选（防全样本选择偏差）。"),
        "data_window": meta,
        "corr_method": ("average weekly cross-sectional rank correlation (Spearman-type): "
                        "ranks computed per factor within its own weekly valid set "
                        "(eligible & finite factor), pairwise-complete Pearson on the ranks, "
                        "Fisher-z mean across weeks; weeks with <100 valid stocks skipped. "
                        "Note: rank-compression from factor-specific missing cells makes it "
                        "differ from strict Spearman on the intersection by <~0.02."),
        "corr_summary": {
            "threshold_cluster": CORR_CLUSTER,
            "mean_abs_offdiag": float(np.nanmean(corr_abs)),
            "max_abs_offdiag": float(np.nanmax(corr_abs)),
            "n_pairs_abs_ge_0.7": int(np.nansum(corr_abs >= 0.70) // 2),
            "n_pairs_abs_ge_0.5": int(np.nansum(corr_abs >= 0.50) // 2),
            "mean_pair_obs_weeks": float(np.nanmean(cnt)),
        },
        "n_weeks": int(n_weeks_valid),
    }

    if screen is not None:
        summary["n_pass"] = {
            "dev": pass_count(screen, "dev_icir_5d", "dev_coverage"),
            "full": pass_count(screen, "icir_5d", "coverage"),
            "candidates_dev": pass_count(metrics, "dev_icir_5d", "dev_coverage"),
            "candidates_full": pass_count(metrics, "icir_5d", "coverage"),
        }
        summary["pass_rate"] = {
            "dev": summary["n_pass"]["dev"] / max(len(screen), 1),
            "full": summary["n_pass"]["full"] / max(len(screen), 1),
        }
        summary["ic_distribution"] = {"dev": _dist(screen["dev_ic_5d"]),
                                      "full": _dist(screen["ic_5d"])}
        summary["icir_distribution"] = {"dev": _dist(screen["dev_icir_5d"]),
                                        "full": _dist(screen["icir_5d"])}
        summary["coverage_distribution"] = {"dev": _dist(screen["dev_coverage"]),
                                            "full": _dist(screen["coverage"])}
        summary["turnover_distribution"] = {"full": _dist(screen["turnover_top20"]),
                                            "dev": _dist(screen["dev_turnover_top20"])}
        # 族覆盖
        fc = {}
        for col in ("lib", "category_en", "base", "transform"):
            if col not in screen.columns:
                continue
            tot = screen[col].astype(str).value_counts()
            cand = metrics[col].astype(str).value_counts()
            fc[col] = {k: {"n_library": int(v), "n_candidates": int(cand.get(k, 0))}
                       for k, v in tot.items()}
        summary["family_coverage"] = fc
        summary["ic_distribution_candidates"] = {"dev": _dist(metrics["dev_ic_5d"]),
                                                 "full": _dist(metrics["ic_5d"])}
        summary["icir_distribution_candidates"] = {"dev": _dist(metrics["dev_icir_5d"]),
                                                  "full": _dist(metrics["icir_5d"])}

    def pack(df: pd.DataFrame, n: int) -> list[dict]:
        d = df.copy()
        d["dev_abs_icir"] = d["dev_icir_5d"].abs()
        d = d.sort_values("dev_abs_icir", ascending=False).head(n)
        return _jsonable(d[[c for c in CAND_FIELDS if c in d.columns]].to_dict("records"))

    summary["top20"] = pack(metrics, 20)
    summary["bottom20"] = pack(metrics.sort_values("dev_ic_5d"), 20)

    clusters = []
    for c, rep_i in enumerate(reps):
        members = [keys[j] for j in range(len(keys)) if cid[j] == c]
        if not members:
            continue
        sub = m.loc[[k for k in members if k in m.index]]
        best = sub.loc[sub["dev_icir_5d"].abs().idxmax()] if len(sub) else None
        within = [abs(float(corr[rep_i, j])) for j in range(len(keys))
                  if cid[j] == c and j != rep_i]
        clusters.append({
            "cluster_id": int(c),
            "representative": keys[rep_i],
            "representative_name_cn": str(m.loc[keys[rep_i], "name_cn"]) if keys[rep_i] in m.index else "",
            "n_members": len(members),
            "members": members,
            "best_member_by_dev_icir": None if best is None else str(best.name),
            "best_dev_icir": None if best is None else float(best["dev_icir_5d"]),
            "mean_abs_corr_within": float(np.mean(within)) if within else None,
            "max_corr_with_other_representatives": None,
        })
    # 簇代表之间的最大相关
    for cl in clusters:
        i = keys.index(cl["representative"])
        others = [abs(float(corr[i, keys.index(o["representative"])]))
                  for o in clusters if o["cluster_id"] != cl["cluster_id"]]
        cl["max_corr_with_other_representatives"] = float(max(others)) if others else None
    summary["cluster_representatives"] = clusters
    summary["n_clusters"] = len(clusters)
    summary["artifacts"] = [str(paths[k].relative_to(ROOT)) for k in
                            ("corr", "family", "summary", "metrics", "weekly_top")]
    return summary


# ================================================================ 卡片/索引回填
def update_cards(paths: dict, keys: list[str], corr: np.ndarray, cid: np.ndarray) -> int:
    keys = list(keys)
    n_upd = 0
    for i, k in enumerate(keys):
        fn = paths["cards_dir"] / (str(k).replace(":", "_") + ".json")
        if not fn.exists():
            continue
        card = json.loads(fn.read_text(encoding="utf-8"))
        row = np.abs(corr[i]).astype(np.float64)
        row[i] = np.nan
        card["corr_max_cluster"] = None if not np.isfinite(row).any() else float(np.nanmax(row))
        card["cluster_id"] = int(cid[i])
        fn.write_text(json.dumps(_jsonable(card), ensure_ascii=False, indent=1), encoding="utf-8")
        n_upd += 1
    return n_upd


def append_index_section(paths: dict, summary: dict) -> None:
    if not paths["index"].exists():
        return
    txt = paths["index"].read_text(encoding="utf-8")
    marker = "## 5. 相关性聚类（evaluate_factors.py）"
    if marker in txt:
        txt = txt[: txt.index(marker)]
    cs = summary.get("corr_summary", {})
    lines = [marker, ""]
    lines.append(f"- 相关口径：{summary['corr_method']}")
    lines.append(f"- 候选因子数：{summary['n_candidates']}；相关簇数：{summary.get('n_clusters')}"
                 f"（|corr| >= {cs.get('threshold_cluster')} 归为一簇，按 DEV |ICIR| 优先级贪心）")
    lines.append(f"- 平均 |相关|（非对角）：{cs.get('mean_abs_offdiag'):.3f}；"
                 f"最大 |相关|：{cs.get('max_abs_offdiag'):.3f}；"
                 f"|corr|>=0.7 的因子对：{cs.get('n_pairs_abs_ge_0.7')}")
    lines.append("")
    reps = summary.get("cluster_representatives", [])
    if reps:
        tab = pd.DataFrame([{
            "cluster": c["cluster_id"],
            "representative": c["representative"],
            "name": c["representative_name_cn"],
            "n_members": c["n_members"],
            "best_dev_icir": c["best_dev_icir"],
            "mean_abs_corr_within": c["mean_abs_corr_within"],
            "max_corr_other_reps": c["max_corr_with_other_representatives"],
        } for c in reps])
        lines.append(_md_table(tab, ".3f"))
    lines.append("")
    paths["index"].write_text(txt.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


# ================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="候选因子相关性 / 族统计 / 复盘量化输入")
    ap.add_argument("--smoke", action="store_true", help="使用 build/*/smoke 下的产物")
    ap.add_argument("--cluster-threshold", type=float, default=CORR_CLUSTER)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_all = time.time()
    paths = _resolve_paths(args.smoke)
    print("=" * 88)
    print(f"[factor-lab] evaluate_factors start={_now()} smoke={args.smoke}")

    metrics = pd.read_parquet(paths["metrics"])
    keys = metrics["factor_key"].tolist()
    meta = np.load(paths["weekly_meta"], allow_pickle=False)
    dates = pd.DatetimeIndex(meta["dates"])
    rows = meta["rows"]
    eligible = meta["eligible"]
    label = meta["label"]
    n_w, n_cols = eligible.shape
    n_weeks_valid = int((eligible.sum(axis=1) >= MIN_OBS).sum())
    print(f"[load] 候选 {len(keys)} 因子；周频行 {n_w}（{dates[0].date()} ~ {dates[-1].date()}），"
          f"有效周 {n_weeks_valid}；股票数 {n_cols}")

    # ---- 1) 相关矩阵
    corr, cnt = compute_corr_matrix(paths["weekly_top"], keys, eligible, n_cols, verbose=True)
    kstr = max(len(k) for k in keys) if keys else 8
    np.savez(paths["corr"],
             corr=corr,
             factor_keys=np.array(keys, dtype=f"U{kstr}"),
             n_weeks=np.int64(n_w),
             n_weeks_valid=np.int64(n_weeks_valid),
             mean_pair_obs=np.asarray(cnt.mean(), dtype=np.float32),
             method=np.array("average_weekly_cross_sectional_spearman_fisher_z"))
    print(f"[corr] 矩阵 {corr.shape}，平均 |非对角相关| {np.nanmean(np.abs(corr - np.eye(len(keys)))):.3f} "
          f"→ {paths['corr']}")

    # ---- 2) 聚类（按 DEV |ICIR| 优先级）
    order = metrics.assign(_a=metrics["dev_icir_5d"].abs()).sort_values(
        "_a", ascending=False)["factor_key"].tolist()
    cid, reps = greedy_clusters(keys, corr, order, threshold=float(args.cluster_threshold))
    print(f"[cluster] {len(reps)} 个相关簇（阈值 {args.cluster_threshold}）；"
          f"最大簇成员数 {max(np.bincount(cid)) if len(cid) else 0}")

    # ---- 3) 族统计
    screen = pd.read_parquet(paths["screen"]) if paths["screen"].exists() else None
    fam = build_family_stats(metrics, screen)
    fam.to_parquet(paths["family"])
    print(f"[family] {len(fam)} 行 → {paths['family']}")

    # ---- 4) review_summary
    data_meta = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "dev_end": str(C.DEV_END),
        "n_weeks": int(n_w),
        "n_weeks_valid": int(n_weeks_valid),
        "n_stocks": int(n_cols),
        "label": f"fwd_open_ret({C.FWD_HORIZON}) / t+1 开盘 → t+1+{C.FWD_HORIZON} 开盘",
    }
    summary = build_review_summary(metrics, screen, corr, cnt, cid, reps, data_meta,
                                   paths, n_weeks_valid)
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, ensure_ascii=False, indent=1)
    print(f"[summary] → {paths['summary']}  n_pass={summary.get('n_pass')} "
          f"n_clusters={summary.get('n_clusters')}")

    # ---- 5) 回填因子卡 + INDEX
    n_cards = update_cards(paths, keys, corr, cid)
    print(f"[cards] 回填 {n_cards} 张因子卡的 corr_max_cluster / cluster_id")
    append_index_section(paths, summary)
    print(f"[index] 追加相关性聚类章节 → {paths['index']}")

    # ---- 6) journal
    _append_jsonl(paths["runs"], {
        "run_id": f"{_now()}_lab_evaluate_factors",
        "stage": "factor_eval",
        "status": "ok",
        "n_factors": int(len(metrics)),
        "n_candidates": int(len(metrics)),
        "elapsed_sec": round(time.time() - t_all, 1),
        "artifacts": [str(paths[k].relative_to(ROOT)) for k in ("corr", "family", "summary")],
        "note": (f"corr=逐周横截面Spearman Fisher-z均值; n_clusters={summary.get('n_clusters')}; "
                 f"cluster_threshold={args.cluster_threshold}; top_rep="
                 f"{summary['cluster_representatives'][0]['representative'] if reps else None}"),
    })
    print(f"[journal] → {paths['runs']}")
    print(f"[factor-lab] evaluate_factors 完成，用时 {(time.time()-t_all)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
