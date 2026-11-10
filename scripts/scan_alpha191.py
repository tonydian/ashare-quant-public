# -*- coding: utf-8 -*-
"""Alpha191 全库扫描：计算 → 评估（IC/换手/分位）→ 相关性聚类 → 报告。

用法::

    python scripts/scan_alpha191.py                 # 全量 180 个
    python scripts/scan_alpha191.py --limit 20      # 先跑前 20 个做冒烟
    python scripts/scan_alpha191.py --no-corr       # 跳过相关性聚类（快）

产物（``build/alpha191/``）::

    metrics.csv    逐因子完整指标（周频 5 日标签，口径同 qf.evaluate）
    decay.csv      1/5/20 日标签下的 Rank IC 均值（信号衰减）
    corr.csv       抽样面板上的秩相关矩阵
    clusters.csv   贪心聚类结果（代表因子、簇大小）
    scan.md        分析报告
    ic_hist.png / top_bar.png
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qf import alpha191 as A          # noqa: E402
from qf import config as C            # noqa: E402
from qf import data as qfdata         # noqa: E402
from qf import evaluate as qfe        # noqa: E402


def _setup_cn_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        try:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return plt
        except Exception:
            continue
    return plt


def _corr_matrix(panels_at_rows: dict[int, np.ndarray]) -> pd.DataFrame:
    """抽样面板上的平均秩相关（逐日 z-score 后拉平求 Pearson）。

    入参为已按「抽样日期 × 固定股票子集」切好的面板。
    """
    ks = sorted(panels_at_rows)
    cols = []
    for k in ks:
        x = np.asarray(panels_at_rows[k]).astype(np.float64)
        ok = np.isfinite(x)
        r = np.where(ok, x, np.nan)
        # 逐日秩 → 逐日 z-score（缺失用 0 即中位）
        order = np.argsort(np.where(ok, x, np.inf), axis=1, kind="stable")
        rr = np.empty_like(x)
        np.put_along_axis(rr, order, np.broadcast_to(np.arange(x.shape[1], dtype=float),
                                                    x.shape), axis=1)
        rr = np.where(ok, rr, np.nan)
        mu = np.nanmean(rr, axis=1, keepdims=True)
        sd = np.nanstd(rr, axis=1, keepdims=True)
        z = np.where(ok, (rr - mu) / np.where(sd > 0, sd, 1.0), 0.0)
        cols.append(z.ravel())
    M = np.vstack(cols)
    M = M - M.mean(axis=1, keepdims=True)
    M /= np.where(M.std(axis=1, keepdims=True) > 0, M.std(axis=1, keepdims=True), 1.0)
    corr = (M @ M.T) / M.shape[1]
    return pd.DataFrame(corr, index=[f"alpha_{k:03d}" for k in ks],
                        columns=[f"alpha_{k:03d}" for k in ks])


def _greedy_clusters(corr: pd.DataFrame, strength: pd.Series, thr: float = 0.7) -> pd.DataFrame:
    """按 |IC| 从强到弱贪心成簇：与簇首 |ρ|≥thr 即归入该簇。"""
    names = list(strength.sort_values(ascending=False).index)
    assigned: dict[str, int] = {}
    reps: list[str] = []
    for n in names:
        if n in assigned:
            continue
        cid = len(reps)
        reps.append(n)
        assigned[n] = cid
        for m in names:
            if m in assigned:
                continue
            if abs(corr.loc[n, m]) >= thr:
                assigned[m] = cid
    df = pd.DataFrame({"alpha": list(assigned), "cluster": [assigned[a] for a in assigned]})
    df["representative"] = df["cluster"].map({i: r for i, r in enumerate(reps)})
    size = df.groupby("cluster").size().rename("cluster_size")
    df = df.merge(size, on="cluster")
    df["abs_ic"] = df["alpha"].map(strength)
    return df.sort_values(["cluster_size", "abs_ic"], ascending=[False, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "build" / "alpha191"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-corr", action="store_true")
    ap.add_argument("--corr-thr", type=float, default=0.7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    print("[1/5] 载入行情 + 构造输入面板 ...", flush=True)
    md = qfdata.load_market(verbose=False)
    ns = A.build_inputs(md, verbose=True)
    print(f"      面板 {md.T}×{md.N}", flush=True)

    rows = qfe.weekly_rows(md)
    labels = {h: md.fwd_open_ret(h) for h in (1, 5, 20)}
    labels_at = {h: np.asarray(labels[h])[rows] for h in labels}
    elig_at = np.asarray(md.eligible)[rows]
    caches = {h: qfe.LabelRankCache(labels_at[h], elig_at) for h in labels}
    print(f"      周频行 {len(rows)} 个（{md.dates[rows[0]].date()} ~ {md.dates[rows[-1]].date()}）", flush=True)

    ks = A.available()
    if args.limit:
        ks = ks[:args.limit]

    # 相关性抽样：每 4 个周频行取 1 个，股票只留覆盖最全的 1000 只（控制内存）
    corr_rows = rows[::4]
    elig_cnt = np.asarray(md.eligible).sum(axis=0)
    keep_cols = np.sort(np.argsort(-elig_cnt)[:1000])

    records, decays, errs = [], [], {}
    sampled: dict[int, np.ndarray] = {}
    print(f"[2/5] 计算并评估 {len(ks)} 个因子 ...", flush=True)
    for i, k in enumerate(ks, 1):
        t0 = time.time()
        try:
            x = A.compute_one(k, ns)
        except Exception as exc:  # noqa: BLE001
            errs[k] = f"{type(exc).__name__}: {exc}"
            print(f"  [{i:3d}/{len(ks)}] alpha_{k:03d} 失败: {errs[k][:60]}", flush=True)
            continue
        try:
            m = qfe.factor_metrics(md, x, labels[5], rows, name=f"alpha_{k:03d}",
                                   label_cache=caches[5])
        except Exception as exc:  # noqa: BLE001
            errs[k] = f"metrics: {type(exc).__name__}: {exc}"
            print(f"  [{i:3d}/{len(ks)}] alpha_{k:03d} 评估失败: {errs[k][:60]}", flush=True)
            continue
        x_at = x[rows]
        for h in (1, 20):
            ic = qfe.rank_ic(x_at, labels_at[h], elig_at, None, label_cache=caches[h])
            m[f"ic_{h}d"] = float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan
        m["secs"] = round(time.time() - t0, 2)
        records.append(m)

        if not args.no_corr:
            sampled[k] = x[corr_rows][:, keep_cols].astype(np.float32)
        del x, x_at
        gc.collect()
        print(f"  [{i:3d}/{len(ks)}] alpha_{k:03d} IC={m['ic_5d']:+.4f} "
              f"ICIR={m['icir_5d']:+.2f} t={m['tstat']:+.1f} turn={m['turnover_top20']:.2f} "
              f"cov={m['coverage']:.2f} ({m['secs']}s)", flush=True)

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("没有任何因子计算成功")
    df = df.sort_values("ic_5d", key=lambda s: s.abs(), ascending=False)
    df.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    print(f"[3/5] 指标已写出：{out / 'metrics.csv'}", flush=True)

    # ------------------------------------------------ 相关性 + 聚类
    clusters = None
    corr = None
    if not args.no_corr and len(sampled) > 5:
        t0 = time.time()
        print(f"[4/5] 相关性聚类（{len(sampled)} 个因子，抽样行 {len(next(iter(sampled.values())))}）...",
              flush=True)
        corr = _corr_matrix(sampled)
        corr.to_csv(out / "corr.csv", encoding="utf-8-sig")
        strength = df.set_index("name")["ic_5d"].abs()
        clusters = _greedy_clusters(corr, strength, thr=args.corr_thr)
        clusters.to_csv(out / "clusters.csv", index=False, encoding="utf-8-sig")
        print(f"      {len(strength)} 个因子 → {clusters['cluster'].nunique()} 个簇 "
              f"({time.time()-t0:.1f}s)", flush=True)
    else:
        print("[4/5] 跳过相关性聚类", flush=True)

    # ------------------------------------------------ 图
    plt = _setup_cn_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ic = df["ic_5d"].dropna()
    axes[0].hist(ic, bins=40, color="#4C78A8", alpha=0.85)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_title(f"Alpha191 周频 Rank IC 分布（n={len(ic)}）")
    axes[0].set_xlabel("IC（5 日标签）")
    axes[0].set_ylabel("因子数")

    top = df.head(20).iloc[::-1]
    axes[1].barh(top["name"], top["ic_5d"], color=np.where(top["ic_5d"] > 0, "#C44E52", "#55A868"))
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title("|IC| 最大的 20 个因子")
    axes[1].set_xlabel("IC（红=正向，绿=反向）")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "ic_hist.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------ 报告
    write_report(out, df, clusters, corr, errs, md, rows, args)
    print(f"[5/5] 完成，用时 {(time.time()-t_all)/60:.1f} 分钟 → {out}", flush=True)


def write_report(out: Path, df: pd.DataFrame, clusters, corr, errs, md, rows, args) -> None:
    n = len(df)
    ic = df["ic_5d"]
    sig = (df["tstat"].abs() >= 2)
    pos = (ic > 0).mean()

    lines = ["# Alpha191 全库扫描报告", ""]
    lines += [f"- 面板：{md.T} 交易日 × {md.N} 只，评估区间 "
              f"{md.dates[rows[0]].date()} ~ {md.dates[rows[-1]].date()}（周频 {len(rows)} 期）",
              f"- 标签：t+1 开盘买入、持有 5 日（`md.fwd_open_ret(5)`），与主模型口径一致",
              f"- 可实现因子：**{n}** 个；排除 {len(A.EXCLUDED)} 个（见文末）",
              ""]

    lines += ["## 1. 整体表现", "",
              "| 指标 | 数值 |", "|---|---|",
              f"| IC 均值（截面平均） | {ic.mean():+.4f} |",
              f"| IC 中位数 | {ic.median():+.4f} |",
              f"| \\|IC\\| ≥ 0.02 的因子数 | {(ic.abs() >= 0.02).sum()} 个（{(ic.abs()>=0.02).mean():.0%}） |",
              f"| \\|IC\\| ≥ 0.03 的因子数 | {(ic.abs() >= 0.03).sum()} 个（{(ic.abs()>=0.03).mean():.0%}） |",
              f"| \\|t\\| ≥ 2 的因子数 | {sig.sum()} 个（{sig.mean():.0%}） |",
              f"| IC 为正的因子占比 | {pos:.0%} |",
              ""]

    lines += ["## 2. 最强的 20 个因子", "",
              "| 因子 | IC(5d) | ICIR | t | IC(1d) | IC(20d) | 换手 | 覆盖 | 单调性 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.head(20).iterrows():
        lines.append(
            f"| {r['name']} | {r['ic_5d']:+.4f} | {r['icir_5d']:+.2f} | {r['tstat']:+.1f} | "
            f"{r.get('ic_1d', float('nan')):+.4f} | {r.get('ic_20d', float('nan')):+.4f} | "
            f"{r['turnover_top20']:.2f} | {r['coverage']:.2f} | {r['monotonicity']:+.2f} |")
    lines.append("")

    lines += ["## 3. 最弱的 10 个因子", "",
              "| 因子 | IC(5d) | t | 换手 |", "|---|---|---|---|"]
    for _, r in df.tail(10).iterrows():
        lines.append(f"| {r['name']} | {r['ic_5d']:+.4f} | {r['tstat']:+.1f} | {r['turnover_top20']:.2f} |")
    lines.append("")

    if clusters is not None:
        n_cl = clusters["cluster"].nunique()
        big = clusters[clusters["cluster_size"] >= 4].sort_values(
            ["cluster_size", "abs_ic"], ascending=[False, False])
        lines += ["## 4. 相关性聚类（去冗余）", "",
                  f"以 |Spearman ρ| ≥ {args.corr_thr} 作为「同一信号」判据，"
                  f"**{len(clusters)} 个因子塌缩为 {n_cl} 个独立簇**——"
                  f"实际自由度只有约 {n_cl/len(clusters):.0%}。", ""]
        if len(big):
            lines += ["| 簇 | 大小 | 代表因子（\\|IC\\|最强） | 代表 IC |", "|---|---|---|---|"]
            seen = set()
            for _, r in big.iterrows():
                if r["cluster"] in seen:
                    continue
                seen.add(r["cluster"])
                lines.append(f"| {int(r['cluster'])} | {int(r['cluster_size'])} | "
                             f"{r['representative']} | {r['abs_ic']:+.4f} |")
        lines.append("")

    if errs:
        lines += ["## 5. 计算失败", "", "| 因子 | 原因 |", "|---|---|"]
        for k, v in errs.items():
            lines.append(f"| alpha_{k:03d} | {v[:80]} |")
        lines.append("")

    lines += ["## 6. 未实现的因子", "", "| 因子 | 原因 |", "|---|---|"]
    for k, v in sorted(A.EXCLUDED.items()):
        lines.append(f"| alpha_{k:03d} | {v} |")
    lines.append("")

    (out / "scan.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
