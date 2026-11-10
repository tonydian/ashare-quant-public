# -*- coding: utf-8 -*-
"""两阶段因子构建：800 因子周频初筛 → Top200 候选细分 + 周频面板/因子卡/索引落盘。

流程（严格按 docs/INTERFACES.md §2.2）::

    1. md = load_market(); inp = FactorInputs(md); cat = load_catalog()
    2. rows_weekly = 2017-01-03 起每 C.REBALANCE_DAYS(5) 个交易日
       label = md.fwd_open_ret(5)（t+1 开盘买 → t+6 开盘卖，只用于评估）
    3. Stage-1: qf.factors.iter_groups 逐 (base,window) 组复用 base 信号，
       800 因子在全部周频行上评估 → build/eval/factor_metrics_screen.parquet
    4. 选候选：**只用开发期 DEV(2017-01-03~2019-12-31) 的 |Rank ICIR| + 覆盖率** 排序
       （防全样本选择偏差），并保证每个 lib / category 至少一个代表 → 200 个；
       全样本指标只作报告字段。
    5. Stage-2: 重放选中组 → 复核指标（全样本 + 分年度）→ factor_metrics.parquet；
       逐因子落盘周频面板 weekly_top.npz（键=factor_key，(n_weeks,N) float32，
       流式 zip 写入，内存只驻留单面板）+ weekly_meta.npz
       （dates / rows / label / eligible）。
    6. 因子卡 journal/factor_cards/<key>.json、journal/INDEX.md、
       journal/runs.jsonl、build/factors/catalog.parquet。

用法::

    python scripts/build_factors.py                 # 全量（Stage-1+2）
    python scripts/build_factors.py --smoke 12      # 12 因子端到端自检（写 *\\smoke\\ 子目录）
    python scripts/build_factors.py --stage1-only
    python scripts/build_factors.py --stage2-only   # 复用已有 screen 结果
    python scripts/build_factors.py --resume        # Stage-1 断点续跑

单进程串行；所有随机过程固定 C.RANDOM_SEED（本脚本无随机过程）。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C          # noqa: E402
from qf import data as qfdata       # noqa: E402
from qf import evaluate as qfe      # noqa: E402
from qf import factors as qff       # noqa: E402

N_CANDIDATES = 200            # Stage-2 候选数
MIN_DEV_WEEKS = 52            # 选候选要求的 DEV 有效周数下限（约 1 年）
MIN_DEV_COVERAGE = 0.50       # 选候选要求的 DEV 覆盖率下限
PROGRESS_EVERY = 10           # 每 N 组落盘进度

META_COLS = ["lib", "factor_id", "name_cn", "name_en", "category_cn",
             "category_en", "base", "transform", "window", "norm"]


# ================================================================ 工具
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _jsonable(obj):
    """递归把 numpy / NaN 转成可 JSON 序列化的 Python 对象。"""
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


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """极简 markdown 表格（不依赖 tabulate）。"""
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


def _resolve_paths(smoke: int | None) -> dict:
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
        "eval_dir": eval_dir,
        "fac_dir": fac_dir,
        "journal_dir": journal_dir,
        "cards_dir": cards_dir,
        "screen": eval_dir / "factor_metrics_screen.parquet",
        "screen_progress": eval_dir / "_screen_progress.parquet",
        "metrics": eval_dir / "factor_metrics.parquet",
        "catalog": fac_dir / "catalog.parquet",
        "weekly_top": fac_dir / "weekly_top.npz",
        "weekly_meta": fac_dir / "weekly_meta.npz",
        "index": journal_dir / "INDEX.md",
        "runs": journal_dir / "runs.jsonl",
    }


# ================================================================ 候选选择（仅 DEV 期）
def select_candidates(screen: pd.DataFrame, n_candidates: int = N_CANDIDATES,
                      ensure_representative: bool = True) -> pd.DataFrame:
    """按**开发期** |dev_icir_5d| 排序选 Top N；保证每个 lib/category 至少一个代表。

    全样本字段只随行返回用于报告，不参与排序。返回按 |dev_icir| 降序排列的候选表。
    """
    df = screen.copy()
    df["dev_abs_icir"] = df["dev_icir_5d"].abs()
    n_target = min(int(n_candidates), len(df))
    selectable = df[(df["dev_ic_n"] >= MIN_DEV_WEEKS) & (df["dev_coverage"] >= MIN_DEV_COVERAGE)]
    if len(selectable) == 0:
        selectable = df
    order = selectable.sort_values("dev_abs_icir", ascending=False)["factor_key"].tolist()
    selected = order[:n_target]

    if ensure_representative:
        # 每个 (lib, category_en) 至少一个代表（若全被淘汰则强行加入）
        idx = df.set_index("factor_key", drop=False)
        group_of = {k: (str(idx.loc[k, "lib"]), str(idx.loc[k, "category_en"])) for k in idx.index}
        for (lib, catg), grp in df.groupby(["lib", "category_en"], sort=False):
            if any(group_of[k] == (str(lib), str(catg)) for k in selected):
                continue
            g = grp[grp["dev_ic_n"] >= MIN_DEV_WEEKS]
            if len(g) == 0:
                g = grp
            rep = g.sort_values("dev_abs_icir", ascending=False)["factor_key"].iloc[0]
            if rep not in selected:
                selected.append(rep)
        # 裁回 n_target：优先删掉"同族已有其他代表"的低分因子
        if len(selected) > n_target:
            counts = Counter(group_of[k] for k in selected)
            for k in sorted(selected, key=lambda kk: idx.loc[kk, "dev_abs_icir"]):
                if len(selected) <= n_target:
                    break
                gk = group_of[k]
                if counts[gk] > 1:
                    selected.remove(k)
                    counts[gk] -= 1
            # 若仍超（每个族只有一个成员且族数 > n_target），只能保留
    out = df.set_index("factor_key").loc[selected].reset_index()
    out = out.sort_values("dev_abs_icir", ascending=False).reset_index(drop=True)
    return out


# ================================================================ Stage-1
def run_stage1(md, cat, inp, rows, label, cat_use, paths, resume: bool,
               verbose: bool = True) -> pd.DataFrame:
    screen_path = paths["screen"]
    prog_path = paths["screen_progress"]
    existing = None
    done_keys: list[str] = []
    if resume and prog_path.exists():
        existing = pd.read_parquet(prog_path)
        done_keys = existing["factor_key"].tolist()
        if verbose:
            print(f"[stage1] resume：已有 {len(done_keys)} 个因子结果 → {prog_path}")
    t0 = time.time()
    df_new = qfe.early_screen(md, cat_use, inp, rows, label, verbose=verbose,
                              progress_path=prog_path, progress_every=PROGRESS_EVERY,
                              done_keys=done_keys)
    if existing is not None and len(df_new):
        df = pd.concat([existing, df_new], ignore_index=True)
    elif existing is not None:
        df = existing
    else:
        df = df_new
    # 按 catalog 顺序排列（可复现），并校验完整性
    idx = df.set_index("factor_key", drop=False)
    order = [k for k in cat_use["factor_key"].tolist() if k in idx.index]
    missing = [k for k in cat_use["factor_key"].tolist() if k not in idx.index]
    df = idx.loc[order].reset_index(drop=True)
    df.to_parquet(screen_path)
    if prog_path.exists():
        prog_path.unlink()
    if verbose:
        print(f"[stage1] 完成 {len(df)}/{len(cat_use)} 因子，用时 {(time.time()-t0)/60:.1f} 分钟 → {screen_path}")
        if missing:
            print(f"[stage1][warn] 缺失 {len(missing)} 个因子：{missing[:5]} ...")
    return df


# ================================================================ Stage-2
def run_stage2(md, cat, inp, rows, label, selected: pd.DataFrame, paths,
               verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """重放选中 (base,window) 组 → 复核指标 + 流式写周频面板 npz。"""
    keys = selected["factor_key"].tolist()
    n_keys = len(keys)
    key_pos = {k: i for i, k in enumerate(keys)}
    meta = selected.set_index("factor_key")
    label_rows = qfe._align_rows(label, rows, md.T, "label")
    mask_w = md.eligible[rows]
    label_cache = qfe.LabelRankCache(label_rows, mask_w)
    cache = qfe.TransformCache(inp)

    results: list[dict | None] = [None] * n_keys
    sel_groups = {tuple(k) for k in
                  meta.reset_index()[["base", "window"]].drop_duplicates().itertuples(index=False)}
    weekly_path = paths["weekly_top"]
    t0 = time.time()
    t_group_sum = 0.0
    n_groups_done = 0
    n_groups_total = len(sel_groups)
    problems: list[str] = []

    zf = zipfile.ZipFile(weekly_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
    try:
        for base, w, g in qff.iter_groups(cat):
            g_sel = g[g["factor_key"].isin(key_pos)]
            if len(g_sel) == 0:
                continue
            tg = time.time()
            p0 = qfe.entry_params(g.iloc[0], window=w)
            sig = qff._base_signal(inp, base, p0)
            for _, r in g_sel.iterrows():
                p = qfe.entry_params(r, window=w)
                x = qfe.transform_at_rows(sig, inp, r["transform"], p, rows, cache)
                # ---- 周频面板落盘（键=factor_key，值 (n_weeks,N) float32）
                buf = io.BytesIO()
                np.save(buf, np.ascontiguousarray(x, dtype=np.float32), allow_pickle=False)
                zf.writestr(r["factor_key"] + ".npy", buf.getvalue())
                del buf
                # ---- 复核指标（全样本 + 分年度 + DEV 子集）
                met = qfe._factor_metrics_aligned(md, x, label_rows, mask_w, rows,
                                                  name=r["factor_key"], label_cache=label_cache)
                for c in META_COLS:
                    met[c] = r[c]
                met["factor_key"] = r["factor_key"]
                results[key_pos[r["factor_key"]]] = met
                del x
            del sig
            dt = time.time() - tg
            t_group_sum += dt
            n_groups_done += 1
            if verbose:
                avg = t_group_sum / n_groups_done
                eta = avg * (n_groups_total - n_groups_done) / 60.0
                print(f"[stage2] {n_groups_done:3d}/{n_groups_total} {base:16s} w={int(w):2d} "
                      f"{len(g_sel)}f {dt:5.1f}s | elapsed {(time.time()-t0)/60:.1f}min "
                      f"ETA {eta:.1f}min", flush=True)
    finally:
        zf.close()

    metrics = pd.DataFrame([r for r in results if r is not None])
    idx = metrics.set_index("factor_key", drop=False)
    metrics = idx.loc[[k for k in keys if k in idx.index]].reset_index(drop=True)
    if len(metrics) != n_keys:
        problems.append(f"Stage-2 只完成 {len(metrics)}/{n_keys} 个因子")
    metrics.to_parquet(paths["metrics"])
    if verbose:
        print(f"[stage2] 完成 {len(metrics)} 因子，用时 {(time.time()-t0)/60:.1f} 分钟 → {paths['metrics']}")
        print(f"[stage2] 周频面板 → {weekly_path} "
              f"({weekly_path.stat().st_size/1e6:.0f} MB)")
    return metrics, problems


def write_weekly_meta(md, rows, label, paths) -> None:
    rows = np.asarray(rows, dtype=np.int64)
    meta = {
        "dates": np.asarray(md.dates[rows].values, dtype="datetime64[ns]"),
        "rows": rows,
        "label": np.ascontiguousarray(label[rows], dtype=np.float32),
        "eligible": np.ascontiguousarray(md.eligible[rows], dtype=bool),
        "horizon": np.int64(C.FWD_HORIZON),
        "rebalance_days": np.int64(C.REBALANCE_DAYS),
        "start_date": np.array(str(md.dates[rows[0]].date())),
        "dev_end": np.array(str(C.DEV_END)),
        "n_stocks": np.int64(md.N),
    }
    np.savez(paths["weekly_meta"], **meta)


# ================================================================ 因子卡 / 索引
CARD_FIELDS = [
    "factor_key", "lib", "factor_id", "slug", "name_cn", "name_en",
    "category_cn", "category_en", "base", "transform", "window", "norm", "params",
    "ic_5d", "ic_std_5d", "icir_5d", "tstat", "ic_pos_ratio", "ic_n", "coverage",
    "ls_mean_5d", "ls_std_5d", "ls_ir_5d", "turnover_top20", "monotonicity",
    "ic_by_year", "q1_5d", "q2_5d", "q3_5d", "q4_5d", "q5_5d",
    "dev_ic_5d", "dev_icir_5d", "dev_tstat", "dev_ic_pos_ratio", "dev_ic_n",
    "dev_coverage", "dev_ls_mean_5d", "dev_ls_ir_5d", "dev_turnover_top20",
    "dev_monotonicity", "dev_q1_5d", "dev_q2_5d", "dev_q3_5d", "dev_q4_5d", "dev_q5_5d",
    "corr_max_cluster", "cluster_id", "created_at",
]


def write_factor_cards(cat: pd.DataFrame, metrics: pd.DataFrame, cards_dir: Path) -> int:
    cat_idx = cat.set_index("factor_key", drop=False)
    n = 0
    now = _now()
    for _, r in metrics.iterrows():
        key = r["factor_key"]
        card = {k: r[k] for k in CARD_FIELDS if k in r.index}
        for c in META_COLS:
            card.setdefault(c, r.get(c))
        # catalog 补充字段
        if key in cat_idx.index:
            for c in ("slug", "params"):
                if c in cat_idx.columns:
                    card[c] = cat_idx.loc[key, c]
        card["corr_max_cluster"] = None      # 由 scripts/evaluate_factors.py 回填
        card["cluster_id"] = None
        card["created_at"] = now
        # INTERFACES §2.5 字段名 `category`（同时保留 category_cn/category_en）
        card["category"] = card.get("category_en")
        fn = str(key).replace(":", "_") + ".json"
        with open(cards_dir / fn, "w", encoding="utf-8") as f:
            json.dump(_jsonable(card), f, ensure_ascii=False, indent=1)
        n += 1
    return n


def write_index_md(md, cat: pd.DataFrame, screen: pd.DataFrame, metrics: pd.DataFrame,
                   paths: dict, extra: dict | None = None) -> None:
    extra = extra or {}
    rows = extra.get("rows")
    n_all = len(screen)
    n_cand = len(metrics)
    lines = []
    lines.append("# 因子库索引（Factor Lab）\n")
    lines.append(f"- 生成时间：{_now()}")
    if rows is not None:
        lines.append(f"- 数据窗口：{md.dates[rows[0]].date()} ~ {md.dates[rows[-1]].date()}"
                     f"（{len(rows)} 个周频行；标签 = fwd_open_ret(5)，t+1 开盘→t+6 开盘）")
    lines.append(f"- 因子总数：**{n_all}**（volume-stat / directional / risk-pattern 三库）")
    lines.append(f"- Stage-2 候选：**{n_cand}**（选择依据：**DEV 期 2017-01-03~2019-12-31** "
                 f"|Rank ICIR| + 覆盖率，保证每 lib / category 至少一个代表；全样本指标仅作报告）")
    lines.append("")
    lines.append("口径：Rank IC = 周频横截面 Spearman；ICIR = mean(IC)/std(IC)；"
                 "分组 = 按因子值横截面排名 5 等分（等权，5 日收益）；"
                 "换手 = Top20% 组相邻周成员变化均值；覆盖率 = finite(factor)&eligible / eligible；"
                 "单调性 = 5 组收益与组序号的 Spearman。有效股票数 < 100 的周跳过。")
    lines.append("")

    def fam_table(df_src: pd.DataFrame, by: str, title: str) -> str:
        if by not in df_src.columns or len(df_src) == 0:
            return f"### {title}\n\n（无数据）\n"
        g = df_src.groupby(by, dropna=False)
        tab = pd.DataFrame({
            "n": g.size(),
            "dev_ic_mean": g["dev_ic_5d"].mean(),
            "dev_icir_mean": g["dev_icir_5d"].mean(),
            "dev_icir_median": g["dev_icir_5d"].median(),
            "dev_abs_icir_mean": g["dev_icir_5d"].apply(lambda s: s.abs().mean()),
            "coverage_mean": g["coverage"].mean(),
            "turnover_mean": g["turnover_top20"].mean(),
        }).reset_index()
        return f"### {title}\n\n" + _md_table(tab) + "\n"

    lines.append("## 1. 候选因子族分布\n")
    lines.append(fam_table(metrics, "lib", "按因子库（候选）"))
    lines.append(fam_table(metrics, "category_en", "按类别（候选）"))
    lines.append(fam_table(metrics, "base", "按 base 信号（候选）"))
    lines.append(fam_table(metrics, "transform", "按 transform（候选）"))
    lines.append("## 2. 全库（800）族分布\n")
    lines.append(fam_table(screen, "lib", "按因子库（全部 800）"))
    lines.append(fam_table(screen, "category_en", "按类别（全部 800）"))

    top = metrics.sort_values("dev_icir_5d", key=lambda s: s.abs(), ascending=False).head(20)
    show = top[["factor_key", "name_cn", "lib", "category_en", "base", "transform", "window",
                "dev_ic_5d", "dev_icir_5d", "dev_ic_n", "ic_5d", "icir_5d",
                "coverage", "turnover_top20", "monotonicity"]].copy()
    show = show.rename(columns={"name_cn": "名称", "lib": "库", "category_en": "类别",
                                "base": "base", "transform": "变换", "window": "窗口",
                                "dev_ic_5d": "DEV_IC", "dev_icir_5d": "DEV_ICIR",
                                "dev_ic_n": "DEV周数", "ic_5d": "全样本IC",
                                "icir_5d": "全样本ICIR", "coverage": "覆盖率",
                                "turnover_top20": "Top20%换手", "monotonicity": "单调性"})
    lines.append("## 3. Top 20 候选（按 |DEV ICIR|）\n")
    lines.append(_md_table(show, floatfmt=".3f"))
    lines.append("")
    lines.append("## 4. 产物清单\n")
    for name, p in [
        ("因子目录", paths["catalog"]),
        ("Stage-1 初筛指标（全 800）", paths["screen"]),
        ("Stage-2 候选指标（复核）", paths["metrics"]),
        ("候选周频面板（weekly_top.npz）", paths["weekly_top"]),
        ("周频元信息（weekly_meta.npz）", paths["weekly_meta"]),
        ("因子卡目录", paths["cards_dir"]),
        ("实验记录", paths["runs"]),
    ]:
        lines.append(f"- {name}：`{p}`")
    if extra.get("corr_note"):
        lines.append("")
        lines.append("## 5. 相关性聚类（evaluate_factors.py）\n")
        lines.append(extra["corr_note"])
    lines.append("")
    paths["index"].write_text("\n".join(lines), encoding="utf-8")


# ================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="两阶段 800 因子构建与评估（factor-lab）")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="小规模端到端自检：只取 N 个因子，产物写 *\\smoke\\ 子目录")
    ap.add_argument("--stage1-only", action="store_true", help="只跑 Stage-1 初筛")
    ap.add_argument("--stage2-only", action="store_true", help="跳过 Stage-1，复用已有 screen 结果")
    ap.add_argument("--resume", action="store_true", help="Stage-1 断点续跑（复用 _screen_progress.parquet）")
    ap.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_all = time.time()
    paths = _resolve_paths(args.smoke)
    smoke = args.smoke is not None

    print("=" * 88)
    print(f"[factor-lab] build_factors  start={_now()}  smoke={args.smoke}  "
          f"stage1_only={args.stage1_only} stage2_only={args.stage2_only} resume={args.resume}")
    print(f"[factor-lab] 输出目录: eval={paths['eval_dir']}  factors={paths['fac_dir']}  journal={paths['journal_dir']}")

    # ---- 1) 数据 + 因子目录
    t0 = time.time()
    md = qfdata.load_market(verbose=True)
    inp = qff.FactorInputs(md)
    cat = qff.load_catalog()
    print(f"[load] market+inp {time.time()-t0:.1f}s；catalog {cat.shape}")
    paths["catalog"].parent.mkdir(parents=True, exist_ok=True)
    try:
        cat.to_parquet(paths["catalog"])
    except Exception as e:  # params 列是 dict，pyarrow 可能不支持 → 退化为 JSON 字符串
        print(f"[catalog][warn] 直接落盘失败（{e}），params 改为 JSON 字符串列")
        cat2 = cat.copy()
        cat2["params"] = cat2["params"].map(lambda d: json.dumps(_jsonable(d), ensure_ascii=False))
        cat2.to_parquet(paths["catalog"])
    print(f"[catalog] → {paths['catalog']}")

    # ---- 2) 周频行 + 标签
    rows = qfe.weekly_rows(md, C.DEV_START, C.REBALANCE_DAYS)
    label = md.fwd_open_ret(C.FWD_HORIZON)
    dev = qfe.dev_week_mask(md, rows)
    print(f"[rows] 周频行 {len(rows)} 个：{md.dates[rows[0]].date()} ~ {md.dates[rows[-1]].date()}；"
          f"DEV 周 {int(dev.sum())}（{C.DEV_START}~{C.DEV_END}）")
    n_elig = md.eligible[rows].sum(axis=1)
    print(f"[rows] 每周 eligible 均值 {n_elig.mean():.0f}，有效周（>=100）{int((n_elig>=100).sum())}")

    # ---- 3) 因子子集（smoke）
    if smoke:
        cat_use = pd.concat([cat.iloc[: max(1, args.smoke - 1)], cat.iloc[[-1]]],
                            ignore_index=True) if args.smoke > 1 else cat.iloc[:1]
        print(f"[smoke] 使用 {len(cat_use)} 个因子（跨 {cat_use.groupby(['base','window']).ngroups} 个组）")
    else:
        cat_use = cat

    # ---- 4) Stage-1
    screen = None
    if not args.stage2_only:
        if args.resume and not smoke and paths["screen"].exists() and not paths["screen_progress"].exists():
            screen = pd.read_parquet(paths["screen"])
            print(f"[stage1] resume：直接复用已完成 {len(screen)} 行 → {paths['screen']}")
        else:
            screen = run_stage1(md, cat, inp, rows, label, cat_use, paths,
                                resume=args.resume, verbose=True)
        assert len(screen) == len(cat_use), f"Stage-1 行数 {len(screen)} != {len(cat_use)}"
    if args.stage1_only:
        print(f"\n[factor-lab] --stage1-only 完成，用时 {(time.time()-t_all)/60:.1f} 分钟")
        return 0
    if screen is None:
        screen = pd.read_parquet(paths["screen"])
        print(f"[stage2] 复用 {paths['screen']}（{len(screen)} 行）")

    # ---- 5) 候选选择（仅 DEV 期指标）
    n_cand = min(int(args.n_candidates), len(screen)) if smoke and args.smoke else int(args.n_candidates)
    selected = select_candidates(screen, n_candidates=n_cand,
                                 ensure_representative=(n_cand >= 50))
    print(f"[select] 候选 {len(selected)}/{len(screen)}；规则：DEV 期 |ICIR| + 覆盖率，"
          f"lib/category 保代表；DEV 有效周>= {MIN_DEV_WEEKS}，覆盖率>= {MIN_DEV_COVERAGE}")
    print(f"[select] 候选 lib 分布：{selected['lib'].value_counts().to_dict()}")
    print(f"[select] 候选 category 分布：{selected['category_en'].value_counts().to_dict()}")

    # ---- 6) Stage-2 + 周频面板
    metrics, problems = run_stage2(md, cat_use, inp, rows, label, selected, paths, verbose=True)
    write_weekly_meta(md, rows, label, paths)
    print(f"[stage2] weekly_meta → {paths['weekly_meta']}")

    # ---- 7) 一致性复核：Stage-1 vs Stage-2 指标
    sc = screen.set_index("factor_key")
    mm = metrics.set_index("factor_key")
    common = [k for k in mm.index if k in sc.index]
    diffs = {}
    for col in ("ic_5d", "icir_5d", "dev_icir_5d", "coverage", "turnover_top20"):
        a = sc.loc[common, col].to_numpy(dtype=np.float64)
        b = mm.loc[common, col].to_numpy(dtype=np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        diffs[col] = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0
    print(f"[check] Stage-1 vs Stage-2 最大绝对差：{diffs}")
    if any(v > 1e-12 for v in diffs.values()):
        problems.append(f"Stage-1/2 指标不一致：{diffs}")

    # ---- 8) 因子卡
    n_cards = write_factor_cards(cat_use, metrics, paths["cards_dir"])
    print(f"[cards] {n_cards} 张因子卡 → {paths['cards_dir']}")

    # ---- 9) INDEX.md
    write_index_md(md, cat_use, screen, metrics, paths, extra={"rows": rows})
    print(f"[index] → {paths['index']}")

    # ---- 10) 实验记录
    artifacts = [str(paths[k].relative_to(ROOT)) for k in
                 ("catalog", "screen", "metrics", "weekly_top", "weekly_meta", "index", "runs")]
    record = {
        "run_id": f"{_now()}_lab_build_factors",
        "stage": "factor_build",
        "status": "ok" if not problems else "warn",
        "n_factors": int(len(screen)),
        "n_candidates": int(len(metrics)),
        "elapsed_sec": round(time.time() - t_all, 1),
        "artifacts": artifacts,
        "note": ("两阶段：Stage-1 全库周频 RankIC/ICIR/分组/换手/覆盖率/分年度；"
                 "Stage-2 候选按 DEV 期(2017-01-03~2019-12-31)|ICIR|+覆盖率选择"
                 "（lib/category 保代表）并复核；全样本指标仅报告字段。"
                 f"stage1_vs_stage2_maxdiff={diffs}；problems={problems}"),
    }
    _append_jsonl(paths["runs"], record)
    print(f"[journal] → {paths['runs']}")

    print("-" * 88)
    top = metrics.sort_values("dev_icir_5d", key=lambda s: s.abs(), ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"  {r['factor_key']:20s} {str(r['name_cn'])[:14]:14s} "
              f"DEV IC={r['dev_ic_5d']:+.4f} ICIR={r['dev_icir_5d']:+.3f} | "
              f"全样本 IC={r['ic_5d']:+.4f} ICIR={r['icir_5d']:+.3f} | "
              f"覆盖率={r['coverage']:.3f} 换手={r['turnover_top20']:.3f}")
    print("-" * 88)
    if problems:
        print(f"[factor-lab][warn] problems={problems}")
    print(f"[factor-lab] 全流程完成，用时 {(time.time()-t_all)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
