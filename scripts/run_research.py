# -*- coding: utf-8 -*-
"""Walk-forward 因子选择 + 权重估计 + 复合打分。

用法:
    python scripts/run_research.py --out build/models/wf_v1 --method blend

输入:
    build/factors/weekly_top.npz   （factor-lab 产出：factor_key -> (n_weeks, N) 因子面板）
    build/factors/weekly_meta.npz  （dates/rows/label/eligible）
输出:
    build/models/wf_v1/composite_score.npz  (T,N) 周频行有值的复合分
    build/models/wf_v1/model_weights.parquet / pool_meta.parquet / composite_ic.parquet / model_meta.json
"""
from __future__ import annotations

import argparse
import sys
import time
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C
from qf import data as qfdata
from qf import factors as qff
from qf import model as qfm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.BUILD_MODELS / "wf_v1"))
    ap.add_argument("--factors", default=str(C.BUILD_FACTORS / "weekly_top.npz"))
    ap.add_argument("--meta", default=str(C.BUILD_FACTORS / "weekly_meta.npz"))
    ap.add_argument("--method", default="blend", choices=["blend", "icir", "ridge"])
    ap.add_argument("--max-factors", type=int, default=C.MAX_FACTORS)
    ap.add_argument("--max-per-category", type=int, default=None,
                    help="单类因子上限（默认 4）。设为较大值即「取消族配额」，"
                         "用于检验按 |ICIR| 全局竞争是否优于强行凑类目")
    ap.add_argument("--corr-max", type=float, default=0.70)
    ap.add_argument("--train-weeks", type=int, default=156)
    ap.add_argument("--refit-weeks", type=int, default=None,
                    help="再估计最小间隔（周）；默认用 ModelConfig 的 12 周")
    ap.add_argument("--horizon", type=int, default=C.FWD_HORIZON)
    ap.add_argument("--smooth-alpha", type=float, default=0.6)
    ap.add_argument("--fdr-q", type=float, default=None,
                    help="BH-FDR 控制水平（如 0.05）；默认不启用，用于控制多因子筛选假阳性")
    ap.add_argument("--extra", action="append", default=None,
                    help="额外因子族面板 npz（可重复指定；默认自动加载 momentum/fundamental 两个 extra）")
    ap.add_argument("--no-fundamental", action="store_true",
                    help="不并入 fundamental-extra 因子族（主模型既定设计：量价+动量；"
                         "基本面只作实验分支 wf_h10_fund）")
    ap.add_argument("--pool-guard", action="store_true",
                    help="启用因子池缓冲（防抖动）：连续 N 次确认才换人")
    ap.add_argument("--pool-streak", type=int, default=2,
                    help="缓冲确认次数（默认 2）")
    ap.add_argument("--pool-entry-rank", type=int, default=None,
                    help="新因子入场排名线（默认 k*0.75）")
    ap.add_argument("--pool-exit-rank", type=int, default=None,
                    help="旧因子退出排名线（默认 k*1.25）")
    ap.add_argument("--pool-from", default=None,
                    help="复用某个已有模型的因子池（做同池对比/NaN中性化复跑时使用）")
    ap.add_argument("--family-budget", action="store_true",
                    help="按族分配权重预算（族间等分 1/F，族内按原权重比例）；默认关闭")
    ap.add_argument("--family-strength", type=float, default=1.0,
                    help="族预算强度，0=不干预、1=完全族等权（线性插值）")
    ap.add_argument("--family-cap", type=float, default=0.0,
                    help="单族权重上限（如 0.25）：只在某族超过上限时削峰，未越界族不动；"
                         "0=关闭。必须 >= 1/有效族数，否则退化为族等分")
    ap.add_argument("--require-oos-nondecay", action="store_true",
                    help="选池第三判据：|OOS ICIR| 必须 >= |DEV ICIR|×(1-tol)，默认关闭")
    ap.add_argument("--oos-decay-tol", type=float, default=0.0,
                    help="允许衰减比例（0=严格 OOS>=DEV；0.2=允许衰减 20%%）")
    ap.add_argument("--oos-window-start", default="2019-01-01",
                    help="衰减检验窗起（必须在 DEV 内，默认 2019-01-01）")
    ap.add_argument("--oos-window-end", default="2019-12-31",
                    help="衰减检验窗止（默认 2019-12-31）")
    ap.add_argument("--start", default=C.VALID_START)
    args = ap.parse_args()

    t0 = time.time()
    print("[1/4] 加载行情数据 ...")
    md = qfdata.load_market(verbose=True)
    cat = qff.load_catalog()

    print("[2/4] 加载周频因子面板 ...")
    zf = np.load(args.factors)
    zm = np.load(args.meta)
    rows_weekly = zm["rows"].astype(np.int64)
    panels = {k: zf[k] for k in zf.files}
    print(f"    {len(panels)} 个因子 × {len(rows_weekly)} 周，日期 "
          f"{md.dates[rows_weekly[0]].date()} ~ {md.dates[rows_weekly[-1]].date()}")
    # 额外因子族：momentum-extra（中期反转/动量）+ fundamental-extra（基本面，PIT）
    extra_sets = [
        (C.BUILD_FACTORS / "momentum_extra.npz", C.BUILD_FACTORS / "momentum_catalog.parquet"),
        (C.BUILD_FACTORS / "fundamental_extra.npz", C.BUILD_FACTORS / "fundamental_catalog.parquet"),
    ]
    # --extra 可重复（2026-11-03）：换手族与 momentum 族需要同时竞争，只能传一个时会互相挤掉
    for _p in (args.extra or []):
        p = pathlib.Path(_p)
        extra_sets.append((p, p.with_name(p.stem.replace("_extra", "_catalog") + ".parquet")))
    for extra_path, cat_path in extra_sets:
        if not (extra_path.exists() and cat_path.exists()):
            continue
        if args.no_fundamental and "fundamental" in extra_path.stem:
            print(f"    跳过 {extra_path.stem}（--no-fundamental：主模型只用量价+动量）")
            continue
        ze = np.load(extra_path)
        extra_cat = pd.read_parquet(cat_path)
        n_extra = 0
        n_skip = []
        for k in ze.files:
            if k in panels:
                continue
            a = ze[k]
            if a.shape[1] != md.N:
                n_skip.append((k, f"N={a.shape[1]}"))
                continue
            if a.shape[0] > len(rows_weekly):
                n_skip.append((k, f"{a.shape[0]}周>meta"))
                continue
            if a.shape[0] < len(rows_weekly):
                # 早期构建的面板比 weekly_meta 少几周（如 475 vs 477）。原实现直接
                # `shape[0] == len(rows_weekly)` 过滤，导致 momentum(8)/trend(16)
                # 共 24 个因子被**静默丢弃**、从未进入选池竞争。见 qff.align_panel_rows。
                a = qff.align_panel_rows(a, len(rows_weekly), k)
            panels[k] = a
            n_extra += 1
        # 同一因子可能在多个族里重复登记（newf-extra 的 N001~N005 与 newf-turnover
        # 是同一批换手因子的两份副本）。panels 是字典、天然去重，但 catalog 直接 concat
        # 会留下重复行，污染后续「按类配额」计数与池内去重逻辑 → 以先登记者优先去重。
        add = extra_cat[extra_cat["factor_key"].isin(panels)]
        add = add.drop_duplicates(subset="factor_key", keep="first")
        dup = sorted(set(add["factor_key"]) & set(cat["factor_key"]))
        if dup:
            print(f"    [warn] {extra_path.stem} 有 {len(dup)} 个因子与已有 catalog 重名，已跳过：{dup[:5]}")
            add = add[~add["factor_key"].isin(set(cat["factor_key"]))]
        cat = pd.concat([cat, add], ignore_index=True)
        print(f"    额外因子族 {extra_path.stem}: {n_extra} 个（已并入候选池）")
        if n_skip:
            print(f"    [warn] 跳过 {len(n_skip)} 个形状不符：{n_skip[:4]}")
    # 只保留 catalog 中存在的因子
    panels = {k: v for k, v in panels.items() if k in set(cat["factor_key"])}

    # 因子池缓冲：从已有模型目录读上一次的池 + 状态
    _prev_pool = None
    _pool_state = {}
    if args.pool_guard:
        import json as _json
        from qf import pool_guard as _pg
        _out_dir = pathlib.Path(args.out)
        _prev_pool = _pg.prev_pool_from_meta(_out_dir)
        _pool_state = _pg.load_state(ROOT)
        if _prev_pool:
            print(f"[pool-guard] 上次池 {len(_prev_pool)} 个，"
                  f"状态条目 {len(_pool_state)}")
        else:
            print("[pool-guard] 首次构建（无历史池）")

    _kw = dict(method=args.method, max_factors=args.max_factors,
               corr_max=args.corr_max, train_weeks=args.train_weeks,
               horizon=args.horizon, score_smooth_alpha=args.smooth_alpha,
               fdr_q=args.fdr_q,
               pool_guard=bool(args.pool_guard),
               pool_prev=_prev_pool,
               pool_state=_pool_state,
               pool_need_streak=int(args.pool_streak),
               pool_entry_rank=args.pool_entry_rank,
               pool_exit_rank=args.pool_exit_rank,
               family_budget=bool(args.family_budget),
               family_strength=float(args.family_strength),
               family_cap=float(args.family_cap),
               require_oos_nondecay=bool(args.require_oos_nondecay),
               oos_decay_tol=float(args.oos_decay_tol),
               oos_window_start=args.oos_window_start,
               oos_window_end=args.oos_window_end)
    if args.max_per_category is not None:
        _kw["max_per_category"] = int(args.max_per_category)
    if args.refit_weeks is not None:
        _kw["refit_weeks"] = int(args.refit_weeks)
    cfg = qfm.ModelConfig(**_kw)
    pool_override = None
    if args.pool_from:
        _base = qfm.load_result(args.pool_from)
        pool_override = _base.pool
        print(f"[3/4] 复用因子池：{args.pool_from}（{len(pool_override)} 个因子）")
    print(f"[3/4] walk-forward（method={cfg.method}, K≤{cfg.max_factors}, "
          f"train={cfg.train_weeks}w, refit={cfg.refit_weeks}w, start={args.start}）...")
    res = qfm.walk_forward(md, rows_weekly, panels, cat, cfg=cfg, start_date=args.start,
                           pool_override=pool_override)

    print("[4/4] 保存 ...")
    qfm.save_result(res, args.out)

    # 持久化缓冲状态
    if args.pool_guard:
        try:
            from qf import pool_guard as _pg
            _st = getattr(cfg, "pool_state_out", None)
            if _st:
                _pg.save_state(ROOT, _st, meta={
                    "at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                    "pool": list(res.pool),
                    "n": len(res.pool),
                })
                print(f"[pool-guard] 状态已保存（{len(_st)} 条）")
        except Exception as _e:
            print(f"[pool-guard] 状态保存失败: {_e}")
    print(f"    pool={len(res.pool)} 因子；refit {res.weights['refit_date'].nunique()} 次；"
          f"OOS IC 周数 {len(res.composite_ic)}")
    if len(res.composite_ic):
        ic = res.composite_ic["ic"].to_numpy()
        print(f"    复合分 OOS RankIC: mean={ic.mean():.4f}  ICIR={ic.mean()/max(ic.std(),1e-9):.3f}  "
              f"t={ic.mean()/max(ic.std(),1e-9)*np.sqrt(len(ic)):.2f}  正IC占比={np.mean(ic>0):.2%}")
        ic2 = res.composite_ic.copy()
        ic2["year"] = pd.to_datetime(ic2["date"]).dt.year
        print("    分年度 IC:")
        print(ic2.groupby("year")["ic"].agg(["mean", "std", "count"]).round(4).to_string())
    print("\n池内权重（最近一次 refit）:")
    last = res.weights["refit_date"].max()
    w = res.weights[res.weights["refit_date"] == last].sort_values("weight", ascending=False)
    print(w[["factor_key", "weight"]].head(45).to_string(index=False))
    print(f"\n完成，用时 {time.time()-t0:.1f}s → {args.out}")


if __name__ == "__main__":
    main()
