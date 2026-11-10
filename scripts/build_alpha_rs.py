# -*- coding: utf-8 -*-
"""构建 alpha-rs（Alpha158/Alpha101 右侧因子族）的周频面板与统计。

用法::

    python scripts/build_alpha_rs.py

输出（仅这三个文件）::

    build/factors/alpha_rs.npz             factor_key -> (475, N) float32
    build/factors/alpha_rs_catalog.parquet 因子目录（16 行）
    build/eval/alpha_rs_stats.json         每个因子 DEV(2017-2019)/全期 IC、ICIR、覆盖率

口径:
- 周频行 = ``np.arange(md.date_index("2017-01-03"), md.T, C.REBALANCE_DAYS)``（475 行，
  与 build/factors/weekly_meta.npz 完全同相位；若数据包更长则截取前 475 行）；
- 标签 = ``qf.model.weekly_label(md, rows, mask, horizon=10, rank=True)``（t+1 开盘买入、
  t+11 开盘卖出的 10 日收益的横截面 rank，含 t+1 可买掩码；右侧信号通常对应较长持有期，
  与 trend-extra 的 h10 一致）；
- IC 统计口径与 ``scripts/build_fundamental_extra.py`` 一致：特征 = 因子横截面 rank，
  ``qf.model.weekly_ic`` 逐周 RankIC；DEV = 2017-01-03~2019-12-31；覆盖率 =
  选股域内因子有限值占比；
- 另附 ``dev_model``/``full_model``：按模型实际使用口径（``qf.model.build_feature_block``
  横截面 rank + 行业/市值中性化 + z-score）计算的 IC，供交叉核对；
- 因子只用 t 日及以前数据，无未来函数。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import alpha_rs as qfars         # noqa: E402
from qf import config as C              # noqa: E402
from qf import data as qfdata           # noqa: E402
from qf import model as qfm             # noqa: E402

N_WEEKS = 475       # 与 weekly_meta 对齐的周频行数
HORIZON = 10        # 标签持有期（右侧模型约定，与 trend-extra 一致）
MIN_IC_OBS = 100    # 逐周横截面有效股票数下限（qf.model.weekly_ic 内部同口径）


def _ic_metrics(ic: np.ndarray) -> dict:
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"ic": None, "icir": None, "t": None, "pos": None, "n": 0}
    m = float(np.mean(ic))
    s = float(np.std(ic))
    icir = m / s if s > 1e-12 else 0.0
    return {"ic": m, "icir": icir, "t": icir * float(np.sqrt(len(ic))),
            "pos": float(np.mean(ic > 0)), "n": int(len(ic))}


def _fmt(x, nd=4) -> str:
    return "   nan  " if x is None else f"{x:+.{nd}f}"


def main() -> None:
    t0 = time.time()
    print("=" * 92)
    print("[alpha-rs] 构建 Alpha158/Alpha101 右侧因子族（16 个）")

    # ---------------------------------------------------------------- 1) 数据 + 周频行
    print("[1/5] 加载行情 ...")
    md = qfdata.load_market(verbose=True)
    rows_all = np.arange(md.date_index("2017-01-03"), md.T, C.REBALANCE_DAYS, dtype=np.int64)
    if len(rows_all) > N_WEEKS:
        print(f"    [warn] 数据包较长（T={md.T}），rows={len(rows_all)} > {N_WEEKS}，"
              f"截取前 {N_WEEKS} 行以与 weekly_meta 对齐")
        rows = rows_all[:N_WEEKS]
    elif len(rows_all) < N_WEEKS:
        raise RuntimeError(f"数据包太短：只有 {len(rows_all)} 个周频行（需 {N_WEEKS}）；"
                           f"请使用默认数据包 data_extract/was_stock_data")
    else:
        rows = rows_all
    mask = md.eligible[rows]
    dates_w = md.dates[rows]
    print(f"    T={md.T}, N={md.N}, 数据 {md.dates[0].date()} ~ {md.dates[-1].date()}"
          f"（{C.DATA_RAW.name}）")
    print(f"    周频行 {len(rows)}：{dates_w[0].date()} ~ {dates_w[-1].date()}；"
          f"选股域均值 {mask.sum(axis=1).mean():.0f} 只/周")
    zm_path = C.BUILD_FACTORS / "weekly_meta.npz"
    if zm_path.exists():
        zm_rows = np.load(zm_path)["rows"].astype(np.int64)
        print(f"    与 weekly_meta rows 对齐：{np.array_equal(zm_rows, rows)}"
              f"（meta {len(zm_rows)} 行）")

    # ---------------------------------------------------------------- 2) 因子面板
    print("[2/5] 计算 alpha-rs 因子面板（全量面板 → 周频行切片）...")
    panels = qfars.build_panels(md, rows)
    cat = qfars.catalog()
    keys = cat["factor_key"].tolist()
    assert list(panels.keys()) == keys, "panels 与 catalog 顺序不一致"
    cat_idx = cat.set_index("factor_key")
    for k in keys:
        v = panels[k]
        assert v.shape == (len(rows), md.N), f"{k} shape {v.shape}"
        assert v.dtype == np.float32, f"{k} dtype {v.dtype}"
        cov = float((np.isfinite(v) & mask).sum() / max(mask.sum(), 1))
        print(f"    {k:16s} {str(cat_idx.loc[k, 'name_cn']):14s} "
              f"w={int(cat_idx.loc[k, 'window']):2d} 覆盖率(选股域)={cov:.1%}")
    out_npz = C.BUILD_FACTORS / "alpha_rs.npz"
    np.savez_compressed(out_npz, **panels)
    out_cat = C.BUILD_FACTORS / "alpha_rs_catalog.parquet"
    cat.to_parquet(out_cat, index=False)
    print(f"    → {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB)")
    print(f"    → {out_cat}")

    # ---------------------------------------------------------------- 3) IC 统计（h10）
    print("[3/5] 因子 IC 评估（标签 = qf.model.weekly_label(h10, rank)，2020 后不参与调参）...")
    label = qfm.weekly_label(md, rows, mask, horizon=HORIZON, rank=True)
    dev = dates_w <= pd.Timestamp(C.DEV_END)
    years = pd.DatetimeIndex(dates_w).year.to_numpy()
    print(f"    标签有效周 {int(np.isfinite(label).any(axis=1).sum())}/{len(rows)}；"
          f"DEV 周 {int(dev.sum())}")
    stats: dict[str, dict] = {}
    for k in keys:
        panel = panels[k]
        # (a) 与 fundamental_stats 一致的口径：横截面 rank 特征 + h10 rank 标签
        feat = qfm.cs_rank_rows(panel, np.arange(len(rows)), mask=mask)
        ic = qfm.weekly_ic(feat, label, mask)
        # (b) 模型实际口径：rank + 行业/市值中性化 + z-score
        feat_model = qfm.build_feature_block(md, rows, panels, k, mask)
        ic_model = qfm.weekly_ic(feat_model, label, mask)
        by_year = {}
        for y in sorted(set(years.tolist())):
            sel = (years == y) & np.isfinite(ic)
            if sel.sum():
                by_year[int(y)] = round(float(np.mean(ic[sel])), 5)
        coverage = float((np.isfinite(panel) & mask).sum() / max(mask.sum(), 1))
        stats[k] = {
            "factor_key": k,
            "slug": str(cat_idx.loc[k, "key"]),
            "name_cn": str(cat_idx.loc[k, "name_cn"]),
            "name_en": str(cat_idx.loc[k, "name_en"]),
            "family": str(cat_idx.loc[k, "category_en"]),
            "window": int(cat_idx.loc[k, "window"]),
            "expr": str(cat_idx.loc[k, "params"]["expr"]),
            "expr_alpha158": str(cat_idx.loc[k, "params"]["expr_alpha158"]),
            "dev": _ic_metrics(ic[dev]),
            "full": _ic_metrics(ic),
            "dev_model": _ic_metrics(ic_model[dev]),
            "full_model": _ic_metrics(ic_model),
            "ic_by_year": by_year,
            "coverage": round(coverage, 4),
        }
        d = stats[k]["dev"]
        f = stats[k]["full"]
        print(f"    {k:16s} {str(cat_idx.loc[k, 'name_cn']):14s} "
              f"DEV IC {_fmt(d['ic'])} ICIR {_fmt(d['icir'], 3)} | "
              f"全期 IC {_fmt(f['ic'])} ICIR {_fmt(f['icir'], 3)} 覆盖 {coverage:.1%}")

    # ---------------------------------------------------------------- 4) live_row 自检
    print("[4/5] live_row（最新交易日截面）自检 ...")
    lr1 = qfars.live_row(md)
    lr2 = qfars.live_row(md)
    assert set(lr1) == set(keys)
    assert lr2 is lr1, "live_row 未命中 id(md) 缓存"
    for k in keys:
        assert lr1[k].shape == (md.N,), f"{k} live shape {lr1[k].shape}"
        assert lr1[k].dtype == np.float32
    print(f"    {len(lr1)} 个因子 × {md.N} 只股票；缓存命中 ✓；示例 "
          f"{keys[0]} finite={np.isfinite(lr1[keys[0]]).mean():.1%}")

    # ---------------------------------------------------------------- 5) 统计落盘
    print("[5/5] 保存统计 ...")
    out_stats = C.BUILD_EVAL / "alpha_rs_stats.json"
    result = {
        "lib": qfars.LIB,
        "horizon": HORIZON,
        "label": "md.fwd_open_ret(10) 横截面 rank（qf.model.weekly_label，含 t+1 可买掩码）",
        "feature": ("cs_rank（与 build_fundamental_extra.py 统计口径一致）；"
                    "dev_model/full_model 为 qf.model.build_feature_block 中性化口径"),
        "rows": int(len(rows)),
        "date_range": [str(dates_w[0].date()), str(dates_w[-1].date())],
        "data_range": [str(md.dates[0].date()), str(md.dates[-1].date())],
        "data_source": str(C.DATA_RAW),
        "n_stocks": int(md.N),
        "factors": stats,
        "skipped": qfars.SKIPPED,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    C.BUILD_EVAL.mkdir(parents=True, exist_ok=True)
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"    → {out_stats}")

    # ---------------------------------------------------------------- 汇总
    d_ic = [stats[k]["dev"]["ic"] for k in keys]
    d_ir = [stats[k]["dev"]["icir"] for k in keys]
    print("-" * 92)
    print(f"[alpha-rs] DEV IC>0 的因子：{sum(1 for x in d_ic if x and x > 0)}/{len(keys)}；"
          f"DEV |ICIR| 中位数 {np.median([abs(x) for x in d_ir if x]):.3f}")
    best = sorted(keys, key=lambda k: -abs(stats[k]["dev"]["icir"] or 0))[:5]
    for k in best:
        d = stats[k]["dev"]
        print(f"    {k:16s} {stats[k]['name_cn']:14s} DEV IC {_fmt(d['ic'])} "
              f"ICIR {_fmt(d['icir'], 3)} 覆盖 {stats[k]['coverage']:.1%}")
    print(f"\n[alpha-rs] 完成，用时 {time.time() - t0:.1f}s")
    print("=" * 92)


if __name__ == "__main__":
    main()
