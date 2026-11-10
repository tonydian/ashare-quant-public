# -*- coding: utf-8 -*-
"""构建 trend-extra（右侧趋势 / 动量）因子的周频面板。

用法:
    python scripts/build_trend_extra.py

输出:
    build/factors/trend_extra.npz      factor_key -> (475, N) float32
    build/factors/trend_catalog.parquet
    build/eval/trend_stats.json        DEV / 全期 RankIC、覆盖率、分年度 IC

口径:
- 周频行沿用 build/factors/weekly_meta.npz 的 rows（与主线 wf_h10 完全同相位）；
- 标签 = md.fwd_open_ret(10) 横截面 rank（t+1 开盘买入，t+11 开盘卖出的 10 日收益；
  右侧趋势信号通常对应更长的持有周期，故用 h10；主模型仍用 h5）；
- 因子只用 t 日及以前数据，无未来函数。
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

from qf import config as C
from qf import data as qfdata
from qf import model as qfm
from qf import trend as qftrend


def _ic_metrics(ic: np.ndarray) -> dict:
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"ic": None, "icir": None, "t": None, "pos": None, "n": 0}
    m = float(np.mean(ic))
    s = float(np.std(ic))
    icir = m / s if s > 1e-12 else 0.0
    return {"ic": m, "icir": icir, "t": icir * np.sqrt(len(ic)),
            "pos": float(np.mean(ic > 0)), "n": int(len(ic))}


def main() -> None:
    t0 = time.time()
    print("[1/4] 加载行情（ext 数据源）...")
    md = qfdata.load_market(verbose=True)
    zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
    rows = zm["rows"].astype(np.int64)
    mask = md.eligible[rows]
    print(f"    周频行 {len(rows)}：{md.dates[rows[0]].date()} ~ {md.dates[rows[-1]].date()}；"
          f"数据 {md.dates[0].date()} ~ {md.dates[-1].date()}，T={md.T}")

    print("[2/4] 计算 trend-extra 因子面板 ...")
    panels = qftrend.build_panels(md, rows)
    for k, v in panels.items():
        print(f"    {k:22s} shape={v.shape} 覆盖率={np.isfinite(v).mean():.1%}")
    np.savez_compressed(C.BUILD_FACTORS / "trend_extra.npz", **panels)
    cat = qftrend.catalog()
    cat.to_parquet(C.BUILD_FACTORS / "trend_catalog.parquet", index=False)
    print(f"    → {C.BUILD_FACTORS / 'trend_extra.npz'}")
    print(f"    → {C.BUILD_FACTORS / 'trend_catalog.parquet'}")

    print("[3/4] 因子 IC 评估（h10 标签，中性化特征）...")
    label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
    dev = md.dates[rows] <= pd.Timestamp(C.DEV_END)
    years = pd.Series(pd.to_datetime(md.dates[rows])).dt.year.to_numpy()
    stats: dict[str, dict] = {}
    cat_idx = cat.set_index("factor_key")
    for k, panel in panels.items():
        feat = qfm.build_feature_block(md, rows, panels, k, mask)
        ic = qfm.weekly_ic(feat, label, mask)
        dev_ic = ic[dev]
        full_ic = ic
        by_year = {}
        for y in sorted(set(years.tolist())):
            sel = (years == y) & np.isfinite(ic)
            if sel.sum() > 0:
                by_year[int(y)] = round(float(np.mean(ic[sel])), 5)
        coverage = float((np.isfinite(panel) & mask).sum() / max(mask.sum(), 1))
        stats[k] = {
            "factor_key": k,
            "slug": str(cat_idx.loc[k, "key"]),
            "name_cn": str(cat_idx.loc[k, "name_cn"]),
            "family": str(cat_idx.loc[k, "category_en"]),
            "window": int(cat_idx.loc[k, "window"]),
            "dev": _ic_metrics(dev_ic),
            "full": _ic_metrics(full_ic),
            "ic_by_year": by_year,
            "coverage": round(coverage, 4),
        }
        d = stats[k]["dev"]
        f = stats[k]["full"]
        print(f"    {k:22s} {stats[k]['name_cn']:18s} DEV IC {d['ic'] or 0:+.4f} "
              f"ICIR {d['icir'] or 0:+.3f} | 全期 IC {f['ic'] or 0:+.4f} "
              f"ICIR {f['icir'] or 0:+.3f} 覆盖 {coverage:.1%}")

    print("[4/4] 保存统计 ...")
    out = {
        "lib": qftrend.LIB,
        "horizon": 10,
        "rows": len(rows),
        "date_range": [str(md.dates[rows[0]].date()), str(md.dates[rows[-1]].date())],
        "data_range": [str(md.dates[0].date()), str(md.dates[-1].date())],
        "n_stocks": int(md.N),
        "factors": stats,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    C.BUILD_EVAL.mkdir(parents=True, exist_ok=True)
    with open(C.BUILD_EVAL / "trend_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"    → {C.BUILD_EVAL / 'trend_stats.json'}")
    print(f"\n完成，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
