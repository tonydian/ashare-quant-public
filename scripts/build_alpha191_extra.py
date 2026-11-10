# -*- coding: utf-8 -*-
"""构建 Alpha191 增量因子面板（供 walk-forward 模型以 --extra 并入）。

为什么只接这 2 个
-----------------
`build/alpha191/incremental.csv` 的两级正交化结果：180 个因子里，剥离
[log市值, log股价, log20日均额] 三风格、**再**剥离现有库 top30 之后，
|IC| 仍在 0.02 附近的只有 alpha_046（+0.0211）与 alpha_062（+0.0193）。
其余高位因子要么是纯规模/成交额代理（alpha_132 风格调整后仅 -0.0058），
要么归零到 0.009 以下。

用法:
    python scripts/build_alpha191_extra.py
输出:
    build/factors/alpha191_extra.npz       (factor_key -> (n_weeks, N) float32)
    build/factors/alpha191_catalog.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qf import alpha191 as a191          # noqa: E402
from qf import config as C              # noqa: E402
from qf import model as qfm             # noqa: E402
from qf.data import MarketData          # noqa: E402

LIB = "alpha191-extra"

DEFS = [
    # (factor_id, 因子号, 中文名, 英文名, 公式简述)
    ("A046", 46, "均线位置", "MA position",
     "SUM(CLOSE,3)+SUM(CLOSE,6)+2*SUM(CLOSE,12)+2*SUM(CLOSE,24) 的加权均价相对收盘的位置"),
    ("A062", 62, "量价背离", "-CORR(HIGH, RANK(VOL), 5)",
     "5 日内最高价与成交量横截面秩的负相关"),
]


def main() -> None:
    md = MarketData(verbose=True)
    zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
    rows = zm["rows"].astype(np.int64)
    print(f"周频行 {len(rows)}：{md.dates[rows[0]].date()} ~ {md.dates[rows[-1]].date()}")

    knum = [d[1] for d in DEFS]
    print(f"计算 Alpha191 因子 {knum} ...")
    raw = a191.compute_panel(md, knum, rows=rows, verbose=True)

    elig = md.eligible[rows]
    lab = md.fwd_open_ret(C.FWD_HORIZON)[rows]
    dev = md.dates[rows] <= pd.Timestamp(C.DEV_END)

    panels = {}
    cat_rows = []
    print("\n覆盖率与 IC 预检：")
    for fid, k, cn, en, desc in DEFS:
        key = f"{LIB}:{fid}"
        x = np.asarray(raw[k], dtype=np.float32)
        assert x.shape[0] == len(rows), f"{key} 行数不符 {x.shape[0]} vs {len(rows)}"
        panels[key] = x

        # 覆盖率只统计「选股域内」的有效值（因子在停牌/退市股上也可能有 ffill 值）
        cov = (np.isfinite(x) & elig).sum(axis=1) / np.maximum(elig.sum(axis=1), 1)
        cov_dev = cov[dev]
        f = qfm.cs_rank_rows(x, np.arange(len(rows)), mask=elig)
        labr = qfm.cs_rank_rows(lab, np.arange(len(rows)), mask=elig)
        ic = qfm.weekly_ic(f, labr, elig)
        icd = ic[dev & np.isfinite(ic)]
        ica = ic[np.isfinite(ic)]
        print(f"  {key:<16} 覆盖 DEV {np.nanmean(cov_dev):.3f} / 全期 {np.nanmean(cov):.3f} | "
              f"DEV IC {np.nanmean(icd):+.4f} ICIR {np.nanmean(icd)/max(np.nanstd(icd),1e-9):+.3f} (n={len(icd)}) | "
              f"全期 IC {np.nanmean(ica):+.4f} ICIR {np.nanmean(ica)/max(np.nanstd(ica),1e-9):+.3f}")

        cat_rows.append({
            "lib": LIB, "factor_id": fid, "slug": f"alpha191_{k:03d}",
            "name_cn": cn, "name_en": en,
            "category_cn": "Alpha191", "category_en": "Alpha191-extra",
            "base": desc, "transform": "raw", "window": 0, "norm": 0,
            "params": {"alpha191_no": k, "formula": a191.FORMULAS.get(k, "")},
            "path": "", "factor_key": key,
        })

    out_npz = C.BUILD_FACTORS / "alpha191_extra.npz"
    out_cat = C.BUILD_FACTORS / "alpha191_catalog.parquet"
    np.savez_compressed(out_npz, **panels)
    pd.DataFrame(cat_rows).to_parquet(out_cat, index=False)
    print(f"\n→ {out_npz}")
    print(f"→ {out_cat}")


if __name__ == "__main__":
    main()
