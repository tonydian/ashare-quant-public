# -*- coding: utf-8 -*-
"""构建 fundamental-extra 基本面因子族的周频面板（供 walk-forward 模型使用）。

用法:
    python scripts/build_fundamental_extra.py
输出:
    build/factors/fundamental_extra.npz    factor_key -> (475, N) float32
    build/factors/fundamental_catalog.parquet
    build/eval/fundamental_stats.json      DEV/全期 IC/ICIR/覆盖率
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C
from qf import data as qfdata
from qf import fundamentals as qff
from qf import model as qfmodel


def main():
    md = qfdata.load_market(verbose=True)
    rows = np.arange(md.date_index("2017-01-03"), md.T, C.REBALANCE_DAYS, dtype=np.int64)
    print(f"周频行 {len(rows)}：{md.dates[rows[0]].date()} ~ {md.dates[rows[-1]].date()}")

    fd = qff.FundamentalData(md)
    panels = fd.panels_at_rows(rows)
    out = C.BUILD_FACTORS / "fundamental_extra.npz"
    np.savez_compressed(out, **panels)
    cat = qff.catalog()
    cat.to_parquet(C.BUILD_FACTORS / "fundamental_catalog.parquet", index=False)

    elig = md.eligible[rows]
    lab_raw = md.fwd_open_ret(C.FWD_HORIZON)[rows]
    # 标签口径与模型一致：要求 t+1 可买
    buy = np.zeros_like(elig)
    buy[:-1] = md.buyable[rows[:-1] + 1]
    mask = elig & buy
    lab = qfmodel.cs_rank_rows(lab_raw, np.arange(len(rows)), mask=mask)
    dev = md.dates[rows] <= pd.Timestamp(C.DEV_END)
    stats = {}
    for k, v in panels.items():
        f = qfmodel.cs_rank_rows(v, np.arange(len(rows)), mask=mask)
        ic = qfmodel.weekly_ic(f, lab, mask)
        ic_dev = ic[dev & np.isfinite(ic)]
        ic_all = ic[np.isfinite(ic)]
        d_ic = float(np.nanmean(ic_dev)) if len(ic_dev) else np.nan
        d_ir = float(np.nanmean(ic_dev) / max(np.nanstd(ic_dev), 1e-9)) if len(ic_dev) else np.nan
        a_ic = float(np.nanmean(ic_all)) if len(ic_all) else np.nan
        a_ir = float(np.nanmean(ic_all) / max(np.nanstd(ic_all), 1e-9)) if len(ic_all) else np.nan
        cov = float(np.mean(np.isfinite(v[elig]))) if elig.any() else np.nan
        stats[k] = {"dev_ic_5d": d_ic, "dev_icir_5d": d_ir, "ic_5d": a_ic,
                    "icir_5d": a_ir, "coverage": cov, "n_ic": int(np.isfinite(ic).sum())}
        print(f"  {k:22s} DEV IC {d_ic:+.4f} ICIR {d_ir:+.3f} | 全期 IC {a_ic:+.4f} "
              f"ICIR {a_ir:+.3f} | 覆盖率 {cov:.1%}")
    (C.BUILD_EVAL / "fundamental_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}\n→ {C.BUILD_FACTORS / 'fundamental_catalog.parquet'}")
    print(f"→ {C.BUILD_EVAL / 'fundamental_stats.json'}")


if __name__ == "__main__":
    main()
