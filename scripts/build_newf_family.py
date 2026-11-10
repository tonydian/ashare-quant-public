# -*- coding: utf-8 -*-
"""按「族」落盘新因子（换手 / 流动性 / 量价 / 隔夜），生成标准 catalog 格式。

背景
----
2026-11-03 换手族进了生产池，但 ``build/factors/newf_turnover_extra.npz`` 当时
是实验脚本一次性产物、**没有可复现的生成入口**。本脚本补上这一环：算式只从
``qf.newf`` 取，落盘逻辑按族切分，保证
``--extra build/factors/newf_turnover_extra.npz`` 随时可重算。

产物::

    --family all           build/factors/newf_extra.npz + newf_catalog.parquet
    --family turnover      build/factors/newf_turnover_extra.npz + newf_turnover_catalog.parquet
    --family liquidity     ...newf_liquidity_*
    --family pricevolume   ...newf_pricevolume_*
    --family overnight     ...newf_overnight_*

用法::

    python scripts/build_newf_family.py --family turnover
    python scripts/build_newf_family.py --family all
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C
from qf import data as qfdata
from qf import newf as qfn

FAMILIES = ("all", "turnover", "liquidity", "pricevolume", "overnight")


def _slug_base() -> dict[str, str]:
    return {s: fam for fam, slugs in qfn.FAMILY.items() for s in slugs}


def out_paths(family: str) -> tuple[pathlib.Path, pathlib.Path]:
    stem = "newf" if family == "all" else f"newf_{family}"
    return C.BUILD_FACTORS / f"{stem}_extra.npz", C.BUILD_FACTORS / f"{stem}_catalog.parquet"


def build_and_save(family: str = "all", verbose: bool = True) -> pd.DataFrame:
    if family not in FAMILIES:
        raise SystemExit(f"未知族 {family!r}，可选 {FAMILIES}")
    slugs = (list(qfn.SLUG_TO_ID) if family == "all"
             else [s for s in qfn.FAMILY[family] if s in qfn.SLUG_TO_ID])
    if not slugs:
        raise SystemExit(f"族 {family} 没有因子")

    if verbose:
        print("[1/3] 加载行情并计算各因子族面板 ...")
    md = qfdata.load_market(verbose=verbose)
    panels = qfn.build_panels(md)
    rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)

    npz_path, cat_path = out_paths(family)
    base_of = _slug_base()
    out: dict[str, np.ndarray] = {}
    cat_rows = []
    meta = {s: (cn, en, ccn, cen, win) for s, cn, en, ccn, cen, win in qfn.SPEC}
    if verbose:
        print(f"[2/3] 取周频行（{len(rows)} 周）并体检 ...")
        print(f"{'slug':<18}{'覆盖':>8}{'周数':>6}")
    for slug in slugs:
        panel = panels[slug][rows]
        key = f"{qfn.PREFIX}{qfn.SLUG_TO_ID[slug]}"
        out[key] = panel.astype(np.float32)
        cn, en, ccn, cen, win = meta[slug]
        if verbose:
            print(f"{slug:<18}{np.isfinite(panel).mean():>7.1%}{len(panel):>6}")
        cat_rows.append({
            "lib": qfn.LIB, "factor_id": qfn.SLUG_TO_ID[slug], "slug": slug,
            "name_cn": cn, "name_en": en, "category_cn": ccn, "category_en": cen,
            "base": base_of[slug], "transform": "raw", "window": int(win), "norm": 20,
            "params": json.dumps({"window": int(win), "norm": 20}, ensure_ascii=False),
            "path": f"{npz_path.relative_to(ROOT).as_posix()}#{key}".replace("\\", "/"),
            "factor_key": key,
        })

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **out)
    cat = pd.DataFrame(cat_rows)
    cat.to_parquet(cat_path, index=False)
    if verbose:
        print(f"[3/3] 写出 {npz_path.name}（{len(out)} 因子 × {len(rows)} 周）与 {cat_path.name}")
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all", choices=FAMILIES)
    args = ap.parse_args()
    build_and_save(args.family)


if __name__ == "__main__":
    main()
