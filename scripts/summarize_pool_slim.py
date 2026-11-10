# -*- coding: utf-8 -*-
"""汇总因子池精简实验：槽位数 × (IC / ICIR / 相位带 CAGR 分布)。

读取 build/eval/phase_band*.json 与 build/models/*/composite_ic.parquet，
输出一张对齐口径的对比表（Markdown + JSON）。

用法::
    python scripts/summarize_pool_slim.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import config as C  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "build" / "models"
EVAL = ROOT / "build" / "eval"

# (显示名, 模型目录, 相位带 json)
CASES = [
    ("34 个（现状）", "wf_h10", "phase_band.json"),
    ("24 个（每类3个）", "wf_cap3_24", "phase_band_cap3_24.json"),
    ("20 个（每类2个）", "wf_cap2_20", "phase_band_cap2_20.json"),
    ("20 个（按|w|）", "wf_slim20", "phase_band_slim20.json"),
    ("16 个（按|w|）", "wf_slim16", "phase_band_slim16.json"),
    ("12 个（按|w|）", "wf_slim12", "phase_band_slim12.json"),
    ("16 个（按DEV）", "wf_dev16", "phase_band_dev16.json"),
    ("16 个（无配额）", "wf_nocap16", "phase_band_nocap16.json"),
    ("12 个（重选池）", "wf_rs12", None),
]


def ic_of(model: str) -> dict:
    p = MODELS / model / "composite_ic.parquet"
    if not p.exists():
        return {}
    ic = pd.read_parquet(p)
    v = ic["ic"].to_numpy()
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return {}
    return {"ic": float(v.mean()), "icir": float(v.mean() / v.std(ddof=1)),
            "t": float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))), "n": int(len(v))}


def band_of(fn: str | None) -> dict:
    if not fn:
        return {}
    p = EVAL / fn
    if not p.exists():
        return {}
    b = json.loads(p.read_text(encoding="utf-8"))
    return {
        "n_pool": int(b["prod"]["n_pool"]) if "n_pool" in b.get("prod", {}) else None,
        "cagr_mean": b["cagr"]["mean"], "cagr_med": b["cagr"]["median"],
        "cagr_min": b["cagr"]["min"], "cagr_max": b["cagr"]["max"],
        "cagr_std": b["cagr"]["std"],
        "sharpe_mean": b["sharpe"]["mean"],
        "prod_cagr": b["prod"]["cagr"] if isinstance(b.get("prod"), dict) else None,
    }


def main():
    rows = []
    for name, model, fn in CASES:
        d = {"name": name, "model": model}
        d.update(ic_of(model))
        d.update(band_of(fn))
        mp = MODELS / model / "model_meta.json"
        if mp.exists():
            d["k"] = len(json.loads(mp.read_text(encoding="utf-8")).get("pool", []))
        rows.append(d)
    df = pd.DataFrame(rows)

    out = df.copy()
    pd.set_option("display.width", 240)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Markdown
    md = ["| 方案 | 因子数 | OOS IC | ICIR | 相位 CAGR 均值 | 中位 | 最差 | 最好 | 生产相位 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append("| {} | {} | {:.4f} | {:.3f} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} |".format(
            r["name"], int(r["k"]) if pd.notna(r.get("k")) else "-",
            r.get("ic", float("nan")), r.get("icir", float("nan")),
            r.get("cagr_mean", float("nan")), r.get("cagr_med", float("nan")),
            r.get("cagr_min", float("nan")), r.get("cagr_max", float("nan")),
            r.get("prod_cagr", float("nan"))))
    text = "\n".join(md)
    print()
    print(text)
    (EVAL / "pool_slim_summary.md").write_text(text + "\n", encoding="utf-8")
    (EVAL / "pool_slim_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {EVAL/'pool_slim_summary.md'}")


if __name__ == "__main__":
    main()
