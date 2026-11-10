# -*- coding: utf-8 -*-
"""对照测试：向量化因子实现 vs QuantSkills 参考 factor.py（逐股票 pandas 实现）。

用法: python scripts/check_factor_equiv.py [n_factor] [n_stock]
"""
from __future__ import annotations

import importlib.util
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import data as qfdata
from qf import factors as qff

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_ref(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    n_factor = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_stock = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    md = qfdata.load_market(verbose=True)
    inp = qff.FactorInputs(md)
    cat = qff.load_catalog()

    # 每个 base 选一个因子，transform 轮换覆盖
    trs = ["z", "rank", "confirm_volume", "compress", "vol_scaled", "stability", "smooth", "delta", "raw"]
    picked = []
    for k, (lib, sub) in enumerate(cat.groupby("lib", sort=False)):
        for i, base in enumerate(sub["base"].unique()):
            s = sub[sub["base"] == base]
            tr = trs[(i + k) % len(trs)]
            cand_tr = s[s["transform"] == tr]
            e = (cand_tr if len(cand_tr) else s).iloc[0]
            picked.append(e)
    picked = picked[:n_factor]
    print(f"测试 {len(picked)} 个因子 × {n_stock} 只股票")

    # 选股票：上市早、数据全
    rng = np.random.default_rng(7)
    cand = np.flatnonzero(md.first_valid < 200)
    cand = cand[np.isfinite(md.close[-1, cand])]
    pick_stocks = rng.choice(cand, size=min(n_stock, len(cand)), replace=False)

    # 构造参考 df（价格按引擎口径 ffilled，volume 保留 NaN）
    df_all = pd.DataFrame({
        "symbol": np.repeat(md.codes[pick_stocks], md.T),
        "date": np.tile(md.dates.values, len(pick_stocks)),
        "open": inp.open[:, pick_stocks].T.reshape(-1),
        "high": inp.high[:, pick_stocks].T.reshape(-1),
        "low": inp.low[:, pick_stocks].T.reshape(-1),
        "close": inp.close[:, pick_stocks].T.reshape(-1),
        "volume": inp.volume[:, pick_stocks].T.reshape(-1),
    })
    df_all = df_all.sort_values(["symbol", "date"]).reset_index(drop=True)

    rows = []
    for i, e in enumerate(picked):
        ref = load_ref(e["path"] + "/scripts/factor.py", f"ref_{i}")
        out_ref = ref.compute_factor(df_all.copy())
        col = ref.FACTOR_COLUMN
        ref_panel = out_ref.pivot(index="date", columns="symbol", values=col).to_numpy(dtype=np.float64)
        ref_codes = np.array(out_ref["symbol"].unique())
        # 对齐到 md 顺序
        ref_panel_full = np.full((md.T, md.N), np.nan)
        idx = np.searchsorted(md.codes, ref_codes)  # codes 已排序? 若非有序则用 map
        # codes 顺序来自 npz，未排序；用 dict 对齐
        cmap = {c: j for j, c in enumerate(md.codes)}
        col_idx = np.array([cmap[c] for c in ref_codes])
        ref_panel_full[:, col_idx] = ref_panel
        mine = qff.compute_factor(inp, e)
        a = mine[:, pick_stocks].astype(np.float64)
        b = ref_panel_full[:, pick_stocks]
        m = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a[m] - b[m])
        denom = np.maximum(np.abs(b[m]), 1e-9)
        rel = diff / denom
        # 相关系数
        if m.sum() > 10:
            corr = np.corrcoef(a[m], b[m])[0, 1]
        else:
            corr = np.nan
        rows.append({
            "factor": e["factor_key"], "base": e["base"], "transform": e["transform"],
            "win": e["window"], "n": int(m.sum()),
            "max_abs": float(diff.max()) if len(diff) else np.nan,
            "p99_abs": float(np.percentile(diff, 99)) if len(diff) else np.nan,
            "p99_rel": float(np.percentile(rel, 99)) if len(rel) else np.nan,
            "corr": corr,
        })
        print(f"  {e['factor_key']:24s} {e['base']:16s} {e['transform']:14s} w={e['window']:2d} "
              f"n={m.sum():8d} p99|diff|={rows[-1]['p99_abs']:.5f} corr={corr:.6f}")
    res = pd.DataFrame(rows)
    print("\n汇总:")
    print(res[["factor", "p99_abs", "corr"]].to_string(index=False))
    ok = (res["corr"] > 0.9995).all()
    print("\n等价性:", "PASS" if ok else "CHECK")


if __name__ == "__main__":
    main()
