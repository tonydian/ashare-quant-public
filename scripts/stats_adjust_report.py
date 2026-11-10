# -*- coding: utf-8 -*-
"""用真实数据做两项统计校正：
  1) 标签重叠 → IC / Sharpe 的 Newey-West t 与 block bootstrap 置信区间；
  2) 800 因子 DEV 筛选 → BH-FDR 校正（对比未校正的"期望假阳性"）。
输出 build/eval/stats_adjust.json + 控制台报告。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data"))
sys.path.insert(0, str(ROOT))

from qf import config as C, model as qfm, stats_adjust as qsa  # noqa: E402

OUT = C.BUILD_EVAL / "stats_adjust.json"
res = {}

# ============================================================ 1) 标签重叠
ic = pd.read_parquet(C.BUILD_MODELS / "wf_h10/composite_ic.parquet")
ic_vals = ic["ic"].to_numpy(dtype=np.float64)
ic_vals = ic_vals[np.isfinite(ic_vals)]
row_step = 5          # 周频行步长（交易日）
horizon = int(json.load(open(C.BUILD_MODELS / "wf_h10/model_meta.json", encoding="utf-8"))["cfg"]["horizon"])
lag = int(np.ceil(horizon / row_step))   # =2

print("=" * 72)
print("1) 标签重叠校正：复合分 IC 序列")
print("=" * 72)
print(f"样本：{len(ic_vals)} 周；标签 h={horizon} 交易日；周频步长 {row_step} → 重叠 "
      f"{(horizon - row_step) / horizon:.1%}；NW lag={lag}")
nw = qsa.newey_west_tstat(ic_vals, lag=lag)
print(f"  naive t = {nw['naive_t']:.2f}  →  Newey-West t = {nw['t']:.2f}（lag={nw['lag']}）")
eff = qsa.effective_n(len(ic_vals), row_step, horizon)
t_eff = float(ic_vals.mean() / ic_vals.std(ddof=1) * np.sqrt(eff))
print(f"  有效样本数 n_eff = {eff:.0f}（{len(ic_vals)}×{row_step}/{horizon}）→ t_eff = {t_eff:.2f}")
boot_ic = qsa.circular_block_bootstrap_ci(ic_vals, lambda s: float(np.mean(s)),
                                          n_boot=2000, block_len=max(3, horizon // row_step))
print(f"  block bootstrap 95% CI: mean IC [{boot_ic['lo']:.4f}, {boot_ic['hi']:.4f}]")
res["ic_overlap"] = {"n": len(ic_vals), "mean_ic": float(ic_vals.mean()),
                     "naive_t": nw["naive_t"], "newey_west_t": nw["t"], "nw_lag": lag,
                     "effective_n": eff, "t_effective": t_eff,
                     "bootstrap_ci": [boot_ic["lo"], boot_ic["hi"]],
                     "overlap_pct": (horizon - row_step) / horizon}

# ---- Sharpe 的 Newey-West / bootstrap
nav = pd.read_parquet(C.BUILD_RUNS / "main/nav.parquet")["nav"].astype(float)
ret = nav.pct_change().dropna().to_numpy()
sh = qsa.sharpe_ratio(ret, rf=C.CASH_YIELD)
print("\n策略日收益（build/runs/main）")
print(f"  Sharpe(年化) = {sh:.3f}；naive t = {sh * np.sqrt(len(ret) / 252):.2f}（≈Sharpe×√years）")
nw_sh = qsa.newey_west_tstat(ret, lag=5)   # 日频收益 NW(5)
print(f"  日收益均值 NW t = {nw_sh['t']:.2f}（naive {nw_sh['naive_t']:.2f}）")
boot_sh = qsa.circular_block_bootstrap_ci(ret, lambda s: qsa.sharpe_ratio(s, rf=C.CASH_YIELD),
                                          n_boot=2000, block_len=21)
print(f"  Sharpe block bootstrap 95% CI: [{boot_sh['lo']:.3f}, {boot_sh['hi']:.3f}]")
res["sharpe"] = {"sharpe": sh, "naive_t": sh * np.sqrt(len(ret) / 252),
                 "mean_nw_t": nw_sh["t"], "bootstrap_ci": [boot_sh["lo"], boot_sh["hi"]],
                 "n_days": len(ret)}

# ============================================================ 2) BH-FDR
print()
print("=" * 72)
print("2) 800 因子多重筛选校正（DEV 期）")
print("=" * 72)
scr = pd.read_parquet(C.BUILD_EVAL / "factor_metrics_screen.parquet")
d = scr[scr.lib == "directional-alpha"] if False else scr  # 全部 800 个候选
dev_ic = d["dev_ic_5d"].to_numpy(dtype=np.float64)
dev_n = d["dev_ic_n"].to_numpy(dtype=np.float64)
dev_icir = d["dev_icir_5d"].to_numpy(dtype=np.float64)
valid = np.isfinite(dev_ic) & np.isfinite(dev_icir) & (dev_n > 0)
tstat = dev_icir * np.sqrt(dev_n)
pvals = np.array([qsa.p_value_two_sided(t) for t in tstat])

for q in (0.05, 0.10):
    r = qsa.bh_fdr(np.where(valid, pvals, np.nan), q=q)
    print(f"  q={q:.2f}: 800 个候选中 BH-FDR 判显著 {r['k']} 个"
          f"（单侧 5% 未校正期望假阳性约 {int(round(int(valid.sum()) * 0.05))} 个）")
    if q == 0.05:
        res["bh_fdr_005"] = {"k": r["k"], "n_tests": r["n_tests"],
                             "threshold": r["threshold"],
                             "expected_false_pos": int(round(int(valid.sum()) * 0.05))}

# 入池 34 个因子在 BH-FDR 下的命运
pool = json.load(open(C.BUILD_MODELS / "wf_h10/model_meta.json", encoding="utf-8"))["pool"]
pm = scr.set_index("factor_key")
in_pool = [k for k in pool if k in pm.index]
p_pool = np.array([qsa.p_value_two_sided(pm.loc[k, "dev_icir_5d"] * np.sqrt(pm.loc[k, "dev_ic_n"]))
                   for k in in_pool])
r05 = qsa.bh_fdr(np.where(valid, pvals, np.nan), q=0.05)
pool_sig = 0
for k, pv in zip(in_pool, p_pool):
    i = int(np.flatnonzero(scr["factor_key"].to_numpy() == k)[0])
    pool_sig += int(r05["reject"][i])
print(f"  现行入池 {len(pool)} 个因子中，BH-FDR(q=0.05) 仍显著：{pool_sig} 个")
res["pool_bh_fdr"] = {"pool_size": len(pool), "significant_after_bh": int(pool_sig)}

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
