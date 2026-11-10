# -*- coding: utf-8 -*-
"""最终验证：BOP 加入右侧模型 wf_rs10 的实际效果。

基线：wf_rs10（6 个 pullback 因子 + above_ma60 选股域）
试验：6 因子 + BOP（按 ICIR 分配权重）

对比：全期 / 验证期 / 样本外 的 CAGR / Sharpe / MaxDD
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, metrics as qfm_metrics  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci  # noqa: E402

OUT = ROOT / "build" / "eval" / "bop_rs_final.json"

md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(C.DEV_END))

O = md.price_open.astype(np.float64)
H = md.high.astype(np.float64)
L = md.low.astype(np.float64)
Cl = md.close.astype(np.float64)
rng = np.maximum(H - L, 1e-9)
bop1 = np.where(rng > 0, (Cl - O) / rng, np.nan)
bop10 = pd.DataFrame(bop1).rolling(10, min_periods=5).mean().to_numpy().astype(np.float32)

# above_ma60 掩码
ma60 = pd.DataFrame(Cl).rolling(60, min_periods=40).mean().to_numpy()
above60 = np.isfinite(Cl) & np.isfinite(ma60) & (Cl > ma60)
mask_above = mask & above60[rows]

base = qfm.load_result(C.BUILD_MODELS / "wf_rs10")
print(f"基线 wf_rs10：{len(base.pool)} 个因子，universe={base.cfg.universe}")

scores_base = np.asarray(base.scores)
if scores_base.shape[0] < md.T:
    scores_base = np.vstack([scores_base,
                             np.full((md.T - scores_base.shape[0], md.N), np.nan, np.float32)])
elif scores_base.shape[0] > md.T:
    scores_base = scores_base[:md.T]

# BOP 在 above_ma60 口径下的 ICIR
print("\n[1/3] 计算 BOP 在 above_MA60 口径的 IC…", flush=True)
f_bop = qfm.build_feature_block(md, rows, {"BOP10": bop10[rows]}, "BOP10", mask_above)
ic_bop = qfm.weekly_ic(f_bop, label, mask_above)
v = np.isfinite(ic_bop)
dev_ic = float(np.mean(ic_bop[v & dev]))
icir = dev_ic / max(float(np.std(ic_bop[v & dev])), 1e-9)
print(f"  DEV IC {dev_ic:+.4f} | ICIR {icir:+.3f}")
w_bop = float(np.clip(icir * 0.03, 0.0, 0.15))
print(f"  → BOP 权重 {w_bop:.3f}")

# BOP 周频 z 分（above_ma60 内中性化）
print("[2/3] 计算 BOP z 分…", flush=True)
bop_z = np.full((md.T, md.N), np.nan, np.float32)
for wi, t in enumerate(rows):
    row = f_bop[wi]
    m = mask_above[wi]
    ok = np.isfinite(row) & m
    if ok.sum() > 30:
        r = qfm.neutralize_rows(row[None, :], md.industry[[t]], md.logmcap[[t]], m[None, :])
        z = qfm.zscore_rows(r, m[None, :])[0]
        bop_z[t] = z

# 合成新打分（只在 above_ma60 内混合）
scores_new = scores_base.copy()
mz = np.isfinite(bop_z) & np.isfinite(scores_base)
scores_new[mz] = (1 - w_bop) * scores_base[mz] + w_bop * bop_z[mz]

# 回测
start = int(md.date_index(C.VALID_START))
phase = int(base.rows_weekly[0]) % 5
while start % 5 != phase:
    start += 1
end = md.T - 1
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)

print("[3/3] 回测对比…", flush=True)
res = {}
navs = {}
for tag, sc in (("wf_rs10_基线", scores_base), ("wf_rs10_加BOP", scores_new)):
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=(phase - start) % 10,
                              cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    nav = r.nav
    navs[tag] = nav
    row = {}
    for seg, (s, e) in {"full": ("2020-01-08", "2026-10-14"),
                        "val": ("2020-01-08", "2021-12-31"),
                        "oos": ("2022-01-01", "2026-10-14")}.items():
        sub = nav[(nav.index >= s) & (nav.index <= e)]
        p = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        row[seg] = {k: float(p[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}
    res[tag] = row
    print(f"\n  [{tag}]")
    for seg in ("full", "val", "oos"):
        p = row[seg]
        print(f"    {seg:5s} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")

# 显著性
common = navs["wf_rs10_基线"].index.intersection(navs["wf_rs10_加BOP"].index)
diff = (navs["wf_rs10_加BOP"][common].pct_change() -
        navs["wf_rs10_基线"][common].pct_change()).dropna().to_numpy()
ci = circular_block_bootstrap_ci(diff, lambda x: float(np.mean(x) * 252),
                                 n_boot=2000, block_len=21)
t = float(np.mean(diff) / np.std(diff, ddof=1) * np.sqrt(len(diff)))

print("\n" + "=" * 90)
print("差异显著性")
print("=" * 90)
print(f"  年化收益差: {np.mean(diff)*252*100:+.2f}%")
print(f"  95% CI: [{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%]")
print(f"  t = {t:.2f}")
d_full = (res["wf_rs10_加BOP"]["full"]["cagr"] - res["wf_rs10_基线"]["full"]["cagr"]) * 100
d_oos = (res["wf_rs10_加BOP"]["oos"]["cagr"] - res["wf_rs10_基线"]["oos"]["cagr"]) * 100
print(f"\n  全期 CAGR: {d_full:+.2f}pp | 样本外: {d_oos:+.2f}pp")

res["meta"] = {"bop_ic_above60": dev_ic, "icir": icir, "w_bop": w_bop,
               "diff_annual": float(np.mean(diff) * 252),
               "ci": [float(ci["lo"]), float(ci["hi"])], "t": t}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
