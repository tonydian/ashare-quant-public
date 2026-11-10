# -*- coding: utf-8 -*-
"""最终验证：BOP 加入因子池后，样本外是否真有改善。

方法（严格）：
  1) 基线：现有 wf_h10（34 因子）的历史 scores
  2) 试验：34 因子打分 + BOP 打分（按 ICIR 分配权重，上限 8%）
     合成新 scores → 重新回测
  3) 对比：全期 / 验证期 / 样本外 的 CAGR / Sharpe / MaxDD
  4) 日收益差 bootstrap（CI 是否含 0）
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

OUT = ROOT / "build" / "eval" / "bop_final_test.json"

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

base = qfm.load_result(C.BUILD_MODELS / "wf_h10")
scores_base = np.asarray(base.scores)
if scores_base.shape[0] < md.T:
    scores_base = np.vstack([scores_base,
                             np.full((md.T - scores_base.shape[0], md.N), np.nan, np.float32)])
elif scores_base.shape[0] > md.T:
    scores_base = scores_base[:md.T]

# BOP 因子块 + 每周 z 分（与模型同口径：中性化 + zscore）
print("[1/4] 计算 BOP 特征并标准化…", flush=True)
f_bop = qfm.build_feature_block(md, rows, {"BOP10": bop10[rows]}, "BOP10", mask)
ic_bop = qfm.weekly_ic(f_bop, label, mask)
v = np.isfinite(ic_bop)
dev_ic = float(np.mean(ic_bop[v & dev]))
dev_icir = dev_ic / max(float(np.std(ic_bop[v & dev])), 1e-9)
print(f"  BOP DEV IC {dev_ic:+.4f} | ICIR {dev_icir:+.3f}")

# BOP 周频 z 分（中性化 + 横截面标准化，与模型一致）
print("[2/4] 按周计算 BOP 的 z 分…", flush=True)
bop_z = np.full((md.T, md.N), np.nan, np.float32)
for wi, t in enumerate(rows):
    row = f_bop[wi]
    m = mask[wi]
    ok = np.isfinite(row) & m
    if ok.sum() > 100:
        r = qfm.neutralize_rows(row[None, :], md.industry[[t]], md.logmcap[[t]],
                                m[None, :])
        z = qfm.zscore_rows(r, m[None, :])[0]
        bop_z[t] = z

# 合成：scores = (1-w)*base + w*bop_z（权重按 ICIR 上限 8%）
w_bop = float(np.clip(dev_icir * 0.02, 0.0, 0.08))
print(f"[3/4] 合成新打分：BOP 权重 {w_bop:.4f}（{(1-w_bop)*100:.1f}% 基线 + {w_bop*100:.1f}% BOP）")

scores_new = scores_base.copy()
mz = np.isfinite(bop_z)
scores_new[mz] = (1 - w_bop) * scores_base[mz] + w_bop * bop_z[mz]

# 回测
start = int(md.date_index(C.VALID_START))
phase = int(base.rows_weekly[0]) % 5
while start % 5 != phase:
    start += 1
end = md.T - 1
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)

print("[4/4] 回测对比…", flush=True)
out = {}
navs = {}
for tag, sc in (("baseline_34", scores_base), ("with_bop_35", scores_new)):
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
    out[tag] = row
    print(f"\n  [{tag}]")
    for seg in ("full", "val", "oos"):
        p = row[seg]
        print(f"    {seg:5s} CAGR {p['cagr']:7.2%} Sharpe {p['sharpe']:6.3f} "
              f"MaxDD {p['max_dd']:7.2%} Calmar {p['calmar']:5.2f}")

# 差异显著性
a = navs["baseline_34"].pct_change().dropna()
b = navs["baseline_34"].index
nb = navs["with_bop_35"].pct_change().dropna()
common = navs["baseline_34"].index.intersection(navs["with_bop_35"].index)
diff = (navs["with_bop_35"][common].pct_change() -
        navs["baseline_34"][common].pct_change()).dropna().to_numpy()
ci = circular_block_bootstrap_ci(diff,
                                 lambda x: float(np.mean(x) * 252),
                                 n_boot=2000, block_len=21)
t = float(np.mean(diff) / np.std(diff, ddof=1) * np.sqrt(len(diff)))

print("\n" + "=" * 90)
print("差异显著性（BOP 版本 - 基线）")
print("=" * 90)
print(f"  年化收益差点估计: {np.mean(diff)*252*100:+.2f}%")
print(f"  95% CI: [{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%] / 年")
print(f"  t 统计量: {t:.2f}")
print(f"  P(Δ>0) ≈ {'>95%' if t > 2 else ('不显著' if abs(t) < 2 else '弱')}")

d_cagr = (out["with_bop_35"]["full"]["cagr"] - out["baseline_34"]["full"]["cagr"]) * 100
d_oos = (out["with_bop_35"]["oos"]["cagr"] - out["baseline_34"]["oos"]["cagr"]) * 100
print(f"\n  CAGR 变化: {d_cagr:+.2f}pp | 样本外变化: {d_oos:+.2f}pp")

out["meta"] = {"bop_dev_ic": dev_ic, "bop_icir": dev_icir, "w_bop": w_bop,
               "diff_annual": float(np.mean(diff) * 252),
               "ci": [float(ci["lo"]), float(ci["hi"])], "t": t}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
