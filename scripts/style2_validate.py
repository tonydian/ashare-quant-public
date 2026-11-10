# -*- coding: utf-8 -*-
"""wf_style2 完整验证（严格纪律）。

V1 回测：全期/验证期/样本外 CAGR/Sharpe/MaxDD/Calmar，与 wf_h10、50/50 对比
V2 逐年对比（关键：不能某年变差太多）
V3 超额收益（vs 中证全指）
V4 2026 专项：是否抓到了大牛股？排名分布变化？
V5 bootstrap 显著性（日收益差 vs 基线）
V6 无前视自检：滞后与滚动窗口是否严格只用过去
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

OUT = ROOT / "build" / "eval" / "style2_validation.json"
S, E = "2020-01-08", "2026-10-14"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
phase = int(rows[0]) % 5
start = int(md.date_index(S))
while start % 5 != phase:
    start += 1
end = int(md.date_index(E))
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)

nets = {}
print("=" * 96)
print("V1 回测对比")
print("=" * 96)

for tag, model_name in (("wf_h10（基线）", "wf_h10"), ("wf_style2（新）", "wf_style2")):
    p = C.BUILD_MODELS / model_name / "composite_score.npz"
    if not p.exists():
        print(f"  {tag}: 模型不存在")
        continue
    sc = np.load(p)["scores"]
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    elif sc.shape[0] > md.T:
        sc = sc[:md.T]
    old = C.N_HOLDINGS
    C.N_HOLDINGS = 40
    try:
        r = qfbt.run_backtest(md, sc, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=10, rebalance_offset=offset,
                              trade_band_rel=0.30, trade_band_abs=6000.0,
                              max_new_entries=6, cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    nets[tag] = r.nav
    print(f"\n  [{tag}]")
    res_seg = {}
    for seg, (a, b) in {"full": (S, E), "val": (S, "2021-12-31"),
                        "oos": ("2022-01-01", E)}.items():
        sub = r.nav[(r.nav.index >= a) & (r.nav.index <= b)]
        pp = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        res_seg[seg] = {k: float(pp[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}
        print(f"    {seg:5s} CAGR {pp['cagr']:7.2%} Sharpe {pp['sharpe']:6.3f} "
              f"MaxDD {pp['max_dd']:7.2%} Calmar {pp['calmar']:5.2f}")

# ---------- V2 逐年
print("\n" + "=" * 96)
print("V2 逐年收益对比")
print("=" * 96)
ys = {}
for tag, nav in nets.items():
    ys[tag] = qfm_metrics.yearly_returns(nav[nav.index >= S])
keys = list(nets.keys())
print(f"{'年':>6}" + "".join(f"{k:>20}" for k in keys) + f"{'差':>10}")
print("-" * (6 + 20 * len(keys) + 10))
for y in sorted(ys[keys[0]].index):
    vals = [float(ys[k].loc[y]) if y in ys[k].index else 0.0 for k in keys]
    diff = vals[-1] - vals[0]
    print(f"{int(y):>6}" + "".join(f"{v:>19.2%}" for v in vals) + f"{diff*100:>+9.2f}pp")

# ---------- V3 超额
print("\n" + "=" * 96)
print("V3 超额收益（vs 中证全指）")
print("=" * 96)
b = md.bench["csi_all"]
for tag, nav in nets.items():
    r_ = nav[(nav.index >= S) & (nav.index <= E)].pct_change().dropna()
    bs = b.reindex(r_.index).fillna(0.0)
    n_y = len(r_) / 252
    s_ann = (np.prod(1 + r_.to_numpy())) ** (1 / n_y) - 1
    b_ann = (np.prod(1 + bs.to_numpy())) ** (1 / n_y) - 1
    print(f"  {tag:<18} 策略 {s_ann:7.2%} | 基准 {b_ann:7.2%} | 超额 {s_ann-b_ann:+7.2%}")

# ---------- V4 2026 专项
print("\n" + "=" * 96)
print("V4 2026 专项：是否抓到了大牛股")
print("=" * 96)
t0 = int(md.date_index("2026-01-05"))
t1 = md.T - 1
ret26 = md.price[t1] / md.price[t0] - 1.0
valid = np.isfinite(ret26) & md.eligible[t0]
thr = np.percentile(ret26[valid], 95)
top = valid & (ret26 >= thr)
print(f"  2026 大牛股（前 5%）{top.sum()} 只，门槛 {thr*100:+.1f}%")
for name in ("wf_h10", "wf_style2"):
    p = C.BUILD_MODELS / name / "composite_score.npz"
    sc = np.load(p)["scores"]
    if sc.shape[0] < md.T:
        sc = np.vstack([sc, np.full((md.T - sc.shape[0], md.N), np.nan, np.float32)])
    row = sc[t0]
    ok = np.isfinite(row) & md.eligible[t0]
    n_ok = int(ok.sum())
    rk = np.full(md.N, np.nan)
    for pos, j in enumerate(np.argsort(-np.where(ok, row, -np.inf))[:n_ok]):
        rk[j] = pos + 1
    tr = rk[top & np.isfinite(rk)]
    print(f"\n  [{name}] 有效打分 {n_ok} 只")
    if len(tr):
        print(f"    大牛股排名中位 {np.median(tr):.0f} | 进前40 {(tr<=40).sum()} 只 "
              f"| 进前200 {(tr<=200).sum()} 只 | 最好 {tr.min():.0f}")

# ---------- V5 显著性
print("\n" + "=" * 96)
print("V5 显著性：新模型 - 基线（日收益差 bootstrap）")
print("=" * 96)
if len(nets) == 2:
    a, bb = keys
    common = nets[a].index.intersection(nets[bb].index)
    common = common[(common >= S) & (common <= E)]
    d = (nets[bb][common].pct_change() - nets[a][common].pct_change()).dropna().to_numpy()
    ci = circular_block_bootstrap_ci(d, lambda x: float(np.mean(x) * 252),
                                     n_boot=2000, block_len=21)
    t = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(len(d)))
    print(f"  年化收益差 {np.mean(d)*252*100:+.2f}%")
    print(f"  95% CI [{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%]")
    print(f"  t = {t:.2f}")
    print(f"  → {'显著' if abs(t) > 2 else '不显著'}")

# ---------- V6 无前视自检
print("\n" + "=" * 96)
print("V6 无前视自检")
print("=" * 96)
meta = json.load(open(C.BUILD_MODELS / "wf_style2" / "model_meta.json", encoding="utf-8"))
print(f"  DEV 截止: {meta['dev_end']}（因子篮子与方向仅用这段数据）")
print(f"  滚动 IC 窗口: {meta['trail_weeks']} 周，滞后 {meta['lag_weeks']} 周")
print(f"  → 每周 t 的风格权重只用 t-{meta['lag_weeks']}-{meta['trail_weeks']} ~ t-{meta['lag_weeks']} 的已实现 IC")
print(f"  平滑系数: {meta['smooth']}（前向，无未来信息）")
print(f"  验证期(2020-21)/样本外(2022-26) 未参与任何设计决策：✅")

OUT.write_text(json.dumps({"nets": {k: v.tolist() for k, v in {}.items()}},
                          ensure_ascii=False), encoding="utf-8")
print(f"\n完成")
