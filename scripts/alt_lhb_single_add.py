# -*- coding: utf-8 -*-
"""单独测试 lhb:HIT_20（龙虎榜上榜频率）加入主模型的边际贡献。

为什么只加这 1 个：
  * 它是全部另类因子中唯一"三期 IC 都稳定"的（DEV -0.0336 / 验证 -0.0310 / 样本外 -0.0380）
  * 逻辑清晰：上榜 = 炒作高峰 = 未来下跌
  * 正交性中等（0.269），与现有因子不重复

对比：
  A = wf_h10（34 量价）
  B = wf_h10（34 量价）+ lhb:HIT_20（35 因子）
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

from qf import backtest as qfbt, config as C, data as qfdata, factors as qff, model as qfm, metrics as qfm_metrics  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci  # noqa: E402

ALT = ROOT / "build" / "alt"
S, E = "2020-01-08", "2026-10-14"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
n_w = len(rows)
phase = int(rows[0]) % 5

# ---------------- 构造 lhb:HIT_20
lhb = np.load(ALT / "billboard_panel.npz")
hit = lhb["hit"]
hit20 = pd.DataFrame(hit).rolling(20, min_periods=10).sum().to_numpy().astype(np.float32)
print(f"lhb:HIT_20 非零比例 {np.count_nonzero(hit20)/hit20.size:.2%}")

# ---------------- 基础因子
zf = np.load(C.BUILD_FACTORS / "weekly_top.npz")
panels = {k: zf[k] for k in zf.files}
cat = qff.load_catalog()
for ep, cp in ((C.BUILD_FACTORS / "momentum_extra.npz",
                C.BUILD_FACTORS / "momentum_catalog.parquet"),):
    if ep.exists() and cp.exists():
        ze = np.load(ep)
        ec = pd.read_parquet(cp)
        for k in ze.files:
            if k not in panels and ze[k].shape[0] == n_w:
                panels[k] = ze[k]
        cat = pd.concat([cat, ec[ec["factor_key"].isin(panels)]], ignore_index=True)

cat_lhb = pd.DataFrame({
    "lib": ["lhb"], "factor_id": [0], "slug": ["lhb:HIT_20"],
    "name_cn": ["龙虎榜20日上榜次数"], "name_en": ["lhb_hit_20"],
    "category_cn": "龙虎榜", "category_en": "lhb", "base": ["lhb"],
    "transform": ["none"], "window": [20], "norm": ["cs_rank"],
    "params": ["{}"], "path": [""], "factor_key": ["lhb:HIT_20"],
})

panels2 = dict(panels)
panels2["lhb:HIT_20"] = hit20[rows]
cat2 = pd.concat([cat, cat_lhb], ignore_index=True)
cat2 = cat2[cat2["factor_key"].isin(panels2)]

# ---------------- 建模
print("\n[1/3] 建模：34 量价 + 1 龙虎榜…")
cfg = qfm.ModelConfig(horizon=10, method="blend", max_factors=35,
                      corr_max=0.70, train_weeks=156,
                      score_smooth_alpha=0.6, fdr_q=0.05)
res = qfm.walk_forward(md, rows, panels2, cat2, cfg=cfg, start_date="2020-01-01",
                       verbose=False)
out_dir = C.BUILD_MODELS / "wf_h10_lhb"
qfm.save_result(res, out_dir)
print(f"  pool={len(res.pool)} | 含 lhb: {'lhb:HIT_20' in res.pool}")
if "lhb:HIT_20" in res.pool:
    pm = pd.read_parquet(out_dir / "pool_meta.parquet")
    if "lhb:HIT_20" in pm.index:
        print(f"  该因子在池中排名: {list(pm.index).index('lhb:HIT_20')+1} / {len(pm)}")
        print(f"  详情: ic={pm.loc['lhb:HIT_20','ic']:+.4f} "
              f"icir={pm.loc['lhb:HIT_20','icir']:+.3f} "
              f"t={pm.loc['lhb:HIT_20','t']:+.2f}")

# ---------------- 回测
print("\n[2/3] 回测对比…")
start = int(md.date_index(S))
while start % 5 != phase:
    start += 1
end = int(md.date_index(E))
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)


def run(tag):
    p = C.BUILD_MODELS / tag / "composite_score.npz"
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
    d = {}
    for k, (a, b) in {"full": (S, E), "val": (S, "2021-12-31"),
                      "os25": ("2022-01-01", "2025-12-31"),
                      "oos": ("2022-01-01", E)}.items():
        sub = r.nav[(r.nav.index >= a) & (r.nav.index <= b)]
        pp = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        d[k] = {kk: float(pp[kk]) for kk in ("cagr", "sharpe", "max_dd", "calmar")}
    return r.nav, d


R = {}
for tag in ("wf_h10", "wf_h10_lhb"):
    R[tag] = run(tag)

print(f"\n  {'模型':<18}{'全期':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
      f"{'验证20-21':>11}{'样本外22-25':>12}{'样本外22-26':>12}")
print("  " + "-" * 88)
for tag in ("wf_h10", "wf_h10_lhb"):
    d = R[tag][1]
    print(f"  {tag:<18}{d['full']['cagr']:>8.2%}{d['full']['sharpe']:>8.3f}"
          f"{d['full']['max_dd']:>8.2%}{d['full']['calmar']:>8.2f}"
          f"{d['val']['cagr']:>10.2%}{d['os25']['cagr']:>11.2%}"
          f"{d['oos']['cagr']:>11.2%}")

# ---------------- 逐年
print("\n[3/3] 逐年 + 显著性…")
ys = {t: qfm_metrics.yearly_returns(R[t][0][R[t][0].index >= S]) for t in R}
print(f"\n  {'年':>6}{'基线':>12}{'+龙虎榜':>12}{'差':>10}")
print("  " + "-" * 42)
for y in sorted(ys["wf_h10"].index):
    a = float(ys["wf_h10"].loc[y]) if y in ys["wf_h10"].index else 0.0
    b = float(ys["wf_h10_lhb"].loc[y]) if y in ys["wf_h10_lhb"].index else 0.0
    print(f"  {int(y):>6}{a:>11.2%}{b:>11.2%}{(b-a)*100:>+9.2f}pp")

# 显著性
for seg, (a, b) in {"全期": (S, E), "样本外22-25": ("2022-01-01", "2025-12-31"),
                    "验证期": (S, "2021-12-31")}.items():
    common = R["wf_h10"][0].index.intersection(R["wf_h10_lhb"][0].index)
    common = common[(common >= a) & (common <= b)]
    d = (R["wf_h10_lhb"][0][common].pct_change() -
         R["wf_h10"][0][common].pct_change()).dropna().to_numpy()
    ci = circular_block_bootstrap_ci(d, lambda x: float(np.mean(x) * 252),
                                     n_boot=2000, block_len=21)
    t = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(len(d)))
    print(f"\n  {seg}: 年化差 {np.mean(d)*252*100:+.2f}% | "
          f"CI [{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%] | t={t:+.2f}"
          f" → {'✅显著' if abs(t) > 2 else '不显著'}")

(ALT / "lhb_single_add.json").write_text(
    json.dumps({t: R[t][1] for t in R}, ensure_ascii=False, indent=1, default=float),
    encoding="utf-8")
print(f"\n→ {ALT / 'lhb_single_add.json'}")
