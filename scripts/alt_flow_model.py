# -*- coding: utf-8 -*-
"""C 方案第三步：把资金流因子并入模型，严格 walk-forward 验证。

设计（严格遵守只看 DEV 期）：
  1) 因子池 = 主模型 34 因子 + 资金流 7 个达标因子（方向由 DEV 期 IC 符号决定）
     —— 但这里做**对照实验**，避免污染主模型：
        M0 = wf_h10（基线，34 量价因子）
        M1 = wf_h10 + 7 资金流因子（38~41 因子，walk-forward 重新选池）
        M2 = 纯资金流因子（7 个，看单独有效性）
  2) walk-forward 用 qfm.walk_forward（与主模型同一套流程，quarterly refit）
  3) 对比全期/验证期/样本外，含 bootstrap 显著性
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

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, factors as qff, metrics as qfm_metrics  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci  # noqa: E402

ALT = ROOT / "build" / "alt"
OUT = ALT / "flow_model_result.json"
S, E = "2020-01-08", "2026-10-14"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
n_w = len(rows)
phase = int(rows[0]) % 5

# ---------------- 载入资金流因子面板
print("[1/4] 载入资金流因子…")
zp = np.load(ALT / "flow_panel.npz")
dates_flow = pd.DatetimeIndex(zp["dates"])
net_pct, r0_pct, turn = zp["net_pct"], zp["r0_pct"], zp["turnover"]
assert len(dates_flow) == md.T, f"日期轴不一致 {len(dates_flow)} vs {md.T}"


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.sum()).to_numpy()


def roll_absmean(a, w):
    return pd.DataFrame(np.abs(a)).rolling(w, min_periods=max(2, w // 2)).mean().to_numpy()


flow_fac = {
    "flow:FLOW_MA5": roll(net_pct, 5),
    "flow:FLOW_MA20": roll(net_pct, 20),
    "flow:R0_MA5": roll(r0_pct, 5),
    "flow:R0_MA20": roll(r0_pct, 20),
    "flow:FLOW_TURN": roll(net_pct, 20) / np.where(roll(turn, 20) > 0, roll(turn, 20), np.nan),
    "flow:TURN_Z": (turn - roll(turn, 20)) / np.where(roll_absmean(turn, 20) > 0,
                                                     roll_absmean(turn, 20), np.nan),
    "flow:FLOW_SURGE": np.where(roll_absmean(net_pct, 20) > 0.05,
                                net_pct / roll_absmean(net_pct, 20), np.nan),
}
for k in flow_fac:
    flow_fac[k] = np.asarray(flow_fac[k], dtype=np.float32)

# 周频化（取 rows 对应的那些天）
flow_weekly = {k: v[rows] for k, v in flow_fac.items()}
print(f"  资金流因子 {len(flow_weekly)} 个")

# ---------------- DEV 期 IC 定方向（只用于判断有效性，方向交给 walk_forward）
print("\n[2/4] DEV 期验证（仅确认有效，不做选择）…")
dev = np.asarray(dates_w <= pd.Timestamp("2019-12-31"))
flow_ok = []
for k, pan in flow_weekly.items():
    f = qfm.build_feature_block(md, rows, {k: pan}, k, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    d = ic[v & dev]
    if len(d) < 10:
        continue
    m = float(np.mean(d))
    icir = m / max(float(np.std(d)), 1e-9)
    if abs(m) > 0.015 and abs(icir) > 0.25:
        flow_ok.append(k)
        print(f"  ✅ {k:<22} DEV IC {m:+.4f} ICIR {icir:+.3f}")
    else:
        print(f"  ❌ {k:<22} DEV IC {m:+.4f} ICIR {icir:+.3f}（不达标）")
print(f"  达标 {len(flow_ok)} 个")

# ---------------- 对照组 1：纯资金流因子模型
print("\n[3/4] 建模（walk-forward）…")
cat_flow = pd.DataFrame({
    "lib": ["flow"] * len(flow_ok),
    "factor_id": list(range(len(flow_ok))),
    "slug": flow_ok,
    "name_cn": flow_ok,
    "name_en": flow_ok,
    "category_cn": "资金流",
    "category_en": "flow",
    "base": ["flow"] * len(flow_ok),
    "transform": ["none"] * len(flow_ok),
    "window": [20] * len(flow_ok),
    "norm": ["cs_rank"] * len(flow_ok),
    "params": ["{}"] * len(flow_ok),
    "path": [""] * len(flow_ok),
    "factor_key": flow_ok,
})

cfg_flow = qfm.ModelConfig(horizon=10, method="icir", max_factors=7,
                           corr_max=0.85, train_weeks=156,
                           score_smooth_alpha=0.6)
res_flow = qfm.walk_forward(md, rows, flow_weekly, cat_flow, cfg=cfg_flow,
                            start_date="2020-01-01", verbose=False)
qfm.save_result(res_flow, C.BUILD_MODELS / "wf_flow")
print(f"  wf_flow: pool={len(res_flow.pool)}")

# ---------------- 对照组 2：主模型 + 资金流（合池重选）
print("\n  [合池模型：34 量价 + 资金流]")
zf = np.load(C.BUILD_FACTORS / "weekly_top.npz")
panels_all = {k: zf[k] for k in zf.files}
cat_all = qff.load_catalog()
for ep, cp in ((C.BUILD_FACTORS / "momentum_extra.npz",
                C.BUILD_FACTORS / "momentum_catalog.parquet"),):
    if ep.exists() and cp.exists():
        ze = np.load(ep)
        ec = pd.read_parquet(cp)
        for k in ze.files:
            if k not in panels_all and ze[k].shape[0] == n_w:
                panels_all[k] = ze[k]
        cat_all = pd.concat([cat_all, ec[ec["factor_key"].isin(panels_all)]], ignore_index=True)
for k, v in flow_weekly.items():
    panels_all[k] = v
cat_all = pd.concat([cat_all, cat_flow], ignore_index=True)
cat_all = cat_all[cat_all["factor_key"].isin(panels_all)]

cfg_mix = qfm.ModelConfig(horizon=10, method="blend", max_factors=40,
                          corr_max=0.70, train_weeks=156,
                          score_smooth_alpha=0.6, fdr_q=0.05)
res_mix = qfm.walk_forward(md, rows, panels_all, cat_all, cfg=cfg_mix,
                           start_date="2020-01-01", verbose=False)
qfm.save_result(res_mix, C.BUILD_MODELS / "wf_h10_flow")
print(f"  wf_h10_flow: pool={len(res_mix.pool)}")
print(f"    池中的资金流因子: {[k for k in res_mix.pool if k.startswith('flow:')]}")

# ---------------- 回测对比
print("\n[4/4] 回测对比…")
start = int(md.date_index(S))
while start % 5 != phase:
    start += 1
end = int(md.date_index(E))
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)


def run(model_dir, tag):
    p = model_dir / "composite_score.npz"
    if not p.exists():
        return None
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
    out = {}
    print(f"\n  [{tag}]")
    for seg, (a, b) in {"full": (S, E), "val": (S, "2021-12-31"),
                        "oos": ("2022-01-01", E)}.items():
        sub = r.nav[(r.nav.index >= a) & (r.nav.index <= b)]
        pp = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        out[seg] = {k: float(pp[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}
        print(f"    {seg:5s} CAGR {pp['cagr']:7.2%} Sharpe {pp['sharpe']:6.3f} "
              f"MaxDD {pp['max_dd']:7.2%} Calmar {pp['calmar']:5.2f}")
    return r.nav, out


r_base = run(C.BUILD_MODELS / "wf_h10", "wf_h10（基线）")
r_flow = run(C.BUILD_MODELS / "wf_flow", "wf_flow（纯资金流）")
r_mix = run(C.BUILD_MODELS / "wf_h10_flow", "wf_h10_flow（量价+资金流）")

# ---------------- 显著性
print("\n" + "=" * 96)
print("显著性检验（vs 基线，日收益差 bootstrap）")
print("=" * 96)
for tag, rr in (("纯资金流", r_flow), ("量价+资金流", r_mix)):
    if rr is None or r_base is None:
        continue
    common = r_base[0].index.intersection(rr[0].index)
    common = common[(common >= S) & (common <= E)]
    d = (rr[0][common].pct_change() - r_base[0][common].pct_change()).dropna().to_numpy()
    ci = circular_block_bootstrap_ci(d, lambda x: float(np.mean(x) * 252),
                                     n_boot=2000, block_len=21)
    t = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(len(d)))
    print(f"  {tag}: 年化差 {np.mean(d)*252*100:+.2f}% | "
          f"95%CI [{ci['lo']*100:+.2f}%, {ci['hi']*100:+.2f}%] | t={t:.2f}")
    print(f"    → {'显著' if abs(t) > 2 else '不显著'}")

result = {
    "flow_factors_ok": flow_ok,
    "wf_flow_pool": list(res_flow.pool),
    "wf_mix_pool": list(res_mix.pool),
    "mix_flow_factors": [k for k in res_mix.pool if k.startswith("flow:")],
    "base": r_base[1] if r_base else None,
    "flow_only": r_flow[1] if r_flow else None,
    "mixed": r_mix[1] if r_mix else None,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
