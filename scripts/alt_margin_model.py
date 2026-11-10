# -*- coding: utf-8 -*-
"""融资融券模型：严格 walk-forward 验证。

三个模型对比：
  M0 = wf_h10（基线，34 量价）
  M1 = wf_margin（纯融资融券 6 因子）
  M2 = wf_h10_margin（量价 + 融资融券合池）
另外追加：
  M3 = wf_h10_flow_margin（量价 + 资金流 + 融资融券）
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
OUT = ALT / "margin_model_result.json"
S, E = "2020-01-08", "2026-10-14"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
n_w = len(rows)
phase = int(rows[0]) % 5
dev = np.asarray(dates_w <= pd.Timestamp("2019-12-31"))

# ---------------- 1) 重建融资融券因子
print("[1/4] 载入融资融券因子…")
zp = np.load(ALT / "margin_panel.npz")
f_bal, f_rzjm, f_rq = zp["bal"], zp["rzjm"], zp["rq"]
f_total, f_buy = zp["total"], zp["buy"]
assert len(zp["dates"]) == md.T

amt = md.amount.astype(np.float64)
mc = md.mcap.astype(np.float64)


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.sum()).to_numpy()


def pct_chg(a, w):
    df = pd.DataFrame(a)
    return (df / df.shift(w) - 1.0).to_numpy()


with np.errstate(divide="ignore", invalid="ignore"):
    bal_ratio = np.where(mc > 0, f_bal / mc, np.nan)
    rzjm_ratio = np.where(amt > 0, f_rzjm / amt, np.nan)
    total_ratio = np.where(f_total > 0, f_bal / f_total, np.nan)
    buy_ratio = np.where(amt > 0, f_buy / amt, np.nan)

fac = {
    "mgn:BAL_RATIO": bal_ratio,
    "mgn:BAL_RATIO_MA5": roll(bal_ratio, 5),
    "mgn:BAL_CHG5": pct_chg(f_bal, 5),
    "mgn:BAL_CHG20": pct_chg(f_bal, 20),
    "mgn:RZJM_5": roll(rzjm_ratio, 5, "sum"),
    "mgn:RZJM_20": roll(rzjm_ratio, 20, "sum"),
    "mgn:BAL_TREND": roll(bal_ratio, 5) - roll(bal_ratio, 20),
    "mgn:RQ_CHG5": pct_chg(f_rq, 5),
    "mgn:FIN_RATIO": total_ratio,
    "mgn:BUY_RATIO": buy_ratio,
}
fac = {k: np.asarray(v, np.float32) for k, v in fac.items()}
fac_w = {k: v[rows] for k, v in fac.items()}

# DEV 期筛选（只在这里用 DEV）
mgn_ok = []
for k, pan in fac_w.items():
    f = qfm.build_feature_block(md, rows, {k: pan}, k, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    d = ic[v & dev]
    if len(d) < 15:
        continue
    m = float(np.mean(d))
    icir = m / max(float(np.std(d)), 1e-9)
    if abs(m) > 0.015 and abs(icir) > 0.25:
        mgn_ok.append(k)
print(f"  DEV 达标 {len(mgn_ok)} 个: {mgn_ok}")

cat_mgn = pd.DataFrame({
    "lib": ["mgn"] * len(mgn_ok), "factor_id": list(range(len(mgn_ok))),
    "slug": mgn_ok, "name_cn": mgn_ok, "name_en": mgn_ok,
    "category_cn": "融资融券", "category_en": "margin",
    "base": ["margin"] * len(mgn_ok), "transform": ["none"] * len(mgn_ok),
    "window": [20] * len(mgn_ok), "norm": ["cs_rank"] * len(mgn_ok),
    "params": ["{}"] * len(mgn_ok), "path": [""] * len(mgn_ok),
    "factor_key": mgn_ok,
})

# ---------------- 2) 建模
print("\n[2/4] 建模…")

# M1 纯融资融券
cfg1 = qfm.ModelConfig(horizon=10, method="icir", max_factors=8,
                       corr_max=0.85, train_weeks=156, score_smooth_alpha=0.6)
res1 = qfm.walk_forward(md, rows, {k: fac_w[k] for k in mgn_ok}, cat_mgn,
                        cfg=cfg1, start_date="2020-01-01", verbose=False)
qfm.save_result(res1, C.BUILD_MODELS / "wf_margin")
print(f"  wf_margin: pool={len(res1.pool)}")

# 全部基础因子
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
        cat_all = pd.concat([cat_all, ec[ec["factor_key"].isin(panels_all)]],
                            ignore_index=True)
base_panels = dict(panels_all)
base_cat = cat_all.copy()

# M2 量价 + 融资融券
p2 = dict(base_panels)
p2.update({k: fac_w[k] for k in mgn_ok})
c2 = pd.concat([base_cat, cat_mgn], ignore_index=True)
c2 = c2[c2["factor_key"].isin(p2)]
cfg2 = qfm.ModelConfig(horizon=10, method="blend", max_factors=42,
                       corr_max=0.70, train_weeks=156, score_smooth_alpha=0.6,
                       fdr_q=0.05)
res2 = qfm.walk_forward(md, rows, p2, c2, cfg=cfg2, start_date="2020-01-01", verbose=False)
qfm.save_result(res2, C.BUILD_MODELS / "wf_h10_margin")
print(f"  wf_h10_margin: pool={len(res2.pool)} | "
      f"融资融券因子 {[k for k in res2.pool if k.startswith('mgn:')]}")

# M3 三者合一
flow_weekly = {}
fp = ALT / "flow_panel.npz"
if fp.exists():
    zf2 = np.load(fp)
    net_pct, r0_pct, turn = zf2["net_pct"], zf2["r0_pct"], zf2["turnover"]

    def rabs(a, w):
        return pd.DataFrame(np.abs(a)).rolling(w, min_periods=max(2, w // 2)).mean().to_numpy()

    flow_full = {
        "flow:FLOW_MA20": roll(net_pct, 20),
        "flow:FLOW_TURN": roll(net_pct, 20) / np.where(roll(turn, 20) > 0, roll(turn, 20), np.nan),
        "flow:R0_MA20": roll(r0_pct, 20),
        "flow:R0_MA5": roll(r0_pct, 5),
        "flow:TURN_Z": (turn - roll(turn, 20)) / np.where(rabs(turn, 20) > 0, rabs(turn, 20), np.nan),
    }
    flow_weekly = {k: np.asarray(v, np.float32)[rows] for k, v in flow_full.items()}
    cat_flow = pd.DataFrame({
        "lib": ["flow"] * len(flow_weekly), "factor_id": list(range(len(flow_weekly))),
        "slug": list(flow_weekly.keys()), "name_cn": list(flow_weekly.keys()),
        "name_en": list(flow_weekly.keys()),
        "category_cn": "资金流", "category_en": "flow",
        "base": ["flow"] * len(flow_weekly), "transform": ["none"] * len(flow_weekly),
        "window": [20] * len(flow_weekly), "norm": ["cs_rank"] * len(flow_weekly),
        "params": ["{}"] * len(flow_weekly), "path": [""] * len(flow_weekly),
        "factor_key": list(flow_weekly.keys()),
    })
    p3 = dict(p2)
    p3.update(flow_weekly)
    c3 = pd.concat([c2, cat_flow], ignore_index=True)
    c3 = c3[c3["factor_key"].isin(p3)]
    cfg3 = qfm.ModelConfig(horizon=10, method="blend", max_factors=46,
                           corr_max=0.70, train_weeks=156, score_smooth_alpha=0.6,
                           fdr_q=0.05)
    res3 = qfm.walk_forward(md, rows, p3, c3, cfg=cfg3, start_date="2020-01-01",
                            verbose=False)
    qfm.save_result(res3, C.BUILD_MODELS / "wf_h10_margin_flow")
    print(f"  wf_h10_margin_flow: pool={len(res3.pool)}")

# ---------------- 3) 回测
print("\n[3/4] 回测…")
start = int(md.date_index(S))
while start % 5 != phase:
    start += 1
end = int(md.date_index(E))
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)


def run(model_dir, tag):
    p = model_dir / "composite_score.npz"
    if not p.exists():
        print(f"  [{tag}] 模型不存在")
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


R = {}
R["baseline"] = run(C.BUILD_MODELS / "wf_h10", "wf_h10（基线）")
R["margin"] = run(C.BUILD_MODELS / "wf_margin", "wf_margin（纯融资融券）")
R["h10_margin"] = run(C.BUILD_MODELS / "wf_h10_margin", "wf_h10_margin（量价+两融）")
R["h10_margin_flow"] = run(C.BUILD_MODELS / "wf_h10_margin_flow",
                          "wf_h10_margin_flow（量价+两融+资金流）")

# ---------------- 4) 显著性
print("\n" + "=" * 96)
print("[4/4] 显著性（vs 基线）")
print("=" * 96)
base = R.get("baseline")
if base:
    for tag in ("margin", "h10_margin", "h10_margin_flow"):
        rr = R.get(tag)
        if not rr:
            continue
        common = base[0].index.intersection(rr[0].index)
        common = common[(common >= S) & (common <= E)]
        d = (rr[0][common].pct_change() - base[0][common].pct_change()).dropna().to_numpy()
        ci = circular_block_bootstrap_ci(d, lambda x: float(np.mean(x) * 252),
                                         n_boot=2000, block_len=21)
        t = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(len(d)))
        print(f"  {tag:<20} 年化差 {np.mean(d)*252*100:+6.2f}% | "
              f"95%CI [{ci['lo']*100:+6.2f}%, {ci['hi']*100:+6.2f}%] | t={t:+5.2f} "
              f"→ {'✅显著' if abs(t) > 2 else '不显著'}")

result = {
    "margin_factors_ok": mgn_ok,
    "wf_margin_pool": list(res1.pool),
    "wf_h10_margin_pool": list(res2.pool),
    "h10_margin_mgn_factors": [k for k in res2.pool if k.startswith("mgn:")],
    **{k: (v[1] if v else None) for k, v in R.items()},
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float),
               encoding="utf-8")
print(f"\n→ {OUT}")
