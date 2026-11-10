# -*- coding: utf-8 -*-
"""全量另类数据模型：量价 + 两融 + 龙虎榜 + 快照（户数/解禁/预告）+ 资金流。

模型阶梯（逐步加数据，看边际贡献）：
  M0 wf_h10                          34 量价（基线）
  M1 wf_h10_margin                   量价 + 两融
  M2 wf_h10_margin_bb                量价 + 两融 + 龙虎榜
  M3 wf_h10_alt_all                  量价 + 两融 + 龙虎榜 + 快照
  M4 wf_h10_all_plus_flow            全部（含资金流）

验证纪律：
  * 所有另类因子的"选入"只看 DEV 期（2017-2019）
  * walk-forward 权重估计只用 t 之前已实现标签
  * 2020-2021 验证期、2022-2026 样本外不参与任何决策
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
OUT = ALT / "all_alt_result.json"
S, E = "2020-01-08", "2026-10-14"
DEV_END = "2019-12-31"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
n_w = len(rows)
phase = int(rows[0]) % 5
dev = np.asarray(dates_w <= pd.Timestamp(DEV_END))
T, N = md.T, md.N
dates = pd.DatetimeIndex(md.dates)
code_idx = {str(c): j for j, c in enumerate(md.codes)}
date_pos = {d.normalize(): i for i, d in enumerate(dates)}

print("=" * 96)
print("全量另类数据模型构建")
print("=" * 96)


def next_idx(d):
    d = pd.Timestamp(d).normalize()
    if d in date_pos:
        return date_pos[d]
    for i, dd in enumerate(dates):
        if dd.normalize() >= d:
            return i
    return None


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.sum()).to_numpy()


def pct_chg(a, w):
    df = pd.DataFrame(a)
    return (df / df.shift(w) - 1.0).to_numpy()


# ============================================================
# 1) 载入所有另类因子面板
print("\n[1/6] 载入各数据源因子…")
fac_all = {}

# --- 两融
zp = np.load(ALT / "margin_panel.npz")
f_bal, f_rzjm, f_rq = zp["bal"], zp["rzjm"], zp["rq"]
f_total, f_buy = zp["total"], zp["buy"]
amt = md.amount.astype(np.float64)
mc = md.mcap.astype(np.float64)
with np.errstate(divide="ignore", invalid="ignore"):
    bal_ratio = np.where(mc > 0, f_bal / mc, np.nan)
    rzjm_ratio = np.where(amt > 0, f_rzjm / amt, np.nan)
    total_ratio = np.where(f_total > 0, f_bal / f_total, np.nan)
    buy_ratio = np.where(amt > 0, f_buy / amt, np.nan)
fac_margin = {
    "mgn:BAL_RATIO": bal_ratio,
    "mgn:BAL_RATIO_MA5": roll(bal_ratio, 5),
    "mgn:BAL_CHG5": pct_chg(f_bal, 5),
    "mgn:BAL_CHG20": pct_chg(f_bal, 20),
    "mgn:FIN_RATIO": total_ratio,
    "mgn:BUY_RATIO": buy_ratio,
}
print(f"  两融: {len(fac_margin)}")

# --- 龙虎榜
lhb = np.load(ALT / "billboard_panel.npz")
hit = lhb["hit"]
fac_lhb = {"lhb:HIT_20": roll(hit, 20, "sum")}
print(f"  龙虎榜: {len(fac_lhb)}")

# --- 快照（户数/解禁/预告）
snap = np.load(ALT / "snapshot_panel.npz")
fac_snap = {
    "alt:HOLDER_CHG": snap["holder_chg"],
    "alt:HOLDER_AVGMC": snap["holder_avgmc"],
    "alt:FCST_GROWTH": snap["fcst_growth"],
    "alt:FCST_POS": snap["fcst_pos"],
}
print(f"  快照: {len(fac_snap)}")

# --- 资金流
fac_flow = {}
fp = ALT / "flow_panel.npz"
if fp.exists():
    zf2 = np.load(fp)
    net_pct, r0_pct, turn = zf2["net_pct"], zf2["r0_pct"], zf2["turnover"]

    def rabs(a, w):
        return pd.DataFrame(np.abs(a)).rolling(w, min_periods=max(2, w // 2)).mean().to_numpy()

    fac_flow = {
        "flow:FLOW_MA20": roll(net_pct, 20),
        "flow:R0_MA20": roll(r0_pct, 20),
        "flow:R0_MA5": roll(r0_pct, 5),
        "flow:FLOW_TURN": roll(net_pct, 20) / np.where(roll(turn, 20) > 0, roll(turn, 20), np.nan),
        "flow:TURN_Z": (turn - roll(turn, 20)) / np.where(rabs(turn, 20) > 0, rabs(turn, 20), np.nan),
    }
print(f"  资金流: {len(fac_flow)}")

# ============================================================
# 2) DEV 期筛选（每个因子集独立筛，只看 DEV）
print("\n[2/6] DEV 期筛选…")


def dev_screen(d: dict, min_ic=0.012, min_icir=0.20) -> list[str]:
    ok = []
    for k, pan in d.items():
        pan32 = np.asarray(pan, np.float32)
        f = qfm.build_feature_block(md, rows, {k: pan32[rows]}, k, mask)
        ic = qfm.weekly_ic(f, label, mask)
        v = np.isfinite(ic)
        dd = ic[v & dev]
        if len(dd) < 15:
            continue
        m = float(np.mean(dd))
        icir = m / max(float(np.std(dd)), 1e-9)
        if abs(m) > min_ic and abs(icir) > min_icir:
            ok.append(k)
    return ok


sel = {
    "margin": dev_screen(fac_margin),
    "lhb": dev_screen(fac_lhb, min_ic=0.010, min_icir=0.15),
    "snap": dev_screen(fac_snap, min_ic=0.010, min_icir=0.15),
    "flow": dev_screen(fac_flow),
}
for k, v in sel.items():
    print(f"  {k}: {v}")

# ============================================================
# 3) 基础因子
print("\n[3/6] 载入基础量价因子…")
zf = np.load(C.BUILD_FACTORS / "weekly_top.npz")
base_panels = {k: zf[k] for k in zf.files}
base_cat = qff.load_catalog()
for ep, cp in ((C.BUILD_FACTORS / "momentum_extra.npz",
                C.BUILD_FACTORS / "momentum_catalog.parquet"),):
    if ep.exists() and cp.exists():
        ze = np.load(ep)
        ec = pd.read_parquet(cp)
        for k in ze.files:
            if k not in base_panels and ze[k].shape[0] == n_w:
                base_panels[k] = ze[k]
        base_cat = pd.concat([base_cat, ec[ec["factor_key"].isin(base_panels)]],
                             ignore_index=True)
print(f"  基础因子 {len(base_panels)}")


def mk_cat(keys, tag):
    return pd.DataFrame({
        "lib": [tag] * len(keys), "factor_id": list(range(len(keys))),
        "slug": keys, "name_cn": keys, "name_en": keys,
        "category_cn": tag, "category_en": tag,
        "base": [tag] * len(keys), "transform": ["none"] * len(keys),
        "window": [20] * len(keys), "norm": ["cs_rank"] * len(keys),
        "params": ["{}"] * len(keys), "path": [""] * len(keys),
        "factor_key": keys,
    })


# ============================================================
# 4) 逐级建模
print("\n[4/6] 建模（逐级加数据）…")
MODELS = []


def build(tag, extra_sets, max_f):
    panels = dict(base_panels)
    cat = base_cat.copy()
    for name, keys, d in extra_sets:
        if not keys:
            continue
        for k in keys:
            panels[k] = np.asarray(d[k], np.float32)[rows]
        cat = pd.concat([cat, mk_cat(keys, name)], ignore_index=True)
    cat = cat[cat["factor_key"].isin(panels)]
    cfg = qfm.ModelConfig(horizon=10, method="blend", max_factors=max_f,
                          corr_max=0.70, train_weeks=156,
                          score_smooth_alpha=0.6, fdr_q=0.05)
    r = qfm.walk_forward(md, rows, panels, cat, cfg=cfg, start_date="2020-01-01",
                         verbose=False)
    d = C.BUILD_MODELS / tag
    qfm.save_result(r, d)
    n_alt = len([k for k in r.pool if ":" in k and not k.startswith(
        ("volume-stat:", "directional:", "risk-pattern:", "momentum-extra:"))])
    print(f"  {tag}: pool={len(r.pool)}（另类 {n_alt} 个）")
    MODELS.append((tag, len(r.pool), n_alt))
    return r


build("wf_h10_m_all2", [("margin", sel["margin"], fac_margin),
                        ("lhb", sel["lhb"], fac_lhb)], 44)
build("wf_h10_m_all3", [("margin", sel["margin"], fac_margin),
                        ("lhb", sel["lhb"], fac_lhb),
                        ("snap", sel["snap"], fac_snap)], 48)
build("wf_h10_m_all4", [("margin", sel["margin"], fac_margin),
                        ("lhb", sel["lhb"], fac_lhb),
                        ("snap", sel["snap"], fac_snap),
                        ("flow", sel["flow"], fac_flow)], 52)

# ============================================================
# 5) 回测
print("\n[5/6] 回测…")
start = int(md.date_index(S))
while start % 5 != phase:
    start += 1
end = int(md.date_index(E))
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, C.CASH_BOND_WEIGHT, C.CASH_GOLD_WEIGHT)


def run(tag):
    p = C.BUILD_MODELS / tag / "composite_score.npz"
    if not p.exists():
        return None
    sc = np.load(p)["scores"]
    if sc.shape[0] < T:
        sc = np.vstack([sc, np.full((T - sc.shape[0], N), np.nan, np.float32)])
    elif sc.shape[0] > T:
        sc = sc[:T]
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
    for seg, (a, b) in {"full": (S, E), "val": (S, "2021-12-31"),
                        "os25": ("2022-01-01", "2025-12-31"),
                        "oos": ("2022-01-01", E)}.items():
        sub = r.nav[(r.nav.index >= a) & (r.nav.index <= b)]
        pp = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        out[seg] = {k: float(pp[k]) for k in ("cagr", "sharpe", "max_dd", "calmar")}
    return r.nav, out


ALL = ["wf_h10", "wf_h10_margin", "wf_h10_m_all2", "wf_h10_m_all3", "wf_h10_m_all4"]
R = {}
print(f"\n  {'模型':<22}{'全期':>9}{'Sharpe':>8}{'MaxDD':>9}{'2020-25':>10}"
      f"{'样本外22-25':>12}{'样本外22-26':>12}")
print("  " + "-" * 84)
for tag in ALL:
    rr = run(tag)
    if not rr:
        print(f"  {tag:<22} 不存在")
        continue
    R[tag] = rr
    o = rr[1]
    print(f"  {tag:<22} 全期 {o['full']['cagr']:>7.2%} Sharpe {o['full']['sharpe']:>6.3f} "
          f"MaxDD {o['full']['max_dd']:>7.2%} 样本外22-25 {o['os25']['cagr']:>7.2%}")

print("\n" + "=" * 96)
print("汇总表")
print("=" * 96)
print(f"{'模型':<24}{'全期CAGR':>10}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
      f"{'样本外22-25':>12}{'样本外22-26':>12}")
print("-" * 96)
for tag in ALL:
    if tag not in R:
        continue
    o = R[tag][1]
    print(f"{tag:<24}{o['full']['cagr']:>9.2%}{o['full']['sharpe']:>8.3f}"
          f"{o['full']['max_dd']:>9.2%}{o['full']['calmar']:>8.2f}"
          f"{o['os25']['cagr']:>11.2%}{o['oos']['cagr']:>11.2%}")

# ============================================================
# 6) 显著性
print("\n" + "=" * 96)
print("[6/6] 显著性（vs wf_h10 基线，样本外 2022-2025）")
print("=" * 96)
base = R.get("wf_h10")
if base:
    for tag in ALL[1:]:
        if tag not in R:
            continue
        common = base[0].index.intersection(R[tag][0].index)
        common = common[(common >= "2022-01-01") & (common <= "2025-12-31")]
        d = (R[tag][0][common].pct_change() -
             base[0][common].pct_change()).dropna().to_numpy()
        ci = circular_block_bootstrap_ci(d, lambda x: float(np.mean(x) * 252),
                                         n_boot=2000, block_len=21)
        t = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(len(d)))
        print(f"  {tag:<24} 年化差 {np.mean(d)*252*100:+6.2f}% | "
              f"CI [{ci['lo']*100:+6.2f}%, {ci['hi']*100:+6.2f}%] | t={t:+5.2f} "
              f"→ {'✅' if abs(t) > 2 else '不显著'}")

result = {tag: R[tag][1] for tag in R}
result["_selection"] = sel
result["_pools"] = {t: (p, a) for t, p, a in MODELS}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float),
               encoding="utf-8")
print(f"\n→ {OUT}")
