# -*- coding: utf-8 -*-
"""B 方案第二步：把 size/vol 暴露加入模型，看真实效果。

DEV 期结果（第一步）已经很清楚：
  * 纯市值（sv:SIZE，负对数市值=小市值偏好）：DEV IC -0.0125（小市值在 DEV 期**无效**）
    但 2026 IC +0.0348 → 方向相反，用 2026 选择就是过拟合
  * 波动率 sv:VOL20/VOL60：DEV IC -0.067 / -0.066（**强负**，高波动 = 跌得多）
  * 高成交额 sv:AMT_LOG：DEV IC +0.067（正向，流动性溢价）
  * 合成 sv:SMALL_VOL：DEV IC -0.067（负，说明波动主导）

注意：主模型**已做了市值中性化**，所以 sv:SIZE 与现有因子正交。
      但波动类因子（realized_vol 等）已在池中，方向也是负的。

所以 B 方案有两个变体：
  B1 = 主模型 + sv:SIZE（放开市值暴露，买小市值）
  B2 = 主模型 + sv:SMALL_VOL（小市值 + 高波动合成）
  B3 = 主模型 + sv:AMT_LOG（流动性正向暴露）← DEV 期最强正向
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
T, N = md.T, md.N
Cl = md.price.astype(np.float64)


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.std()).to_numpy()

logmc = np.where(md.mcap > 0, np.log(md.mcap), np.nan)
ret_d = pd.DataFrame(Cl).pct_change().to_numpy()
amt20 = pd.DataFrame(md.amount).rolling(20, min_periods=10).mean().to_numpy()

vol20 = roll(ret_d, 20, "std")
size_z = np.where(np.isfinite(logmc), -logmc, np.nan)
vol_z = vol20

def cs_z(a):
    a = np.asarray(a, float)
    df = pd.DataFrame(a)
    mu = df.mean(axis=1); sd = df.std(axis=1).replace(0, np.nan)
    z = (df.sub(mu, axis=0)).div(sd, axis=0).to_numpy()
    return np.where(np.isfinite(z), z, np.nan)

zz_size = cs_z(size_z); zz_vol = cs_z(vol_z)
small_vol = np.where(np.isfinite(zz_size) & np.isfinite(zz_vol),
                     0.5 * (np.nan_to_num(zz_size) + np.nan_to_num(zz_vol)), np.nan)

FAC = {
    "sv:SIZE": np.asarray(size_z, np.float32),
    "sv:SMALL_VOL": np.asarray(small_vol, np.float32),
    "sv:AMT_LOG": np.asarray(-np.log(np.maximum(amt20, 1.0)), np.float32),
}

# 基础因子
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


def mk_cat(keys):
    return pd.DataFrame({
        "lib": ["sv"] * len(keys), "factor_id": list(range(len(keys))),
        "slug": keys, "name_cn": keys, "name_en": keys,
        "category_cn": "市值波动", "category_en": "size_vol",
        "base": ["size_vol"] * len(keys), "transform": ["none"] * len(keys),
        "window": [20] * len(keys), "norm": ["cs_rank"] * len(keys),
        "params": ["{}"] * len(keys), "path": [""] * len(keys),
        "factor_key": keys,
    })


def build(tag, add_keys):
    p2 = dict(panels)
    c2 = cat.copy()
    for k in add_keys:
        p2[k] = FAC[k][rows]
    c2 = pd.concat([c2, mk_cat(add_keys)], ignore_index=True)
    c2 = c2[c2["factor_key"].isin(p2)]
    cfg = qfm.ModelConfig(horizon=10, method="blend",
                          max_factors=len(add_keys) + 34,
                          corr_max=0.70, train_weeks=156,
                          score_smooth_alpha=0.6, fdr_q=0.05)
    r = qfm.walk_forward(md, rows, p2, c2, cfg=cfg, start_date="2020-01-01",
                         verbose=False)
    qfm.save_result(r, C.BUILD_MODELS / tag)
    inpool = [k for k in r.pool if k.startswith("sv:")]
    print(f"  {tag}: pool={len(r.pool)} | 进池的 sv 因子: {inpool}")
    return r


print("=" * 96)
print("[1/3] 建模（三个变体）")
print("=" * 96)
build("wf_h10_size", ["sv:SIZE"])
build("wf_h10_smallvol", ["sv:SMALL_VOL"])
build("wf_h10_liq", ["sv:AMT_LOG"])

# ---------------- 回测
print("\n[2/3] 回测…")
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
    d = {}
    for k, (a, b) in {"full": (S, E), "val": (S, "2021-12-31"),
                      "os25": ("2022-01-01", "2025-12-31"),
                      "oos": ("2022-01-01", E)}.items():
        sub = r.nav[(r.nav.index >= a) & (r.nav.index <= b)]
        pp = qfm_metrics.performance(sub, rf=C.CASH_YIELD)
        d[k] = {kk: float(pp[kk]) for kk in ("cagr", "sharpe", "max_dd", "calmar")}
    return r.nav, d


TAGS = ["wf_h10", "wf_h10_size", "wf_h10_smallvol", "wf_h10_liq"]
R = {}
print(f"\n  {'模型':<20}{'全期':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
      f"{'验证20-21':>11}{'样本外22-25':>12}{'2026':>9}")
print("  " + "-" * 88)
for tag in TAGS:
    rr = run(tag)
    if not rr:
        print(f"  {tag:<20} 不存在")
        continue
    R[tag] = rr
    d = rr[1]
    y26 = qfm_metrics.yearly_returns(rr[0][rr[0].index >= S])
    v26 = float(y26.loc[2026]) if 2026 in y26.index else 0.0
    print(f"  {tag:<20}{d['full']['cagr']:>8.2%}{d['full']['sharpe']:>8.3f}"
          f"{d['full']['max_dd']:>8.2%}{d['full']['calmar']:>8.2f}"
          f"{d['val']['cagr']:>10.2%}{d['os25']['cagr']:>11.2%}{v26:>8.2%}")

# ---------------- 显著性
print("\n[3/3] 显著性（vs 基线）")
print("=" * 96)
base = R.get("wf_h10")
for tag in TAGS[1:]:
    if tag not in R or not base:
        continue
    for seg, (a, b) in (("全期", (S, E)), ("样本外22-25", ("2022-01-01", "2025-12-31"))):
        common = base[0].index.intersection(R[tag][0].index)
        common = common[(common >= a) & (common <= b)]
        dv = (R[tag][0][common].pct_change() -
              base[0][common].pct_change()).dropna().to_numpy()
        ci = circular_block_bootstrap_ci(dv, lambda x: float(np.mean(x) * 252),
                                         n_boot=2000, block_len=21)
        t = float(np.mean(dv) / np.std(dv, ddof=1) * np.sqrt(len(dv)))
        print(f"  {tag:<20} {seg:<14} 年化差 {np.mean(dv)*252*100:+6.2f}% | "
              f"t={t:+5.2f} → {'✅显著' if abs(t) > 2 else '不显著'}")

(ALT / "size_vol_model.json").write_text(
    json.dumps({t: R[t][1] for t in R}, ensure_ascii=False, indent=1, default=float),
    encoding="utf-8")
print(f"\n→ {ALT / 'size_vol_model.json'}")
