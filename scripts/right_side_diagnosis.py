# -*- coding: utf-8 -*-
"""右侧模型收益率低的根因诊断 + 优化实验。

根因假设：
  1) 方向库 296 因子窗口只有 5/7/10/14 天 —— 测的是**短周期**趋势，而 A 股短周期是反转；
     所以"右侧选股"用的其实是短周期反转因子（R241/R076/R340/R465/R729/R498 全是 breakdown/reversal）。
  2) **横截面排序**（选分最高的 40 只）与趋势跟踪的逻辑相悖：趋势跟踪要"让赢家跑"，
     横截面每天重排会把强势股卖掉去买"更强"的。
  3) 持有期 10 天 + 固定 -15% 止损：趋势跟踪需要更长持有 + 移动止损。

优化方向（本脚本逐个实测）：
  A. 长窗口趋势因子（trend-extra 的 120/200/250 日因子）替代短窗口；
  B. 时序动量（绝对趋势）而非横截面排序；
  C. 更长持有期（20/40 日）+ 移动止损（trailing stop）；
  D. 上述组合。
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
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, model as qfm, backtest as qfbt, metrics as qfm_metrics  # noqa: E402
from qf import trend as qftrend  # noqa: E402

OUT = C.BUILD_EVAL / "right_side_diagnosis.json"
res_out = {}

md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
phase = int(rows[0]) % 5
start = max(int(rows[0]), md.date_index(C.VALID_START))
while start % 5 != phase:
    start += 1
end = md.T - 1
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)


def run(scores, tag, rb=10, n_hold=40):
    old = C.N_HOLDINGS
    C.N_HOLDINGS = n_hold
    try:
        r = qfbt.run_backtest(md, scores, start, end, risk_overlay=True, stop_loss=True,
                              rebalance_days=rb, rebalance_offset=(phase - start) % rb,
                              cash_asset_returns=cash, verbose=False)
    finally:
        C.N_HOLDINGS = old
    m = r.metrics
    row = {"tag": tag, "rb": rb, "n_hold": n_hold,
           "cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
           "max_dd": float(m["max_dd"]), "calmar": float(m["calmar"]),
           "turnover": float(m["turnover_ann"]), "n_trades": int(m["n_trades"])}
    print(f"  [{tag:34s}] CAGR={m['cagr']:7.2%} Sharpe={m['sharpe']:6.3f} "
          f"MaxDD={m['max_dd']:7.2%} Cal={m['calmar']:5.2f} 换手={m['turnover_ann']:.2f}x")
    return row


print("=" * 86)
print("0) 基线：现行 wf_rs10（pull_ma60：MA60 上方 + 短窗口回调因子）")
print("=" * 86)
cur = qfm.load_result(C.BUILD_MODELS / "wf_rs10")
cur_scores = np.asarray(cur.scores)
if cur_scores.shape[0] < md.T:
    cur_scores = np.vstack([cur_scores, np.full((md.T - cur_scores.shape[0], md.N), np.nan, np.float32)])
base = run(cur_scores, "现行 wf_rs10 (pull_ma60)")

print()
print("=" * 86)
print("1) 长窗口趋势因子（trend-extra 120/200/250）作为横截面打分")
print("=" * 86)
tp = qftrend.build_panels(md, rows)
feats = {}
for k in tp:
    feats[k] = qfm.build_feature_block(md, rows, tp, k, mask)
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dev = np.asarray(pd.to_datetime(md.dates[rows]) <= pd.Timestamp(C.DEV_END))
stats = {}
for k, f in feats.items():
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    icd = ic[v & dev]
    stats[k] = (float(np.mean(icd)) if len(icd) else np.nan,
                float(np.mean(icd) / max(np.std(icd), 1e-9)) if len(icd) else np.nan)
print("  trend-extra 因子 DEV(h10) IC：")
for k, (a, b) in sorted(stats.items(), key=lambda x: -(x[1][0] or -9))[:8]:
    print(f"    {k:24s} IC={a:+.4f} ICIR={b:+.3f}")

pos = [k for k, (a, b) in stats.items() if a > 0]
print(f"  DEV IC>0 的长窗口因子：{len(pos)} 个")
if pos:
    w = {k: 1.0 / len(pos) for k in pos}
    F = np.stack([feats[k] for k in pos], axis=0)
    sc = np.full((md.T, md.N), np.nan, dtype=np.float32)
    for wi in range(len(rows)):
        row = np.tensordot(np.array([w[k] for k in pos]), F[:, wi, :], axes=(0, 0)).astype(np.float32)[None, :]
        mm = mask[wi][None, :]
        if np.isfinite(row[mm]).sum() > 20:
            rr = qfm.neutralize_rows(row, md.industry[rows[wi]][None, :], md.logmcap[rows[wi]][None, :], mm)
            sc[rows[wi]] = qfm.zscore_rows(rr, mm)[0]
    r1 = run(sc, "长窗口趋势因子(横截面,10日)")
    res_out["long_window_cs"] = r1

print()
print("=" * 86)
print("2) 时序动量（绝对趋势）：只买 12 个月动量为正 + 价格在 MA200 上方")
print("=" * 86)
close = md.close.astype(np.float64)
ma200 = pd.DataFrame(close).rolling(200, min_periods=150).mean().to_numpy()
mom250_20 = np.full_like(close, np.nan)
mom250_20[250:] = close[250:] / close[:-250] - 1.0
for rb, n_hold, tag in ((10, 40, "时序动量 rb10"), (20, 40, "时序动量 rb20"), (40, 40, "时序动量 rb40")):
    sc = np.full((md.T, md.N), np.nan, dtype=np.float32)
    for wi, t in enumerate(rows):
        if t < 250:
            continue
        ok = (mask[wi] & np.isfinite(mom250_20[t]) & (mom250_20[t] > 0)
              & np.isfinite(ma200[t]) & (close[t] > ma200[t]))
        if ok.sum() < 40:
            continue
        # 打分 = 时序动量（越大越强），不做横截面 rank，保留"强者恒强"
        v = np.where(ok, mom250_20[t], np.nan)
        mm = ok[None, :]
        rr = qfm.neutralize_rows(v[None, :], md.industry[rows[wi]][None, :],
                                 md.logmcap[rows[wi]][None, :], mm)
        sc[t] = qfm.zscore_rows(rr, mm)[0]
    r2 = run(sc, tag, rb=rb, n_hold=n_hold)
    res_out[tag] = r2

print()
print("=" * 86)
print("3) 组合：时序动量 + 回调买点（MA60 上方 + 12M 动量>0 + 短期回撤）")
print("=" * 86)
pull = qftrend.build_panels(md, rows)
pull_feats = {}
for k in cur.pool:
    pan = None
    try:
        from qf import factors as qff
        cat_all = qff.load_catalog().set_index("factor_key")
        pan = qff.compute_factor(qff.FactorInputs(md), cat_all.loc[k].to_dict())[rows]
    except Exception:
        continue
    if pan is not None:
        pull_feats[k] = qfm.build_feature_block(md, rows, {k: np.asarray(pan, np.float32)}, k, mask)
for rb in (10, 20):
    sc = np.full((md.T, md.N), np.nan, dtype=np.float32)
    for wi, t in enumerate(rows):
        if t < 250:
            continue
        ok = (mask[wi] & np.isfinite(mom250_20[t]) & (mom250_20[t] > 0)
              & np.isfinite(ma200[t]) & (close[t] > ma200[t]))
        if ok.sum() < 40:
            continue
        trend_z = np.where(ok, mom250_20[t], np.nan)
        pz = np.nanmean([pull_feats[k][wi] for k in pull_feats], axis=0) if pull_feats else np.zeros(md.N)
        v = 0.7 * np.nan_to_num(trend_z) + 0.3 * np.nan_to_num(pz)
        v = np.where(ok, v, np.nan)
        mm = ok[None, :]
        rr = qfm.neutralize_rows(v[None, :], md.industry[rows[wi]][None, :],
                                 md.logmcap[rows[wi]][None, :], mm)
        sc[t] = qfm.zscore_rows(rr, mm)[0]
    r3 = run(sc, f"时序动量+回调 rb{rb}", rb=rb)
    res_out[f"tsmom_pull_rb{rb}"] = r3

OUT.write_text(json.dumps(res_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
