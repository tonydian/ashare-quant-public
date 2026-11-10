# -*- coding: utf-8 -*-
"""快照类另类数据 → 因子 → DEV 期 IC。

三类数据（都是"事件型"，关键是时点可知）：

1) 股东户数（季度）
   HOLDER_NUM_RATIO  户数变化率（负=筹码集中）
   AVG_MARKET_CAP    户均市值
   注意：END_DATE 是报告期，实际**公告日**要滞后 1-2 个月 → 用 END_DATE + 60 天作为可知日

2) 限售解禁（月度）
   LIFT_MARKET_CAP / 流通市值  解禁规模占比
   FREE_RATIO                  解禁股占总股本比例
   注意：FREE_DATE 是解禁日，**提前可知**（公告早于解禁日）→ 可直接用 FREE_DATE

3) 业绩预告（季度）
   ADD_AMP_LOWER/UPPER  预告净利增速
   PREDICT_TYPE         预增/预减/扭亏
   注意：NOTICE_DATE 是公告日 → 从该日起可知

因子设计（全部只用 t 时点已知的信息）：
  HOLDER_CHG      最近一期股东户数变化率（筹码集中度）
  HOLDER_CHG_Q    同上，按公告日映射
  LIFT_RATIO      未来 30/60 日解禁市值占流通市值比例（解禁压力）
  LIFT_NEAR       是否临近解禁（30 日内）
  FCST_GROWTH     最近一期业绩预告增速中值
  FCST_POS        预告是否正面（预增/扭亏=1，预减/亏损=-1）
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

from qf import config as C, data as qfdata, model as qfm, factors as qff  # noqa: E402

ALT = ROOT / "build" / "alt"
DEV_END = "2019-12-31"

md = qfdata.load_market(verbose=False)
dates = pd.DatetimeIndex(md.dates)
T, N = md.T, md.N
code_idx = {str(c): j for j, c in enumerate(md.codes)}
date_pos = {d.normalize(): i for i, d in enumerate(dates)}


def next_idx(d):
    """给定日期，返回 >= 该日 的第一个交易日索引。"""
    d = pd.Timestamp(d).normalize()
    if d in date_pos:
        return date_pos[d]
    for i, dd in enumerate(dates):
        if dd.normalize() >= d:
            return i
    return None


# ============================================================
print("[1/5] 股东户数…")
holder_chg = np.full((T, N), np.nan, np.float32)
holder_avgmc = np.full((T, N), np.nan, np.float32)
p = ALT / "holder_num.jsonl"
n = 0
if p.exists():
    with open(p, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            end = pd.Timestamp(obj["end_date"])
            # 公告滞后：报告期 + 60 天起可知
            di = next_idx(end + pd.Timedelta(days=60))
            if di is None:
                continue
            for r in obj["rows"]:
                j = code_idx.get(str(r.get("SECURITY_CODE") or "").zfill(6))
                if j is None:
                    continue
                n += 1
                try:
                    v = r.get("HOLDER_NUM_RATIO")
                    if v is not None:
                        holder_chg[di:, j] = float(v)   # 该期数据一直有效到下期
                except Exception:
                    pass
                try:
                    v2 = r.get("AVG_MARKET_CAP")
                    if v2 is not None:
                        holder_avgmc[di:, j] = float(v2)
                except Exception:
                    pass
print(f"  记录 {n} 条")

# ============================================================
print("\n[2/5] 限售解禁…")
lift_ratio = np.zeros((T, N), np.float32)     # 解禁市值/流通市值
lift_near = np.zeros((T, N), np.float32)      # 30 日内是否有解禁
p = ALT / "lift.jsonl"
n = 0
mcap = md.mcap.astype(np.float64)
if p.exists():
    with open(p, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for r in obj["rows"]:
                j = code_idx.get(str(r.get("SECURITY_CODE") or "").zfill(6))
                if j is None:
                    continue
                fd = r.get("FREE_DATE")
                if not fd:
                    continue
                di = next_idx(fd)
                if di is None:
                    continue
                n += 1
                try:
                    lmc = float(r.get("LIFT_MARKET_CAP") or 0)
                except Exception:
                    lmc = 0.0
                mc = mcap[di, j] if np.isfinite(mcap[di, j]) else np.nan
                if np.isfinite(mc) and mc > 0 and lmc > 0:
                    lift_ratio[di, j] += lmc / mc
                # 解禁前 30 个交易日的压力标记
                lo = max(0, di - 30)
                lift_near[lo:di, j] = 1.0
print(f"  记录 {n} 条")

# ============================================================
print("\n[3/5] 业绩预告…")
fcst_growth = np.full((T, N), np.nan, np.float32)
fcst_pos = np.full((T, N), np.nan, np.float32)
p = ALT / "forecast.jsonl"
n = 0
POS_TYPES = {"预增", "略增", "扭亏", "续盈"}
NEG_TYPES = {"预减", "略减", "首亏", "续亏", "增亏"}
if p.exists():
    with open(p, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for r in obj["rows"]:
                j = code_idx.get(str(r.get("SECURITY_CODE") or "").zfill(6))
                if j is None:
                    continue
                nd = r.get("NOTICE_DATE")
                if not nd:
                    continue
                di = next_idx(nd)
                if di is None:
                    continue
                n += 1
                try:
                    lo = r.get("ADD_AMP_LOWER")
                    hi = r.get("ADD_AMP_UPPER")
                    vals = [float(x) for x in (lo, hi) if x is not None]
                    if vals:
                        fcst_growth[di:, j] = float(np.mean(vals))
                except Exception:
                    pass
                pt = str(r.get("PREDICT_TYPE") or "")
                if pt in POS_TYPES:
                    fcst_pos[di:, j] = 1.0
                elif pt in NEG_TYPES:
                    fcst_pos[di:, j] = -1.0
print(f"  记录 {n} 条")

# ============================================================
print("\n[4/5] 周频 IC…")
fac = {
    "alt:HOLDER_CHG": holder_chg,
    "alt:HOLDER_AVGMC": holder_avgmc,
    "alt:LIFT_RATIO": lift_ratio,
    "alt:LIFT_NEAR": lift_near,
    "alt:FCST_GROWTH": fcst_growth,
    "alt:FCST_POS": fcst_pos,
}
for k, v in fac.items():
    vv = np.asarray(v, np.float32)
    fac[k] = vv
    print(f"  {k:<20} 非NaN {np.isfinite(vv).mean():6.1%}")

zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(DEV_END))
val = np.asarray((dates_w >= "2020-01-01") & (dates_w <= "2021-12-31"))
oos = np.asarray(dates_w >= "2022-01-01")

res = {}
print(f"\n  {'因子':<20}{'DEV IC':>10}{'ICIR':>9}{'验证IC':>10}{'样本外IC':>10}{'判定':>6}")
print("  " + "-" * 68)
for k, pan in fac.items():
    f = qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    if v.sum() < 20:
        print(f"  {k:<20} 数据不足")
        continue
    d = ic[v & dev]
    if len(d) < 15:
        print(f"  {k:<20} DEV 周数不足({len(d)})")
        continue
    d_ic = float(np.mean(d))
    d_icir = d_ic / max(float(np.std(d)), 1e-9)
    v_ic = float(np.mean(ic[v & val])) if (v & val).sum() else float("nan")
    o_ic = float(np.mean(ic[v & oos])) if (v & oos).sum() else float("nan")
    flag = "✅" if abs(d_ic) > 0.012 and abs(d_icir) > 0.20 else (
        "🟡" if abs(d_ic) > 0.006 else "❌")
    res[k] = {"dev_ic": d_ic, "dev_icir": d_icir, "val_ic": v_ic, "oos_ic": o_ic}
    print(f"  {k:<20}{d_ic:>+10.4f}{d_icir:>+9.3f}{v_ic:>+10.4f}{o_ic:>+10.4f}{flag:>6}")

# ============================================================
print("\n[5/5] 正交性…")
cat = qff.load_catalog().set_index("factor_key")
base_model = qfm.load_result(C.BUILD_MODELS / "wf_h10")
inp = qff.FactorInputs(md)
good = [k for k, v in res.items() if abs(v["dev_ic"]) > 0.012 and abs(v["dev_icir"]) > 0.20]
print(f"  达标 {len(good)} / {len(res)}: {good}")
for gk in good:
    fg = qfm.build_feature_block(md, rows, {gk: fac[gk][rows]}, gk, mask)
    mx, mxk = 0.0, None
    for bk in base_model.pool[:15]:
        try:
            pan = qff.compute_factor(inp, cat.loc[bk].to_dict())
            fb = qfm.build_feature_block(md, rows, {bk: pan[rows]}, bk, mask)
            a, b2 = fg.ravel(), fb.ravel()
            m = np.isfinite(a) & np.isfinite(b2)
            if m.sum() > 5000:
                c = abs(float(np.corrcoef(a[m], b2[m])[0, 1]))
                if c > mx:
                    mx, mxk = c, bk
        except Exception:
            pass
    res[gk]["max_corr_base"] = mx
    print(f"    {gk:<20} 最高相关 {mx:+.3f}（{mxk}）")

np.savez_compressed(ALT / "snapshot_panel.npz",
                    dates=dates.values, codes=np.array([str(c) for c in md.codes]),
                    holder_chg=holder_chg, holder_avgmc=holder_avgmc,
                    lift_ratio=lift_ratio, lift_near=lift_near,
                    fcst_growth=fcst_growth, fcst_pos=fcst_pos)

(ALT / "snapshot_factor_ic.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {ALT / 'snapshot_factor_ic.json'}")
