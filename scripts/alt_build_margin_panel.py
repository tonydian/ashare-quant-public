# -*- coding: utf-8 -*-
"""融资融券 → 面板 → 因子 → DEV 期 IC 检验。

因子设计（全部时点可知，只用 t 日及之前的数据）：
  MARGIN_BAL     融资余额 / 流通市值（杠杆存量，需市值）
  MARGIN_BAL_MA5 融资余额 5 日均值占比
  MARGIN_CHG     融资余额日变化率（5 日）
  MARGIN_CHG20   20 日变化率
  RZJME_5        融资净买入额 / 成交额（5 日累计）
  RZJME_20       同上 20 日
  RQYL_CHG       融券余量变化率
  MARGIN_RATIO   融资余额 / 融资融券总余额（纯多头杠杆占比）
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

from qf import config as C, data as qfdata, model as qfm  # noqa: E402

ALT = ROOT / "build" / "alt"
DEV_END = "2019-12-31"

md = qfdata.load_market(verbose=False)
dates = pd.DatetimeIndex(md.dates)
T, N = md.T, md.N
code_idx = {str(c): j for j, c in enumerate(md.codes)}
date_pos = {d.normalize(): i for i, d in enumerate(dates)}
print(f"行情面板 {T}×{N}，{dates[0].date()} ~ {dates[-1].date()}")

# ---------------- 1) 原表 → 面板
raw = ALT / "margin_raw.jsonl"
if not raw.exists():
    raise SystemExit(f"缺 {raw}")

print("\n[1/5] 构建面板…", flush=True)
f_bal = np.full((T, N), np.nan, np.float32)     # 融资余额
f_rzjm = np.full((T, N), np.nan, np.float32)    # 融资净买入
f_rq = np.full((T, N), np.nan, np.float32)      # 融券余量
f_total = np.full((T, N), np.nan, np.float32)   # 融资融券总余额
f_buy = np.full((T, N), np.nan, np.float32)     # 融资买入额

n_days = n_rows = 0
with open(raw, encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        di = date_pos.get(pd.Timestamp(obj["date"]).normalize())
        if di is None:
            continue
        n_days += 1
        for r in obj["rows"]:
            sc = str(r.get("SCODE") or "").zfill(6)
            j = code_idx.get(sc)
            if j is None:
                continue
            n_rows += 1

            def num(k):
                v = r.get(k)
                try:
                    return float(v) if v is not None else np.nan
                except Exception:
                    return np.nan

            f_bal[di, j] = num("RZYE")
            f_rzjm[di, j] = num("RZJME")
            f_rq[di, j] = num("RQYL")
            f_total[di, j] = num("RZRQYE")
            f_buy[di, j] = num("RZMRE")

print(f"  读取 {n_days} 天 / {n_rows} 行")
print(f"  覆盖率: 融资余额 {np.isfinite(f_bal).mean():.1%} | "
      f"净买入 {np.isfinite(f_rzjm).mean():.1%}")
cov = np.isfinite(f_bal).sum(axis=1) / max(N, 1)
ok_days = np.flatnonzero(cov > 0.02)
if len(ok_days):
    print(f"  有效区间 {dates[ok_days[0]].date()} ~ {dates[ok_days[-1]].date()}"
          f"（覆盖率>2%：{len(ok_days)} 天）")

np.savez_compressed(ALT / "margin_panel.npz",
                    dates=dates.values, codes=np.array([str(c) for c in md.codes]),
                    bal=f_bal, rzjm=f_rzjm, rq=f_rq, total=f_total, buy=f_buy)
print(f"  → {ALT / 'margin_panel.npz'}")

# ---------------- 2) 因子
print("\n[2/5] 构造融资融券因子…", flush=True)


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.sum()).to_numpy()


def pct_chg(a, w):
    df = pd.DataFrame(a)
    return (df / df.shift(w) - 1.0).to_numpy()


amt = md.amount.astype(np.float64)
mc = md.mcap.astype(np.float64)

with np.errstate(divide="ignore", invalid="ignore"):
    bal_ratio = np.where(mc > 0, f_bal / mc, np.nan)           # 融资余额/市值
    rzjm_ratio = np.where(amt > 0, f_rzjm / amt, np.nan)       # 净买入/成交额
    total_ratio = np.where(f_total > 0, f_bal / f_total, np.nan)  # 融资占比

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
    "mgn:BUY_RATIO": np.where(amt > 0, f_buy / amt, np.nan),
}
fac = {k: np.asarray(v, np.float32) for k, v in fac.items()}
for k, v in fac.items():
    print(f"  {k:<20} 非NaN {np.isfinite(v).mean():6.1%}")

# ---------------- 3) 周频 IC
print("\n[3/5] 周频 IC 检验（仅 DEV 期决策）…", flush=True)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(DEV_END))
val = np.asarray((dates_w >= "2020-01-01") & (dates_w <= "2021-12-31"))
oos = np.asarray(dates_w >= "2022-01-01")
print(f"  周频 {len(rows)} | DEV {dev.sum()} | 验证 {val.sum()} | 样本外 {oos.sum()}")

res = {}
print(f"\n  {'因子':<22}{'DEV IC':>10}{'ICIR':>9}{'验证IC':>10}{'样本外IC':>10}{'判定':>6}")
print("  " + "-" * 70)
for k, pan in fac.items():
    f = qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    if v.sum() < 20:
        continue
    d = ic[v & dev]
    if len(d) < 15:
        print(f"  {k:<22} DEV 数据不足（{len(d)} 周）")
        continue
    d_ic = float(np.mean(d))
    d_icir = d_ic / max(float(np.std(d)), 1e-9)
    v_ic = float(np.mean(ic[v & val])) if (v & val).sum() else float("nan")
    o_ic = float(np.mean(ic[v & oos])) if (v & oos).sum() else float("nan")
    flag = "✅" if abs(d_ic) > 0.015 and abs(d_icir) > 0.25 else (
        "🟡" if abs(d_ic) > 0.008 else "❌")
    res[k] = {"dev_ic": d_ic, "dev_icir": d_icir, "val_ic": v_ic, "oos_ic": o_ic}
    print(f"  {k:<22}{d_ic:>+10.4f}{d_icir:>+9.3f}{v_ic:>+10.4f}{o_ic:>+10.4f}{flag:>6}")

# ---------------- 4) 与量价因子的相关性（是否冗余）
print("\n[4/5] 与现有量价因子相关性（检验是否冗余）…", flush=True)
from qf import factors as qff  # noqa: E402

cat = qff.load_catalog().set_index("factor_key")
base_model = qfm.load_result(C.BUILD_MODELS / "wf_h10")
inp = qff.FactorInputs(md)
good = [k for k, v in res.items() if abs(v["dev_ic"]) > 0.015 and abs(v["dev_icir"]) > 0.25]
for gk in good[:4]:
    fg = qfm.build_feature_block(md, rows, {gk: fac[gk][rows]}, gk, mask)
    mx, mxk = 0.0, None
    for bk in base_model.pool[:12]:
        try:
            pan = qff.compute_factor(inp, cat.loc[bk].to_dict())
            fb = qfm.build_feature_block(md, rows, {bk: pan[rows]}, bk, mask)
            a, b = fg.ravel(), fb.ravel()
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 5000:
                c = abs(float(np.corrcoef(a[m], b[m])[0, 1]))
                if c > mx:
                    mx, mxk = c, bk
        except Exception:
            pass
    print(f"  {gk:<22} 与量价因子最高相关 {mx:+.3f}（{mxk}）")

# ---------------- 5) 结论
print("\n" + "=" * 96)
print("[5/5] 结论")
print("=" * 96)
print(f"  DEV 期达标: {len(good)} / {len(res)}")
for k in sorted(good, key=lambda x: -abs(res[x]["dev_icir"])):
    d = "正" if res[k]["dev_ic"] > 0 else "负"
    print(f"    {k:<22} DEV IC {res[k]['dev_ic']:+.4f} ICIR {res[k]['dev_icir']:+.3f}"
          f"（{d}向）样本外 {res[k]['oos_ic']:+.4f}")

if not good:
    print("\n  ❌ 融资融券因子在 DEV 期无效 → 停止")
else:
    print(f"\n  ✅ {len(good)} 个因子可用 → 进入建模")

(ALT / "margin_factor_ic.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {ALT / 'margin_factor_ic.json'}")
