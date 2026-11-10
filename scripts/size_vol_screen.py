# -*- coding: utf-8 -*-
"""B 方案第一步：小市值 + 高波动暴露，在 DEV 期（2017-2019）有效吗？

关键背景：
  * 主模型对 logmcap 和行业做了**中性化**，即故意剥离了市值暴露
  * 波动类因子（realized_vol/atr_ratio/range_ratio）在因子库里，但 DEV 期多为负 IC
  * 2026 大牛股的特征是"前期强势 + 高波动 + 小市值"

必须先回答：
  Q1 纯市值因子（小市值溢价）在 DEV 期有效吗？方向如何？
  Q2 纯波动因子在 DEV 期有效吗？
  Q3 "小市值 + 高波动"合成打分在 DEV 期 IC 是多少？
  Q4 如果 DEV 期无效，只用 2026 做选择就是前视 → B 方案不成立

严格纪律：只用 DEV 期（2017-01-01 ~ 2019-12-31）做判断。
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
OUT = ALT / "size_vol_screen.json"

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(DEV_END))
val = np.asarray((dates_w >= "2020-01-01") & (dates_w <= "2021-12-31"))
oos = np.asarray(dates_w >= "2022-01-01")
print(f"周频 {len(rows)} | DEV {dev.sum()} | 验证 {val.sum()} | 样本外 {oos.sum()}")

T, N = md.T, md.N
Cl = md.price.astype(np.float64)

# ============================================================
# 构造候选暴露因子（全部时点可知）
print("\n[1/4] 构造市值/波动因子…", flush=True)


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.std()).to_numpy()


logmc = np.where(md.mcap > 0, np.log(md.mcap), np.nan)     # 市值（越小越好）
ret_d = pd.DataFrame(Cl).pct_change().to_numpy()

fac = {
    "sv:SIZE": -logmc,                                  # 负对数市值 = 小市值暴露
    "sv:VOL20": roll(ret_d, 20, "std"),                 # 20 日波动率
    "sv:VOL60": roll(ret_d, 60, "std"),
    "sv:ATR": roll((md.high - md.low) / np.maximum(Cl, 1e-9), 20),  # 振幅
    "sv:AMT_LOG": -np.log(np.maximum(
        pd.DataFrame(md.amount).rolling(20, min_periods=10).mean().to_numpy(), 1.0)),
}
# 合成：小市值 + 高波动的简单等权（每期横截面 z 分后相加）
def cs_z(a):
    a = np.asarray(a, dtype=np.float64)
    df = pd.DataFrame(a)
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z = (df.sub(mu, axis=0)).div(sd, axis=0).to_numpy()
    return np.where(np.isfinite(z), z, np.nan)

z_size = cs_z(fac["sv:SIZE"])
z_vol = cs_z(fac["sv:VOL20"])
with np.errstate(invalid="ignore"):
    fac["sv:SMALL_VOL"] = np.where(
        np.isfinite(z_size) & np.isfinite(z_vol),
        0.5 * (np.nan_to_num(z_size) + np.nan_to_num(z_vol)), np.nan)

fac = {k: np.asarray(v, np.float32) for k, v in fac.items()}
for k, v in fac.items():
    print(f"  {k:<16} 非NaN {np.isfinite(v).mean():6.1%}")

# ============================================================
print("\n[2/4] DEV 期 IC（决定性判断）…", flush=True)
res = {}
print(f"\n  {'因子':<16}{'DEV IC':>10}{'ICIR':>9}{'验证IC':>10}{'样本外IC':>10}"
      f"{'2026IC':>10}{'判定':>6}")
print("  " + "-" * 72)
ic_rows = {}
for k, pan in fac.items():
    f = qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    if v.sum() < 20:
        continue
    ic_rows[k] = ic
    d = ic[v & dev]
    if len(d) < 15:
        print(f"  {k:<16} DEV 不足")
        continue
    d_ic = float(np.mean(d))
    d_icir = d_ic / max(float(np.std(d)), 1e-9)
    v_ic = float(np.mean(ic[v & val])) if (v & val).sum() else float("nan")
    o_ic = float(np.mean(ic[v & oos])) if (v & oos).sum() else float("nan")
    y26 = np.asarray(dates_w.year == 2026)
    y26_ic = float(np.mean(ic[v & y26])) if (v & y26).sum() > 3 else float("nan")
    flag = "✅" if abs(d_ic) > 0.012 and abs(d_icir) > 0.20 else (
        "🟡" if abs(d_ic) > 0.006 else "❌")
    res[k] = {"dev_ic": d_ic, "dev_icir": d_icir, "val_ic": v_ic,
              "oos_ic": o_ic, "y2026_ic": y26_ic}
    print(f"  {k:<16}{d_ic:>+10.4f}{d_icir:>+9.3f}{v_ic:>+10.4f}{o_ic:>+10.4f}"
          f"{y26_ic:>+10.4f}{flag:>6}")

# ============================================================
print("\n[3/4] 关键检验：小市值/高波动在 2026 是否突然有效？")
print("=" * 96)
for k in ("sv:SIZE", "sv:VOL20", "sv:SMALL_VOL"):
    if k not in res:
        continue
    r = res[k]
    print(f"\n  {k}:")
    print(f"    DEV(2017-19)  IC {r['dev_ic']:+.4f}  ← 决定能否进模型")
    print(f"    验证(2020-21) IC {r['val_ic']:+.4f}")
    print(f"    样本外(2022+) IC {r['oos_ic']:+.4f}")
    print(f"    2026 单年     IC {r['y2026_ic']:+.4f}")
    if r["dev_ic"] * r["y2026_ic"] < 0:
        print(f"    ⚠️ DEV 期与 2026 方向相反 → 用 2026 选择就是前视")

# ============================================================
print("\n[4/4] 结论")
print("=" * 96)
good = [k for k, v in res.items() if abs(v["dev_ic"]) > 0.012 and abs(v["dev_icir"]) > 0.20]
print(f"  DEV 期达标: {len(good)} / {len(res)}")
for k in good:
    d = "正" if res[k]["dev_ic"] > 0 else "负"
    print(f"    {k:<16} DEV IC {res[k]['dev_ic']:+.4f}（{d}向）"
          f" | 2026 IC {res[k]['y2026_ic']:+.4f}")

if not good:
    print("""
  ❌ B 方案在 DEV 期无效。

  含义：小市值/高波动这两个暴露，在 2017-2019 的历史数据里**没有正向预测力**。
  2026 大牛股恰好是小市值高波动股，属于"事后归因"。
  如果据此建模型，就是把 2026 的结果写进规则 → 典型前视偏差。
""")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
               encoding="utf-8")
print(f"\n→ {OUT}")
