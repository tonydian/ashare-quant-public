# -*- coding: utf-8 -*-
"""新模型 design-2「风格自适应」：不预设动量或反转，由市场状态决定。

为什么需要它：
  2026 是动量年（大牛股前期 120 日涨幅 +29% vs 市场 +15%），
  但 DEV 期（2017-2019）34 个动量因子 100% 负 IC —— 说明动量在 A 股长期无效。
  任何"用 IC 选因子"的静态模型都只能选到一种风格。

设计（严格无前视）：
  1) 构造两个方向性组合：
       M = 动量篮子得分（用 DEV 期 IC 符号定向）
       R = 反转篮子得分（同上）
  2) 每周 t，用 **t-2 周及之前** 的已实现 IC 计算两篮子的滚动 ICIR
       w_M ∝ max(0, ICIR_M_trail),  w_R ∝ max(0, ICIR_R_trail)
     并做平滑与下限，避免频繁翻转
  3) 若两者都 ≤0，退回等权（不赌方向）
  4) score = w_M·M + w_R·R

验证纪律：
  * 篮子成员与定向符号：只用 DEV 期（2017-2019）
  * 自适应权重：只用 t 时点之前的已实现 IC（滞后 2 周）
  * 2020-2021 验证期、2022-2026 样本外：**不参与任何设计决策**
  * 2026 单列检查，不回头调参
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

from qf import config as C, data as qfdata, factors as qff, model as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "tech2_design.json"
DEV_END = "2019-12-31"
TRAIL_WEEKS = 26        # 滚动 IC 窗口（半年）
LAG_WEEKS = 2           # 标签滞后（与主模型一致，避免用到未实现标签）
SMOOTH = 0.6            # 风格权重平滑

md = qfdata.load_market(verbose=False)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
n_w = len(rows)
dev = np.asarray(dates_w <= pd.Timestamp(DEV_END))

# ---------------- 载入候选因子
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
panels = {k: v for k, v in panels.items() if k in set(cat["factor_key"])}
base_of = dict(zip(cat["factor_key"], cat["base"]))

PANEL_MOM = {"return", "skip_return", "breakout", "sma_gap", "ema_gap",
             "trend_strength", "sma_slope", "dual_ema_gap"}
PANEL_REV = {"reversal", "breakdown", "rsi_reversal", "ts_rank_close",
             "range_position", "drawdown"}
PANEL_VOL = {"realized_vol", "atr_ratio", "range_ratio", "downside_vol",
             "ret_skew", "ret_kurt", "gap_sum", "upper_wick", "lower_wick"}
PANEL_VOLU = {"volume_ratio", "dollar_volume", "obv_slope", "volume_z",
              "ts_rank_volume", "price_volume_corr"}

print("=" * 96)
print("Design-2「风格自适应」模型")
print("=" * 96)

# ---------------- 1) DEV 期算每个因子的 IC 与方向
print("\n[1/5] DEV 期因子 IC（仅用于定方向，不用来选股）…", flush=True)
rows_dev, mask_dev, label_dev = rows[dev], mask[dev], label[dev]
info = {}
for k, pan in panels.items():
    try:
        f = qfm.build_feature_block(md, rows_dev, {k: pan[dev]}, k, mask_dev)
        ic = qfm.weekly_ic(f, label_dev, mask_dev)
        v = np.isfinite(ic)
        if v.sum() < 30:
            continue
        info[k] = {"ic": float(np.mean(ic[v])),
                   "icir": float(np.mean(ic[v]) / max(np.std(ic[v]), 1e-9))}
    except Exception:
        continue
print(f"  有效 {len(info)} 个因子")

# 每个篮子：DEV 期 |ICIR| 前 N（这是"用 DEV 期选"，合法）
def pick(bases, n):
    cand = [(k, v) for k, v in info.items() if base_of.get(k) in bases]
    cand.sort(key=lambda x: -abs(x[1]["icir"]))
    return cand[:n]

BASKETS = {
    "M_mom": (pick(PANEL_MOM, 8), "动量"),
    "R_rev": (pick(PANEL_REV, 8), "反转"),
    "V_vol": (pick(PANEL_VOL, 8), "波动"),
    "Q_volu": (pick(PANEL_VOLU, 8), "成交"),
}
for name, (lst, cn) in BASKETS.items():
    print(f"  {name}（{cn}）{len(lst)} 个：")
    for k, v in lst[:5]:
        print(f"      {k:<24} IC {v['ic']:+.4f} ICIR {v['icir']:+.3f}")

# ---------------- 2) 构造每周篮子得分（中性化 + z 分 + 定向）
print("\n[2/5] 构造篮子周频得分…", flush=True)
TH = int(md.T)


def basket_score(lst):
    """加权（按 DEV |ICIR|）合成篮子得分，方向按 DEV IC 符号翻正。"""
    tot = np.zeros((n_w, md.N), dtype=np.float32)
    wsum = 0.0
    for k, v in lst:
        pan = panels[k]
        f = qfm.build_feature_block(md, rows, {k: pan}, k, mask)
        # 逐周中性化 + z
        z = np.full((n_w, md.N), np.nan, dtype=np.float32)
        for wi, t in enumerate(rows):
            row = f[wi]
            m = mask[wi] & np.isfinite(row)
            if m.sum() > 50:
                r_ = qfm.neutralize_rows(row[None, :], md.industry[[t]], md.logmcap[[t]],
                                         m[None, :])
                z[wi] = qfm.zscore_rows(r_, m[None, :])[0]
        sign = 1.0 if v["ic"] > 0 else -1.0
        w = abs(v["icir"])
        tot += sign * w * np.nan_to_num(z, nan=0.0)
        wsum += w
    return tot / max(wsum, 1e-9)


scores_b = {}
for name, (lst, cn) in BASKETS.items():
    if not lst:
        continue
    scores_b[name] = basket_score(lst)
    print(f"  {name} 完成", flush=True)

# ---------------- 3) 每周计算滚动 IC（滞后 2 周）
print("\n[3/5] 计算滚动 IC（滞后 %d 周）…" % LAG_WEEKS, flush=True)
trail_ic = {}
for name, sc in scores_b.items():
    ic = qfm.weekly_ic(sc, label, mask)
    s = pd.Series(ic)
    # shift(LAG) 后再 rolling：只用 t-LAG 及之前
    trail_ic[name] = s.shift(LAG_WEEKS).rolling(TRAIL_WEEKS, min_periods=8).mean().to_numpy()

# ---------------- 4) 风格权重（只有 M 和 R 参与方向博弈）
print("\n[4/5] 合成自适应权重…", flush=True)
icM = trail_ic.get("M_mom")
icR = trail_ic.get("R_rev")
w_mom = np.zeros(n_w)
w_rev = np.zeros(n_w)
for i in range(n_w):
    a = icM[i] if icM is not None and np.isfinite(icM[i]) else 0.0
    b = icR[i] if icR is not None and np.isfinite(icR[i]) else 0.0
    a, b = max(0.0, a), max(0.0, b)
    if a + b <= 1e-9:
        w_mom[i] = w_rev[i] = 0.5      # 都没效 → 等权（保持方向不赌）
    else:
        w_mom[i] = a / (a + b)
        w_rev[i] = b / (a + b)
# 平滑
for i in range(1, n_w):
    w_mom[i] = SMOOTH * w_mom[i] + (1 - SMOOTH) * w_mom[i - 1]
    w_rev[i] = 1.0 - w_mom[i]

print(f"  动量权重：均值 {w_mom.mean():.2f} | 最高 {w_mom.max():.2f} | 最低 {w_mom.min():.2f}")
for y in (2020, 2021, 2022, 2023, 2024, 2025, 2026):
    m = dates_w.year == y
    if m.sum() > 3:
        print(f"    {y}: 动量权重均值 {w_mom[m].mean():.2f}")

# ---------------- 5) 合成最终打分
print("\n[5/5] 合成最终打分…", flush=True)
SM = scores_b["M_mom"]
SR = scores_b["R_rev"]
final = np.full((n_w, md.N), np.nan, dtype=np.float32)
for i in range(n_w):
    if not np.isfinite(SM[i]).any() and not np.isfinite(SR[i]).any():
        continue
    final[i] = w_mom[i] * np.nan_to_num(SM[i], nan=0.0) + \
        w_rev[i] * np.nan_to_num(SR[i], nan=0.0)
# 映射回 T×N
scores_T = np.full((md.T, md.N), np.nan, dtype=np.float32)
scores_T[rows] = final

# 保存
out_model = C.BUILD_MODELS / "wf_style2"
out_model.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_model / "composite_score.npz", scores=scores_T)
import json as _json
_meta = {
    "name": "wf_style2",
    "desc": "风格自适应（动量/反转按滚动IC切换）",
    "dev_end": DEV_END, "trail_weeks": TRAIL_WEEKS, "lag_weeks": LAG_WEEKS,
    "smooth": SMOOTH, "rows_weekly": [int(rows[0]), int(rows[-1])],
    "baskets": {k: [x[0] for x in v[0]] for k, v in BASKETS.items()},
    "cfg": {"horizon": 10, "method": "style-adaptive"},
}
(out_model / "model_meta.json").write_text(_json.dumps(_meta, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
np.savez_compressed(out_model / "weekly_meta.npz", rows=rows)
print(f"\n→ {out_model}")

json.dump({
    "w_mom_mean": float(w_mom.mean()),
    "w_mom_by_year": {int(y): float(w_mom[dates_w.year == y].mean())
                      for y in range(2017, 2027) if (dates_w.year == y).sum() > 3},
    "baskets": {k: [x[0] for x in v[0]] for k, v in BASKETS.items()},
}, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"→ {OUT}")
