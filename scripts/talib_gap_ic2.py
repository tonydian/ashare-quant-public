# -*- coding: utf-8 -*-
"""TA-Lib 缺失指标 IC 实测（向量化版，避免 rolling.apply 太慢）。

用 numpy 全向量化实现，T×N 矩阵一次算完。
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

OUT = ROOT / "build" / "eval" / "talib_gap_ic.json"

md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dev = np.asarray(pd.to_datetime(md.dates[rows]) <= pd.Timestamp(C.DEV_END))

O = md.price_open.astype(np.float64)
H = md.high.astype(np.float64)
L = md.low.astype(np.float64)
Cl = md.close.astype(np.float64)
V = md.volume.astype(np.float64)
T, N = Cl.shape
print(f"面板 {T}×{N}，周频 {len(rows)} 行", flush=True)


def f32(x):
    return np.asarray(x, dtype=np.float32)


def ic_of(panel, name, res):
    key = name
    f = qfm.build_feature_block(md, rows, {key: panel[rows]}, key, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    if v.sum() < 10:
        print(f"  {name:26s} 数据不足", flush=True)
        return
    d = ic[v & dev]
    st = {"dev_ic": float(np.mean(d)) if len(d) else np.nan,
          "dev_icir": float(np.mean(d) / max(np.std(d), 1e-9)) if len(d) else np.nan,
          "full_ic": float(np.mean(ic[v])),
          "pos_share": float((d > 0).mean()) if len(d) else np.nan}
    flag = "✅" if st["dev_ic"] > 0.005 else ("🟡" if st["dev_ic"] > 0 else "❌")
    print(f"  {name:26s} DEV IC {st['dev_ic']:+.4f} ICIR {st['dev_icir']:+.3f} "
          f"| 全期 {st['full_ic']:+.4f} | 正IC {st['pos_share']:.0%} {flag}", flush=True)
    res[key] = st


def roll_mean(a, w):
    """向量化滚动均值（nan 感知）。"""
    df = pd.DataFrame(a)
    return df.rolling(w, min_periods=max(3, w // 2)).mean().to_numpy()


def roll_sum(a, w):
    df = pd.DataFrame(a)
    return df.rolling(w, min_periods=max(3, w // 2)).sum().to_numpy()


def roll_max_idx(H_, w):
    """向量化：过去 w+1 天里最高点距今多少天。"""
    T_, N_ = H_.shape
    out = np.full((T_, N_), np.nan)
    # 分块计算，避免内存爆炸
    for start in range(w, T_):
        seg = H_[start - w:start + 1]           # (w+1, N)
        out[start] = w - np.argmax(seg, axis=0)
    return out


res = {}
body = Cl - O
rng = np.maximum(H - L, 1e-9)
upper = H - np.maximum(O, Cl)
lower = np.minimum(O, Cl) - L

print("\n" + "=" * 96, flush=True)
print("一、形态识别族（TA-Lib 61 个 CDL*，取代表性）", flush=True)
print("=" * 96, flush=True)

hammer = np.where((lower > 2 * np.abs(body)) & (upper < 0.3 * np.abs(body)),
                  lower / rng, 0.0)
ic_of(f32(hammer), "CDLHAMMER", res)

prev_c = np.roll(Cl, 1, axis=0)
prev_o = np.roll(O, 1, axis=0)
prev_b = np.roll(body, 1, axis=0)
bull_engulf = np.where((body > 0) & (prev_b < 0) & (O <= prev_c) & (Cl >= prev_o),
                       np.abs(body) / rng, 0.0)
bull_engulf[0] = 0
ic_of(f32(bull_engulf), "CDLENGULFING", res)

doji = np.where(np.abs(body) < 0.1 * rng, 1.0, 0.0)
ic_of(f32(doji), "CDLDOJI", res)

c1 = (Cl > O).astype(float)
three_up = np.zeros((T, N))
for i in range(2, T):
    three_up[i] = ((c1[i] + c1[i - 1] + c1[i - 2]) >= 3) & (Cl[i] > Cl[i - 1]) & (Cl[i - 1] > Cl[i - 2])
ic_of(f32(three_up), "CDL3WHITESOLDIERS", res)

ic_of(f32(lower / rng), "lower_shadow_ratio", res)
ic_of(f32(upper / rng), "upper_shadow_ratio", res)

print("\n" + "=" * 96, flush=True)
print("二、AROON（趋势新鲜度）", flush=True)
print("=" * 96, flush=True)
for w in (10, 20):
    ds = roll_max_idx(H, w)
    aroon_up = (w - ds) / w * 100.0
    ic_of(f32(aroon_up), f"AROON_UP_{w}", res)

print("\n" + "=" * 96, flush=True)
print("三、CCI（顺势指标）", flush=True)
print("=" * 96, flush=True)
for w in (14, 20):
    tp = (H + L + Cl) / 3.0
    ma_tp = roll_mean(tp, w)
    dev_tp = np.abs(tp - ma_tp)
    md_tp = roll_mean(dev_tp, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = np.where(md_tp > 0, (tp - ma_tp) / (0.015 * md_tp), np.nan)
    ic_of(f32(cci), f"CCI_{w}", res)

print("\n" + "=" * 96, flush=True)
print("四、MFI（资金流量）", flush=True)
print("=" * 96, flush=True)
for w in (14, 20):
    tp = (H + L + Cl) / 3.0
    mf = tp * V
    prev_tp = np.roll(tp, 1, axis=0)
    up_flow = np.where(tp > prev_tp, mf, 0.0)
    dn_flow = np.where(tp < prev_tp, mf, 0.0)
    up_flow[0] = 0
    dn_flow[0] = 0
    us = roll_sum(up_flow, w)
    ds2 = roll_sum(dn_flow, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        mfi = np.where(us + ds2 > 0, 100.0 * us / (us + ds2), 50.0)
    ic_of(f32(mfi - 50.0), f"MFI_{w}", res)

print("\n" + "=" * 96, flush=True)
print("五、BOP（均势）", flush=True)
print("=" * 96, flush=True)
for w in (10, 20):
    bop1 = np.where(rng > 0, body / rng, 0.0)
    ic_of(f32(roll_mean(bop1, w)), f"BOP_{w}", res)

print("\n" + "=" * 96, flush=True)
print("六、周期强度（HT_DCPERIOD 近似）", flush=True)
print("=" * 96, flush=True)
ret = Cl[1:] / Cl[:-1] - 1.0
ret = np.vstack([np.full((1, N), np.nan), ret])
for w in (20, 40):
    r = pd.DataFrame(ret)
    x = r.to_numpy()
    # 向量化自相关：lag1 相关系数 = cov(r_t, r_{t-1}) / (std * std)
    r1 = np.vstack([np.full((1, N), np.nan), x[:-1]])
    def roll_cov(a, b, ww):
        return roll_mean(a * b, ww) - roll_mean(a, ww) * roll_mean(b, ww)
    cov = roll_cov(x, r1, w)
    sx = np.sqrt(np.maximum(roll_mean(x * x, w) - roll_mean(x, w) ** 2, 0))
    sy = np.sqrt(np.maximum(roll_mean(r1 * r1, w) - roll_mean(r1, w) ** 2, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ac = np.where((sx > 0) & (sy > 0), cov / (sx * sy), np.nan)
    ic_of(f32(ac), f"HT_CYCLE_AC_{w}", res)

print("\n" + "=" * 96, flush=True)
print("七、汇总", flush=True)
print("=" * 96, flush=True)
valid = {k: v for k, v in res.items() if v}
good = {k: v for k, v in valid.items() if v["dev_ic"] > 0.005}
mid = {k: v for k, v in valid.items() if 0 < v["dev_ic"] <= 0.005}
bad = {k: v for k, v in valid.items() if v["dev_ic"] <= 0}

print(f"\n  ✅ 显著正 IC (>0.005)：{len(good)} 个", flush=True)
for k, v in sorted(good.items(), key=lambda x: -x[1]["dev_ic"]):
    print(f"     {k:24s} DEV IC {v['dev_ic']:+.4f}", flush=True)
print(f"\n  🟡 微弱正 IC：{len(mid)} 个", flush=True)
for k, v in sorted(mid.items(), key=lambda x: -x[1]["dev_ic"]):
    print(f"     {k:24s} DEV IC {v['dev_ic']:+.4f}", flush=True)
print(f"\n  ❌ 负 IC：{len(bad)} 个", flush=True)
for k, v in sorted(bad.items(), key=lambda x: x[1]["dev_ic"]):
    print(f"     {k:24s} DEV IC {v['dev_ic']:+.4f}", flush=True)

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}", flush=True)
