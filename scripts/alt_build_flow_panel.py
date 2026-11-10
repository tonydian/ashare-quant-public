# -*- coding: utf-8 -*-
"""C 方案第二步：资金流原始数据 → 面板 → 因子 → IC 检验。

流程：
  1) flow_raw.jsonl → (T, N) 面板（净流入占比、超大单占比、换手）
  2) 构造 6 个资金流因子
  3) 严格用 DEV 期（2017-2019）检验 IC
  4) 若 DEV 期有效 → 才继续做模型；否则停在这里

因子设计（全部为"时点可知"）：
  FLOW_MA5     近 5 日净流入占成交额比例的均值
  FLOW_MA20    近 20 日均值
  FLOW_SURGE   今日净流入占比 / 过去 20 日均值绝对值（放量流入）
  R0_MA5       近 5 日超大单净占比均值
  R0_SURGE     超大单放量
  FLOW_REV     净流入 20 日累计 / 换手率（标准化）
  TURN_Z       换手率 z 分（存量数据，用于对照）
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
print(f"行情面板 {T}×{N}，{dates[0].date()} ~ {dates[-1].date()}")

# ---------------- 1) 读原始 JSONL → 面板
raw = ALT / "flow_raw.jsonl"
if not raw.exists():
    raise SystemExit(f"缺 {raw}，先跑 alt_fetch_flow.py")

print("\n[1/4] 构建面板…", flush=True)
net_pct = np.full((T, N), np.nan, dtype=np.float32)   # 净流入 / 成交额
r0_pct = np.full((T, N), np.nan, dtype=np.float32)    # 超大单净流入占比
turn = np.full((T, N), np.nan, dtype=np.float32)      # 换手率

date_pos = {d.normalize(): i for i, d in enumerate(dates)}
n_rec = n_ok = 0
with open(raw, encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        j = code_idx.get(str(obj["code"]).zfill(6))
        if j is None:
            continue
        n_ok += 1
        for r in obj["rows"]:
            n_rec += 1
            d = pd.Timestamp(r.get("opendate"))
            di = date_pos.get(d.normalize())
            if di is None:
                continue
            try:
                net_pct[di, j] = float(r.get("ratioamount") or "nan") * 100
            except Exception:
                pass
            try:
                r0_pct[di, j] = float(r.get("r0_ratio") or "nan") * 100
            except Exception:
                pass
            try:
                turn[di, j] = float(r.get("turnover") or "nan")
            except Exception:
                pass

print(f"  读取 {n_ok} 只 / {n_rec} 条记录")
print(f"  面板覆盖率: net {np.isfinite(net_pct).mean():.1%} | "
      f"r0 {np.isfinite(r0_pct).mean():.1%} | turn {np.isfinite(turn).mean():.1%}")

# 有效起始日（覆盖率 > 30% 的第一天）
cov = np.isfinite(net_pct).sum(axis=1) / max(N, 1)
ok_days = np.flatnonzero(cov > 0.3)
print(f"  有效日期: {dates[ok_days[0]].date()} ~ {dates[ok_days[-1]].date()}"
      f"（{len(ok_days)} 天覆盖率 >30%）")

np.savez_compressed(ALT / "flow_panel.npz",
                    dates=dates.values, codes=np.array([str(c) for c in md.codes]),
                    net_pct=net_pct, r0_pct=r0_pct, turnover=turn)
print(f"  → {ALT / 'flow_panel.npz'}")

# ---------------- 2) 构造因子
print("\n[2/4] 构造资金流因子…", flush=True)


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.sum()).to_numpy()


def roll_absmean(a, w):
    return pd.DataFrame(np.abs(a)).rolling(w, min_periods=max(2, w // 2)).mean().to_numpy()


fac = {}
fac["FLOW_MA5"] = roll(net_pct, 5)
fac["FLOW_MA20"] = roll(net_pct, 20)
fac["FLOW_SUM20"] = roll(net_pct, 20, "sum")
with np.errstate(divide="ignore", invalid="ignore"):
    fac["FLOW_SURGE"] = np.where(roll_absmean(net_pct, 20) > 0.05,
                                 net_pct / roll_absmean(net_pct, 20), np.nan)
fac["R0_MA5"] = roll(r0_pct, 5)
fac["R0_MA20"] = roll(r0_pct, 20)
with np.errstate(divide="ignore", invalid="ignore"):
    fac["R0_SURGE"] = np.where(roll_absmean(r0_pct, 20) > 0.05,
                               r0_pct / roll_absmean(r0_pct, 20), np.nan)
fac["FLOW_TURN"] = roll(net_pct, 20) / np.where(roll(turn, 20) > 0, roll(turn, 20), np.nan)
fac["TURN_Z"] = (turn - roll(turn, 20)) / np.where(roll_absmean(turn, 20) > 0,
                                                   roll_absmean(turn, 20), np.nan)

for k, v in fac.items():
    v = np.asarray(v, dtype=np.float32)
    fac[k] = v
    print(f"  {k:<12} 非NaN {np.isfinite(v).mean():6.1%}")

# ---------------- 3) 映射到周频 + 算 IC
print("\n[3/4] 周频 IC 检验（仅 DEV 期）…", flush=True)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(DEV_END))
val = np.asarray((dates_w >= "2020-01-01") & (dates_w <= "2021-12-31"))
oos = np.asarray(dates_w >= "2022-01-01")
print(f"  周频 {len(rows)} 行 | DEV {dev.sum()} | 验证 {val.sum()} | 样本外 {oos.sum()}")

res = {}
print(f"\n  {'因子':<12}{'DEV IC':>10}{'DEV ICIR':>10}{'验证IC':>10}{'样本外IC':>10}{'判定':>6}")
print("  " + "-" * 62)
for k, pan in fac.items():
    f = qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    if v.sum() < 20:
        continue
    d = ic[v & dev]
    va = ic[v & val]
    oo = ic[v & oos]
    if len(d) < 10:
        continue
    d_ic = float(np.mean(d))
    d_icir = d_ic / max(float(np.std(d)), 1e-9)
    v_ic = float(np.mean(va)) if len(va) else float("nan")
    o_ic = float(np.mean(oo)) if len(oo) else float("nan")
    flag = "✅" if abs(d_ic) > 0.015 and abs(d_icir) > 0.25 else ("🟡" if abs(d_ic) > 0.008 else "❌")
    res[k] = {"dev_ic": d_ic, "dev_icir": d_icir, "val_ic": v_ic, "oos_ic": o_ic}
    print(f"  {k:<12}{d_ic:>+10.4f}{d_icir:>+10.3f}{v_ic:>+10.4f}{o_ic:>+10.4f}{flag:>6}")

# ---------------- 4) 结论
print("\n" + "=" * 96)
print("[4/4] 结论")
print("=" * 96)
good = {k: v for k, v in res.items() if abs(v["dev_ic"]) > 0.015 and abs(v["dev_icir"]) > 0.25}
print(f"  DEV 期达标（|IC|>0.015 且 |ICIR|>0.25）: {len(good)} / {len(res)} 个")
for k, v in sorted(good.items(), key=lambda x: -abs(x[1]["dev_icir"])):
    direction = "正向" if v["dev_ic"] > 0 else "反向"
    print(f"    {k:<12} DEV IC {v['dev_ic']:+.4f} ICIR {v['dev_icir']:+.3f}（{direction}）"
          f" | 样本外 IC {v['oos_ic']:+.4f}")

if not good:
    print("\n  ❌ 资金流因子在 DEV 期无效 → C 方案在方法论上不成立，停止。")
else:
    print(f"\n  ✅ 有 {len(good)} 个因子可用 → 进入建模阶段")
    print(f"  注意：样本外 IC 仅作记录，**不参与选因子决策**")

(ALT / "flow_factor_ic.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {ALT / 'flow_factor_ic.json'}")
