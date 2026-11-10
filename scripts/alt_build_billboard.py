# -*- coding: utf-8 -*-
"""龙虎榜因子构建 + DEV 期 IC + 正交性检验。

龙虎榜的经济含义：
  * 上榜 = 当日异常波动被交易所公开披露（通常涨跌幅达 7%/换手 20% 等）
  * 席位明细揭示"谁在买"：机构专用 vs 游资营业部
  * 上榜后 1-10 日表现（D1/D2/D5/D10_CLOSE_ADJCHRATE）已被东财算好，但**那是未来数据，不可用**

可用（时点可知）：
  当日上榜次数、上榜成交额占比、买卖净额、机构席位净买入、游资席位净买入
  以及过去 N 日的累计上榜频率

因子设计：
  LHB_HIT        当日是否上榜（0/1）
  LHB_HIT_20     过去 20 日上榜次数
  LHB_NET        当日龙虎榜净买入额 / 成交额
  LHB_NET_20     过去 20 日累计净买入 / 累计成交额
  LHB_RATIO      上榜成交额 / 当日总成交额（被关注度）
  LHB_INST_NET   机构专用席位净买入 / 成交额
  LHB_YOUZI_NET  游资席位净买入 / 成交额
  LHB_INST_20    过去 20 日机构净买入累计
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

# ---------------- 1) 龙虎榜明细 → 面板
print("[1/5] 龙虎榜明细 → 面板…", flush=True)
hit = np.zeros((T, N), np.float32)          # 上榜标记
deal_amt = np.zeros((T, N), np.float32)     # 上榜成交额
net_amt = np.zeros((T, N), np.float32)      # 净买入额
buy_amt = np.zeros((T, N), np.float32)
sell_amt = np.zeros((T, N), np.float32)
deal_ratio = np.zeros((T, N), np.float32)   # 上榜成交额 / 总成交额

p = ALT / "billboard_detail.jsonl"
n_days = n_rows = 0
if p.exists():
    with open(p, encoding="utf-8") as f:
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
                sc = str(r.get("SECURITY_CODE") or "").zfill(6)
                j = code_idx.get(sc)
                if j is None:
                    continue
                n_rows += 1

                def num(k):
                    v = r.get(k)
                    try:
                        return float(v) if v is not None else 0.0
                    except Exception:
                        return 0.0

                hit[di, j] = 1.0
                deal_amt[di, j] += num("BILLBOARD_DEAL_AMT")
                net_amt[di, j] += num("BILLBOARD_NET_AMT")
                buy_amt[di, j] += num("BILLBOARD_BUY_AMT")
                sell_amt[di, j] += num("BILLBOARD_SELL_AMT")
                deal_ratio[di, j] += num("DEAL_AMOUNT_RATIO")

print(f"  读取 {n_days} 天 / {n_rows} 行")
print(f"  上榜覆盖: {hit.mean():.3%} 股票-日有上榜")

# ---------------- 2) 席位 → 机构/游资净买入
print("\n[2/5] 席位明细 → 机构/游资分化…", flush=True)
inst_net = np.zeros((T, N), np.float32)
youzi_net = np.zeros((T, N), np.float32)
seat_cnt = np.zeros((T, N), np.float32)

p2 = ALT / "billboard_seat.jsonl"
n_seat = 0
if p2.exists():
    with open(p2, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            di = date_pos.get(pd.Timestamp(obj["date"]).normalize())
            if di is None:
                continue
            for r in obj["rows"]:
                sc = str(r.get("SECURITY_CODE") or "").zfill(6)
                j = code_idx.get(sc)
                if j is None:
                    continue
                n_seat += 1
                name = str(r.get("OPERATEDEPT_NAME") or "")
                try:
                    net = float(r.get("NET") or 0.0)
                except Exception:
                    net = 0.0
                seat_cnt[di, j] += 1
                if "机构专用" in name:
                    inst_net[di, j] += net
                else:
                    youzi_net[di, j] += net
    print(f"  席位记录 {n_seat} 条")
else:
    print("  席位数据缺失（跳过机构/游资分化）")

# ---------------- 3) 构造因子
print("\n[3/5] 构造因子…", flush=True)
amt = md.amount.astype(np.float64)


def roll(a, w, fn="mean"):
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=max(2, w // 2))
    return (r.mean() if fn == "mean" else r.sum()).to_numpy()


with np.errstate(divide="ignore", invalid="ignore"):
    lhb_net_ratio = np.where(amt > 0, net_amt / amt, np.nan)
    lhb_inst_ratio = np.where(amt > 0, inst_net / amt, np.nan)
    lhb_yz_ratio = np.where(amt > 0, youzi_net / amt, np.nan)

fac = {
    "lhb:HIT": hit,
    "lhb:HIT_20": roll(hit, 20, "sum"),
    "lhb:NET_RATIO": lhb_net_ratio,
    "lhb:NET_20": roll(lhb_net_ratio, 20, "sum"),
    "lhb:DEAL_RATIO": deal_ratio,
    "lhb:INST_NET": lhb_inst_ratio,
    "lhb:YZ_NET": lhb_yz_ratio,
    "lhb:INST_20": roll(lhb_inst_ratio, 20, "sum"),
    "lhb:SEAT_CNT": seat_cnt,
    "lhb:INST_SHARE": np.where(
        np.abs(inst_net) + np.abs(youzi_net) > 0,
        np.abs(inst_net) / (np.abs(inst_net) + np.abs(youzi_net)), np.nan),
}
fac = {k: np.asarray(v, np.float32) for k, v in fac.items()}
for k, v in fac.items():
    print(f"  {k:<18} 非NaN {np.isfinite(v).mean():6.1%} 非零 {np.count_nonzero(np.nan_to_num(v))/v.size:6.2%}")

np.savez_compressed(ALT / "billboard_panel.npz",
                    dates=dates.values, codes=np.array([str(c) for c in md.codes]),
                    hit=hit, net_amt=net_amt, deal_amt=deal_amt,
                    inst_net=inst_net, youzi_net=youzi_net, seat_cnt=seat_cnt)

# ---------------- 4) DEV 期 IC
print("\n[4/5] 周频 IC（仅 DEV 期决策）…", flush=True)
zm = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
rows = zm["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(DEV_END))
val = np.asarray((dates_w >= "2020-01-01") & (dates_w <= "2021-12-31"))
oos = np.asarray(dates_w >= "2022-01-01")

res = {}
print(f"\n  {'因子':<18}{'DEV IC':>10}{'ICIR':>9}{'验证IC':>10}{'样本外IC':>10}{'判定':>6}")
print("  " + "-" * 66)
for k, pan in fac.items():
    f = qfm.build_feature_block(md, rows, {k: pan[rows]}, k, mask)
    ic = qfm.weekly_ic(f, label, mask)
    v = np.isfinite(ic)
    if v.sum() < 20:
        continue
    d = ic[v & dev]
    if len(d) < 15:
        print(f"  {k:<18} DEV 不足({len(d)}周)")
        continue
    d_ic = float(np.mean(d))
    d_icir = d_ic / max(float(np.std(d)), 1e-9)
    v_ic = float(np.mean(ic[v & val])) if (v & val).sum() else float("nan")
    o_ic = float(np.mean(ic[v & oos])) if (v & oos).sum() else float("nan")
    flag = "✅" if abs(d_ic) > 0.012 and abs(d_icir) > 0.20 else (
        "🟡" if abs(d_ic) > 0.006 else "❌")
    res[k] = {"dev_ic": d_ic, "dev_icir": d_icir, "val_ic": v_ic, "oos_ic": o_ic}
    print(f"  {k:<18}{d_ic:>+10.4f}{d_icir:>+9.3f}{v_ic:>+10.4f}{o_ic:>+10.4f}{flag:>6}")

# ---------------- 5) 正交性
print("\n[5/5] 与现有一切因子的正交性（越正交越有价值）…", flush=True)
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
    print(f"    {gk:<18} 与量价最高相关 {mx:+.3f}（{mxk}）")

(ALT / "billboard_factor_ic.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {ALT / 'billboard_factor_ic.json'}")
