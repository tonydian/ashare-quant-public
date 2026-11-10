# -*- coding: utf-8 -*-
"""精确定价验证：主策略成交价是否 == 成交日开盘价（含滑点）。"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
os.environ.setdefault("QF_DATA_RAW", str(ROOT.parent / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm  # noqa: E402

md = qfdata.load_market(verbose=False)
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
scores = np.asarray(res.scores)
if scores.shape[0] < md.T:
    scores = np.vstack([scores, np.full((md.T - scores.shape[0], md.N), np.nan, np.float32)])
elif scores.shape[0] > md.T:
    scores = scores[:md.T]

start = int(md.date_index("2020-01-08"))
end = int(md.date_index("2026-10-14"))
phase = int(res.rows_weekly[0]) % 5
offset = (phase - start) % 10
cash = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
old = C.N_HOLDINGS
C.N_HOLDINGS = 40
try:
    r = qfbt.run_backtest(md, scores, start, end, risk_overlay=True, stop_loss=True,
                          rebalance_days=10, rebalance_offset=offset,
                          cash_asset_returns=cash, verbose=False)
finally:
    C.N_HOLDINGS = old

tr = r.trades
print(f"交易 {len(tr)} 笔")
print(f"\n{'日期':<12}{'代码':<8}{'方向':<6}{'成交价':>10}{'当日开盘':>10}{'当日收盘':>10}{'前日收盘':>10}{'匹配':>10}")
print("-" * 86)

n_open = n_close = n_prev = n_other = 0
for row in tr.head(25).itertuples():
    d = pd.Timestamp(row.date)
    j = md.code_index(str(row.code))
    if j < 0:
        continue
    t = int(md.date_index(d))
    po = md.price_open[t, j]
    pc = md.price[t, j]
    pp = md.price[t - 1, j] if t > 0 else np.nan
    px = float(row.price)
    # 判断匹配（含滑点约 5bp）
    def near(a, b):
        return np.isfinite(a) and np.isfinite(b) and abs(a / b - 1) < 0.002
    if near(px, po):
        tag = "开盘价✅"; n_open += 1
    elif near(px, pc):
        tag = "收盘价⚠️"; n_close += 1
    elif near(px, pp):
        tag = "前收盘⚠️⚠️"; n_prev += 1
    else:
        tag = "其他"; n_other += 1
    print(f"{str(d.date()):<12}{row.code:<8}{row.side:<6}{px:>10.4f}{po:>10.4f}"
          f"{pc:>10.4f}{pp:>10.4f}{tag:>10}")

print("-" * 86)
print(f"前 25 笔统计：开盘价 {n_open} | 收盘价 {n_close} | 前日收盘 {n_prev} | 其他 {n_other}")

# 全量验证
cnt = {"open": 0, "close": 0, "prev": 0, "other": 0}
for row in tr.itertuples():
    d = pd.Timestamp(row.date)
    j = md.code_index(str(row.code))
    if j < 0:
        continue
    t = int(md.date_index(d))
    po = md.price_open[t, j] if np.isfinite(md.price_open[t, j]) else np.nan
    pc = md.price[t, j]
    pp = md.price[t - 1, j] if t > 0 else np.nan
    px = float(row.price)
    def near(a, b):
        return np.isfinite(a) and np.isfinite(b) and abs(a / b - 1) < 0.002
    if near(px, po):
        cnt["open"] += 1
    elif near(px, pc):
        cnt["close"] += 1
    elif near(px, pp):
        cnt["prev"] += 1
    else:
        cnt["other"] += 1

print(f"\n全量 {len(tr)} 笔统计：")
print(f"  成交价 == 当日开盘价 : {cnt['open']}")
print(f"  成交价 == 当日收盘价 : {cnt['close']}")
print(f"  成交价 == 前日收盘价 : {cnt['prev']}")
print(f"  其他（滑点/涨跌停等） : {cnt['other']}")
tot = sum(cnt.values())
print(f"\n判定：开盘价占比 {cnt['open']/tot:.1%}")
if cnt["open"] / tot > 0.8:
    print("  ✅ 主策略使用【次日开盘价】成交 —— 时序正确，无同 bar 前视")
elif cnt["close"] / tot > 0.5:
    print("  ⚠️ 主策略可能使用【当日收盘价】成交 —— 需检查是否有前视")
else:
    print("  需人工确认成交价语义")
