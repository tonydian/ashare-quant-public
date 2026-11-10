# -*- coding: utf-8 -*-
"""一致性盘点：哪些产物基于原始包(10-14)、哪些基于 ext(10-23)、哪些是信号(10-22)。

结论用于决定"正式口径"：模型/回测用原始包（T=2428，与报告 2020-01-08~2026-10-14 一致），
实时信号用 ext（最新交易日）；右侧模型因依赖 ext 周频行，随 ext 更新。
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
print("模型 scores 的长度（=训练数据 T）：")
for n in ("wf_h10", "wf_v1", "wf_h10_fund", "wf_rs10"):
    z = np.load(ROOT / f"build/models/{n}/composite_score.npz")
    rows = z["rows"]
    print(f"  {n:14s} T={z['scores'].shape[0]}  rows={len(rows)}  最后周频行索引={int(rows[-1])}")

m = json.load(open(ROOT / "build/runs/main/metrics.json", encoding="utf-8"))
print(f"\n主回测：{m['start']} ~ {m['end']}（T={m['n_days']}）")

z = np.load(r"D:\wkf\getrich\data_extract\was_stock_data_ext\panel_ohlcv.npz")
d = pd.DatetimeIndex(z["dates"])
print(f"ext 面板：{d[0].date()} ~ {d[-1].date()}，T={len(d)}")

z0 = np.load(r"D:\wkf\getrich\data_extract\was_stock_data\panel_ohlcv.npz")
d0 = pd.DatetimeIndex(z0["dates"])
print(f"原始面板：{d0[0].date()} ~ {d0[-1].date()}，T={len(d0)}")

s = json.load(open(ROOT / "signals/latest_signal.json", encoding="utf-8"))
print(f"\n实盘信号：{s['signal_date']}，持仓 {s['n_holdings']} 只")

print("""
口径说明（重要）：
  * 模型训练/回测报告：用**原始包**（2016-10-14~2026-10-14），保证与报告 §1 的 2020-01-08~2026-10-14 一致；
  * 实时信号：用 ext（最新交易日），模型权重固定（不重训），只做最新截面打分；
  * 右侧模型 wf_rs10 因 build_rs2_model.py 默认读 ext，其 scores 会随 ext 增长
    → 这是"最新数据"，不是口径错误；但若要做严格同区间对比，需固定数据源。
""")
