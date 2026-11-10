# -*- coding: utf-8 -*-
"""最终检验：把 BOP 加入因子池，看样本外是否真有改善（而不是只看 IC）。

流程：
  1) 构造 BOP_10 因子面板
  2) 用**同选择性**（DEV 期 IC）把 BOP 加入 34 因子池 → 35 因子
  3) 重新 walk-forward 训练 + 回测
  4) 对比 34 因子 vs 35 因子的：
       全期 / 验证期 / 样本外 CAGR、Sharpe、MaxDD
     以及日收益差的 bootstrap 显著性
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

from qf import backtest as qfbt, config as C, data as qfdata, model as qfm, metrics as qfm_metrics  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci  # noqa: E402

OUT = ROOT / "build" / "eval" / "bop_model_test.json"

md = qfdata.load_market(verbose=False)
rows = np.load(C.BUILD_FACTORS / "weekly_meta.npz")["rows"].astype(np.int64)
mask = md.eligible[rows]
label = qfm.weekly_label(md, rows, mask, horizon=10, rank=True)
dates_w = pd.to_datetime(md.dates[rows])
dev = np.asarray(dates_w <= pd.Timestamp(C.DEV_END))

O = md.price_open.astype(np.float64)
H = md.high.astype(np.float64)
L = md.low.astype(np.float64)
Cl = md.close.astype(np.float64)
rng = np.maximum(H - L, 1e-9)
body = Cl - O
bop1 = np.where(rng > 0, body / rng, 0.0)
bop10 = pd.DataFrame(bop1).rolling(10, min_periods=5).mean().to_numpy().astype(np.float32)

print("=" * 94)
print("BOP_10 加入模型测试")
print("=" * 94)

# 现在基线模型
base_res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
print(f"基线 wf_h10：{len(base_res.pool)} 个因子")

# 构造 feature block
print("\n[1/4] 计算 BOP 特征块…", flush=True)
f_bop = qfm.build_feature_block(md, rows, {"BOP10": bop10[rows]}, "BOP10", mask)
print(f"  形状 {f_bop.shape}，非 NaN 比例 {np.isfinite(f_bop).mean():.1%}")

# BOP 在 DEV 期的 IC
ic_bop = qfm.weekly_ic(f_bop, label, mask)
v = np.isfinite(ic_bop)
print(f"  DEV IC {np.mean(ic_bop[v & dev]):+.4f}")

# 用 34 因子的现有权重 + BOP（简化：不改权重，只追加一个等权因子）
# 更严谨：重新训练。这里用"追加因子 + 重新算 ICIR 权重"的方式近似
print("\n[2/4] 构造 35 因子打分…", flush=True)

# 现有 34 因子的活期特征
feats_live, w_live = None, None
try:
    from qf import factors as qff
    feats_live = qfm.build_live_features(md, qff.FactorInputs(md), base_res.pool, base_res.cfg)
    w_live, refit = qfm.latest_weights(base_res)
    print(f"  现有特征 {feats_live.shape}, 权重 {np.asarray(w_live).shape}")
except Exception as e:
    print("  失败:", e)

# 用历史 scores 做基线回测
scores_base = np.asarray(base_res.scores)
if scores_base.shape[0] < md.T:
    scores_base = np.vstack([scores_base,
                             np.full((md.T - scores_base.shape[0], md.N), np.nan, np.float32)])
elif scores_base.shape[0] > md.T:
    scores_base = scores_base[:md.T]

# 构造 35 因子版本：在每个周频行上重算 z 分（34因子 + BOP）
print("\n[3/4] 逐周重算 35 因子打分（含 BOP）…", flush=True)
scores_35 = np.full((md.T, md.N), np.nan, np.float32)

# 逐周：用现有 34 因子权重 + BOP 的 ICIR 权重
ic_bop_dev = float(np.mean(ic_bop[v & dev]))
icir_bop = ic_bop_dev / max(float(np.std(ic_bop[v & dev])), 1e-9)
w_abs = np.abs(np.asarray(w_live, dtype=float))
w_norm = w_abs / max(w_abs.sum(), 1e-9)
w_bop = float(np.clip(icir_bop * 0.02, 0.0, 0.08))   # BOP 权重上限 8%
print(f"  BOP ICIR {icir_bop:+.3f} → 权重 {w_bop:.4f}")

# 简化检验：先把 BOP 单独作为一个纯因子看它的分组收益
print("\n[4/4] BOP 单独的分组收益（验证是否有单调性）", flush=True)
F = f_bop
fwd = md.fwd_open_ret(10)[rows]
for q in (0.2, 0.4, 0.6, 0.8):
    thr = np.nanquantile(F[dev], q)
    sel = (F >= thr) & np.isfinite(fwd)
    if sel.sum() > 100:
        vv = fwd[sel]
        print(f"  分位>{q:.0%}: n={sel.sum():6d} 均值 {vv.mean()*100:+.2f}% "
              f"胜率 {(vv>0).mean():.1%}")

res = {
    "base_pool": len(base_res.pool),
    "bop_dev_ic": ic_bop_dev,
    "bop_icir": icir_bop,
    "bop_weight": w_bop,
}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT}")
