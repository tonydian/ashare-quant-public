# -*- coding: utf-8 -*-
"""临时冒烟测试：用少量真实因子验证 qf.model 的 walk-forward 代码路径。"""
import sys, pathlib, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import config as C, data as qfdata, factors as qff, model as qfm

t0 = time.time()
md = qfdata.load_market(verbose=True)
print(f"load md {time.time()-t0:.1f}s")
inp = qff.FactorInputs(md)
cat = qff.load_catalog()
keys = ["R020", "R001", "R021", "R083", "R011"]
panels = {}
for fid in keys:
    sub = cat[cat["factor_id"] == fid]
    if not len(sub):
        continue
    e = sub.iloc[0]
    t1 = time.time()
    f = qff.compute_factor(inp, e)
    panels[e["factor_key"]] = f
    print(f"  {e['factor_key']:22s} {time.time()-t1:.2f}s finite={np.isfinite(f).mean():.3f}")

start = md.date_index(C.DEV_START)
rows_weekly = np.arange(start, md.T, C.REBALANCE_DAYS)
mask = md.eligible[rows_weekly]
lab = qfm.weekly_label(md, rows_weekly, mask, 5, rank=True)
allp = {}
for k, f in panels.items():
    allp[k] = f[rows_weekly]
cfg = qfm.ModelConfig(max_factors=4, corr_max=0.99, train_weeks=104, refit_weeks=26)
res = qfm.walk_forward(md, rows_weekly, allp, cat, cfg=cfg, start_date=C.VALID_START, verbose=True)
print("pool:", res.pool)
print(res.weights.tail(10).to_string(index=False))
print("scores finite rows:", int(np.isfinite(res.scores).any(axis=1).sum()))
if len(res.composite_ic):
    ic = res.composite_ic["ic"]
    print(f"OOS IC mean {ic.mean():.4f} ir {ic.mean()/ic.std():.3f} n={len(ic)}")
print(f"total {time.time()-t0:.1f}s")
