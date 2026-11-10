# -*- coding: utf-8 -*-
import sys, time, numpy as np
sys.path.insert(0, r"D:\wkf\getrich\ashare_quant")
from qf import data as qfdata, factors as qff

t0 = time.time()
md = qfdata.load_market(verbose=False)
print(f"load {time.time()-t0:.1f}s")
inp = qff.FactorInputs(md)
cat = qff.load_catalog()
for fid in ["R020", "R021", "R001", "R083", "R011", "R195", "R097", "R033"]:
    e = cat[cat["factor_id"] == fid].iloc[0]
    t1 = time.time()
    f = qff.compute_factor(inp, e)
    print(f"{e['factor_key']:22s} {time.time()-t1:6.2f}s finite={np.isfinite(f).mean():.3f}")

# 模拟 build_factors 的分组复用路径：逐组算 base 一次 + 8 transforms
t0 = time.time()
n = 0
for base, w, g in qff.iter_groups(cat):
    p_list = [dict(r["params"]) for _, r in g.iterrows()]
    trans = g["transform"].tolist()
    out = qff.compute_group(inp, base, w, trans, p_list)
    n += len(trans)
    if n >= 80:
        break
print(f"group path: {n} factors in {time.time()-t0:.1f}s -> est 800: {(time.time()-t0)/n*800/60:.1f} min")
