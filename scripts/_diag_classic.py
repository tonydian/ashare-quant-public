# -*- coding: utf-8 -*-
"""诊断：经典动量/反转/低波等简单因子的多头超额（分年），评估是否值得作为显式 tilt。"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import data as qfdata

md = qfdata.load_market(verbose=False)
T = md.T
rows = np.arange(md.date_index('2017-01-03'), T, 5)
p = md.price.astype(np.float64)
pe = np.where(md.alive, p, np.nan)
pdf = pd.DataFrame(pe)

def cs_rank_matrix(x, mask):
    out = np.full_like(x, np.nan, dtype=np.float32)
    for i in range(len(rows)):
        t = rows[i]
        v = mask[i] & np.isfinite(x[i])
        if v.sum() < 100:
            continue
        idx = np.flatnonzero(v)
        order = np.argsort(x[i][idx], kind="stable")
        r = np.empty(len(idx)); r[order] = np.arange(len(idx)) / max(len(idx) - 1, 1) - 0.5
        out[i, idx] = r
    return out

# 因子（在 rows 上）
feats = {}
ret20 = (pdf.shift(20) / pdf - 1.0) * -1  # 这里用过去20日收益
ret50 = (pdf.shift(50) / pdf - 1.0) * -1
ret120 = (pdf.shift(120) / pdf - 1.0) * -1
rev5 = (pdf.shift(5) / pdf - 1.0) * -1
for name, f in [("mom20", ret20), ("mom50", ret50), ("mom120", ret120), ("rev5", rev5)]:
    arr = f.to_numpy(dtype=np.float64)[rows]
    feats[name] = arr
elig = md.eligible[rows]
min_mask = elig.copy()
lab = md.fwd_open_ret(5)[rows]
out = {}
for name, x in feats.items():
    ranks = cs_rank_matrix(x, elig)
    rec = []
    for i in range(len(rows)):
        m = elig[i] & np.isfinite(ranks[i]) & np.isfinite(lab[i])
        if m.sum() < 100:
            rec.append(np.nan); continue
        idx = np.flatnonzero(m)
        order = idx[np.argsort(-ranks[i][idx])]
        k = min(50, len(order))
        rec.append(np.nanmean(lab[i][order[:k]]) - np.nanmean(lab[i][idx]))
    out[name] = np.array(rec)
df = pd.DataFrame(out, index=md.dates[rows])
df['year'] = df.index.year
print("=== top50 - univ 平均 5 日超额(%)，分年 ===")
print((df.groupby('year').mean() * 100).round(3).to_string())
print()
print("=== 全期平均（5日 %）与年化 ===")
for c in out:
    v = df[c].mean()
    print(f"  {c:8s}: {v*100:+.4f}%/5d  ann {((1+v)**50-1)*100:+7.2f}%")
