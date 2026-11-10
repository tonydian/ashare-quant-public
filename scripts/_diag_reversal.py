# -*- coding: utf-8 -*-
"""诊断：不同回看窗口的“反转”多头超额（top50 = 过去 k 日跌幅最大者），分年。"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import data as qfdata

md = qfdata.load_market(verbose=False)
rows = np.arange(md.date_index('2017-01-03'), md.T, 5)
p = pd.DataFrame(np.where(md.alive, md.price.astype(np.float64), np.nan))
elig = md.eligible[rows]
lab = md.fwd_open_ret(5)[rows]

def ranks(f):
    arr = f.to_numpy(dtype=np.float64)[rows]
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for i in range(len(rows)):
        v = elig[i] & np.isfinite(arr[i])
        if v.sum() < 100:
            continue
        idx = np.flatnonzero(v)
        order = np.argsort(-arr[i][idx], kind="stable")  # 值大 → 排名高
        r = np.empty(len(idx)); r[order] = np.arange(len(idx)) / max(len(idx) - 1, 1)
        out[i, idx] = r
    return out

factors = {}
for k in (5, 10, 20, 40, 60, 120):
    factors[f"rev{k}"] = -(p / p.shift(k) - 1.0)     # 反转因子：过去跌幅越大值越大
factors["rev20_skip5"] = -((p.shift(5) / p.shift(25)) - 1.0)
rec = {}
for name, f in factors.items():
    rk = ranks(f)
    vals = []
    for i in range(len(rows)):
        m = elig[i] & np.isfinite(rk[i]) & np.isfinite(lab[i])
        if m.sum() < 100:
            vals.append(np.nan); continue
        idx = np.flatnonzero(m)
        order = idx[np.argsort(-rk[i][idx])]
        vals.append(np.nanmean(lab[i][order[:50]]) - np.nanmean(lab[i][idx]))
    rec[name] = vals
df = pd.DataFrame(rec, index=md.dates[rows])
df["year"] = df.index.year
print("=== top50 - univ 平均 5 日超额(%) 分年（正 = 反转有效） ===")
print((df.groupby("year").mean() * 100).round(3).to_string())
print()
print("=== 全期平均 ===")
for c in rec:
    v = df[c].mean()
    print(f"  {c:14s}: {v*100:+.4f}%/5d  ann {((1+v)**50-1)*100:+7.2f}%  "
          f"DEV {(df.loc[df.index.year<=2019, c].mean()*100):+.4f}%")
