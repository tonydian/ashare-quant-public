# -*- coding: utf-8 -*-
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import data as qfdata

md = qfdata.load_market(verbose=False)
rows = np.arange(md.date_index('2017-01-03'), md.T, 5)
p = pd.DataFrame(np.where(md.alive, md.price.astype(np.float64), np.nan))
elig = md.eligible[rows]
lab = md.fwd_open_ret(5)[rows]


def top_excess(f, k=50):
    arr = f.to_numpy(dtype=np.float64)[rows]
    vals = []
    for i in range(len(rows)):
        v = elig[i] & np.isfinite(arr[i]) & np.isfinite(lab[i])
        if v.sum() < 100:
            vals.append(np.nan)
            continue
        idx = np.flatnonzero(v)
        order = idx[np.argsort(-arr[i][idx])]
        vals.append(np.nanmean(lab[i][order[:k]]) - np.nanmean(lab[i][idx]))
    return np.array(vals)


out = {}
for k in (10, 20, 40, 60, 120):
    out[f"mom{k}"] = top_excess(p / p.shift(k) - 1.0)
    out[f"mom{k}s5"] = top_excess(p.shift(5) / p.shift(k + 5) - 1.0)
df = pd.DataFrame(out, index=md.dates[rows])
df["year"] = df.index.year
print("=== top50 相对选股域等权超额(%)，正=动量有效 ===")
print((df.groupby("year").mean() * 100).round(3).to_string())
print()
print("=== 全期 ===")
for c in out:
    v = df[c].mean()
    print(f"  {c:10s}: {v*100:+.4f}%/5d ann {((1+v)**50-1)*100:+7.2f}%  DEV {df.loc[df.year<=2019, c].mean()*100:+.4f}%")
