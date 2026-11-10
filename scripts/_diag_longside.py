# -*- coding: utf-8 -*-
"""诊断：不同持仓数的多头超额（top-k vs 选股域等权），分年度。"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import data as qfdata, model as qfm

md = qfdata.load_market(verbose=False)
res = qfm.load_result('build/models/wf_v1')
sc, rows = res.scores, res.rows_weekly
lab = md.fwd_open_ret(5)
elig = md.eligible[rows]
ks = [5, 10, 20, 50, 100, 200]
rec = {str(k): [] for k in ks}
rec['univ'] = []
dates = []
for i, t in enumerate(rows):
    s = sc[t]
    m = elig[i] & np.isfinite(s) & np.isfinite(lab[t])
    if m.sum() < 100:
        continue
    idx = np.flatnonzero(m)
    order = idx[np.argsort(-s[idx])]
    y = lab[t]
    for k in ks:
        rec[str(k)].append(np.nanmean(y[order[:min(k, len(order))]]))
    rec['univ'].append(np.nanmean(y[order]))
    dates.append(md.dates[t])
df = pd.DataFrame(rec, index=dates)
df['year'] = df.index.year
print("=== 全期平均 5 日收益与年化（未扣费、满仓） ===")
for k in [str(x) for x in ks] + ['univ']:
    v = df[k].mean()
    print(f"  top{k:>4}: 5d {v*100:+.4f}%   ann {((1+v)**50-1)*100:+6.2f}%")
print()
print("=== 分年（5日收益 %） ===")
print((df.groupby('year')[['5', '10', '20', '50', '100', 'univ']].mean() * 100).round(3).to_string())
print()
print("=== 多头超额（top-k - univ，5日 %）分年 ===")
for k in ['5', '10', '20', '50', '100']:
    print("top", k)
    print(((df[k] - df['univ']).groupby(df['year']).mean() * 100).round(3).to_string())
