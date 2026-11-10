# -*- coding: utf-8 -*-
"""调试：单只股票、单因子，定位向量化实现与参考实现的差异。"""
import importlib.util, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qf import data as qfdata, factors as qff, ops

md = qfdata.load_market(verbose=False)
inp = qff.FactorInputs(md)
cat = qff.load_catalog()
code = "000001"
j = md.code_index(code)
print("code", code, "j", j, "first_valid", md.first_valid[j], "last", md.last_valid[j])

df = pd.DataFrame({
    "symbol": code, "date": md.dates,
    "open": inp.open[:, j], "high": inp.high[:, j], "low": inp.low[:, j],
    "close": inp.close[:, j], "volume": inp.volume[:, j],
})

def run_ref(e):
    spec = importlib.util.spec_from_file_location("refx", e["path"] + "/scripts/factor.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = mod.compute_factor(df.copy())
    return out[mod.FACTOR_COLUMN].to_numpy()

for fid in ["R195", "R008", "R009", "R109"]:
    e = cat[cat["factor_id"] == fid].iloc[0]
    ref = run_ref(e)
    mine = qff.compute_factor(inp, e)[:, j]
    m = np.isfinite(ref) & np.isfinite(mine)
    d = np.abs(ref - mine)
    print(f"\n== {fid} {e['base']} {e['transform']} w={e['window']} valid={m.sum()}")
    print("   max diff", np.nanmax(d), " p99", np.nanpercentile(d, 99), " n NaN mismatch:",
          int((np.isfinite(ref) != np.isfinite(mine)).sum()))
    bad = np.flatnonzero(m & (d > 1e-4))[:8]
    for t in bad:
        print(f"   t={t} {md.dates[t].date()} ref={ref[t]:.6f} mine={mine[t]:.6f} close={inp.close[t,j]:.3f}")
    # 打印 warmup 区
    print("   ref[:12]", np.round(ref[:12], 4))
    print("   mine[:12]", np.round(mine[:12], 4))
