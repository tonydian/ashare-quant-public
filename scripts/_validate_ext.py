# -*- coding: utf-8 -*-
"""校验增量更新后的数据目录（was_stock_data_ext）与原始目录的一致性。"""
import sys, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))
import qf.config as C

EXT = pathlib.Path(r"D:\wkf\getrich\data_extract\was_stock_data_ext")
ORIG = C.DATA_RAW
FILE_KEYS = ["PANEL_FILE", "RAWCLOSE_FILE", "MCAP_FILE", "INDUSTRY_FILE",
             "INDUSTRY_CATS", "META_FILE", "INDICES_FILE"]
orig_paths = {k: getattr(C, k) for k in FILE_KEYS}

# 1) 加载扩展数据
from qf import data as qfdata
from qf import factors as qff
from qf import momentum as qfmom

for k in FILE_KEYS:
    setattr(C, k, EXT / pathlib.Path(orig_paths[k]).name)
md_ext = qfdata.load_market(verbose=True)
T_ext, N = md_ext.T, md_ext.N
print(f"[ext] {T_ext} 交易日 × {N} 只，最后 {md_ext.dates[-1].date()}")

# 2) 加载原始数据（同一进程切换 config 路径）
for k in FILE_KEYS:
    setattr(C, k, orig_paths[k])
md_orig = qfdata.load_market(verbose=False)
T0 = md_orig.T
assert np.array_equal(md_ext.dates[:T0], md_orig.dates)
assert np.array_equal(md_ext.codes, md_orig.codes)
for name in ["close", "open", "high", "low", "volume", "amount", "raw_close", "mcap"]:
    a = getattr(md_ext, name)[:T0]
    b = getattr(md_orig, name)
    m = np.isfinite(a) & np.isfinite(b)
    same = np.array_equal(a[m], b[m])
    n_extra = int((np.isfinite(a) & ~np.isfinite(b)).sum())
    print(f"[check] old rows {name}: equal_on_overlap={same}, ext新增有限值={n_extra}"
          f"（来自面板末尾停牌/退市股在更新数据中恢复存续，属预期）")
    assert same, name
print("[check] industry old rows:", np.array_equal(md_ext.industry[:T0], md_orig.industry))
print("[check] bench old rows:", np.allclose(md_ext.bench.iloc[:T0].to_numpy(),
                                            md_orig.bench.to_numpy(), equal_nan=True))
print("[check] eligible counts:", int(md_ext.eligible[-1].sum()), "at", md_ext.dates[-1].date())
print("[check] eligible 10-14:", int(md_orig.eligible[-1].sum()))
print("[check] new days:", [str(d.date()) for d in md_ext.dates[T0:]])
for i in range(T0, T_ext):
    print(f"   {md_ext.dates[i].date()}: eligible={int(md_ext.eligible[i].sum())}, "
          f"alive={int(md_ext.alive[i].sum())}, fin_close={int(np.isfinite(md_ext.close[i]).sum())}, "
          f"fin_mcap={int(np.isfinite(md_ext.mcap[i]).sum())}")
print("[check] bench tail:")
print(md_ext.bench_prices.tail(6).round(2).to_string())

# 3) 因子不变性（旧行在两边都有限处应完全一致；面板末尾停牌股恢复存续会让新增有限值）
inp_ext = qff.FactorInputs(md_ext)
inp_orig = qff.FactorInputs(md_orig)
cat = qff.load_catalog()
for fid in ["R716", "R610", "R223"]:
    e = cat[cat["factor_id"] == fid].iloc[0]
    f_ext = qff.compute_factor(inp_ext, e)
    f_orig = qff.compute_factor(inp_orig, e)
    m = np.isfinite(f_ext[:T0]) & np.isfinite(f_orig)
    same = np.array_equal(f_ext[:T0][m], f_orig[m])
    print(f"[check] factor {e['factor_key']} old rows equal_on_overlap={same}, "
          f"ext新增有限值={int((np.isfinite(f_ext[:T0]) & ~np.isfinite(f_orig)).sum())}")
    assert same
# momentum-extra 因子
for key in ["momentum-extra:M001", "momentum-extra:M004"]:
    full = qfmom.build_live(md_ext, key)
    full_orig = qfmom.build_live(md_orig, key)
    m = np.isfinite(full[:T0]) & np.isfinite(full_orig)
    same = np.array_equal(full[:T0][m], full_orig[m])
    print(f"[check] {key} old rows equal_on_overlap={same}")
    assert same

# 4) 新行收益率 sanity：与指数相关性
new_rets = []
for i in range(T0, T_ext):
    r = md_ext.close[i] / md_ext.close[i - 1] - 1.0
    new_rets.append(np.nanmedian(r))
print("[check] 新交易日全市场收益中位数:", np.round(new_rets, 4).tolist())
print("        同期 csi_all 收益:",
      np.round(md_ext.bench_prices["csi_all"].pct_change().iloc[T0:].to_numpy(), 4).tolist())
print("VALIDATION_DONE")
