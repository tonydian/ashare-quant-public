# -*- coding: utf-8 -*-
"""校验基本面因子 PIT 正确性（每股独立可得日 + 手工重算 + 未来数据扰动测试）。"""
import sys, pathlib, copy
import numpy as np, pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))
from qf import config as C
from qf import data as qfdata
from qf import fundamentals as qff

md = qfdata.load_market(verbose=False)
fd = qff.FundamentalData(md)
fac = fd.build_factors()
P, N = len(fd.periods), md.N
print(f"periods {P}  stocks {N}  days {md.T}")
print("avail_mat", fd.avail_mat.shape, fd.avail_mat.dtype)
print("p_idx", fd.p_idx.shape, fd.p_idx.dtype, "range", int(fd.p_idx.min()), int(fd.p_idx.max()))
q = np.array([pd.Timestamp(p).quarter for p in fd.periods])
print("真实公告日覆盖率：" + "  ".join(
    f"Q{qq}={np.nanmean(fd.disc_cov[q == qq]):.3f}" for qq in (1, 2, 3, 4)))

# 原始长表（用于手工重算）
raw = qff._load_raw()
raw_idx = {(r.code, r.period): r for r in raw.itertuples(index=False)}
periods = fd.periods
code_j = {c: j for j, c in enumerate(md.codes)}


def latest_period_for(code, t):
    """独立实现：该股票在 t 日已知的**最大期号**（只依赖 avail_mat，不看数据有无）"""
    j = code_j.get(code)
    if j is None:
        return -1
    best = -1
    dt = np.datetime64(md.dates[t], "ns")
    for p in range(len(periods)):
        if fd.avail_mat[p, j] <= dt:
            best = p
    return best


def expected_value(code, t, col):
    """独立实现：按 p_idx 取该期原始值，套用与 _to_daily 相同的 ffill + stale 语义"""
    j = code_j.get(code)
    p = int(fd.p_idx[t, j])
    if p < 0 or fd.stale_mask[t, j]:
        return np.nan
    for k in range(p, -1, -1):
        rec = raw_idx.get((code, periods[k]))
        if rec is None:
            continue
        v = getattr(rec, col, np.nan)
        if np.isfinite(v):
            return float(v)
    return np.nan


rng = np.random.default_rng(7)
codes = md.codes
sample_t = rng.choice(np.arange(500, md.T - 1), 30, replace=False)

# 1) p_idx 独立重算
bad_idx = 0
for t in sample_t:
    for _ in range(6):
        j = int(rng.integers(0, md.N))
        if latest_period_for(codes[j], t) != int(fd.p_idx[t, j]):
            bad_idx += 1
print(f"p_idx 独立重算：{len(sample_t)*6} 个抽样格子，不一致 {bad_idx} 个")

# 2) 因子值核对（bp / rev_growth）
errs = []
for t in sample_t:
    for _ in range(6):
        j = int(rng.integers(0, md.N))
        code = codes[j]
        want_bps = expected_value(code, t, "bps")
        got_bp = fac["bp"][t, j]
        if np.isfinite(want_bps) and md.price[t, j] > 0 and np.isfinite(got_bp):
            errs.append(abs(got_bp / (want_bps / md.price[t, j]) - 1))
        want_g = expected_value(code, t, "rev_yoy")
        got_g = fac["rev_growth"][t, j]
        if np.isfinite(want_g) and np.isfinite(got_g):
            errs.append(abs(got_g - want_g))
print(f"PIT 手工重算：n={len(errs)}，最大误差 {max(errs):.3e}")
assert bad_idx == 0 and max(errs) < 1e-5, "PIT 对齐与独立重算不一致"

# 未来数据扰动：按 (股票, 报告期) 格子置空 avail > t0 的部分，t<=t0 的因子应逐位不变
t_test = md.date_index("2022-06-15")
fd2 = copy.deepcopy(fd)
future = fd2.avail_mat > np.datetime64(md.dates[t_test])
print(f"扰动：置空 {int(future.sum())} 个 (股票,报告期) 格子")
for col in fd2.M:
    fd2.M[col][future] = np.nan
for arr in ("ttm_net", "ttm_rev", "sq_net", "sq_rev"):
    getattr(fd2, arr)[future] = np.nan
fac2 = fd2.build_factors()
max_diff = {}
for k in fac:
    a = fac[k][: t_test + 1]
    b = fac2[k][: t_test + 1]
    m = np.isfinite(a) & np.isfinite(b)
    d = np.max(np.abs(a[m] - b[m])) if m.any() else 0.0
    max_diff[k] = float(d)
bad = {k: v for k, v in max_diff.items() if v > 1e-6}
print("未来数据扰动测试（2022-06-15 之前逐格子）：", "PASS" if not bad else f"FAIL {bad}")
print(f"  检查因子数 {len(fac)}，最大差异 {max(max_diff.values()):.2e}")

# 新旧口径对比（固定滞后 vs 真实公告日）
fd0 = qff.FundamentalData(md, use_disclosure=False)
adv = (fd0.avail_mat - fd.avail_mat) / np.timedelta64(1, "D")
print("\n新口径相对固定滞后口径的提前量（天）：")
for qq in (1, 2, 3, 4):
    a = adv[q == qq].ravel()
    a = a[np.isfinite(a)]
    print(f"  Q{qq}  中位 {np.median(a):5.1f}  均值 {a.mean():5.1f}  提前>0 占比 {(a > 0).mean():.3f}")
