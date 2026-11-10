# -*- coding: utf-8 -*-
"""ERC 风险平价深挖 + 组合层面全面优化。

现有实现的问题：只是"波动率倒数 × 固定权重"（不迭代），且用全样本波动率（前视）。

本脚本改进：
  1) **真 ERC**：迭代求解使各资产风险贡献相等（含协方差，不只是波动）
  2) **无前视**：协方差只用 t 之前 252 日（月度滚动更新）
  3) 对比 6 种配置方法：
       EQ     等权（1/3 各）
       5050   主策略+宽基 各半
       50/30/20 现行
       IVP    逆波动率（现有"ERC"）
       ERC    真等风险贡献
       MDP    最大分散化（最小相关）
  4) 参数敏感性：协方差窗口 126/252/504、再平衡频率 20/60 日
  5) 分年度 + bootstrap + 换手成本

三个资产：主策略(个股) / 宽基ETF / 板块轮动
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

from qf import config as C, metrics as qfm  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci, sharpe_ratio  # noqa: E402

OUT = ROOT / "build" / "eval" / "erc_deep.json"
S, E = "2020-01-08", "2026-10-14"

# ============================================================
# 1) 载入三条策略净值
print("[1/5] 载入三条策略净值…")
navs = {}
for tag, rel in (("M", "runs/main/nav.parquet"),
                 ("W", "runs/etf_trend/nav_donchian55.parquet"),
                 ("S", "runs/sector_rotation/nav.parquet")):
    p = C.BUILD / rel
    if not p.exists():
        print(f"  {tag}: 缺 {p}")
        continue
    s = pd.read_parquet(p)["nav"].astype(float)
    s.index = pd.to_datetime(s.index)
    navs[tag] = s
    print(f"  {tag}: {len(s)} 天，{s.index[0].date()} ~ {s.index[-1].date()}")

if len(navs) < 3:
    raise SystemExit("需要三条净值都存在")

# 对齐
idx = None
for v in navs.values():
    idx = v.index if idx is None else idx.intersection(v.index)
idx = idx[(idx >= S) & (idx <= E)]
rets = pd.DataFrame({k: v.reindex(idx).pct_change() for k, v in navs.items()}).fillna(0)
print(f"  对齐后 {len(rets)} 日，{rets.index[0].date()} ~ {rets.index[-1].date()}")
print(f"\n  相关性矩阵:")
print(rets.corr().round(3).to_string())

print(f"\n  各自表现:")
print(f"  {'':<10}{'CAGR':>9}{'Sharpe':>8}{'波动':>9}{'MaxDD':>9}")
for k in rets.columns:
    nav = (1 + rets[k]).cumprod()
    p = qfm.performance(nav, rf=C.CASH_YIELD)
    print(f"  {k:<10}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}{p['ann_vol']:>8.2%}"
          f"{p['max_dd']:>9.2%}")

# ============================================================
# 2) 各种权重计算方法
print("\n[2/5] 计算权重（滚动，无前视）…")


def w_equal(cov, ret_win):
    return np.ones(3) / 3


def w_fixed_5050(cov, ret_win):
    return np.array([0.5, 0.5, 0.0])


def w_fixed_50_30_20(cov, ret_win):
    return np.array([0.5, 0.3, 0.2])


def w_ivp(cov, ret_win):
    """逆波动率（现有所谓 ERC 的实现）。"""
    v = np.sqrt(np.diag(cov))
    v = np.where(v > 0, v, np.nan)
    inv = 1.0 / v
    inv = np.where(np.isfinite(inv), inv, 0.0)
    return inv / max(inv.sum(), 1e-12)


def w_erc(cov, ret_win, iters=300, tol=1e-8):
    """真 ERC：迭代求解使各资产风险贡献相等（长仓、非负）。"""
    n = len(cov)
    # 初值用逆波动
    v = np.sqrt(np.diag(cov))
    v = np.where(v > 0, v, np.nan)
    inv = 1.0 / v
    inv = np.where(np.isfinite(inv), inv, 0.0)
    w = inv / max(inv.sum(), 1e-12)
    if w.sum() <= 0:
        return np.ones(n) / n
    for _ in range(iters):
        mrc = cov @ w                    # 边际风险贡献
        rc = w * mrc                     # 风险贡献
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return np.ones(n) / n
        target = port_var / n
        # 乘性更新（保证非负）
        adj = np.where(rc > 1e-14, target / rc, 1.0)
        w = w * np.power(np.clip(adj, 0.5, 2.0), 0.5)
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0:
            return np.ones(n) / n
        w = w / s
        if np.max(np.abs(rc - target)) / max(port_var / n, 1e-14) < tol:
            break
    return w


def w_mdp(cov, ret_win):
    """最大分散化：w ∝ 1/(σ · Σw) 迭代近似 → 用 inv(σ)·inv(corr) 近似。"""
    try:
        vol = np.sqrt(np.diag(cov))
        corr = cov / np.outer(np.maximum(vol, 1e-12), np.maximum(vol, 1e-12))
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        inv_corr = np.linalg.pinv(corr)
        ones = np.ones(len(cov))
        w = inv_corr @ ones
        w = w / np.maximum(vol, 1e-12)
        w = np.maximum(w, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.ones(len(cov)) / len(cov)
    except Exception:
        return np.ones(len(cov)) / len(cov)


METHODS = {
    "EQ 等权": w_equal,
    "5050 主+宽基": w_fixed_5050,
    "50/30/20": w_fixed_50_30_20,
    "IVP 逆波动": w_ivp,
    "ERC 真等风险": w_erc,
    "MDP 最大分散": w_mdp,
}

# ============================================================
# 3) 回测组合（滚动权重）


def backtest_combo(wfunc, cov_win=252, rebal=60, cost_bp=2.0):
    """滚动权重组合。每 rebal 日用过去 cov_win 日数据重算权重。"""
    n_days = len(rets)
    w = np.ones(3) / 3
    ws = [w.copy()]
    turnover = 0.0
    port = np.zeros(n_days)
    col = list(rets.columns)
    R = rets.to_numpy()

    for i in range(n_days):
        if i > 0 and i % rebal == 0 and i >= cov_win:
            win = R[i - cov_win:i]
            cov = np.cov(win.T) if win.shape[0] > 2 else np.eye(3)
            cov = np.nan_to_num(cov, nan=0.0)
            if np.trace(cov) <= 0:
                cov = np.eye(3) * 1e-6
            try:
                new_w = wfunc(cov, win)
                new_w = np.asarray(new_w, float)
                new_w = np.maximum(new_w, 0.0)
                s = new_w.sum()
                new_w = new_w / s if s > 0 else np.ones(3) / 3
                turnover += float(np.abs(new_w - w).sum())
                w = new_w
            except Exception:
                pass
        ws.append(w.copy())
        # 当日收益（含成本：只在换手日扣）
        r = float(w @ R[i])
        if i > 0 and i % rebal == 0 and i >= cov_win:
            r -= cost_bp / 1e4 * 1.0     # 粗略：每次再平衡 1 个名义单位换手
        port[i] = r

    nav = np.cumprod(1 + port)
    nav_s = pd.Series(nav, index=rets.index)
    return nav_s, np.array(ws), turnover


def perf(nav, s=S, e=E):
    sub = nav[(nav.index >= s) & (nav.index <= e)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino",
                                     "max_dd", "calmar")}


# ============================================================
# 4) 主对比
print("\n[3/5] 六种配置方法对比（协方差 252 日，季度再平衡）…")
print(f"\n  {'方法':<16}{'CAGR':>9}{'Sharpe':>8}{'波动':>9}{'MaxDD':>9}{'Calmar':>8}"
      f"{'验证期':>9}{'样本外':>9}")
print("  " + "-" * 78)
res = {}
navs_c = {}
for name, fn in METHODS.items():
    nav, ws, to = backtest_combo(fn, cov_win=252, rebal=60)
    navs_c[name] = nav
    d = perf(nav)
    dv = perf(nav, s="2020-01-08", e="2021-12-31")
    do = perf(nav, s="2022-01-01", e=E)
    res[name] = {"full": d, "val": dv, "oos": do,
                 "avg_weights": ws.mean(axis=0).tolist(),
                 "turnover": float(to)}
    print(f"  {name:<16}{d['cagr']:>8.2%}{d['sharpe']:>8.3f}{d['ann_vol']:>8.2%}"
          f"{d['max_dd']:>9.2%}{d['calmar']:>8.2f}"
          f"{dv['cagr']:>8.2%}{do['cagr']:>9.2%}")

print(f"\n  平均权重（M/W/S）:")
for name in METHODS:
    if name in res:
        w = res[name]["avg_weights"]
        print(f"    {name:<16} {w[0]:>5.1%} / {w[1]:>5.1%} / {w[2]:>5.1%}"
              f"   （累计换手 {res[name]['turnover']:.1f}）")

# ============================================================
# 5) ERC 参数敏感性
print("\n[4/5] ERC 参数敏感性（协方差窗口 × 再平衡频率）")
print(f"\n  {'窗口':>6}{'再平衡':>7}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}")
print("  " + "-" * 48)
grid = {}
for cov_win in (126, 252, 504):
    for rebal in (20, 60):
        nav, _, _ = backtest_combo(w_erc, cov_win=cov_win, rebal=rebal)
        d = perf(nav)
        grid[f"cov{cov_win}_rb{rebal}"] = d
        print(f"  {cov_win:>6}{rebal:>7}{d['cagr']:>8.2%}{d['sharpe']:>8.3f}"
              f"{d['max_dd']:>9.2%}{d['calmar']:>8.2f}")
sh = [v["sharpe"] for v in grid.values()]
print(f"\n  6 组 Sharpe：中位 {np.median(sh):.3f} | min {min(sh):.3f} | max {max(sh):.3f}")

# ============================================================
# 6) 逐年 + 显著性
print("\n[5/5] 逐年对比 + 显著性")
print(f"\n  {'年':>6}" + "".join(f"{n.split()[0]:>10}" for n in METHODS))
print("  " + "-" * (6 + 10 * len(METHODS)))
ys = {}
for name, nav in navs_c.items():
    ys[name] = qfm.yearly_returns(nav[nav.index >= S])
for y in sorted(ys["EQ 等权"].index):
    line = f"  {int(y):>6}"
    for name in METHODS:
        v = float(ys[name].loc[y]) if y in ys[name].index else 0.0
        line += f"{v:>9.2%} "
    print(line)

print("\n  显著性：ERC vs 50/30/20")
for a, b in (("ERC 真等风险", "50/30/20"), ("ERC 真等风险", "5050 主+宽基")):
    common = navs_c[a].index.intersection(navs_c[b].index)
    d = (navs_c[a][common].pct_change() - navs_c[b][common].pct_change()).dropna().to_numpy()
    ci = circular_block_bootstrap_ci(d, lambda x: float(np.mean(x) * 252),
                                     n_boot=2000, block_len=21)
    t = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(len(d)))
    print(f"    ERC - {b:<14} 年化差 {np.mean(d)*252*100:+6.2f}% | "
          f"CI [{ci['lo']*100:+6.2f}%, {ci['hi']*100:+6.2f}%] | t={t:+5.2f} "
          f"→ {'✅显著' if abs(t) > 2 else '不显著'}")

# bootstrap Sharpe CI
print("\n  Sharpe bootstrap 95% CI:")
for name in ("50/30/20", "ERC 真等风险", "MDP 最大分散"):
    r = navs_c[name].pct_change().dropna().to_numpy()
    ci = circular_block_bootstrap_ci(r, lambda x: sharpe_ratio(x, rf=C.CASH_YIELD),
                                     n_boot=1500, block_len=21)
    print(f"    {name:<16} Sharpe {sharpe_ratio(r, rf=C.CASH_YIELD):.3f} | "
          f"[{ci['lo']:.3f}, {ci['hi']:.3f}]")

json.dump({"methods": res, "grid": grid,
           "corr": rets.corr().round(4).to_dict(),
           "weights": {k: v["avg_weights"] for k, v in res.items()}},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
