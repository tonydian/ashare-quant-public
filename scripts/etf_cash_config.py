# -*- coding: utf-8 -*-
"""测试：去掉黄金，纯国债作现金替代。

对比 4 种现金配置：
  C1 80% 国债 + 20% 黄金（现行）
  C2 100% 国债
  C3 100% 黄金（极端对照）
  C4 货币基金近似（年化 2% 常数）

对每个配置测：
  1) 现金部分本身的年化（空仓期能赚多少）
  2) ETF 右侧策略整体 CAGR / Sharpe / MaxDD
  3) ETF 的纯超额贡献（去掉现金部分后）

注意：现金收益影响巨大 —— 上轮发现 66% 的收益来自现金。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from qf import config as C, metrics as qfm  # noqa: E402
from qf.stats_adjust import circular_block_bootstrap_ci, sharpe_ratio  # noqa: E402
from scripts.etf_trend_strategy import MAX_POS, backtest, load_etf, signals  # noqa: E402

S, E = "2020-01-02", "2026-10-23"
OUT = ROOT / "build" / "eval" / "etf_cash_config.json"

dates, codes, P = load_etf()
idx = pd.DatetimeIndex(dates)

# 载入三种现金资产
bond = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "bond_000012.parquet")
gold = pd.read_parquet(REPO / "data_extract" / "was_stock_data" / "gold_518880.parquet")
bs = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
gs = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
b_ret = bs.reindex(idx).ffill().pct_change().fillna(0)
g_ret = gs.reindex(idx).ffill().pct_change().fillna(0)

print("=" * 96)
print("现金配置对比：去掉黄金的影响")
print("=" * 96)

# ---------- 1. 现金资产本身的收益
print("\n【1】三种现金资产在回测区间的表现")
print("-" * 96)
m = (idx >= S) & (idx <= E)
for name, r in (("国债指数", b_ret), ("黄金ETF", g_ret)):
    sub = r[m]
    tot = (1 + sub).prod() - 1
    yrs = len(sub) / 252
    ann = (1 + tot) ** (1 / yrs) - 1
    vol = sub.std() * np.sqrt(252)
    print(f"  {name:10s}: 累计 {tot:+7.2%} | 年化 {ann:+6.2%} | 波动 {vol:5.2%} | "
          f"最大回撤 {((1+sub).cumprod()/(1+sub).cumprod().cummax()-1).min():7.2%}")

# 组合现金
configs = {
    "C1 80%国债+20%黄金（现行）": 0.8 * b_ret + 0.2 * g_ret,
    "C2 100%国债": b_ret.copy(),
    "C3 100%黄金": g_ret.copy(),
    "C4 货币基金(2%常数)": pd.Series(0.02 / 252, index=idx),
}
print("\n  组合现金的年化：")
for name, r in configs.items():
    sub = r[m]
    tot = (1 + sub).prod() - 1
    yrs = len(sub) / 252
    ann = (1 + tot) ** (1 / yrs) - 1
    print(f"    {name:28s}: 累计 {tot:+7.2%} | 年化 {ann:+6.2%}")

# ---------- 2. 策略整体效果
print("\n【2】ETF 右侧策略（top2）在不同现金配置下的表现")
print("-" * 96)
print(f"{'现金配置':<30}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'空仓年化':>10}")
print("-" * 96)

sig = signals(dates, codes, P, entry_n=55, exit_n=0, max_pos=MAX_POS)
res = {}
navs = {}
for name, r in configs.items():
    nav, pos, tr = backtest(dates, codes, P, sig, r, max_pos=MAX_POS, pos_style="equal")
    sub = nav[m]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    # 空仓期年化
    is_flat = pos[m] == 0
    flat_r = sub.pct_change().fillna(0)[is_flat.values]
    flat_ann = (1 + flat_r.mean()) ** 252 - 1 if len(flat_r) else 0.0
    navs[name] = nav
    res[name] = {"cagr": float(p["cagr"]), "sharpe": float(p["sharpe"]),
                 "max_dd": float(p["max_dd"]), "calmar": float(p["calmar"]),
                 "flat_annual": float(flat_ann)}
    print(f"{name:<30}{p['cagr']:>8.2%}{p['sharpe']:>8.3f}{p['max_dd']:>8.2%}"
          f"{p['calmar']:>8.2f}{flat_ann:>9.2%}")

# ---------- 3. ETF 纯超额贡献
print("\n【3】ETF 操作的纯超额贡献（策略 vs 纯现金）")
print("-" * 96)
print(f"{'现金配置':<30}{'策略累计':>11}{'纯现金累计':>12}{'超额累计':>11}{'超额年化':>10}")
print("-" * 96)
yrs = int(m.sum()) / 252
for name, r in configs.items():
    nav = navs[name]
    sub = nav[m]
    cash_only = (1 + r[m]).prod()
    strat_total = sub.iloc[-1] / sub.iloc[0]
    excess = strat_total / cash_only
    excess_ann = excess ** (1 / yrs) - 1
    res[name]["excess_annual"] = float(excess_ann)
    print(f"{name:<30}{strat_total-1:>10.2%}{cash_only-1:>11.2%}"
          f"{excess-1:>10.2%}{excess_ann:>9.2%}")

# ---------- 4. 关键对比：C1 vs C2
print("\n" + "=" * 96)
print("关键结论：C1（含黄金） vs C2（纯国债）")
print("=" * 96)
c1, c2 = res["C1 80%国债+20%黄金（现行）"], res["C2 100%国债"]
print(f"\n  {'指标':<18}{'C1 含黄金':>14}{'C2 纯国债':>14}{'差':>12}")
print("  " + "-" * 58)
for k, label in (("cagr", "策略 CAGR"), ("sharpe", "Sharpe"),
                 ("max_dd", "MaxDD"), ("calmar", "Calmar"),
                 ("flat_annual", "空仓期年化"), ("excess_annual", "ETF超额年化")):
    v1, v2 = c1[k], c2[k]
    d = v2 - v1
    if k in ("cagr", "max_dd", "flat_annual", "excess_annual"):
        print(f"  {label:<18}{v1:>13.2%}{v2:>13.2%}{d:>11.2%}")
    else:
        print(f"  {label:<18}{v1:>14.3f}{v2:>14.3f}{d:>12.3f}")

# ---------- 5. 结论
print("\n" + "=" * 96)
print("结论")
print("=" * 96)
diff_cagr = (c2["cagr"] - c1["cagr"]) * 100
diff_flat = (c2["flat_annual"] - c1["flat_annual"]) * 100
print(f"""
  去掉黄金（80/20 → 100% 国债）：

  * 策略 CAGR:   {c1['cagr']:.2%} → {c2['cagr']:.2%}  （{diff_cagr:+.2f}pp）
  * 空仓期年化:  {c1['flat_annual']:.2%} → {c2['flat_annual']:.2%}  （{diff_flat:+.2f}pp）
  * Sharpe:      {c1['sharpe']:.3f} → {c2['sharpe']:.3f}
  * MaxDD:       {c1['max_dd']:.2%} → {c2['max_dd']:.2%}
  * ETF 超额:    {c1['excess_annual']:.2%} → {c2['excess_annual']:.2%}
""")

OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"→ {OUT}")
