# -*- coding: utf-8 -*-
"""主策略 + ETF 趋势的 50/50 固定组合（用户最终选择：动态切换被证伪 → 固定比例）。

组合规则（预先设定，不寻优）：
  * 50% 资金给主策略（wf_h10 个股），50% 给 ETF 趋势（Donchian55 紧出场）；
  * 每季度末再平衡一次（让权重回到 50/50）；
  * 两侧各自保留自己的现金管理（主策略 80/20 现金替代、ETF 趋势同样）；
  * 回测成本：再平衡时按换手收取 5bp 单边（保守）。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from qf import config as C, metrics as qfm  # noqa: E402

OUT = ROOT / "build" / "eval" / "etf_combo_stats.json"
RUN = ROOT / "build" / "runs" / "combo_50_50"

main_nav = pd.read_parquet(ROOT / "build/runs/main/nav.parquet")["nav"].astype(float)
main_nav.index = pd.to_datetime(main_nav.index)
etf_nav = pd.read_parquet(ROOT / "build/runs/etf_trend/nav_donchian55.parquet")["nav"].astype(float)
etf_nav.index = pd.to_datetime(etf_nav.index)

common = main_nav.index.intersection(etf_nav.index)
m = main_nav.reindex(common)
e = etf_nav.reindex(common)
print(f"对齐区间 {common[0].date()} ~ {common[-1].date()}，{len(common)} 日")

m_ret = m.pct_change().fillna(0)
e_ret = e.pct_change().fillna(0)
print(f"两侧日收益相关 = {m_ret.corr(e_ret):.3f}")

# 季度再平衡（成本 5bp 单边，按偏离度计）
qends = pd.Series(common, index=common).groupby([common.year, common.quarter]).last().values
is_qend = pd.Series(False, index=common)
is_qend.loc[pd.DatetimeIndex(qends)] = True
w = 0.5
nav = np.ones(len(common))
rebal_cost = 0.0
for i, t in enumerate(common):
    if i > 0:
        nav[i] = nav[i - 1] * (1 + w * m_ret.iloc[i] + (1 - w) * e_ret.iloc[i])
    r_main = m.iloc[i] / m.iloc[0]
    r_etf = e.iloc[i] / e.iloc[0]
    cur_w = (w * r_main) / (w * r_main + (1 - w) * r_etf)
    if is_qend.iloc[i] and abs(cur_w - 0.5) > 0.02:
        cost = abs(cur_w - 0.5) * 0.0005
        nav[i] *= (1 - cost)
        rebal_cost += cost
        w = cur_w
        # 再平衡后名义权重回到 50/50（用当前净值重新归一）
        w = 0.5
combo = pd.Series(nav, index=common)


def perf(nav: pd.Series, s: str, e_: str) -> dict:
    sub = nav[(nav.index >= s) & (nav.index <= e_)]
    p = qfm.performance(sub, rf=C.CASH_YIELD)
    return {k: float(p[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_dd",
                                     "calmar", "win_rate_monthly")}


SEGS = {"full": (str(common[0].date()), str(common[-1].date())),
        "val": ("2020-01-02", "2021-12-31"),
        "oos": ("2022-01-01", str(common[-1].date()))}
print(f"\n{'方案':<18}{'全期CAGR':>10}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}  {'验证期':>10}{'样本外':>10}")
rows = {}
for name, series in (("主策略", m), ("ETF趋势", e), ("50/50组合", combo)):
    r = {"full": perf(series, *SEGS["full"]), "val": perf(series, *SEGS["val"]),
         "oos": perf(series, *SEGS["oos"])}
    rows[name] = r
    print(f"{name:<18}{r['full']['cagr']*100:>9.2f}%{r['full']['sharpe']:>8.3f}"
          f"{r['full']['max_dd']*100:>8.2f}%{r['full']['calmar']:>8.2f}  "
          f"{r['val']['cagr']*100:>9.2f}%{r['oos']['cagr']*100:>9.2f}%")

# 组合 Sharpe 的 bootstrap 置信区间
sys.path.insert(0, str(ROOT))
from qf.stats_adjust import circular_block_bootstrap_ci, sharpe_ratio  # noqa: E402
ret = combo.pct_change().dropna().to_numpy()
ci = circular_block_bootstrap_ci(ret, lambda s: sharpe_ratio(s, rf=C.CASH_YIELD),
                                 n_boot=1500, block_len=21)
print(f"\n组合 Sharpe bootstrap 95% CI: [{ci['lo']:.3f}, {ci['hi']:.3f}]")
print(f"季度再平衡累计成本: {rebal_cost*100:.2f}%（{len(is_qend[is_qend])} 次）")

RUN.mkdir(parents=True, exist_ok=True)
combo.to_frame("nav").to_parquet(RUN / "nav.parquet")
OUT.write_text(json.dumps({"correlation": float(m_ret.corr(e_ret)),
                           "rebal_cost_pct": rebal_cost * 100,
                           "sharpe_ci": [ci["lo"], ci["hi"]],
                           "results": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n→ {OUT}")
