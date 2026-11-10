# -*- coding: utf-8 -*-
"""统计主回测中因涨停/跌停/停牌被拒绝的买卖尝试次数（验证执行约束确实生效）。"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
sys.path.insert(0, str(ROOT))

from qf import config as C, data as qfdata, model as qfm, backtest as qfbt
import qf.execution as qfe

cnt = {"buy_limit_up": 0, "buy_suspended": 0, "buy_other": 0,
       "sell_limit_dn": 0, "sell_suspended": 0, "sell_other": 0}
events = {"buy_limit_up": [], "sell_limit_dn": []}

_ob = qfe.ExecutionEngine._can_buy
_os = qfe.ExecutionEngine._can_sell


def can_buy(self, t, j):
    ok = _ob(self, t, j)
    if not ok:
        md = self.md
        if bool(md.limit_up_at_open[t, j]):
            cnt["buy_limit_up"] += 1
            events["buy_limit_up"].append((str(md.dates[t].date()), str(md.codes[j])))
        elif bool(md.is_suspended[t, j]):
            cnt["buy_suspended"] += 1
        else:
            cnt["buy_other"] += 1
    return ok


def can_sell(self, t, j):
    ok = _os(self, t, j)
    if not ok:
        md = self.md
        if bool(md.limit_dn_at_open[t, j]):
            cnt["sell_limit_dn"] += 1
            events["sell_limit_dn"].append((str(md.dates[t].date()), str(md.codes[j])))
        elif bool(md.is_suspended[t, j]):
            cnt["sell_suspended"] += 1
        else:
            cnt["sell_other"] += 1
    return ok


qfe.ExecutionEngine._can_buy = can_buy
qfe.ExecutionEngine._can_sell = can_sell

md = qfdata.load_market(verbose=False)
res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
scores = np.asarray(res.scores)
if scores.shape[0] < md.T:
    pad = np.full((md.T - scores.shape[0], scores.shape[1]), np.nan, dtype=np.float32)
    scores = np.vstack([scores, pad])
elif scores.shape[0] > md.T:
    scores = scores[:md.T]
start = max(int(np.flatnonzero(np.isfinite(scores).any(axis=1))[0]), md.date_index(C.VALID_START))
phase = int(res.rows_weekly[0]) % 5
while start % 5 != phase:
    start += 1
end = md.T - 1
cash_ret = qfbt.load_cash_asset_returns(md, 0.8, 0.2)
old = C.N_HOLDINGS
C.N_HOLDINGS = 40
try:
    r = qfbt.run_backtest(md, scores, start, end, risk_overlay=True, stop_loss=True,
                          rebalance_days=10, rebalance_offset=(phase - start) % 10,
                          cash_asset_returns=cash_ret, verbose=False)
finally:
    C.N_HOLDINGS = old
print("区间", md.dates[start].date(), "~", md.dates[end].date())
print("被拒尝试次数:", cnt)
print("涨停买不进（尝试）:", len(events["buy_limit_up"]), "次；涉及股票/日期示例:",
      events["buy_limit_up"][:8])
print("跌停卖不出（尝试）:", len(events["sell_limit_dn"]), "次；涉及股票/日期示例:",
      events["sell_limit_dn"][:8])
tr = r.trades
print("成交笔数:", len(tr), "| 卖出中 reason 分布:", tr[tr.side == "sell"]["reason"].value_counts().to_dict())
print("最终 CAGR/Sharpe/MaxDD:", {k: round(r.metrics[k], 4) for k in ("cagr", "sharpe", "max_dd")})
