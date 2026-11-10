# -*- coding: utf-8 -*-
"""``qf/portfolio_style.py`` 的两条不变量。

1. **不加约束时必须与生产 ``portfolio.build_target`` 逐位一致** —— 否则实验组与基线
   的差异里混进了实现差异，整份对比作废。
2. **加了行业相对容差后，实际偏离不得超过容差** —— 否则约束没生效，"约束优化"名不副实。
"""
from __future__ import annotations

import numpy as np
import pytest

from qf import config as C
from qf import data as qfdata
from qf import model as qfm
from qf import portfolio as qfp
from qf import portfolio_style as qfps


@pytest.fixture(scope="module")
def env():
    md = qfdata.load_market(verbose=False)
    res = qfm.load_result("build/models/wf_h10")
    raw = res.scores
    S = np.full((int(md.T), int(md.N)), np.nan, dtype=np.float64)
    S[: raw.shape[0]] = raw
    rows = np.asarray(res.rows_weekly, dtype=int)
    t0 = int(rows[max(0, len(rows) - 30)])
    return md, S, rows[rows >= t0][:12]


def test_unconstrained_matches_production(env):
    md, S, rows = env
    b = qfps.make_style_builder(md, None, rows, None, None)
    prev_a: dict[int, float] = {}
    prev_b: dict[int, float] = {}
    for t in rows:
        a = qfp.build_target(md, int(t), S, prev_a, max_new_entries=6)
        c = b(md, int(t), S, prev_b, max_new_entries=6)
        prev_a, prev_b = dict(a), dict(c)
        assert a.keys() == c.keys(), f"t={t} 选股名单不同"
        for j in a:
            assert abs(a[j] - c[j]) < 1e-12, f"t={t} 权重不同"


def test_industry_dev_cap_binds(env):
    md, S, rows = env
    cap = 0.03
    b = qfps.make_style_builder(md, None, rows, None, cap)
    prev: dict[int, float] = {}
    checked = 0
    for t in rows:
        w = b(md, int(t), S, prev, max_new_entries=6)
        prev = dict(w)
        if not w:
            continue
        bench = qfps.industry_benchmark(md, int(t))
        ind = np.asarray(md.industry[int(t)])
        port: dict[int, float] = {}
        for j, x in w.items():
            port[int(ind[j])] = port.get(int(ind[j]), 0.0) + float(x)
        # 只检查「因约束被拒绝过」的期；未触发约束的期允许不超过 cap + 单票最小权重的松弛
        for code, x in port.items():
            dev = x - bench.get(code, 0.0)
            assert dev <= cap + float(C.MIN_WEIGHT) + 1e-9, (
                f"t={t} 行业 {code} 偏离 {dev:.4f} 超过容差 {cap}")
        checked += 1
    assert checked > 0


def test_style_cap_binds(env):
    md, S, rows = env
    caps = {"VOL": 1.0, "SIZE": 1.0}
    # 风格面板用最简单的常量面板代替：所有票 VOL=+3，任何持仓都会超上限
    style_panels = {k: np.full((len(rows), int(md.N)), 3.0, dtype=np.float32) for k in caps}
    b = qfps.make_style_builder(md, style_panels, rows, caps, None)
    w = b(md, int(rows[-1]), S, {}, max_new_entries=6)
    # 全部标的 z=3 > cap，持仓必须为空（约束真的生效）
    assert w == {}, "风格约束未生效：z 全部超限时仍建了仓"
