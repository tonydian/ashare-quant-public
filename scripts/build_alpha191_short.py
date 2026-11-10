# -*- coding: utf-8 -*-
"""Alpha191 负面清单（避雷）合成与剔除效果验证 —— Step 1 因子层核心检验。

背景
----
`scripts/diag_alpha191_short.py` 已给出决定性证据：Alpha191 全库 **多头 alpha
中位数为负（-0.00034/5日）、空头 alpha 中位数为正（+0.00215/5日）**，
145/163 个因子「空头端强于多头端」。即这个库天生是避雷库。

本脚本验证由此推出的策略形态是否成立：

    **不预测谁涨，只剔除谁要跌** —— 在全市场里按「垃圾分」剔除最差 q%，
    剩余等权持有；alpha = 剩余组合收益 - 全市场等权收益。

检验项
------
1. 垃圾分合成的 5 分组单调性；
2. 剔除 10% / 20% / 30% 后的排雷 alpha 与其 t 值；
3. 分年稳定性（每年独立看符号）；
4. 被剔除名单的换手（成本可行性）；
5. 垃圾分与「规模/股价/成交额」风格的相关性（避免变相做空小市值）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qf import alpha191 as a191          # noqa: E402
from qf import config as C              # noqa: E402
from qf import evaluate as qfe          # noqa: E402
from qf.data import MarketData          # noqa: E402

OUT = ROOT / "build" / "alpha191"
MAX_FACTORS = 12
MAX_CORR = 0.60          # 贪心去相关的阈值
TURNOVER_CAP = 0.40      # 负面清单的换手上限


def pick_factors(verbose: bool = True) -> list[str]:
    """从空头端强度 + 稳定性 + 低换手 + 低共线筛出合成因子。"""
    df = pd.read_csv(OUT / "metrics.csv")
    corr = pd.read_csv(OUT / "corr.csv", index_col=0)

    q = df[["q1_5d", "q2_5d", "q3_5d", "q4_5d", "q5_5d"]].to_numpy(dtype=float)
    neg = df["ic_5d"].to_numpy(dtype=float) < 0
    good = np.where(neg, q[:, 0], q[:, 4])
    bad = np.where(neg, q[:, 4], q[:, 0])
    df["short_alpha"] = q[:, 2] - bad
    df["long_alpha"] = good - q[:, 2]

    def same_sign_ratio(s) -> float:
        if not isinstance(s, str):
            return np.nan
        by = eval(s)
        sign = np.sign(df.loc[by_idx, "ic_5d"])
        return sum(1 for v in by.values() if np.sign(v) == sign) / len(by)

    keep = []
    rows = []
    for by_idx, r in df.iterrows():
        if not np.isfinite(r["coverage"]) or r["coverage"] < 0.999:
            continue
        if not np.isfinite(r["turnover_top20"]) or r["turnover_top20"] > TURNOVER_CAP:
            continue
        if not (r["short_alpha"] > 0):
            continue
        if not np.isfinite(r["tstat"]) or abs(r["tstat"]) < 4.0:
            continue
        by = eval(r["ic_by_year"]) if isinstance(r["ic_by_year"], str) else {}
        if by and sum(1 for v in by.values() if np.sign(v) == np.sign(r["ic_5d"])) < len(by):
            continue          # 要求逐年同向，一次都不许破
        rows.append((r["name"], r["short_alpha"], r["long_alpha"],
                     r["turnover_top20"], r["ic_5d"]))
    cand = pd.DataFrame(rows, columns=["name", "short_alpha", "long_alpha", "turnover", "ic"])
    cand = cand.sort_values("short_alpha", ascending=False).reset_index(drop=True)

    for _, r in cand.iterrows():
        n = r["name"]
        if all(abs(corr.loc[n, k]) < MAX_CORR for k in keep):
            keep.append(n)
        if len(keep) >= MAX_FACTORS:
            break

    if verbose:
        print(f"候选（逐年同向 + 换手<={TURNOVER_CAP} + |t|>=4）：{len(cand)} 个")
        print(f"贪心去相关（|rho|<{MAX_CORR}）后取 {len(keep)} 个：")
        sub = cand[cand["name"].isin(keep)].set_index("name").loc[keep]
        with pd.option_context("display.width", 200):
            print(sub.round(5).to_string())
    return keep


def main() -> None:
    md = MarketData(verbose=True)
    rows = qfe.weekly_rows(md)                       # 477 期
    dates = qfe.weekly_dates(md, rows)
    ks = pick_factors()
    knum = [int(k.split("_")[1]) for k in ks]

    print(f"\n计算 {len(knum)} 个因子 @ {len(rows)} 期 ...")
    panels = a191.compute_panel(md, knum, rows=rows, verbose=True)

    md_df = pd.read_csv(OUT / "metrics.csv").set_index("name")
    sign = np.array([np.sign(md_df.loc[k, "ic_5d"]) for k in ks])
    # 负 IC → 因子值大 = 垃圾 → rank 越大越垃圾（方向 +1）
    # 正 IC → 因子值小 = 垃圾 → 方向 -1
    direction = np.where(sign < 0, 1.0, -1.0)

    elig = md.eligible[rows]
    y = md.fwd_open_ret(5)[rows]
    valid = elig & np.isfinite(y)

    # ---- 合成垃圾分：各因子横截面秩（居中到 [-0.5,0.5]）按方向加权平均
    score = np.zeros(y.shape, dtype=np.float32)
    wsum = np.zeros(y.shape, dtype=np.float32)
    for (k, x), d in zip(panels.items(), direction):
        ok = valid & np.isfinite(x)
        r = qfe.row_rank(x.astype(np.float64), ok)      # 平均秩
        # 归一到 (0,1)
        n = ok.sum(axis=1, keepdims=True).astype(np.float32)
        rn = np.where(ok, (r - 1.0) / np.maximum(n - 1.0, 1.0), np.nan)
        score += np.where(ok, (d * rn).astype(np.float32), 0.0)
        wsum += ok.astype(np.float32)
    garbage = np.where(valid & (wsum > 0), score / np.maximum(wsum, 1e-6), np.nan)
    # garbage 越大 = 越垃圾

    # ---- 1. 五分组（按垃圾分，q0=最干净 … q4=最垃圾）
    print("\n===== 1. 垃圾分五分组（5 日收益，%）=====")
    qs = np.full((len(rows), 5), np.nan)
    for i in range(len(rows)):
        m = valid[i] & np.isfinite(garbage[i])
        if m.sum() < qfe.MIN_IC_STOCKS:
            continue
        g = garbage[i][m]
        order = np.argsort(g)
        yy = y[i][m]
        for j in range(5):
            lo = int(len(order) * j / 5)
            hi = int(len(order) * (j + 1) / 5)
            if hi > lo:
                qs[i, j] = yy[order[lo:hi]].mean()
    names = ["Q0最干净", "Q1", "Q2", "Q3", "Q4最垃圾"]
    for j in range(5):
        v = qs[:, j]
        v = v[np.isfinite(v)]
        print(f"  {names[j]:>8}: 均值 {v.mean()*100:+.3f}%  年化 {v.mean()*244/5*100:+6.2f}%  "
              f"t={v.mean()/v.std(ddof=1)*np.sqrt(len(v)):+6.2f}  n={len(v)}")
    spread = qs[:, 4] - qs[:, 0]
    sv = spread[np.isfinite(spread)]
    print(f"  最垃圾-最干净: 均值 {-sv.mean()*100:+.3f}%(反向)  年化 {-sv.mean()*244/5*100:+.2f}%  "
          f"t={-sv.mean()/sv.std(ddof=1)*np.sqrt(len(sv)):+.2f}")

    # ---- 2. 剔除效果（核心）
    print("\n===== 2. 剔除最差 q% 后的等权组合 vs 全市场等权（5 日）=====")
    mkt = np.array([y[i][valid[i]].mean() if valid[i].sum() else np.nan
                    for i in range(len(rows))])
    print(f"  全市场等权基准: 年化 {np.nanmean(mkt)*244/5*100:+.2f}%")
    res = []
    for drop in (0.10, 0.20, 0.30):
        keep_ret = np.full(len(rows), np.nan)
        kept_sets = []
        for i in range(len(rows)):
            m = valid[i] & np.isfinite(garbage[i])
            if m.sum() < qfe.MIN_IC_STOCKS:
                continue
            idx = np.where(m)[0]
            g = garbage[i][idx]
            cut = np.quantile(g, 1.0 - drop)
            sel = idx[g < cut]
            if len(sel) == 0:
                continue
            keep_ret[i] = y[i][sel].mean()
            kept_sets.append(set(sel.tolist()))
        diff = keep_ret - mkt
        dv = diff[np.isfinite(diff)]
        t = dv.mean() / dv.std(ddof=1) * np.sqrt(len(dv))
        # 名单换手（相邻两期）
        tno = np.nan
        if kept_sets:
            tno = np.mean([1.0 - len(a & b) / max(len(a), 1)
                           for a, b in zip(kept_sets[:-1], kept_sets[1:])])
        res.append({"drop": f"{drop:.0%}", "ann_excess": dv.mean() * 244 / 5,
                    "t": t, "turnover": tno, "n": len(dv)})
        print(f"  剔除最差 {drop:.0%}: 年化超额 {dv.mean()*244/5*100:+.2f}%  "
              f"t={t:+.2f}  名单换手={tno:.2f}  n={len(dv)}")

    # ---- 3. 分年
    print("\n===== 3. 剔除 20% 的年化超额（分年）=====")
    keep_ret = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        m = valid[i] & np.isfinite(garbage[i])
        if m.sum() < qfe.MIN_IC_STOCKS:
            continue
        idx = np.where(m)[0]
        g = garbage[i][idx]
        sel = idx[g < np.quantile(g, 0.80)]
        keep_ret[i] = y[i][sel].mean()
    diff = keep_ret - mkt
    ser = pd.Series(diff, index=dates)
    for yr, g in ser.groupby(ser.index.year):
        gv = g.dropna()
        if len(gv) < 4:
            continue
        print(f"  {yr}: 年化超额 {gv.mean()*244/5*100:+6.2f}%  t={gv.mean()/gv.std(ddof=1)*np.sqrt(len(gv)):+5.2f}  n={len(gv)}")

    # ---- 5. 风格相关性
    print("\n===== 5. 垃圾分与风格的横截面相关性 =====")
    for nm, panel in (("log市值", np.log(md.mcap)),
                      ("log股价", np.log(np.where(md.raw_close > 0, md.raw_close, np.nan))),
                      ("log20日均额", np.log(md.amount_20d))):
        p = np.asarray(panel, dtype=np.float64)[rows]
        cs = []
        for i in range(len(rows)):
            m = valid[i] & np.isfinite(garbage[i]) & np.isfinite(p[i])
            if m.sum() < qfe.MIN_IC_STOCKS:
                continue
            a = pd.Series(garbage[i][m]).rank()
            b = pd.Series(p[i][m]).rank()
            cs.append(a.corr(b))
        print(f"  {nm}: 平均截面 rho = {np.nanmean(cs):+.3f}  (|rho| 中位 {np.nanmedian(np.abs(cs)):.3f})")

    np.save(OUT / "garbage_score.npy", garbage.astype(np.float32))
    (OUT / "garbage_factors.txt").write_text("\n".join(ks), encoding="utf-8")
    print(f"\n已保存 garbage_score.npy ({garbage.shape}) 与 garbage_factors.txt")


if __name__ == "__main__":
    main()
