# -*- coding: utf-8 -*-
"""风格归因：读取一次已完成的主回测，拆解收益来源。

用法::

    python scripts/style_attribution.py                      # 默认 build/runs/main
    python scripts/style_attribution.py --run build/runs/main --out build/attr

输出（out 目录）::

    style_exposures.csv   逐期组合/基准风格暴露
    style_contrib.csv     逐期贡献（MARKET/SIZE/VALUE/MOM/VOL/LIQ/SPECIFIC/UNEXPLAINED）
    summary.csv           各分量年化贡献与 t 值
    STYLE_ATTRIBUTION.md  可读报告
    contrib_cum.png       累积贡献曲线

口径说明见 ``qf/attribution.py`` 模块文档。所有数字由本脚本现场计算，不含硬编码。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import attribution as qfa          # noqa: E402
from qf import config as C                 # noqa: E402
from qf import data as qfdata              # noqa: E402


def _load_run(run: pathlib.Path):
    hold = pd.read_parquet(run / "holdings.parquet")
    nav = pd.read_parquet(run / "nav.parquet")["nav"]
    hold["date"] = pd.to_datetime(hold["date"])
    nav.index = pd.to_datetime(nav.index)
    return hold, nav


def _setup_cn_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        try:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return plt
        except Exception:
            continue
    return plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(C.BUILD_RUNS / "main"))
    ap.add_argument("--out", default=str(C.BUILD / "attr"))
    ap.add_argument("--start", default=None, help="归因起始日（默认回测首日）")
    ap.add_argument("--end", default=None)
    ap.add_argument("--ret-mode", default="close", choices=["close", "open"],
                    help="收益口径：close=t1→t2 收盘；open=t1+1→t2+1 开盘（贴近执行时点）")
    ap.add_argument("--no-compare", action="store_true", help="跳过两种收益口径的对照")
    ap.add_argument("--no-industry", action="store_true",
                    help="不加行业哑变量（用于检验 SPECIFIC 对行业因子的敏感性）")
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载行情 ...")
    md = qfdata.load_market(verbose=False)
    hold, nav = _load_run(run)
    if args.start:
        hold = hold[hold["date"] >= pd.Timestamp(args.start)]
    if args.end:
        hold = hold[hold["date"] <= pd.Timestamp(args.end)]
    nav = nav[(nav.index >= hold["date"].min()) & (nav.index <= nav.index.max())]

    dates = np.array(sorted(hold["date"].unique()))
    rows = np.array([md.date_index(d) for d in dates], dtype=int)
    # 最后一行留作区间终点：若持仓快照最后日期已接近数据末端，用最后交易日补一个终点
    if rows[-1] + 1 < md.T:
        rows = np.append(rows, md.T - 1)
    else:
        rows = np.append(rows, rows[-1])
    print(f"[2/4] 调仓期 {len(rows) - 1} 个：{dates[0].date()} ~ {dates[-1].date()}")

    print("[3/4] 构造风格因子（含 PIT 财务数据）...")
    from qf import fundamentals as qff
    fd = qff.FundamentalData(md)
    value_panels = fd.build_factors()
    styles = qfa.build_styles(md, rows[:-1], value_panels=value_panels)

    print("[4/4] 横截面回归 + 归因 ...")
    from qf import backtest as qfbt
    cash_ret = qfbt.load_cash_asset_returns(md, bond_w=C.CASH_BOND_WEIGHT,
                                           gold_w=C.CASH_GOLD_WEIGHT)
    use_ind = not args.no_industry
    exp, con, ind = qfa.attribute_periods(md, rows, hold, nav, styles,
                                          cash_daily_ret=cash_ret,
                                          ret_mode=args.ret_mode,
                                          use_industry=use_ind)
    if con.empty:
        raise SystemExit("归因结果为空：检查持仓日期与行情日期是否对齐")

    summ = qfa.summarize(con, exp)

    # 收益口径稳健性：另一种口径的归因（只影响 y，风格因子不变）
    alt_mode = "open" if args.ret_mode == "close" else "close"
    alt_summ = None
    if not args.no_compare:
        try:
            e2, c2, _ = qfa.attribute_periods(md, rows, hold, nav, styles,
                                              cash_daily_ret=cash_ret, ret_mode=alt_mode,
                                              use_industry=use_ind)
            alt_summ = qfa.summarize(c2, e2)
            if alt_summ:
                alt_summ["table"].to_csv(out / f"summary_{alt_mode}.csv", encoding="utf-8")
        except Exception as e:
            print(f"[warn] 对照口径失败：{e}")
    exp.to_csv(out / "style_exposures.csv", encoding="utf-8")
    con.to_csv(out / "style_contrib.csv", encoding="utf-8")
    summ["table"].to_csv(out / "summary.csv", encoding="utf-8")
    if not ind.empty:
        ind.to_csv(out / "industry_exposure.csv", encoding="utf-8")

    # ---------------- 报告 ----------------
    t = summ["table"]
    comps = list(t.index)
    lines = ["# 风格归因报告", "",
             f"- 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}",
             f"- 回测来源：`{run.as_posix()}`",
             f"- 样本区间：{con.index[0].date()} ~ {con['end_date'].iloc[-1].date()}",
             f"- 期数：{summ['n_periods']} 个调仓期（约 {summ['years']:.2f} 年）",
             f"- 方法：横截面 WLS（权重 √总市值）；风格 = SIZE/VALUE/MOM/VOL/LIQ；"
             f"行业 = 哑变量（基线取样本内市值最大行业）；收益口径 = {args.ret_mode}", "",
             "## 1. 收益拆解（年化口径）", "",
             "| 分量 | 每期均值 | 累计 | 年化 | 波动 | t 值 | 胜率 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in comps:
        r = t.loc[k]
        lines.append(f"| {k} | {r['mean_per_period']:.2%} | {r['sum']:.2%} | "
                     f"{r['annual']:.2%} | {r['std']:.2%} | {r['t_stat']:.2f} | {r['win_rate']:.0%} |")
    lines += ["", f"- 实际复利累计收益：**{summ['total_actual']:.2%}**",
              f"- 分量复利累加核对：{summ['total_decomp_check']:.2%}",
              "", "## 2. 平均风格暴露（组合 vs 市值加权基准）", "",
              "| 风格 | 组合 | 基准 | 差 |", "| --- | --- | --- | --- |"]
    for k in qfa.STYLES:
        a = summ["avg_exposure"].get(k, np.nan)
        b = summ["avg_bench_exposure"].get(k, np.nan)
        lines.append(f"| {k} | {a:+.3f} | {b:+.3f} | {a - b:+.3f} |")
    lines += ["", "> 暴露为标准化单位（横截面 z-score）。SIZE 正=偏大盘，VALUE 正=偏低估，",
              "> MOM 正=偏强势股，VOL 正=偏高波动，LIQ 正=偏高换手。基准为股票池内总市值加权代理，",
              "> 非真实指数成分暴露。", ""]

    # 结论句（基于数字，不预设）
    def _pct(k):
        return t.loc[k, "annual"] if k in t.index else np.nan
    style_sum = sum(_pct(k) for k in qfa.STYLES)
    spec = _pct("SPECIFIC")
    mkt = _pct("MARKET")
    inds = _pct("INDUSTRY")
    cash = _pct("CASH")
    unexp = _pct("UNEXPLAINED")
    lines += ["## 3. 结论", "",
              f"- 市场（含仓位与基线行业）{mkt:+.2%}；行业配置 {inds:+.2%}；"
              f"风格合计 {style_sum:+.2%}；特异收益 {spec:+.2%}；"
              f"现金替代 {cash:+.2%}；未解释 {unexp:+.2%}。",
              "- " + ("收益中风格暴露占比高于特异收益，需关注风格反转（尤其 SIZE/VOL 同时转向）。"
                      if abs(style_sum) > abs(spec) else
                      "特异收益（选股残差）不高于风格合计贡献，说明超额更多来自暴露与配置而非个股选择。"),
              f"- 单看选股残差：SPECIFIC {spec:+.2%}（t={t.loc['SPECIFIC', 't_stat']:.2f}），"
              f"行业配置 {inds:+.2%}（t={t.loc['INDUSTRY', 't_stat']:.2f}）。"
              "两者符号/量级决定了超额是「选股型」还是「配置型」。",
              "- SPECIFIC 为行业内部残差；UNEXPLAINED 含交易成本、整手取整与 t→t+1 执行时点差。",
              ""]

    if alt_summ:
        t2 = alt_summ["table"]
        lines += [f"## 4. 收益口径对照（{args.ret_mode} vs {alt_mode}）", "",
                  "同一套风格因子，仅换区间收益口径。若「未解释」在两种口径间大幅收敛，",
                  "说明它主要来自执行时点差而非成本。", "",
                  "| 分量 | 年化(" + args.ret_mode + ") | 年化(" + alt_mode + ") | 差 |",
                  "| --- | --- | --- | --- |"]
        for k in list(t.index):
            if k in t2.index:
                a, b = t.loc[k, "annual"], t2.loc[k, "annual"]
                lines.append(f"| {k} | {a:+.2%} | {b:+.2%} | {a - b:+.2%} |")
        lines += [""]

    lines += ["## 5. 已知限制", "",
              "1. 收益区间与「t 收盘决策 / t+1 开盘执行」存在一日错位（第 4 节给出对照）。",
              "2. 基准暴露为股票池内总市值加权代理，非指数真实成分权重。",
              "3. 行业哑变量丢弃基线行业（样本内市值最大者）；MARKET/INDUSTRY 的拆分以",
              "   市值加权行业组合为参照，INDUSTRY 即相对该参照的配置超额。",
              "4. 风格/行业因子收益为样本内横截面估计，含估计误差（t 值见表 1）。",
              "5. 未做风格中性再平衡，不能据此直接构造「风格中性组合」。", ""]
    (out / "STYLE_ATTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------- 图 ----------------
    try:
        plt = _setup_cn_font()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        cum = con[comps].cumsum()
        for k in comps:
            if k == "UNEXPLAINED":
                continue
            axes[0].plot(cum.index, cum[k].to_numpy(), label=k, lw=1.4)
        axes[0].axhline(0, color="#888", lw=0.8, ls="--")
        axes[0].set_title("累积贡献（逐期简单累加）")
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].grid(alpha=0.25)
        axes[0].tick_params(axis="x", rotation=30, labelsize=8)

        xs = np.arange(len(qfa.STYLES))
        p = [summ["avg_exposure"].get(k, np.nan) for k in qfa.STYLES]
        b = [summ["avg_bench_exposure"].get(k, np.nan) for k in qfa.STYLES]
        axes[1].bar(xs - 0.2, p, 0.4, label="组合", color="#c0392b")
        axes[1].bar(xs + 0.2, b, 0.4, label="基准", color="#7f8c8d")
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(qfa.STYLES, fontsize=9)
        axes[1].axhline(0, color="#888", lw=0.8)
        axes[1].set_title("平均风格暴露")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out / "contrib_cum.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[warn] 绘图失败：{e}")

    # ---------------- 控制台摘要 ----------------
    print("\n" + "=" * 78)
    print(f"风格归因  {con.index[0].date()} ~ {con['end_date'].iloc[-1].date()}  "
          f"{summ['n_periods']} 期 / {summ['years']:.2f} 年")
    print("=" * 78)
    print(t[["mean_per_period", "sum", "annual", "t_stat", "win_rate"]]
          .to_string(float_format=lambda x: f"{x:+.4f}"))
    print(f"\n实际复利累计 {summ['total_actual']:.2%}；分量核对 {summ['total_decomp_check']:.2%}")
    print(f"\n平均暴露（组合 / 基准）：")
    for k in qfa.STYLES:
        print(f"  {k:6s} {summ['avg_exposure'].get(k, np.nan):+.3f} / "
              f"{summ['avg_bench_exposure'].get(k, np.nan):+.3f}")
    print(f"\n输出目录：{out}")
    (out / "meta.json").write_text(json.dumps(
        {"run": str(run), "periods": int(summ["n_periods"]), "years": round(summ["years"], 3),
         "total_actual": summ["total_actual"], "generated": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
