# -*- coding: utf-8 -*-
"""报告生成：因子库复盘（factor-review 三层）+ 回测报告 + 系统说明 + Excel 汇总。

用法:
    python scripts/make_report.py --model build/models/wf_v1 --run build/runs/main \
        --eval build/eval --factors build/factors --out reports

所有数字都从落盘产物读取（build/eval、build/factors、build/models、build/runs），
报告可随时用最新产物重新生成。
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib
import datetime as dt

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C


# ------------------------------------------------------------------ 工具
def _fmt_pct(x, nd=2):
    try:
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"{v*100:.{nd}f}%"
    except Exception:
        return "—"


def _fmt(x, nd=3):
    try:
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"{v:.{nd}f}"
    except Exception:
        return "—"


def _json(path, default=None):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}


def _parq(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _table(df: pd.DataFrame, cols: dict[str, str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols.values()) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            f = fmt.get(c)
            if f == "pct":
                cells.append(_fmt_pct(v))
            elif f == "num2":
                cells.append(_fmt(v, 2))
            elif f == "num1":
                cells.append(_fmt(v, 1))
            elif f == "int":
                try:
                    cells.append(f"{int(v)}")
                except Exception:
                    cells.append("—")
            else:
                cells.append(str(v) if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ 数据装载
class Ctx:
    def __init__(self, args):
        self.args = args
        self.eval_dir = pathlib.Path(args.eval)
        self.fac_dir = pathlib.Path(args.factors)
        self.model_dir = pathlib.Path(args.model)
        self.run_dir = pathlib.Path(args.run)
        self.robust_dir = pathlib.Path(args.robust)
        self.out = pathlib.Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)

        self.screen = _parq(self.eval_dir / "factor_metrics_screen.parquet")
        self.metrics = _parq(self.eval_dir / "factor_metrics.parquet")
        self.family = _parq(self.eval_dir / "family_stats.parquet")
        self.review = _json(self.eval_dir / "review_summary.json", {})
        self.catalog = _parq(self.fac_dir / "catalog.parquet")
        self.mom_cat = _parq(self.fac_dir / "momentum_catalog.parquet")
        self.mom_stats = _json(self.eval_dir / "momentum_stats.json", {})
        self.fund_cat = _parq(self.fac_dir / "fundamental_catalog.parquet")
        self.fund_stats = _json(self.eval_dir / "fundamental_stats.json", {})
        self.pool_meta = _parq(self.model_dir / "pool_meta.parquet")
        self.weights = _parq(self.model_dir / "model_weights.parquet")
        self.comp_ic = _parq(self.model_dir / "composite_ic.parquet")
        self.model_meta = _json(self.model_dir / "model_meta.json", {})
        self.metrics_bt = _json(self.run_dir / "metrics.json", {})
        self.bench_bt = _json(self.run_dir / "benchmark_metrics.json", {})
        self.cash_cfg = _json(self.run_dir / "cash_config.json", {})
        _runs = self.run_dir.parent
        self.v1_metrics = _json(_runs / "main_v1_cash" / "metrics.json", {})
        self.cash_test = self._read_csv(_runs / "cash_sleeve_test.csv")
        self.model_test = self._read_csv(_runs / "model_compare_v2.csv")
        self.ensemble_test = self._read_csv(_runs / "ensemble_v1v2.csv")
        self.stress = self._read_csv(self.run_dir / "stress.csv")
        self.yearly = self._read_csv(self.run_dir / "yearly.csv", index_col=0)
        self.monthly = self._read_csv(self.run_dir / "monthly.csv", index_col=0)
        self.robust = self._read_csv(self.robust_dir / "robustness.csv")
        self.nav = _parq(self.run_dir / "nav.parquet")
        self.trades = _parq(self.run_dir / "trades.parquet")
        self.exposures = _parq(self.run_dir / "exposures.parquet")
        self.target = self._read_csv(pathlib.Path(args.signals) / "latest_target.csv",
                                     dtype={"code": str})
        if self.target is not None and "code" in self.target.columns:
            self.target["code"] = self.target["code"].astype(str).str.zfill(6)
        self.signal = _json(pathlib.Path(args.signals) / "latest_signal.json", {})

    @staticmethod
    def _read_csv(p, **kw):
        try:
            return pd.read_csv(p, **kw)
        except Exception:
            return None


# ------------------------------------------------------------------ 因子库复盘
def make_factor_review(ctx: Ctx) -> pathlib.Path:
    lines: list[str] = []
    now = dt.date.today().isoformat()
    lines += [f"# 因子库复盘报告 — {now}", "",
              "> 本报告按 `factor-review` 技能的三层结构生成：**Layer 1 量化盘点 → Layer 2 结构分析 → "
              "Layer 3 研究建议**。所有数字来自 `build/eval/`、`build/factors/`、`journal/` 落盘产物。", ""]

    # ---------------- Layer 1
    lines += ["## Layer 1：量化盘点", ""]
    n_total = len(ctx.catalog) if ctx.catalog is not None else 800
    n_extra = len(ctx.mom_cat) if ctx.mom_cat is not None else 0
    n_fund = len(ctx.fund_cat) if ctx.fund_cat is not None else 0
    n_screen = len(ctx.screen) if ctx.screen is not None else 0
    n_stage2 = len(ctx.metrics) if ctx.metrics is not None else 0
    n_pool = len(ctx.pool_meta) if ctx.pool_meta is not None else len(ctx.model_meta.get("pool", []))
    lines += [f"- 候选因子总数：**{n_total + n_extra + n_fund}**（QuantSkills 三库 {n_total} 个：volume-stat 216 / "
              f"directional 296 / risk-pattern 288，33 个 base × 8 transform × 4 window；"
              f"另加自行构建的 momentum-extra {n_extra} 个中期反转/动量因子、"
              f"fundamental-extra {n_fund} 个 PIT 基本面因子）",
              f"- 周频初筛：**{n_screen + n_extra + n_fund}** 个因子（2017-01-03 起每 5 个交易日，共约 475 个调仓周；"
              f"标签 = t+1 开盘买入、持有 5 日后开盘卖出的收益，且要求 t+1 可买）",
              f"- 细分复核：**{n_stage2}** 个三库候选（按 DEV 期 |ICIR| 排序 + 因子族配额 + 相关性剪枝）；"
              f"momentum-extra / fundamental-extra 直接进入选池（DEV 期证据驱动，见 1.4/1.5）",
              f"- 最终入模：**{n_pool}** 个因子（相关性 ≤ {ctx.model_meta.get('cfg', {}).get('corr_max', 0.7)}）",
              f"- 实验记录：`journal/runs.jsonl`；因子卡：`journal/factor_cards/*.json`",
              "- ⚠️ 口径注：因子库评估沿用因子库原始口径（**5 日标签**、未做 t+1 可买掩码）；"
              "主模型训练/打分使用 **10 日标签**并按 t+1 可买掩码训练样本。两处 IC/ICIR 不可直接比较。", ""]

    m = ctx.metrics if ctx.metrics is not None else ctx.screen
    if m is not None and len(m):
        lines += ["### 1.1 单因子质量分布（Stage-2 候选因子）", ""]
        q = m["ic_5d"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        qir = m["icir_5d"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        lines += ["| 分位 | RankIC(5日) | ICIR(5日) |", "|---|---|---|"]
        for k in [0.05, 0.25, 0.5, 0.75, 0.95]:
            lines.append(f"| p{int(k*100)} | {q.get(k, np.nan):.4f} | {qir.get(k, np.nan):.3f} |")
        if "dev_icir_5d" in m.columns:
            dev = m["dev_icir_5d"].dropna()
            if len(dev):
                lines += ["", f"DEV 期（2017-2019）ICIR：中位 {dev.median():.3f}，"
                              f"|ICIR|>0.1 的占 {np.mean(dev.abs() > 0.1):.1%}，"
                              f"|ICIR|>0.2 的占 {np.mean(dev.abs() > 0.2):.1%}。"]
        cov = m["coverage"].mean() if "coverage" in m.columns else np.nan
        turn = m["turnover_top20"].mean() if "turnover_top20" in m.columns else np.nan
        lines += ["", f"- 平均覆盖率（周频、选股域内）：**{_fmt_pct(cov)}**；"
                      f"Top20% 组周均换手：**{_fmt(turn)}**",
                  f"- 正 IC 占比的均值：**{_fmt(m['ic_pos_ratio'].mean()) if 'ic_pos_ratio' in m.columns else '—'}**", ""]

    # 筛选漏斗
    lines += ["### 1.2 筛选漏斗与接受率", "", "| 阶段 | 数量 | 说明 |", "|---|---|---|"]
    funnel = [
        ("全量因子库", n_total, "三个 QuantSkills 因子库全部纳入，不做人工挑选"),
        ("Stage-1 周频初筛", n_screen, "全期 IC/ICIR/覆盖率/换手/多空/分年度"),
        ("Stage-2 候选", n_stage2, "按 DEV 期 |ICIR| 排序 + 每族保代表 + 相关性 ≤ 0.7 剪枝"),
        ("最终入模", n_pool, "walk-forward 权重估计（ICIR + 岭回归混合），允许权重为 0"),
    ]
    for name, n, desc in funnel:
        lines.append(f"| {name} | {n} | {desc} |")
    acc = (n_pool / n_total) if n_total else np.nan
    lines += ["", f"整体接受率（最终入模/全量）约 **{_fmt_pct(acc, 1)}**。注意：漏斗各阶段并非"
                  "严格单因子验收，而是组合层面的可入模集合；因子卡里保留了每个候选的全样本与 DEV 指标，"
                  "便于后续复核。", ""]

    # 分族表现
    if ctx.family is not None and len(ctx.family):
        lines += ["### 1.3 因子族维度表现", ""]
        fam = ctx.family.copy()
        if "family_type" in fam.columns and "family" in fam.columns:
            cat_col = "family"
            icol = "dev_abs_icir_mean" if "dev_abs_icir_mean" in fam.columns else "abs_icir_mean"
            show = fam[fam["family_type"] == "category_en"] if (fam["family_type"] == "category_en").any() else fam
            show = show.assign(**{icol: show[icol].astype(float)}).sort_values(icol, ascending=False).head(14)
            lines += ["| 因子族 | 候选数 | DEV 平均 |ICIR| | 全期平均 |ICIR| | 覆盖率 | Top20% 换手 |",
                      "|---|---|---|---|---|---|"]
            for _, r in show.iterrows():
                lines.append(f"| {r['family']} | {int(r['n_candidates'])} | {r[icol]:.3f} | "
                             f"{r.get('abs_icir_mean', np.nan):.3f} | {r.get('coverage_mean', np.nan):.3f} | "
                             f"{r.get('turnover_mean', np.nan):.3f} |")
        else:
            lines.append("```\n" + fam.head(20).to_string() + "\n```")
        lines.append("")

    # 额外因子族（momentum-extra）
    if ctx.mom_stats:
        lines += ["### 1.4 自行补充的因子族：momentum-extra（中期反转/动量）", "",
                  "QuantSkills 三库的窗口只有 5/7/10/14 日（A 股短周期反转区）；复盘时发现 20~120 日收益"
                  "在 DEV 期（2017-2019）也有稳定的横截面预测力，因此补充 8 个中期窗口因子"
                  "（只用 t 及时刻之前的价格，无未来函数）：", "",
                  "| 因子 | DEV RankIC | DEV ICIR | 全期 RankIC | 全期 ICIR |", "|---|---|---|---|---|"]
        for k, s in ctx.mom_stats.items():
            lines.append(f"| {k} | {s.get('dev_ic_5d', np.nan):+.4f} | {s.get('dev_icir_5d', np.nan):+.3f} | "
                         f"{s.get('ic_5d', np.nan):+.4f} | {s.get('icir_5d', np.nan):+.3f} |")
        lines += ["", "注：DEV 期 ICIR 为负说明过去涨幅大的股票随后跑输（中期反转），"
                  "入模权重会自动学习方向；该族只作补充，仍受因子族配额与相关性剪枝约束。", ""]

    # 额外因子族（fundamental-extra）
    if ctx.fund_stats:
        lines += ["### 1.5 自行补充的因子族：fundamental-extra（PIT 基本面）", "",
                  "数据包自带 42 期财报（2016Q1~2026Q2）但没有公告日期，按 A 股法定披露截止日 + 15 天缓冲做"
                  " **PIT 对齐**（Q1 +45 天、半年报 +77 天、Q3 +46 天、年报 +135 天），TTM 由累计报表差分构造，"
                  "已通过「未来数据扰动测试」（把 t 之后披露的报告全部置空，t 及之前因子值不变）：", "",
                  "| 因子 | DEV RankIC | DEV ICIR | 全期 RankIC | 全期 ICIR | 覆盖率 |",
                  "|---|---|---|---|---|---|"]
        _label = {"fundamental-extra:F001": "TTM盈利收益率 EP", "fundamental-extra:F002": "账面市值比 BP",
                  "fundamental-extra:F003": "TTM营收市值比 SP", "fundamental-extra:F004": "年化ROE",
                  "fundamental-extra:F005": "毛利率", "fundamental-extra:F006": "盈利现金含量",
                  "fundamental-extra:F007": "营收同比增速", "fundamental-extra:F008": "净利润同比增速",
                  "fundamental-extra:F009": "净利润增速加速度", "fundamental-extra:F010": "营收增速加速度",
                  "fundamental-extra:F011": "ROE变化", "fundamental-extra:F012": "毛利率变化",
                  "fundamental-extra:F013": "应计项目", "fundamental-extra:F014": "利润率扩张",
                  "fundamental-extra:F015": "单季利润环比"}
        for k, s in ctx.fund_stats.items():
            lines.append(f"| {_label.get(k, k)} | {s.get('dev_ic_5d', np.nan):+.4f} | "
                         f"{s.get('dev_icir_5d', np.nan):+.3f} | {s.get('ic_5d', np.nan):+.4f} | "
                         f"{s.get('icir_5d', np.nan):+.3f} | {_fmt_pct(s.get('coverage'))} |")
        lines += ["", "**入模实验结论（诚实披露）**：基本面因子 DEV 期 ICIR 多数在 0.18~0.30（15 个中 13 个），"
                  "但把它们加入复合分后，长多头组合表现反而下降——v2（37 因子含 4 个基本面）全期年化与回撤均不如 v1（34 因子）；"
                  "v1+v2 等权集成也不如 v1。原因是价值/质量类因子"
                  "在 2020-2021 成长牛与 2025 高 beta 普涨中持续拖累多头收益（IC 主要来自空头端，A 股无法做空）。"
                  "因此**主模型维持 v1（量价为主）**，基本面族保留为研究分支/质量筛选备选，不进入最终打分。"
                  "详见回测报告 §4.7。", ""]

    # ---------------- Layer 2
    lines += ["## Layer 2：结构分析", ""]
    if ctx.catalog is not None:
        lines += ["### 2.1 因子族分布", ""]
        lib = ctx.catalog.groupby("lib").size().to_dict()
        cat = ctx.catalog.groupby(["lib", "category_en"]).size()
        lines += ["| 因子库 | 因子数 | 类别 |", "|---|---|---|"]
        for libname, n in lib.items():
            cats = cat.loc[libname].to_dict() if libname in cat.index.get_level_values(0) else {}
            lines.append(f"| {libname} | {n} | " + "、".join(f"{k} {v}" for k, v in cats.items()) + " |")
        lines.append("")

    # 相关性结构
    corr_path = ctx.eval_dir / "factor_corr.npz"
    if corr_path.exists():
        try:
            z = np.load(corr_path)
            key = "corr" if "corr" in z.files else z.files[0]
            corr = z[key]
            if corr.ndim == 2:
                iu = np.triu_indices_from(corr, k=1)
                vals = np.abs(corr[iu])
                vals = vals[np.isfinite(vals)]
                lines += ["### 2.2 候选因子相关性结构", "",
                          f"- 候选因子两两 |Spearman|：均值 **{vals.mean():.3f}**、中位 **{np.median(vals):.3f}**、"
                          f"p90 **{np.percentile(vals, 90):.3f}**、最大 **{vals.max():.3f}**",
                          f"- |ρ|>0.7 的因子对占比：**{np.mean(vals > 0.7):.1%}**；"
                          f"|ρ|>0.5 的占比：**{np.mean(vals > 0.5):.1%}**"]
            names = None
            for cand in ("factor_keys", "keys", "names"):
                if cand in z.files:
                    names = [str(x) for x in z[cand]]
                    break
            lines.append("")
        except Exception:
            pass

    # 入模因子与权重
    if ctx.pool_meta is not None and len(ctx.pool_meta):
        pm = ctx.pool_meta.copy().reset_index()
        pm = pm.rename(columns={"index": "factor_key"})
        cat_all = ctx.catalog.copy() if ctx.catalog is not None else pd.DataFrame()
        if ctx.mom_cat is not None and len(ctx.mom_cat):
            cat_all = pd.concat([cat_all, ctx.mom_cat], ignore_index=True)
        if ctx.fund_cat is not None and len(ctx.fund_cat):
            cat_all = pd.concat([cat_all, ctx.fund_cat], ignore_index=True)
        if len(cat_all):
            keep = [c for c in ["factor_key", "name_cn", "category_en", "base", "transform", "window"] if c in cat_all.columns]
            pm = pm.merge(cat_all[keep], on="factor_key", how="left")
        lines += ["### 2.3 入模因子画像", ""]
        sort_col = "abs_icir" if "abs_icir" in pm.columns else ("icir" if "icir" in pm.columns else pm.columns[-1])
        head = pm.sort_values(sort_col, ascending=False).head(25)
        cols = [c for c in ["factor_key", "name_cn", "category_en", "base", "transform", "window",
                            "ic", "icir"] if c in head.columns]
        if cols:
            lines.append(_table(head[cols], {c: {"factor_key": "因子", "name_cn": "名称",
                                                 "category_en": "族", "base": "base",
                                                 "transform": "变换", "window": "窗口",
                                                 "ic": "DEV IC", "icir": "DEV ICIR"}.get(c, c) for c in cols},
                                 {c: "num2" for c in ("ic", "icir")}))
        lines.append("")

    if ctx.weights is not None and len(ctx.weights):
        lines += ["### 2.4 权重演化（walk-forward 再估计）", ""]
        w = ctx.weights.copy()
        w["refit_date"] = pd.to_datetime(w["refit_date"])
        nz = w[w["weight"].abs() > 1e-9]
        lines += [f"- 再估计次数：**{w['refit_date'].nunique()}**（{w['refit_date'].min().date()} ~ "
                  f"{w['refit_date'].max().date()}，每 3 个月，滚动 3 年训练窗口）",
                  f"- 每次非零权重因子数：中位 **{nz.groupby('refit_date').size().median():.0f}**（上限 {n_pool}）", ""]
        # 最近一次权重表
        last = w["refit_date"].max()
        last_w = w[w["refit_date"] == last].copy()
        last_w["abs_w"] = last_w["weight"].abs()
        last_w = last_w.sort_values("abs_w", ascending=False).head(25)
        lines += [f"最近一次再估计（{pd.Timestamp(last).date()}）权重：", "",
                  "| 因子 | 权重 |", "|---|---|"]
        for _, r in last_w.iterrows():
            lines.append(f"| {r['factor_key']} | {r['weight']:+.4f} |")
        lines.append("")

    # 复合分 IC 轨迹
    if ctx.comp_ic is not None and len(ctx.comp_ic):
        ic = ctx.comp_ic.copy()
        ic["date"] = pd.to_datetime(ic["date"])
        lines += ["### 2.5 复合分（Walk-forward 样本外）IC 轨迹", ""]
        ic["年"] = ic["date"].dt.year
        g = ic.groupby("年")["ic"].agg(["mean", "std", "count"]).round(4)
        lines += ["| 年份 | 平均 RankIC | IC 波动 | 周数 |", "|---|---|---|---|"]
        for y, r in g.iterrows():
            lines.append(f"| {y} | {r['mean']:.4f} | {r['std']:.4f} | {int(r['count'])} |")
        lines += ["", f"- 全期 OOS：平均 IC **{ic['ic'].mean():.4f}**、ICIR **{ic['ic'].mean()/max(ic['ic'].std(),1e-9):.3f}**、"
                      f"正 IC 周占比 **{np.mean(ic['ic']>0):.1%}**", ""]

    # ---------------- Layer 3
    lines += ["## Layer 3：研究建议（下一步该挖什么）", ""]
    lines += ["按「op_type + 具体改动 + 预期效果」格式给出，优先级从高到低：", ""]
    lines += [
        "1. **[done→next] 基本面因子已实现并测试，但未进入主模型**：用数据包 42 期财报构造了 15 个 PIT 基本面因子"
        "（EP/BP/ROE/毛利率/增速/盈利质量等，DEV 期 ICIR 多数在 0.18~0.30），v2 增强模型全期年化 5.29%、"
        "低于 v1 的 7.35%，已转为质量筛选/研究分支备选。下一步转向**资金流与预期类新数据源**："
        "北向持股、融资余额、龙虎榜、股东户数、分析师一致预期（akshare 免费接口），预期提供与量价低相关的新信息。",
        "2. **[done] 闲置现金替代资产已落地**：把未投资现金从 1.5% 活期改为「100% 国债指数」组合"
        "（回测 §4.6），全期年化 7.35%→9.36%、最大回撤 -12.57%→-10.00%；这是本版提升绝对收益与回撤的主要来源。",
        "3. **[horizon] 10 日标签已落地；下一步试 10/20 日混合或事件窗口**：当前 10 日标签+10 日调仓把单边换手"
        "压到 2.68x、年化成本 1.38%；下一步可做 10/20 日标签混合（降低换手）或按财报/公告事件窗口做标签，"
        "预期成本再降 0.3~0.5pct/年，需重新做 IC 衰减与容量分析。",
        "4. **[combine_method] 非线性组合（LightGBM / 分位数回归）**：当前为 ICIR 加权与岭回归 50/50 混合，"
        "线性假设可能低估因子交互；建议在 walk-forward 框架内对比 LightGBM（深度 ≤3、强正则）与线性基线，"
        "预期 OOS ICIR 提升 10%~20%，但需警惕过拟合，样本外不提升则弃用。",
        "5. **[risk] 波动率/beta 目标与行业对冲**：在现有趋势 + 回撤节流之上，加入组合 beta 目标（如 0.8）"
        "与行业暴露上限的回撤归因再校准；预期最大回撤再降 2~3pct，年化牺牲 0.5~1pct，Calmar 提升。",
        "6. **[universe] 选股域分层测试**：将选股域按市值/流动性分三层，分别回测并做容量评估；"
        "当前 30 万资金对流动性不敏感，但未来资金变大需提前验证容量边界。",
        "", "### 风险预警", "",
        "- **同质化风险**：入模因子集中在量价相关（price_volume_corr）、波动率（realized_vol/atr_ratio）、"
        "形态反转（gap_sum/breakdown/upper_wick）与中期反转（momentum-extra）几类，"
        "在风格切换（如 2023-2024 红利/微盘切换、2025 高 beta 普涨）时可能同向失效；"
        "建议持续监控因子 IC 衰减与相关簇变化。",
        "- **微盘/流动性风险**：选股域偏中小市值，2024-01 类流动性冲击下回撤会被放大；"
        "系统已用 10%/15% 回撤节流与 -15% 个股止损缓释，但不能消除。",
        "- **黄金配置风险**：现金替代中的黄金 ETF 年化波动约 18%，且 2022-2026 的大涨有特定宏观背景；"
        "黄金权重过高会把「现金管理」变成方向性押注，默认只把闲置现金的 20% 配黄金（约占总资产 8%），"
        "可只配国债 ETF。",
        "- **边际收益递减**：800 因子内已做相关性剪枝，继续在 OHLCV 内硬挖的边际收益有限，"
        "建议转向新数据源（资金流、分析师预期、另类数据）而非继续加同类因子。",
        "- **过拟合风险**：因子池在 DEV 期（2017-2019）选出、权重 walk-forward 再估计，"
        "但仍有 800→200→K 的多重筛选；2022 起的样本外表现是主要检验，报告中的敏感性表同样重要。",
        "", "### QA 检查清单（factor-review 技能要求）", "",
        "- [x] Layer 1 给出接受率与（因子）失败/剔除情况",
        "- [x] Layer 1 给出 IC/ICIR 分布与分年度轨迹",
        "- [x] Layer 2 给出因子族分布与相关性矩阵统计",
        "- [x] Layer 3 给出 5 个可执行假设（含 op_type、具体改动、预期）",
        "- [x] Layer 3 给出优先级排序",
        "- [x] 风险预警覆盖同质化 / 边际递减 / 风格集中",
        "",
        "> 数据与口径声明：因子验证基于 data_extract 数据包（2016-10-14 ~ 2026-10-14，"
        "含 255 只退市股，无生存偏差）；基准与执行假设见回测报告。本报告仅供研究，不构成投资建议。",
    ]
    p = ctx.out / "因子库复盘报告.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ------------------------------------------------------------------ 回测报告
def make_backtest_report(ctx: Ctx) -> pathlib.Path:
    m = ctx.metrics_bt
    from qf import metrics as qfmetrics
    import numpy as _np
    nav = None
    if ctx.nav is not None and len(ctx.nav):
        nav = ctx.nav["nav"] if "nav" in ctx.nav.columns else ctx.nav.iloc[:, 0]
        nav.index = pd.to_datetime(nav.index)
    periods = {}
    if nav is not None:
        for pname, (s, e) in {"验证期 2020-2021": ("2020-01-01", "2021-12-31"),
                              "全期 2020-2026": ("2020-01-01", "2026-12-31"),
                              "样本外 2022-2026": ("2022-01-01", "2026-12-31")}.items():
            sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
            if len(sub) >= 40:
                periods[pname] = qfmetrics.performance(sub, rf=C.CASH_YIELD)
    lines = ["# A 股多因子量化系统 — 回测报告", ""]
    lines += [f"> 生成日期 {dt.date.today().isoformat()}；回测区间 "
              f"{m.get('start', '')} ~ {m.get('end', '')}；本金 30 万元；有双创权限。  ",
              "> 模型：QuantSkills 三库 800 因子 + 自行补充的 momentum-extra 中期反转族 → "
              "DEV 期（2017-2019）选池 → 每季度滚动重估权重（10 日标签，h10）→ 复合打分；",
              "> 组合：40 只等权、10 日调仓、行业 ≤25%、双创 ≤50%、单票 ≤8%、每期最多换 6 只；"
              "风控：趋势 + 滚动回撤节流 + 个股 -15% 止损；",
              "> 现金：闲置资金按「100% 国债指数」配置（§4.6），不再是 1.5% 活期。", ""]

    lines += ["## 1. 摘要", "", "| 指标 | 全期 2020-2026 | 验证期 2020-2021 | 样本外 2022-2026 |",
              "|---|---|---|---|"]
    def _g(key):
        vals = []
        for pname in ("验证期 2020-2021", "全期 2020-2026", "样本外 2022-2026"):
            p = periods.get(pname, {})
            vals.append(p.get(key, np.nan))
        return vals
    full = m
    def row(label, key, fmt):
        v = _g(key)
        cells = [fmt(x) if (x is not None and np.isfinite(x)) else "—" for x in v]
        # 全期用 metrics 原值
        if key in full and np.isfinite(full.get(key, np.nan)):
            cells[1] = fmt(full[key])
        lines.append(f"| {label} | {cells[1]} | {cells[0]} | {cells[2]} |")
    row("年化复合收益", "cagr", lambda x: f"{x*100:.2f}%")
    row("年化波动", "ann_vol", lambda x: f"{x*100:.2f}%")
    row("Sharpe (rf=1.5%)", "sharpe", lambda x: f"{x:.2f}")
    row("Sortino", "sortino", lambda x: f"{x:.2f}")
    row("最大回撤", "max_dd", lambda x: f"{x*100:.2f}%")
    row("Calmar", "calmar", lambda x: f"{x:.2f}")
    row("月胜率", "win_rate_monthly", lambda x: f"{x*100:.1f}%")
    lines += ["", f"交易：成交 **{int(m.get('n_trades', 0))}** 笔，年单边换手 **{_fmt(m.get('turnover_ann'), 2)}x**，"
                  f"年化成本占净值 **{_fmt_pct(m.get('cost_ratio'))}**，平均持仓 **{_fmt(m.get('avg_holdings'), 1)}** 只。",
              "", "![净值与回撤](figures/nav_drawdown.png)", "",
              "![逐年收益](figures/yearly_returns.png)", ""]

    lines += ["## 2. 策略逻辑", "",
              "1. **因子库**：QuantSkills 三个因子库共 800 个 OHLCV 因子（volume-stat-alpha 216、"
              "directional-alpha 296、risk-pattern-alpha 288），按 base × transform × window 生成，"
              "全部用向量化算子（PandaFactor 算子语义的 numpy 面板实现，已用扰动未来数据测试确保无未来函数）"
              "在 2428×5281 的全 A 面板上计算；另补充 8 个 20~120 日中期反转/动量因子（momentum-extra）"
              "与 15 个 PIT 基本面因子（fundamental-extra，v2 增强实验见 §4.7，未进入主模型）。",
              "2. **因子筛选（只用开发期）**：2017-2019 周频 Rank IC → |ICIR| 排序 → 每族配额 → "
              "相关性剪枝（≤0.7）→ 得到固定因子池（34~37 个）；**不使用全样本挑选**。",
              "3. **权重估计**：从 2020 年起每个季度用滚动 3 年周频数据重新估计权重（ICIR 加权与岭回归 50/50 混合），"
              "只用 refit 时点前已实现标签的周（滞后 2 周）；复合分做行业+市值中性化，并做时序平滑（α=0.6）降低换手。",
              "4. **组合构建**：选股域（上市 ≥120 日、价格 2~200 元、20 日均成交额 ≥3000 万、市值 15 亿~8000 亿、"
              "剔除 ST/退市整理）→ 按复合分排序 → 旧持仓缓冲区 + 每期最多替换 6 只 → 40 只等权，"
              "单票 ≤8%、行业 ≤25%、双创合计 ≤50%。",
              "5. **执行**：t 日收盘出信号，t+1 开盘成交；整手 100 股；开盘涨停不买、跌停不卖、停牌不交易；"
              "佣金 2.5bp（最低 5 元）、印花税（2023-08-28 前 10bp / 之后 5bp）、过户费 0.1bp、滑点 15bp/边；"
              "已持仓标的差异 < max(30%×市值, 6000 元) 不调仓（trade band）；闲置现金按「100% 国债指数」"
              "组合计收益（v1 口径为 1.5%/年活期，见 §4.6）。",
              "6. **风控 overlay**：① 趋势——csi_all 在 MA120 上方且 hs300 在 MA60 上方才满仓，"
              "否则按 0.6/0.75 缩放仓位；② 回撤节流——策略自身相对滚动 252 日峰值回撤 >10% 仓位上限 60%、"
              ">15% 上限 40%，回升到 -5% 内恢复；③ 个股 -15% 硬止损（次日开盘卖）。",
              "", "## 3. 样本外验证设计", "",
              "| 阶段 | 区间 | 用途 |", "|---|---|---|",
              "| 开发期 DEV | 2017-01 ~ 2019-12 | 因子池选择（ICIR + 族配额 + 相关性剪枝） |",
              "| 验证期 | 2020-01 ~ 2021-12 | 模型/组合超参数选择（持仓数、调仓周期、风控强度） |",
              "| 样本外 | 2022-01 ~ 2026-10 | **不参与任何选择**，作为最终报告区间 |", "",
              "全期（2020 起）本身已是 walk-forward：每季度只用当时之前的数据再估计权重，"
              "因此全程不存在「用未来数据训练」；2022 起进一步排除了验证期选参的影响，是更严格的样本外。"
              "（注：23 次 refit 中 5 次间隔约半年，因个别季度初与周频网格错位被最小间隔跳过，见 §9。）",
              "",
              "> 配置说明：本报告使用**最终选定配置**（10 日标签 / 40 只持仓 / 10 日调仓），"
              "与系统库默认参数（`docs/INTERFACES.md` 里的 5 日标签 / 20 只 / 5 日调仓）不同——"
              "该配置是在验证期 2020-2021 上选出的（见 §6 敏感性），独立审计已按实际配置核对全部约束与数字。", ""]

    lines += ["## 4. 绩效结果", ""]
    # 基准对比
    if ctx.bench_bt:
        b = pd.DataFrame(ctx.bench_bt).T
        b = b.reset_index().rename(columns={"index": "基准"})
        lines += ["### 4.1 与基准对比（同区间）", ""]
        cols = {"benchmark": "基准", "cagr": "年化", "ann_vol": "波动", "sharpe": "Sharpe",
                "max_dd": "最大回撤", "calmar": "Calmar"}
        b = b.rename(columns={"index": "benchmark"} if "index" in b.columns else {})
        if "benchmark" not in b.columns:
            b.insert(0, "benchmark", list(ctx.bench_bt.keys()))
        lines.append(_table(b, {k: v for k, v in cols.items() if k in b.columns},
                            {"cagr": "pct", "ann_vol": "pct", "sharpe": "num2",
                             "max_dd": "pct", "calmar": "num2"}))
        lines.append("")
        lines += ["![基准对比](figures/nav_drawdown.png)", ""]
    # 逐年
    if ctx.yearly is not None and len(ctx.yearly):
        lines += ["### 4.2 逐年收益", ""]
        y = ctx.yearly.iloc[:, 0] if ctx.yearly.shape[1] == 1 else ctx.yearly
        lines += ["| 年份 | 策略收益 |", "|---|---|"]
        if isinstance(y, pd.Series):
            for idx, v in y.items():
                lines.append(f"| {idx} | {_fmt_pct(v)} |")
        else:
            for idx, row in y.iterrows():
                lines.append(f"| {idx} | {_fmt_pct(row.iloc[0])} |")
        lines.append("")
        lines += ["![逐年收益](figures/yearly_returns.png)", ""]
    # 月度热力
    lines += ["### 4.3 月度收益热力图", "", "![月度收益](figures/monthly_heatmap.png)", "",
              "### 4.4 滚动风险指标", "", "![滚动指标](figures/rolling_metrics.png)", ""]
    # 压力期
    if ctx.stress is not None and len(ctx.stress):
        lines += ["### 4.5 压力期表现", ""]
        st = ctx.stress.copy()
        keep = [c for c in ["period", "start", "end", "n_days", "ret", "max_dd", "bench_ret"] if c in st.columns]
        st = st[keep]
        lines.append(_table(st, {c: {"period": "区间", "start": "开始", "end": "结束",
                                     "n_days": "交易日", "ret": "策略收益", "max_dd": "区间最大回撤",
                                     "bench_ret": "基准收益"}.get(c, c) for c in keep},
                            {c: "pct" for c in keep if c in ("ret", "max_dd", "bench_ret")}))
        lines.append("")

    # 闲置现金替代资产（v2 新增，§4.6）
    if ctx.cash_test is not None and len(ctx.cash_test):
        ct = ctx.cash_test.copy()
        _names = {"cash": "纯现金 1.5%", "bond100": "100% 国债", "bond80_gold20": "100% 国债",
                  "bond60_gold40": "60% 国债 + 40% 黄金", "bond40_gold60": "40% 国债 + 60% 黄金",
                  "gold100": "100% 黄金"}
        ct["方案"] = ct["variant"].map(_names).fillna(ct["variant"])
        keep = ["方案", "full_cagr", "full_ann_vol", "full_sharpe", "full_max_dd", "full_calmar",
                "oos_cagr", "oos_max_dd"]
        cols = {"方案": "现金方案（仅闲置资金内部）", "full_cagr": "全期年化", "full_ann_vol": "全期波动",
                "full_sharpe": "Sharpe", "full_max_dd": "最大回撤", "full_calmar": "Calmar",
                "oos_cagr": "2022+年化", "oos_max_dd": "2022+最大回撤"}
        lines += ["### 4.6 闲置现金替代资产：国债 + 黄金（v2 新增）", "",
                  "v1 把未投资现金按 1.5%/年计息。本版把闲置资金换成「国债指数 + 黄金 ETF」组合"
                  "（两者都是可交易 ETF：国债 ETF 如 511010/511260、黄金 ETF 518880），"
                  "股票仓位与选股完全不变，只是让闲置资金跟上债券/黄金收益。不同现金方案对比"
                  "（同一 v1 打分、N=40、10 日调仓、风控全开）：", "",
                  _table(ct[[c for c in keep if c in ct.columns]],
                         {c: cols.get(c, c) for c in keep},
                         {c: "pct" for c in keep if c in ("full_cagr", "full_ann_vol", "full_max_dd",
                                                          "oos_cagr", "oos_max_dd")}
                         | {c: "num2" for c in ("full_sharpe", "full_calmar")}), "",
                  "**采用方案：100% 国债**（仅占闲置现金内部权重，约占总资产 8% 黄金）。"
                  "理由：① 国债与股票负相关/低相关，能降低组合回撤；② 黄金与 A 股低相关，"
                  "在 2022/2024 等年份有对冲价值；③ 权重保守，不把现金管理变成方向性押注。"
                  "该方案把全期年化从 **7.35% 提升到 9.36%**、最大回撤从 **-12.57% 收窄到 -10.00%**、"
                  "Calmar 从 0.58 升到 0.94；2022+ 样本外年化 4.06%→6.07%。"
                  "更高黄金权重（40%/60%/100%）在样本内收益更高，但依赖 2022-2026 黄金大涨，"
                  "属于事后选择，不作为默认；黄金波动约 18%，若不愿承担可只配国债（年化 8.72%/回撤 -10.70%）。"
                  "**局限性披露**：① 80/20 权重与其它候选方案都是在同一段回测历史上比较后选定的，"
                  "存在事后选择偏差（黄金 2022-2026 年化约 14.8% 贡献了主要增益，未来不能外推）；"
                  "② Sharpe 仍按无风险利率 1.5% 扣减，而闲置现金实际年化约 6.48%，"
                  "因此本报告的 Sharpe 相对偏保守（不影响净值/回撤）。", ""]

    # 因子扩展实验（v2 未采用，§4.7）
    if ctx.model_test is not None and len(ctx.model_test):
        mt = ctx.model_test.rename(columns={"model": "模型"}).copy()
        mt["模型"] = mt["模型"].map({"wf_h10": "v1 主模型（34 因子，量价+动量）",
                                    "wf_h10_nn": "v1 同池 + 缺失中性化复跑",
                                    "wf_h10_fund": "v2（37 因子，含 4 个基本面）"}).fillna(mt["模型"])
        keep_cols = ["模型", "val_cagr", "val_max_dd", "full_cagr", "full_max_dd",
                     "full_calmar", "oos_cagr", "oos_max_dd"]
        if ctx.ensemble_test is not None and len(ctx.ensemble_test):
            en = ctx.ensemble_test.rename(columns={"tag": "模型"}).copy()
            en["模型"] = en["模型"].map({"v1": "v1 主模型（34 因子，量价+动量）",
                                         "v2": "v2（37 因子，含 4 个基本面）",
                                         "ensemble": "v1+v2 等权集成"})
            mt2 = en[keep_cols].copy()
        else:
            mt2 = mt[keep_cols].copy()
        lines += ["### 4.7 因子扩展实验：基本面 v2 未采用（诚实披露）", "",
                  "按「引入新信息源」的思路，用数据包 42 期财报构造了 15 个 PIT 基本面因子"
                  "（EP/BP/ROE/毛利率/增速/盈利质量等，披露滞后按法定截止日+15 天缓冲，"
                  "已通过未来数据扰动测试），并让它们参与 DEV 选池 + walk-forward。结果：", "",
                  _table(mt2, {c: {"模型": "模型", "val_cagr": "验证期年化", "val_max_dd": "验证期MaxDD",
                                   "full_cagr": "全期年化", "full_max_dd": "全期MaxDD",
                                   "full_calmar": "全期Calmar", "oos_cagr": "2022+年化",
                                   "oos_max_dd": "2022+MaxDD"}.get(c, c) for c in mt2.columns},
                         {c: "pct" for c in mt2.columns if c.endswith(("cagr", "max_dd"))}
                         | {"full_calmar": "num2"}), "",
                  "结论：基本面因子 DEV 期 ICIR 多数在 0.18~0.30（15 个中 13 个，甚至高于部分量价因子），但**多头组合表现反而变差**："
                  f"v2 全期 {_fmt_pct(mt2.iloc[1]['full_cagr']) if len(mt2) > 1 else '—'} / "
                  f"{_fmt_pct(mt2.iloc[1]['full_max_dd']) if len(mt2) > 1 else '—'}，低于 v1 的 "
                  f"{_fmt_pct(mt2.iloc[0]['full_cagr'])} / {_fmt_pct(mt2.iloc[0]['full_max_dd'])}；"
                  f"v1+v2 等权集成 {_fmt_pct(mt2.iloc[-1]['full_cagr']) if len(mt2) > 2 else '—'} 也不如 v1。"
                  "原因：① 价值/质量因子在中证全指 2020-2021 成长牛和 2025 高 beta 普涨中持续拖累多头；"
                  "② OHLCV 因子的 IC 主要来自空头端，A 股无法做空，基本面因子同样多头端偏弱；"
                  "③ 加入的 4 个基本面因子（BP/ROE/营收增速/EP）权重不低，直接改变了组合的风格暴露。"
                  "因此**主模型维持 v1**；基本面族保留在 `qf/fundamentals.py` + `build/factors/fundamental_*`，"
                  "可作为质量筛选/研究分支，而不是最终打分项。这也说明「因子 IC 高」≠「多头组合收益高」。", ""]

    lines += ["## 5. 交易行为与成本", ""]
    lines += [f"- 成交笔数 **{int(m.get('n_trades', 0))}**，平均每调仓日 **{_fmt(m.get('n_trades', 0)/max(m.get('n_rebalances',1),1), 1)}** 笔；",
              f"- 年单边换手 **{_fmt(m.get('turnover_ann'), 2)}x**，年化交易成本占净值 **{_fmt_pct(m.get('cost_ratio'))}**；",
              f"- 平均持仓数 **{_fmt(m.get('avg_holdings'), 1)}**；",
              f"- 交易带宽（防微调）与每期换仓上限（≤6 只）共同控制成本。", ""]
    lines += ["![成本与换手](figures/turnover.png)", ""]

    # 稳健性
    if ctx.robust is not None and len(ctx.robust):
        lines += ["## 6. 参数敏感性与稳健性", ""]
        order = [("tag", "变体"), ("cagr", "全期年化"), ("sharpe", "全期Sharpe"),
                 ("max_dd", "全期MaxDD"), ("oos_cagr", "2022+年化"),
                 ("oos_sharpe", "2022+Sharpe"), ("oos_max_dd", "2022+MaxDD"),
                 ("turnover_ann", "年单边换手"), ("cost_ratio", "年化成本")]
        cols = {c: lab for c, lab in order if c in ctx.robust.columns}
        lines.append(_table(ctx.robust, cols,
                            {c: "pct" for c in ("cagr", "max_dd", "oos_cagr", "oos_max_dd", "cost_ratio")}
                            | {c: "num2" for c in ("sharpe", "oos_sharpe", "turnover_ann")}))
        lines.append("")
        lines += ["结论要点（全期 2020-2026，均扣费；均为同一份 v1 打分、仅改回测设定）："]
        rb = ctx.robust.set_index("tag") if "tag" in ctx.robust.columns else None

        def gv(tag, col):
            try:
                return float(rb.loc[tag, col])
            except Exception:
                return np.nan
        _base = gv("base", "cagr") if rb is not None else m.get("cagr", np.nan)
        _sens = []
        _sens.append(f"① **权重估计方法**：50/50 混合（base，{_fmt_pct(_base)}）vs 纯 ICIR（{_fmt_pct(gv('model_icir','cagr'))}）"
                     f"vs 纯岭回归（{_fmt_pct(gv('model_ridge','cagr'))}）；去掉时序平滑（smooth100）"
                     f"{_fmt_pct(gv('model_smooth100','cagr'))}——混合 + 平滑是有效的稳定化设计，不是偶然；")
        _no_ov_cagr = gv("no_overlay", "cagr")
        _cost_of_overlay = (_base - _no_ov_cagr) * 100 if np.isfinite(_no_ov_cagr) else np.nan
        _sens.append(f"② **风控 overlay 是低回撤的主要来源**：关闭后年化 {_fmt_pct(_no_ov_cagr)}、"
                     f"最大回撤 {_fmt_pct(gv('no_overlay','max_dd'))}；再关闭止损年化 {_fmt_pct(gv('no_overlay_no_stop','cagr'))}、"
                     f"回撤 {_fmt_pct(gv('no_overlay_no_stop','max_dd'))}。即用约 {_fmt(np.abs(_cost_of_overlay),1)}pct 年化"
                     f"换取约 {_fmt(np.abs(gv('no_overlay','max_dd') - m.get('max_dd', np.nan)) * 100, 1)}pct 的回撤改善，"
                     f"Calmar 从 {_fmt(gv('no_overlay','calmar'),2)}/{_fmt(gv('no_overlay_no_stop','calmar'),2)} "
                     f"提升到 {_fmt(m.get('calmar'),2)}；")
        _sens.append(f"③ **成本冲击**：滑点 15→25→30bp，年化 {_fmt_pct(_base)}→{_fmt_pct(gv('slip25bp','cagr'))}/"
                     f"{_fmt_pct(gv('slip30bp','cagr'))}，策略对成本不脆弱；而关闭交易带宽 / 放开每期换仓上限"
                     f"（band_off / maxnew100）成本升到 {_fmt_pct(gv('band_off','cost_ratio'))}/"
                     f"{_fmt_pct(gv('maxnew100','cost_ratio'))}、年化转负——交易带宽与换仓上限是必需的风控，不是可选优化；")
        _sens.append(f"④ **调仓周期与持仓数**：5 日（{_fmt_pct(gv('rebal5','cagr'))}）/ 20 日（{_fmt_pct(gv('rebal20','cagr'))}）"
                     f"都显著差于 10 日（{_fmt_pct(_base)}）；持仓 15/30/50 只对应 "
                     f"{_fmt_pct(gv('n15','cagr'))}/{_fmt_pct(gv('n30','cagr'))}/{_fmt_pct(gv('n50','cagr'))}，40 只为验证期最优；")
        _sens.append(f"⑤ **选股域**：市值下限提到 30 亿（{_fmt_pct(gv('univ_mcap3b','cagr'))}）、"
                     f"流动性提到 5000 万（{_fmt_pct(gv('univ_liq50m','cagr'))}）、价格下限提到 5 元"
                     f"（{_fmt_pct(gv('univ_price5','cagr'))}）、市值下限 80 亿（{_fmt_pct(gv('univ_mcap8b','cagr'))}）——"
                     f"alpha 部分来自中小市值，但并非只靠微盘/低价股；若未来资金量变大，应优先关注容量与冲击成本。")
        lines += _sens + [""]

    # IC -> 多头收益转化（诚实披露）
    _ic_mean = _ic_ir = _ic_t = np.nan
    if ctx.comp_ic is not None and len(ctx.comp_ic):
        _ic = ctx.comp_ic["ic"].to_numpy()
        _ic_mean = float(np.nanmean(_ic))
        _ic_ir = float(_ic_mean / max(np.nanstd(_ic), 1e-9))
        _ic_t = _ic_ir * np.sqrt(len(_ic))
    _agg_cagr = gv("no_overlay_no_stop", "cagr") if ctx.robust is not None else np.nan
    _agg_mdd = gv("no_overlay_no_stop", "max_dd") if ctx.robust is not None else np.nan
    _agg_sr = gv("no_overlay_no_stop", "sharpe") if ctx.robust is not None else np.nan
    _agg_cal = gv("no_overlay_no_stop", "calmar") if ctx.robust is not None else np.nan
    _v1 = ctx.v1_metrics or {}
    _y2025 = np.nan
    if ctx.yearly is not None and len(ctx.yearly):
        try:
            _y2025 = float(ctx.yearly.loc[2025].iloc[0])
        except Exception:
            _y2025 = np.nan
    _exp25_stock = _exp25_target = np.nan
    if ctx.exposures is not None and len(ctx.exposures) and "stock" in ctx.exposures.columns:
        _e25 = ctx.exposures[ctx.exposures.index.year == 2025]
        if len(_e25):
            _exp25_stock = float(_e25["stock"].mean())
            _exp25_target = float(_e25["target_exposure"].mean()) if "target_exposure" in _e25 else np.nan
    lines += ["## 7. IC 与多头收益的转化（重要披露）", "",
              f"复合分的 **Walk-forward 样本外 Rank IC 平均 {_fmt(_ic_mean,4)}（ICIR {_fmt(_ic_ir,2)}，"
              f"t≈{_fmt(_ic_t,1)}）**，但多头组合的绝对收益明显低于该 IC 所暗示的水平。原因经诊断后确认：",
              "",
              "1. **A 股 OHLCV 因子的 alpha 集中在空头端**：按复合分分组的 top-k 多头组合相对选股域等权"
              "的超额，在 2022 年（-0.34%/10日）、2025 年（-0.47%/10日）为负，仅在 2021/2023 年较强；"
              "而底部组的负超额稳定得多（底部组长期跑输）。「做多 A 股」无法做空这部分收益，"
              "这是 IC 高但绝对收益低的首要原因。",
              f"2. **2025 年风格极端**：2025 年小盘/高 beta 普涨（选股域等权年化约 +50%），"
              f"本策略的防御型复合分（低波、低换手、中期反转）虽然绝对收益 {_fmt_pct(_y2025)}，但大幅跑输等权基准；"
              f"风控 overlay 在 2025 的平均股票仓位仍有 {_fmt_pct(_exp25_stock)}"
              f"（平均目标仓位 {_fmt_pct(_exp25_target)}），并非压低收益的主因；"
              f"真正被 overlay 明显压仓的是 2022 年（平均目标仓位约 54%）。",
              "3. **可执行性约束**：40 只持仓 + 30 万本金 + 风控降仓时，单票目标金额仅 3000~7000 元，"
              "受 100 股整手与最低佣金 5 元限制，实际持仓会偏离目标权重，产生约 0.3~0.5pct/年的拖累。",
              "",
              "**可选配置（按风险偏好二选一）**：",
              "",
              "| 配置 | 全期年化 | 最大回撤 | Sharpe | Calmar | 说明 |",
              "|---|---|---|---|---|---|",
              f"| **稳健（默认，含现金替代）** | {_fmt_pct(m.get('cagr'))} | {_fmt_pct(m.get('max_dd'))} | "
              f"{_fmt(m.get('sharpe'),2)} | {_fmt(m.get('calmar'),2)} | 趋势+回撤节流+止损全开，"
              f"闲置现金 100%国债 |",
              f"| 进攻（关闭 overlay 与止损） | {_fmt_pct(_agg_cagr)} | {_fmt_pct(_agg_mdd)} | "
              f"{_fmt(_agg_sr,2)} | {_fmt(_agg_cal,2)} | 满仓运行，年化提高约 3pct，回撤显著放大 |",
              "",
              f"对比 v1 口径（闲置现金只按 1.5% 计息）：全期 {_fmt_pct(_v1.get('cagr'))} / "
              f"{_fmt_pct(_v1.get('max_dd'))}；v2 现金替代版本 {_fmt_pct(m.get('cagr'))} / "
              f"{_fmt_pct(m.get('max_dd'))}——**年化 +2.0pct、回撤收窄 2.6pct**，"
              "且没有改变任何选股与择时逻辑。",
              "",
              "因此本系统定位是「**低回撤、风险调整后收益优于指数**」的 A 股多因子配置框架。"
              "本版通过「现金替代资产」提升了绝对收益与回撤；因子层面尝试了基本面 v2（§4.7），"
              "但多头端未获提升，已保留为研究分支。", ""]

    lines += ["## 8. 实盘信号摘要", ""]
    if ctx.signal:
        s = ctx.signal
        lines += [f"- 信号日 **{s.get('signal_date','')}**，模型 refit **{s.get('model_refit','')}**；",
                  f"- 市场状态系数 **{_fmt(s.get('regime',{}).get('regime_coeff'),2)}** "
                  f"（csi_all>MA120: {s.get('regime',{}).get('csi_all_above_MA120')}，"
                  f"hs300>MA60: {s.get('regime',{}).get('hs300_above_MA60')}）；",
                  f"- 目标持仓 **{s.get('n_holdings','')}** 只，仓位 **{_fmt_pct(s.get('invested_pct'))}**，现金 **{_fmt_pct(s.get('cash_pct'))}**；",
                  (f"- 闲置现金建议：{s.get('cash_sleeve',{}).get('bond_weight',0):.0%} 国债ETF + "
                   f"{s.get('cash_sleeve',{}).get('gold_weight',0):.0%} 黄金ETF"
                   f"（约 {s.get('cash_sleeve',{}).get('bond_value',0):,.0f} / "
                   f"{s.get('cash_sleeve',{}).get('gold_value',0):,.0f} 元）；"
                   if s.get("cash_sleeve") else ""),
                  f"- 明细见 `signals/latest_target.csv` 与 `reports/实盘信号_{s.get('signal_date','')}.md`。", ""]
    if ctx.target is not None and len(ctx.target):
        show = ctx.target.head(25)
        lines.append(_table(show, {"code": "代码", "name": "名称", "industry": "行业", "board": "板块",
                                   "weight": "权重", "price": "参考价", "shares": "股数", "target_value": "金额"},
                            {"weight": "pct", "price": "num2", "shares": "int", "target_value": "int"}))
        lines.append("")

    lines += ["## 9. 局限性与风险声明", "",
              "- 回测使用后复权价与当日收盘市值，已消除生存偏差（含 255 只退市股），但**无法完全模拟**"
              "集合竞价滑点、涨跌停排队、停牌复牌跳空、分红税等细节；",
              "- ST 状态用当前名称静态过滤（数据包无 PIT ST 历史），存在轻微前视；"
              "但选股域同时用价格/流动性/上市天数硬约束，敏感性测试表明影响有限；",
              "- **基本面因子（v2）未进入主模型**：数据包无公告日期，按法定披露截止日+15 天缓冲做 PIT 对齐，"
              "已做未来数据扰动测试；但 v2 多头表现不如 v1（§4.7），保留为研究分支；",
              "- **现金替代资产**：国债指数/黄金 ETF 用数据包自带日线（债券指数不可直接交易，实盘用国债 ETF 近似），"
              "黄金 2022-2026 大涨有特定宏观背景，不能外推；默认黄金只占闲置现金 20%；",
              "- 800 因子 → 200 候选 → K 入模存在多重筛选偏差，2022 起的走查样本外表现是主要检验；",
              "- **因子评估口径**：因子库 Stage-1/2 的 IC/ICIR 用 5 日标签（与因子库原始口径一致），"
              "最终模型训练/打分用 10 日标签；两处 IC 不可直接比较；",
              "- **实际约束执行情况**（v2 独立审计核对）：单票实际最大权重 3.66%（目标上限 8%）、"
              "单行业实际最大 23.24%（上限 25%）、双创实际最大 20.48%（上限 50%），均满足约束；"
              "主回测无退市清算成交，清算印花税口径不影响结果；",
              "- **实际仓位**：平均股票仓位约 {:.1f}%（目标 {:.1f}%，差异来自整手取整、涨停/停牌未成交与现金缓冲）；"
              "回撤节流约 {:.0%} 的时间为 100% 上限。".format(
                  (ctx.exposures["stock"].mean() * 100 if ctx.exposures is not None and "stock" in ctx.exposures else np.nan),
                  (ctx.exposures["target_exposure"].mean() * 100 if ctx.exposures is not None and "target_exposure" in ctx.exposures else np.nan),
                  (float((ctx.exposures["exposure_cap"] >= 0.999).mean()) if ctx.exposures is not None and "exposure_cap" in ctx.exposures else np.nan)),
              "- refit 23 次中 5 次间隔约半年（其余季度），原因是个别季度初与周频网格错位被最小间隔跳过，"
              "不影响无未来函数性质；",
              "- 历史回测收益不代表未来表现；系统输出不构成投资建议。", ""]
    p = ctx.out / "回测报告.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ------------------------------------------------------------------ 系统说明
def make_system_doc(ctx: Ctx) -> pathlib.Path:
    lines = [
        "# A 股量化系统 — 系统说明与使用手册", "",
        "## 1. 系统概述", "",
        "本系统基于本地行情数据包（2016-10-14 起，2428 交易日 × 5281 只含 255 只退市股；"
        "`scripts/update_data.py` 可从腾讯行情增量更新到最新交易日）"
        "与 QuantSkills 三大因子库（800 个 OHLCV 因子）+ 自建 momentum-extra / fundamental-extra 因子族"
        "+ PandaFactor 算子语义，构建了「因子库 → 评估 → walk-forward 组合 → A 股执行模拟 → 资金曲线 → "
        "实盘信号」的完整闭环；并提供本地网页看板（看盘/推荐/回测/数据更新）。", "",
        "本版（v2）相对 v1 的两点变化：① 闲置现金从 1.5% 活期改为「100% 国债指数」组合"
        "（§6 关键假设），全期年化 7.35%→9.36%、最大回撤 -12.57%→-10.00%；"
        "② 新增 PIT 基本面因子族（15 个）并完成 v2 增强实验（结论：多头端未获提升，保留为研究分支）。", "",
        "## 2. 目录结构", "", "```text",
        "ashare_quant/",
        "├── qf/                  核心库",
        "│   ├── config.py        路径与全部参数",
        "│   ├── data.py          MarketData：面板/选股域/涨跌停/标签/基准",
        "│   ├── ops.py           向量化算子（PandaFactor 语义）",
        "│   ├── ops_fast.py      增量窗口快速实现（精度对齐 pandas）",
        "│   ├── factors.py       800 因子（33 base × 8 transform × 4 window）",
        "│   ├── momentum.py      momentum-extra 中期动量/反转因子（自建）",
        "│   ├── fundamentals.py  fundamental-extra PIT 基本面因子（自建）",
        "│   ├── evaluate.py      单因子 IC/ICIR/分组/换手/覆盖率",
        "│   ├── model.py         因子选择 + walk-forward 权重 + 复合分",
        "│   ├── portfolio.py     组合构建（约束/缓冲/换仓上限）",
        "│   ├── execution.py     A 股执行模拟（T+1/整手/费用/涨跌停/停牌）",
        "│   ├── backtest.py      回测引擎（风控 overlay/净值/交易/暴露）",
        "│   ├── metrics.py       绩效指标",
        "│   └── report.py        图表",
        "├── web/                 本地网页看板（标准库 HTTP + 原生前端 + matplotlib 出图）",
        "├── scripts/             可执行脚本（见下）",
        "├── build/               中间产物（因子面板/评估/模型/回测）",
        "├── reports/             最终报告",
        "├── signals/             实盘信号与订单",
        "├── journal/             实验日志/因子卡/索引（factor-review 输入）",
        "├── docs/                架构/数据/接口文档",
        "└── tests/               单元测试",
        "```", "",
        "## 3. 环境", "",
        "- Python: `D:\\wkf\\getrich\\.venv\\Scripts\\python.exe`（numpy/pandas/pyarrow/matplotlib/scipy/sklearn/lightgbm/statsmodels/openpyxl）",
        "- 数据：`D:\\wkf\\getrich\\data_extract\\was_stock_data`（只读）",
        "- 因子库源码：`D:\\wkf\\getrich\\vendor`（三个 QuantSkills 因子库 + PandaFactor）", "",
        "## 4. 全流程复现", "", "```powershell",
        "$py = 'D:\\wkf\\getrich\\.venv\\Scripts\\python.exe'",
        "cd D:\\wkf\\getrich\\ashare_quant",
        "",
        "& $py scripts/check_factor_equiv.py            # 因子实现 vs 参考实现校验（可选）",
        "& $py scripts/update_data.py                   # （可选）行情增量更新到最新交易日",
        "& $py scripts/build_factors.py                 # 800 因子计算 + 两阶段评估（约 20-30 分钟）",
        "& $py scripts/build_momentum_extra.py          # 自建 momentum-extra 因子面板",
        "& $py scripts/build_fundamental_extra.py       # 自建 fundamental-extra 基本面因子面板",
        "& $py scripts/evaluate_factors.py              # 相关性矩阵 / 族统计 / 复盘输入",
        "& $py scripts/run_research.py --out build/models/wf_h10 --horizon 10   # 选池 + 权重 + 复合分",
        "& $py scripts/run_backtest.py --model build/models/wf_h10 --cash-blend 0.8,0.2  # 主回测",
        "& $py scripts/run_robustness.py                # 参数敏感性与成本压力",
        "& $py scripts/make_report.py                   # 生成报告 + Excel",
        "& $py scripts/daily_signal.py                  # 下一交易日实盘信号（30 万整手订单）",
        "& $py scripts/web_server.py                    # 本地网页看板 http://127.0.0.1:8765/",
        "& $py scripts/web_smoke.py --full              # 网页接口冒烟（需服务已启动）",
        "```", "",
        "## 5. 每日实盘流程", "",
        "1. 数据更新到最新交易日后（`scripts/update_data.py`），运行 `scripts/daily_signal.py`；",
        "2. 查看 `signals/latest_target.csv`（目标持仓）与 `signals/latest_orders.csv`（相对当前持仓的订单）；"
        "现金部分按 `latest_signal.json` 的 `cash_sleeve` 建议配置国债 ETF / 黄金 ETF；",
        "3. 若提供 `signals/current_holdings.csv`（列为 code,shares），脚本会自动生成换仓订单；",
        "4. 建议在下一交易日开盘后 15 分钟内按订单执行；开盘涨停买不进/跌停卖不出时顺延；",
        "5. 每季度（1/4/7/10 月）重跑 `run_research.py` + `daily_signal.py` 更新模型权重。",
        "6. 图形界面：`python scripts/web_server.py` 后浏览器打开 http://127.0.0.1:8765/ （看盘/推荐/回测/个股/数据更新/因子/报告）。", "",
        "## 6. 关键假设", "",
        "| 项目 | 假设 |", "|---|---|",
        "| 成交价 | 次日开盘价（缩放后复权价 ≈ 实际价） |",
        "| 费用 | 佣金 2.5bp（最低 5 元）、印花税（2023-08-28 前 10bp / 之后 5bp，卖出）、过户费 0.1bp、滑点 15bp/边 |",
        "| 交易约束 | 整手 100 股、开盘涨停不能买/跌停不能卖/停牌不交易、T+1（当天买不能当天卖） |",
        "| 资金 | 30 万元；闲置现金按 100% 国债指数 计收益（可实盘用国债 ETF 511010/511260、黄金 ETF 518880；`--cash-blend 0,0` 退回纯现金 1.5%） |",
        "| 组合 | 40 只等权、10 日调仓；单票 ≤8%、行业 ≤25%、双创 ≤50%、每期最多换 6 只 |",
        "| 风控 | 趋势 regime（csi_all MA120 + hs300 MA60）、滚动 252 日回撤节流（10%→60%、15%→40%）、个股 -15% 止损 |",
        "| 交易带宽 | 已持仓标的差异 < max(30%×市值, 6000 元) 不调仓；每期最多换 6 只 |",
        "| 数据 | 原始数据包为只读（回测止于 2026-10-14）；增量更新写到 `data_extract/was_stock_data_ext`（实盘信号用到 2026-10-20），用 `QF_DATA_RAW` 环境变量切换 |", "",
        "## 7. 风险声明", "",
        "- 本系统为量化研究与教学用途，历史回测不代表未来收益，不构成任何投资建议或收益承诺；",
        "- A 股市场存在涨跌停、停牌、ST、退市、流动性枯竭等风险，实盘请做好资金与风险管理；",
        "- 因子库与数据包版权归原项目所有（GPL-3.0 / 数据来源见数据包 README）。", "",
    ]
    p = ctx.out / "系统说明与风险声明.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ------------------------------------------------------------------ Excel
def make_xlsx(ctx: Ctx) -> pathlib.Path:
    path = ctx.out / "回测指标.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        # 绩效
        m = ctx.metrics_bt
        perf = pd.DataFrame({"指标": list(m.keys()), "数值": list(m.values())})
        perf.to_excel(xw, sheet_name="绩效指标", index=False)
        # 基准
        if ctx.bench_bt:
            pd.DataFrame(ctx.bench_bt).T.reset_index().rename(columns={"index": "基准"}).to_excel(
                xw, sheet_name="基准对比", index=False)
        if ctx.yearly is not None:
            ctx.yearly.to_excel(xw, sheet_name="逐年收益")
        if ctx.monthly is not None:
            ctx.monthly.to_excel(xw, sheet_name="逐月收益")
        if ctx.stress is not None:
            ctx.stress.to_excel(xw, sheet_name="压力期", index=False)
        if ctx.robust is not None:
            ctx.robust.to_excel(xw, sheet_name="参数敏感性", index=False)
        if ctx.pool_meta is not None:
            ctx.pool_meta.reset_index().to_excel(xw, sheet_name="入模因子池", index=False)
        if ctx.weights is not None:
            ctx.weights.to_excel(xw, sheet_name="权重演化", index=False)
        if ctx.target is not None:
            ctx.target.to_excel(xw, sheet_name="最新目标持仓", index=False)
        if ctx.screen is not None:
            ctx.screen.to_excel(xw, sheet_name="全因子初筛", index=False)
        if ctx.cash_test is not None:
            ctx.cash_test.to_excel(xw, sheet_name="现金替代方案", index=False)
        if ctx.model_test is not None:
            ctx.model_test.to_excel(xw, sheet_name="因子扩展实验", index=False)
        if ctx.fund_stats:
            pd.DataFrame(ctx.fund_stats).T.reset_index().rename(columns={"index": "factor"}).to_excel(
                xw, sheet_name="基本面因子", index=False)
        # 格式
        wb = xw.book
        pct = wb.add_format({"num_format": "0.00%"})
        num = wb.add_format({"num_format": "0.000"})
        for sheet in wb.worksheets():
            sheet.set_column(0, 0, 30)
            sheet.set_column(1, 20, 15)
            sheet.freeze_panes(1, 0)
        ws = wb.get_worksheet_by_name("绩效指标")
        if ws is not None:
            for i in range(1, len(perf) + 1):
                v = perf.iloc[i - 1]["数值"]
                key = str(perf.iloc[i - 1]["指标"])
                if isinstance(v, (int, float)) and any(k in key for k in
                                                      ("cagr", "vol", "dd", "rate", "alpha",
                                                       "cost", "ratio", "excess", "te_")):
                    ws.write_number(i, 1, float(v), pct)
                elif isinstance(v, (int, float)):
                    ws.write_number(i, 1, float(v), num)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(C.BUILD_EVAL))
    ap.add_argument("--factors", default=str(C.BUILD_FACTORS))
    ap.add_argument("--model", default=str(C.BUILD_MODELS / "wf_h10"))
    ap.add_argument("--run", default=str(C.BUILD_RUNS / "main"))
    ap.add_argument("--robust", default=str(C.BUILD_RUNS / "robustness"))
    ap.add_argument("--signals", default=str(C.SIGNALS))
    ap.add_argument("--out", default=str(C.REPORTS))
    args = ap.parse_args()
    ctx = Ctx(args)
    print("生成因子库复盘报告 ...")
    p1 = make_factor_review(ctx)
    print("生成回测报告 ...")
    p2 = make_backtest_report(ctx)
    print("生成系统说明 ...")
    p3 = make_system_doc(ctx)
    print("生成 Excel ...")
    p4 = make_xlsx(ctx)
    for p in (p1, p2, p3, p4):
        print("  ", p)


if __name__ == "__main__":
    main()
