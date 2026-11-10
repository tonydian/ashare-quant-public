# -*- coding: utf-8 -*-
"""按次方量化文章标准，检查我们 ETF 池的"假分散"问题。

文章标准（§3、§4）：
  相关系数 ≥ 0.85  → 🔴 高度相关，几乎重复（触发告警）
  0.60 ~ 0.85      → 🟠 较强同向
  0.30 ~ 0.60      → 🟡 中等同向
  0 ~ 0.30         → 🔵 弱相关（理想）
  负相关           → 🟦 可对冲

  计算口径（§4）：
    · 最近 1 年日线
    · 每日涨跌幅
    · Pearson 相关系数
    · 有效交易日 ≥ 20 天

搭池原则（§5）：
  · 允许有"强相关小分队"，但不能"全是强相关"
  · 看整体结构，不盯单个格子
  · 跨大类资产搭配最好
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
OUT = ROOT / "build" / "eval" / "etf_pool_corr.json"

z = np.load(ROOT / "build" / "etf" / "etf_ohlc_panel.npz", allow_pickle=True)
dates = pd.DatetimeIndex(z["dates"])
codes = [str(c) for c in z["codes"]]
meta = json.load(open(ROOT / "build" / "etf" / "etf_ohlc_meta.json", encoding="utf-8"))
close = z["close"].astype(np.float64)

# 最近 1 年
cut = dates[-1] - pd.Timedelta(days=365)
m = dates >= cut
sub = close[m]
print("=" * 96)
print(f"ETF 池相关性检查（{dates[m][0].date()} ~ {dates[-1].date()}，"
      f"{int(m.sum())} 个交易日）")
print("=" * 96)

# 有效标的（≥20 天有效且非全 NaN）
valid = []
for j, c in enumerate(codes):
    s = sub[:, j]
    if np.isfinite(s).sum() >= 20:
        valid.append(j)
print(f"  有效标的: {len(valid)} / {len(codes)} 只（有效交易日 ≥ 20）")

# 日收益
_df = pd.DataFrame(sub, index=dates[m], columns=codes)
rets = _df.pct_change(fill_method=None)
corr = rets.corr(min_periods=20)

# ============================================================ 高相关告警
print("\n" + "=" * 96)
print("🔴 高相关告警（≥ 0.85，文章标准）")
print("=" * 96)

pairs = []
cols = list(corr.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        c = corr.iloc[i, j]
        if np.isfinite(c) and c >= 0.85:
            pairs.append((cols[i], cols[j], c))

pairs.sort(key=lambda x: -x[2])
if pairs:
    print(f"\n  共 {len(pairs)} 组高相关（≥0.85）：")
    print(f"  {'标的A':<10}{'名称A':<10}{'标的B':<10}{'名称B':<10}{'相关':>8}")
    print("  " + "-" * 56)
    for a, b, c in pairs:
        print(f"  {a:<10}{meta.get(a,'')[:8]:<10}{b:<10}{meta.get(b,'')[:8]:<10}{c:>8.3f}")
else:
    print("  ✅ 无高相关组合")

# ============================================================ 分布
print("\n" + "=" * 96)
print("相关系数分布（文章分级）")
print("=" * 96)

def bucket(c):
    if not np.isfinite(c):
        return "无数据"
    if c >= 0.85:
        return "🔴 ≥0.85 高度同向"
    if c >= 0.60:
        return "🟠 0.60~0.85 较强同向"
    if c >= 0.30:
        return "🟡 0.30~0.60 中等同向"
    if c >= 0:
        return "🔵 0~0.30 弱相关"
    return "🟦 负相关"

vals = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        c = corr.iloc[i, j]
        if np.isfinite(c):
            vals.append((bucket(c), c))

dist = pd.Series([v[0] for v in vals]).value_counts()
print(f"\n  {'等级':<26}{'组数':>8}{'占比':>10}")
print("  " + "-" * 44)
for k in ("🔴 ≥0.85 高度同向", "🟠 0.60~0.85 较强同向",
          "🟡 0.30~0.60 中等同向", "🔵 0~0.30 弱相关", "🟦 负相关"):
    n = int(dist.get(k, 0))
    if n == 0:
        continue
    print(f"  {k:<26}{n:>8}{n/len(vals):>9.1%}")

allv = np.array([v[1] for v in vals])
print(f"\n  平均相关: {allv.mean():.3f}")
print(f"  中位数:   {np.median(allv):.3f}")
print(f"  最小: {allv.min():.3f}  最大: {allv.max():.3f}")

# ============================================================ 关键：资产类别覆盖
print("\n" + "=" * 96)
print("资产类别覆盖（文章建议：跨大类搭配）")
print("=" * 96)

CATS = {
    "A股宽基": ["沪深300", "中证500", "中证1000", "创业板", "科创50"],
    "A股行业": ["电力", "旅游", "农业", "环保", "化工", "消费", "军工", "金融",
                "医药", "地产", "有色", "酒", "银行", "证券", "传媒", "科技",
                "新能源车", "钢铁", "煤炭", "新能源", "通信", "软件", "基建",
                "机器人", "半导体", "AI"],
    "港股": ["港股", "恒生科技"],
    "海外": ["纳指", "标普", "日经"],
    "红利/低波": ["红利"],
}
used = set()
print()
for cat, names in CATS.items():
    hit = [c for c in codes if meta.get(c, "") in names]
    print(f"  {cat:<10} {len(hit):>2} 只: {[meta.get(c,'') for c in hit]}")
    used |= set(meta.get(c, "") for c in hit)
uncovered = [meta.get(c, "") for c in codes if meta.get(c, "") not in used]
if uncovered:
    print(f"  未分类:   {uncovered}")

print(f"\n  覆盖类别: {len(CATS)} 类")
print(f"  ⚠️ 缺少：债券 ETF、黄金 ETF（文章强调的避险资产）")

# ============================================================ 子集检查
print("\n" + "=" * 96)
print("重点子集检查")
print("=" * 96)

def sub_corr(names, label):
    js = [j for j, c in enumerate(codes) if meta.get(c, "") in names]
    if len(js) < 2:
        print(f"  {label}: 标的不足")
        return None
    ss = sub[:, js]
    rr = pd.DataFrame(ss, index=dates[m], columns=[codes[j] for j in js])
    rr = rr.pct_change(fill_method=None)
    cc = rr.corr(min_periods=20)
    v = [cc.iloc[i, j] for i in range(len(js)) for j in range(i+1, len(js))
         if np.isfinite(cc.iloc[i, j])]
    if not v:
        return None
    print(f"\n  【{label}】{len(js)} 只")
    print(f"    平均相关 {np.mean(v):.3f}，"
          f"范围 {min(v):.3f} ~ {max(v):.3f}")
    if np.mean(v) > 0.85:
        print("    🔴 严重同质化（文章说的假分散）")
    elif np.mean(v) > 0.7:
        print("    🟠 高度同质，可作为进攻小分队但不宜全池都是")
    else:
        print(f"    ✅ 分散度可接受")
    return float(np.mean(v))

sub_corr(["沪深300", "中证500", "中证1000"], "A股宽基三兄弟")
sub_corr(["纳指", "标普", "日经"], "海外三兄弟")
sub_corr(["科技", "半导体", "AI", "软件", "通信"], "科技小分队（文章的反面案例）")
sub_corr(["半导体", "AI"], "半导体+AI（文章点名的高危组合）")
sub_corr(["沪深300", "纳指"], "A股 vs 美股")
sub_corr(["沪深300", "煤炭"], "沪深300 vs 煤炭（周期）")

# ============================================================ 我们实际用的池
print("\n" + "=" * 96)
print("我们实际策略用到的池")
print("=" * 96)
tr = pd.read_csv(ROOT / "build" / "runs" / "etf_trend" / "trades_donchian55.csv")
used_codes = sorted(set(tr[tr["date"] >= "2020-01-08"]["code"].astype(str)))
print(f"  ETF 趋势策略用: {[(c, meta.get(c,'')) for c in used_codes]}")

js = [j for j, c in enumerate(codes) if c in used_codes]
if len(js) >= 2:
    ss = sub[:, js]
    rr = pd.DataFrame(ss, index=dates[m], columns=[codes[j] for j in js])
    rr = rr.pct_change(fill_method=None)
    cc = rr.corr(min_periods=20)
    print(f"\n  相关性矩阵:")
    print(cc.round(3).to_string())
    v = [cc.iloc[i, j] for i in range(len(js)) for j in range(i+1, len(js))]
    print(f"\n  平均相关: {np.mean(v):.3f}")
    if np.mean(v) > 0.85:
        print("  🔴 三个宽基高度同质 -> 所谓 top2 轮动实际等于只持有 1 只")
    elif np.mean(v) > 0.7:
        print(f"  🟠 相关偏高，轮动空间有限")
    else:
        print(f"  ✅ 有轮动空间")

json.dump({"n_valid": len(valid), "n_pairs_high": len(pairs),
           "pairs": [{"a": a, "b": b, "corr": float(c)} for a, b, c in pairs],
           "mean_corr": float(allv.mean()), "median": float(np.median(allv)),
           "distribution": dist.to_dict()},
          open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print(f"\n→ {OUT}")
