# -*- coding: utf-8 -*-
"""归因：验证期(2020-2021) vs 样本外(2022-2026) 年化差距分解。

输出：
1) 策略/基准分区间年化与逐年收益
2) 平均仓位/风控状态分区间对比
3) 滚动 24 个月年化分布（说明 2 年期收益的抽样误差）
4) 配置扫描中"验证期选参"的乐观偏差（val vs oos 关系）
"""
import sys, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
sys.path.insert(0, str(ROOT))
from qf import config as C, data as qfdata

md = qfdata.load_market(verbose=False)
nav = pd.read_parquet(ROOT / "build/runs/main/nav.parquet")["nav"]
nav.index = pd.to_datetime(nav.index)
ex = pd.read_parquet(ROOT / "build/runs/main/exposures.parquet")
ex.index = pd.to_datetime(ex.index)
tr = pd.read_parquet(ROOT / "build/runs/main/trades.parquet")
tr["date"] = pd.to_datetime(tr["date"])

PERIODS = {"验证期 2020-2021": ("2020-01-01", "2021-12-31"),
           "样本外 2022-2026": ("2022-01-01", "2026-12-31"),
           "全期 2020-2026": ("2020-01-01", "2026-12-31")}


def cagr_of(series, start, end):
    s = series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
    if len(s) < 30:
        return np.nan
    y = (len(s) - 1) / 252.0
    return (s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1


print("=" * 78)
print("1) 分区间年化：策略 vs 基准（基准用同区间指数/全A等权累计收益换算）")
print("=" * 78)
rows = []
for name, (s, e) in PERIODS.items():
    r = {"period": name, "strategy": cagr_of(nav, s, e)}
    for b in ["all_eq", "hs300", "csi500", "csi1000", "csi_all"]:
        bench_nav = (1 + md.bench[b]).cumprod()
        bench_nav.index = md.dates
        r[b] = cagr_of(bench_nav, s, e)
    rows.append(r)
df = pd.DataFrame(rows).set_index("period")
print((df * 100).round(2).to_string())

print()
print("=" * 78)
print("2) 逐年收益（策略 vs 全A等权 vs 中证1000，%）")
print("=" * 78)
yr_rows = {}
for y in range(2020, 2027):
    s, e = f"{y}-01-01", f"{y}-12-31"
    r = {"year": y, "strategy": cagr_of(nav, s, e)}
    for b in ["all_eq", "csi1000", "csi_all"]:
        bn = (1 + md.bench[b]).cumprod(); bn.index = md.dates
        r[b] = cagr_of(bn, s, e)
    yr_rows[y] = r
print((pd.DataFrame(yr_rows).T * 100).round(2).to_string())

print()
print("=" * 78)
print("3) 仓位与风控状态分区间")
print("=" * 78)
for name, (s, e) in PERIODS.items():
    sub = ex[(ex.index >= pd.Timestamp(s)) & (ex.index <= pd.Timestamp(e))]
    cap = sub["exposure_cap"] if "exposure_cap" in sub.columns else sub.get("dd_cap")
    print(f"{name}: 平均股票仓位 {sub['stock'].mean():.1%}, 平均目标仓位 "
          f"{sub['target_exposure'].mean():.1%}, 平均趋势系数 {sub['regime_scale'].mean():.2f}, "
          f"暴露上限100%占比 {(cap >= 0.999).mean():.0%}, 60%占比 {(cap < 0.999).mean():.0%}, "
          f"平均持仓 {sub['n_positions'].mean():.1f}")

print()
print("=" * 78)
print("4) 滚动 24 个月年化分布（策略净值，说明 2 年期平均的抽样波动）")
print("=" * 78)
monthly = nav.resample("ME").last()
roll2 = (monthly / monthly.shift(24)) ** (1 / 2) - 1
roll2 = roll2.dropna()
print(f"  样本 {len(roll2)} 个滚动 2 年期：min {roll2.min():.2%} / p10 {roll2.quantile(.1):.2%} / "
      f"p25 {roll2.quantile(.25):.2%} / 中位 {roll2.median():.2%} / p75 {roll2.quantile(.75):.2%} / "
      f"max {roll2.max():.2%}")
print(f"  验证期(2020-2021)年化 {cagr_of(nav,'2020-01-01','2021-12-31'):.2%} 在该分布中的分位："
      f"{(roll2 < cagr_of(nav,'2020-01-01','2021-12-31')).mean():.0%}")

print()
print("=" * 78)
print("5) 配置扫描的「验证期选参」乐观偏差（48 个配置）")
print("=" * 78)
cs = pd.read_csv(ROOT / "build/runs/config_scan_h10.csv")
cs = cs.dropna(subset=["val_cagr", "oos_cagr", "val_calmar", "oos_calmar"])
print(f"  全部 {len(cs)} 个配置：验证期 CAGR 范围 {cs['val_cagr'].min():.2%}~{cs['val_cagr'].max():.2%}")
print(f"  验证期 CAGR 与样本外 CAGR 的秩相关："
      f"{cs['val_cagr'].corr(cs['oos_cagr'], method='spearman'):.2f}"
      f"（验证期 Calmar 与样本外 Calmar 秩相关："
      f"{cs['val_calmar'].corr(cs['oos_calmar'], method='spearman'):.2f}）")
top_cagr = cs.sort_values("val_cagr", ascending=False)
print(f"  若按验证期 CAGR 选前 5：验证期均值 {top_cagr.head(5)['val_cagr'].mean():.2%}，"
      f"样本外均值 {top_cagr.head(5)['oos_cagr'].mean():.2%}")
# 实际选型口径：只考虑开启风控 overlay 的配置（用户要求低回撤，off 组先排除），按验证期 Calmar 选
ctrl = cs[cs["overlay"] != "off"]
top_cal = ctrl.sort_values("val_calmar", ascending=False)
print(f"  风控开启（strong/mild）的配置共 {len(ctrl)} 个："
      f"验证期/样本外 CAGR 均值 {ctrl['val_cagr'].mean():.2%} / {ctrl['oos_cagr'].mean():.2%}，"
      f"样本外 CAGR 中位 {ctrl['oos_cagr'].median():.2%}")
print(f"  按验证期 Calmar 选前 5（风控开启组）：验证期 CAGR 均值 {top_cal.head(5)['val_cagr'].mean():.2%}，"
      f"样本外 CAGR 均值 {top_cal.head(5)['oos_cagr'].mean():.2%}，"
      f"样本外 Calmar 均值 {top_cal.head(5)['oos_calmar'].mean():.2f}")
adopted = cs[cs["tag"] == "n40_rb10_strong_csi_all"].iloc[0]
print(f"  最终采用配置 n40_rb10_strong_csi_all：验证期 CAGR {adopted['val_cagr']:.2%}"
      f"（Calmar {adopted['val_calmar']:.2f}），样本外 CAGR {adopted['oos_cagr']:.2%}"
      f"（Calmar {adopted['oos_calmar']:.2f}）")
print(f"  其验证期 Calmar 在风控组中排名 {int((ctrl['val_calmar'] > adopted['val_calmar']).sum()) + 1}/{len(ctrl)}，"
      f"样本外 CAGR 在风控组中排名 {int((ctrl['oos_cagr'] > adopted['oos_cagr']).sum()) + 1}/{len(ctrl)}")

print()
print("=" * 78)
print("6) 交易成本与换手分区间")
print("=" * 78)
for name, (s, e) in PERIODS.items():
    sub = tr[(tr["date"] >= pd.Timestamp(s)) & (tr["date"] <= pd.Timestamp(e))]
    sub_nav = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
    yrs = max(len(sub_nav) - 1, 1) / 252
    cost = (sub["commission"] + sub["stamp"] + sub["transfer"] + sub["slip"]).sum()
    print(f"{name}: 成交 {len(sub)} 笔，成本合计 {cost:,.0f} 元，"
          f"年化成本/平均净值 {cost / sub_nav.mean() / yrs:.2%}")
