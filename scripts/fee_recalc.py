# -*- coding: utf-8 -*-
"""按你的真实费率重算成本结构。

你的费率：佣金万分之0.1（1bp），最低 5 元。

关键问题：
  30 万本金 / 40 只 = 每笔 7,500 元
  按费率应收 7500 × 0.00001 = 0.075 元
  但最低收 5 元 → 实际 5 元 = 6.67 bp

  → 名义费率 1bp，实际 6.67bp，**最低收费主导**

对比当前假设：
  佣金 2.5bp（无最低收费影响，因为按比例算）
  实际：6.67bp 单边（含最低收费）

结论：当前假设的佣金（2.5bp）**低估了**（真实 6.67bp），
      但印花税和滑点不变。
"""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(r"D:\wkf\getrich\ashare_quant\build\eval\fee_recalc.json")

CAP = 300000.0
MIN_COMM = 5.0
COMM_RATE = 0.00001        # 万0.1 = 1bp
STAMP = 0.0005             # 印花税 5bp（卖出）
TRANSFER = 0.00001         # 过户费 万0.1（沪深都收，双向）

print("=" * 88)
print("按真实费率重算（佣金万0.1，最低 5 元）")
print("=" * 88)
print(f"  本金: {CAP:,.0f} 元")
print(f"  佣金率: 万0.1 ({COMM_RATE*1e4:.1f} bp)，最低 {MIN_COMM:.0f} 元")
print(f"  印花税: {STAMP*1e4:.0f} bp（卖出）")
print(f"  过户费: {TRANSFER*1e4:.1f} bp（双向）")

print("\n" + "=" * 88)
print("不同持仓数下，单笔实际佣金率")
print("=" * 88)
print(f"{'持仓数':>6}{'单笔金额':>12}{'费率应收':>12}{'实收':>8}"
      f"{'实际bp':>10}{'vs名义':>10}")
print("-" * 88)
rows = []
for n in (10, 15, 20, 30, 40, 60):
    per = CAP / n
    by_rate = per * COMM_RATE
    actual = max(MIN_COMM, by_rate)
    bp = actual / per * 1e4
    ratio = bp / (COMM_RATE * 1e4)
    print(f"{n:>6}{per:>11,.0f}{by_rate:>11.2f}{actual:>8.2f}{bp:>9.2f}{ratio:>9.1f}x")
    rows.append({"n": n, "per_trade": per, "by_rate": by_rate,
                 "actual": actual, "bp": bp, "ratio": ratio})

# 找到"最低收费不再生效"的临界
crit_per = MIN_COMM / COMM_RATE
print(f"\n  临界单笔金额: {crit_per:,.0f} 元（即 {CAP/crit_per:.0f} 只以上时最低收费失效）")

print("\n" + "=" * 88)
print("往返成本对比（买入 + 卖出）")
print("=" * 88)
print(f"  {'项目':<16}{'当前假设':>14}{'真实(40只)':>14}{'差异':>10}")
print("-" * 88)

# 当前假设
cur = {
    "佣金": 2.5 * 2,        # 双边
    "印花税": 5.0,          # 卖出
    "过户费": 1.0 * 2,      # 假设已有
    "滑点": 15.0 * 2,
}
cur_total = sum(cur.values())

# 真实（40 只）
per40 = CAP / 40
comm40_bp = max(MIN_COMM, per40 * COMM_RATE) / per40 * 1e4
real = {
    "佣金": comm40_bp * 2,
    "印花税": 5.0,
    "过户费": TRANSFER * 1e4 * 2,
    "滑点": 15.0 * 2,
}
real_total = sum(real.values())

for k in cur:
    print(f"  {k:<16}{cur[k]:>13.1f}bp{real[k]:>13.1f}bp{real[k]-cur[k]:>+9.1f}bp")
print("-" * 88)
print(f"  {'合计':<16}{cur_total:>13.1f}bp{real_total:>13.1f}bp"
      f"{real_total-cur_total:>+9.1f}bp")

print(f"\n  → 当前假设往返 {cur_total:.0f}bp，真实（15bp滑点）{real_total:.0f}bp")
print(f"  → 差异主要来自佣金：{cur['佣金']:.1f}bp → {real['佣金']:.1f}bp"
      f"（最低收费 5 元主导）")

print("\n" + "=" * 88)
print("关键洞察：降低最低收费影响的办法")
print("=" * 88)
print("""
  最低 5 元收费 → 单笔金额越小，实际费率越高：

   40 只（7,500元/笔）→ 6.67bp  ← 现在是这个
   20 只（15,000元/笔）→ 3.33bp
   10 只（30,000元/笔）→ 1.67bp
    8 只（37,500元/笔）→ 1.33bp

  但要 37,500 元/笔 才能让万0.1 真正生效（此时费率 0.375 元 < 5 元，仍触发最低）

  实际上要单笔 ≥ 50 万才不触发最低收费 —— 你的资金量达不到。

  → **结论：在 30 万本金下，5 元最低收费是固定成本，最优解是减少持仓数。**
""")

# 年化成本对比（按换手 5.04x）
print("=" * 88)
print("年化成本影响（年换手 5.04 倍）")
print("=" * 88)
TO = 5.038
for label, rc in (("当前假设", cur_total), ("真实(40只)", real_total)):
    ann = TO * rc * 1e-4
    print(f"  {label:<14} 往返 {rc:>6.0f}bp → 年化成本 {ann:>6.2%}")

print("\n  每减少 10bp 往返成本，年化提升约 {:.2%}".format(TO * 10 * 1e-4))

json.dump({"cap": CAP, "comm_rate": COMM_RATE, "min_comm": MIN_COMM,
           "rows": rows, "current_roundtrip_bp": cur_total,
           "real_roundtrip_bp": real_total,
           "turnover": TO,
           "ann_cost_current": TO * cur_total * 1e-4,
           "ann_cost_real": TO * real_total * 1e-4},
          open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n→ {OUT}")
