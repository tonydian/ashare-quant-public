# -*- coding: utf-8 -*-
"""用 panda_data SDK 拉资产负债表截面（全市场）。

为什么用 SDK 而不是 MCP：MCP 单次只回约 3.5 万字节（≈268 行 × 4 字段），
全市场单截面要 20 批、30 个截面 600 次调用，建不了库。见 docs/PANDADATA.md §7.5。

为什么用「截面」而不是日频：get_factor 的财报字段是 PIT 对齐的（值跳变日=披露日），
所以只需在每个报告期披露完毕后取一次快照，就能还原该期的资产负债表。
日频 1200 万行没必要，且会把内存打满。

注意 Q4/年报：宽表在同日/相邻披露多期时只保留最新一期，所以「次年 5 月」取到的
其实是次年 Q1 而非上年年报。因此默认只取 Q1/Q2/Q3 三期（年报用 Q1 近似，差 3 个月）。

用法：
    # 默认：2017-2026 每年 3 期，26 个资产负债表科目
    python scripts/fetch_pandadata_bs.py

    # 只取指定截面
    python scripts/fetch_pandadata_bs.py --dates 20250506,20250901 --fields total_assets,goodwill

    # 全字段 + 指定年份
    python scripts/fetch_pandadata_bs.py --years 2020-2026
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# 资产负债表科目：我们本地 fund/*.parquet 完全没有这些
DEFAULT_FIELDS = [
    "total_assets",
    "total_liabilities",
    "total_equity",
    "current_assets",
    "current_liabilities",
    "non_current_assets",
    "non_current_liabilities",
    "inventory",
    "goodwill",
    "intangible_assets",
    "capital_reserve",
    "surplus_reserve",
    "undistributed_profit",
    "paid_in_capital",
    "minority_interest",
    "short_term_loans",
    "long_term_loans",
    "bond_payable",
    "notes_payable",
    "advance_from_customers",
    "contract_liabilities",
    "deferred_income",
    "construction_in_progress",
    "other_current_assets",
    "other_non_current_assets",
    "oil_and_gas_assets",
]

# 截面日期：报告期披露完毕后第 3 个交易日附近（留出缓冲，避开 4/30 当天的尾巴）
#   Q1 (0331) -> 5/6      Q2 (0630) -> 9/2      Q3 (0930) -> 11/4
SNAPSHOT_MMDD = {1: "0506", 2: "0902", 3: "1104"}


def build_dates(years):
    """生成 (截面日, 对应报告期季度) 列表。"""
    out = []
    for y in years:
        for q, mmdd in SNAPSHOT_MMDD.items():
            out.append((f"{y}{mmdd}", q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2017-2026", help="年份区间，如 2017-2026")
    ap.add_argument("--dates", default="", help="直接指定截面日，逗号分隔，覆盖 --years")
    ap.add_argument("--fields", default="", help="逗号分隔；空则用默认 26 个资产负债表科目")
    ap.add_argument("--out", default="build/pandadata/bs_snapshot.parquet")
    ap.add_argument("--sleep", type=float, default=0.3, help="每次请求间隔秒数")
    args = ap.parse_args()

    import panda_data

    if not panda_data.is_authenticated():
        print("!! SDK 未认证。先执行：")
        print("   python -c \"import panda_data; panda_data.init_token('账号','密码')\"")
        sys.exit(1)
    panda_data.init()
    print("认证状态：", panda_data.auth_info())

    fields = [f.strip() for f in args.fields.split(",") if f.strip()] or DEFAULT_FIELDS
    print(f"字段 {len(fields)} 个：{', '.join(fields[:6])}{' ...' if len(fields) > 6 else ''}")

    if args.dates:
        dates = [(d.strip(), None) for d in args.dates.split(",") if d.strip()]
    else:
        lo, _, hi = args.years.partition("-")
        years = range(int(lo), int(hi or lo) + 1)
        dates = build_dates(years)
    print(f"截面 {len(dates)} 个：{dates[0][0]} ~ {dates[-1][0]}")

    frames, failed = [], []
    for i, (d, q) in enumerate(dates, 1):
        t0 = time.time()
        try:
            df = panda_data.get_factor(
                symbol="", start_date=d, end_date=d, factors=fields, type="stock"
            )
        except Exception as e:
            print(f"[{i}/{len(dates)}] {d}  FAIL  {type(e).__name__}: {e}")
            failed.append((d, str(e)))
            continue
        if df is None or len(df) == 0:
            print(f"[{i}/{len(dates)}] {d}  空")
            failed.append((d, "empty"))
            continue
        df["period_q"] = q
        frames.append(df)
        got = [c for c in fields if c in df.columns]
        miss = [c for c in fields if c not in df.columns]
        print(f"[{i}/{len(dates)}] {d}  {len(df):5d} 行 × {len(got)} 字段  "
              f"{time.time() - t0:.1f}s" + (f"  缺字段 {miss[:4]}" if miss else ""))
        time.sleep(args.sleep)

    if not frames:
        print("!! 一条都没拿到")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out, index=False)
    print(f"\n已写 {out}")
    print(f"  {len(all_df):,} 行 × {all_df.shape[1]} 列  {out.stat().st_size / 1e6:.2f} MB")
    print(f"  截面 {all_df['date'].nunique()} 个，股票 {all_df['symbol'].nunique()} 只")
    if failed:
        print(f"\n失败 {len(failed)} 个：{failed[:5]}")

    # 快速质量检查：资产负债表恒等式 total_assets ≈ total_liabilities + total_equity
    need = {"total_assets", "total_liabilities", "total_equity"}
    if need.issubset(all_df.columns):
        sub = all_df.dropna(subset=list(need))
        rb = (sub["total_liabilities"] + sub["total_equity"]) / sub["total_assets"]
        ok = ((rb - 1).abs() < 0.01).mean()
        print(f"\n恒等式校验（L+E == A，误差<1%）：{ok:.2%} 通过，样本 {len(sub):,}")
        if ok < 0.9:
            print("  !! 通过率偏低，检查是否混入银行/保险（口径不同）或字段单位不一致")


if __name__ == "__main__":
    main()
