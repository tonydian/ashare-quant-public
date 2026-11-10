"""每日采集涨停池 / 炸板池快照（前向数据积累）。

为什么必须采集
--------------
东财涨停池带 `fbt`(首封时间)、`fund`(封单额)、`zbc`(炸板次数)、`lbc`(连板数)、
`hybk`(行业板块)，这些是打板决策的核心输入，**日线无法还原**。
接口只能回溯约 1 个月，所以只能每天存一份，攒出前向样本。

产出
----
`build/limitup/YYYY-MM-DD.parquet`  涨停池
`build/limitup/KB_YYYY-MM-DD.parquet` 炸板池

用法
----
    python tools/collect_limitup.py                # 采最近 1 个交易日
    python tools/collect_limitup.py --backfill 25  # 回补最近 25 个交易日
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "build" / "limitup"
BASE = "https://push2ex.eastmoney.com"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
       "Referer": "https://quote.eastmoney.com/"}
UT = "7eea3edcaed734bea9cbfc24409ed989"
COLS = ["c", "n", "p", "zdp", "amount", "ltsz", "tshare", "hs", "lbc",
        "fbt", "lbt", "fund", "zbc", "hybk", "zttj"]


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def fetch(pool: str, date: str) -> pd.DataFrame:
    api = "getTopicZTPool" if pool == "ZT" else "getTopicKBPool"
    rows, page = [], 0
    while True:
        url = (f"{BASE}/{api}?ut={UT}&dpt=wz.ztzt&Pageindex={page}"
               f"&pagesize=200&sort=fbt%3Aasc&date={date}")
        j = _get(url)
        d = j.get("data") or {}
        got = d.get("pool") or []
        rows += got
        if len(rows) >= (d.get("tc") or 0) or not got:
            break
        page += 1
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df = df[COLS].copy()
    df["zttj_days"] = df["zttj"].map(lambda x: (x or {}).get("days") if isinstance(x, dict) else None)
    df["zttj_ct"] = df["zttj"].map(lambda x: (x or {}).get("ct") if isinstance(x, dict) else None)
    return df.drop(columns=["zttj"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=1, help="回补最近 N 个交易日")
    ap.add_argument("--date", default="", help="指定日期 YYYYMMDD")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.date:
        days = [args.date]
    else:
        # 用现有面板的交易日历（往前取 N 个）
        import numpy as np
        z = np.load(Path(r"D:\wkf\getrich\data_extract\was_stock_data_ext") / "panel_ohlcv.npz",
                    allow_pickle=True)
        ds = pd.DatetimeIndex(z["dates"])[-args.backfill:]
        days = [d.strftime("%Y%m%d") for d in ds]

    for d in days:
        n_zt = n_kb = -1
        for pool, prefix in (("ZT", ""), ("KB", "KB_")):
            fp = OUT / f"{prefix}{d[:4]}-{d[4:6]}-{d[6:]}.parquet"
            if fp.exists():
                n_zt = len(pd.read_parquet(fp))
                continue
            try:
                df = fetch(pool, d)
                if len(df):
                    df.to_parquet(fp, index=False)
                if pool == "ZT":
                    n_zt = len(df)
                else:
                    n_kb = len(df)
            except Exception as e:
                print("  %s %s 失败 %s" % (d, pool, str(e)[:60]))
            time.sleep(0.25)
        print("%s  涨停 %d  炸板 %d" % (d, n_zt, n_kb))


if __name__ == "__main__":
    main()
