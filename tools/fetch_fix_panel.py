"""抓取「前复权」短窗口面板，用于修复增量段的复权价。

为什么需要
----------
`was_stock_data_ext` 的增量段（2026-10-15 起）用腾讯 hfq 拼接，`ext = row * Cj`
是常数缩放、不改变收益率，而腾讯 hfq 的日收益与真实成交价背离 —— 000635 在
11-05 被记成 +12.10%，超过主板 10% 的涨停上限，物理上不可能。

修复必须用「已处理除权」的价格重算收益。前复权的日收益率 = 后复权收益率
（两者只差一个基准缩放），且自动吸收送股/转增/分红，是最省事的口径。

产出
----
`build/eval/fixpanel_qfq.npz`：open/high/low/close 四矩阵 + dates + codes。
仅覆盖修复窗口（默认 2026-10-09 ~ 2026-11-10），很小。

用法
----
    python tools/fetch_fix_panel.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\wkf\getrich\data_extract\was_stock_data_ext")
OUT = Path(__file__).resolve().parent.parent / "build" / "eval" / "fixpanel_qfq.npz"
URL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={sid}"
       "&klt=101&fqt=1&beg={beg}&end={end}&lmt=10000"
       "&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58")
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
       "Referer": "https://quote.eastmoney.com/"}


def secid(code: str) -> str:
    return ("1." if code[0] in "69" else "0.") + code


def fetch_one(code: str, beg: str, end: str) -> tuple[str, dict]:
    try:
        req = urllib.request.Request(URL.format(sid=secid(code), beg=beg, end=end),
                                     headers=HDR)
        with urllib.request.urlopen(req, timeout=25) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
        kl = (j.get("data") or {}).get("klines") or []
        out = {}
        for line in kl:
            p = line.split(",")
            # 日期,开,收,高,低,量,额,振幅
            out[p[0]] = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
        return code, out
    except Exception:
        return code, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--beg", default=None,
                    help="起始日期；默认取面板最后一日回退 45 天（需覆盖增量段起点）")
    ap.add_argument("--end", default=None, help="结束日期；默认面板最后一日")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    z = np.load(ROOT / "panel_ohlcv.npz", allow_pickle=True)
    codes = [str(c) for c in z["codes"]]
    if args.limit:
        codes = codes[: args.limit]
    last = pd.DatetimeIndex(z["dates"])[-1]
    beg = args.beg or (last - pd.Timedelta(days=45)).strftime("%Y%m%d")
    end = args.end or last.strftime("%Y%m%d")
    print(f"窗口 {beg} ~ {end}", flush=True)

    res: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for code, rec in ex.map(lambda c: fetch_one(c, beg, end), codes):
            res[code] = rec
            done += 1
            if done % 500 == 0:
                print("  %d/%d" % (done, len(codes)), flush=True)

    # 交易日轴取所有返回日期的并集
    alld = sorted({d for rec in res.values() for d in rec})
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in alld])
    pos = {d: i for i, d in enumerate(alld)}
    T, N = len(dates), len(codes)
    O = np.full((T, N), np.nan, dtype=np.float32)
    H = np.full((T, N), np.nan, dtype=np.float32)
    L = np.full((T, N), np.nan, dtype=np.float32)
    C = np.full((T, N), np.nan, dtype=np.float32)
    for j, code in enumerate(codes):
        for d, (o, c, h, l) in (res.get(code) or {}).items():
            i = pos[d]
            O[i, j], C[i, j], H[i, j], L[i, j] = o, c, h, l

    print("交易日 %d 天（%s ~ %s），覆盖 %.1f%%"
          % (T, dates[0].date(), dates[-1].date(), np.isfinite(C).mean() * 100))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, dates=dates.values.astype("datetime64[ns]"),
                        codes=np.array(codes, dtype="<U6"),
                        open=O, high=H, low=L, close=C)
    print("已写入", OUT)


if __name__ == "__main__":
    main()
