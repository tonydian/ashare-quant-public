"""重建「不复权」全历史价格面板。

为什么需要
----------
数据包里的 `raw_close.npz` 并不是真的不复权价（茅台 2026-11-09 记为 1441.30，
实际收盘 1444.99）。涨停价必须按交易所的「前收盘 × (1+涨跌幅) 四舍五入到分」算，
用错了价格，涨停判定会大面积漏判。

产出
----
`data_extract/was_stock_data_ext/panel_raw_ohlc.npz`
    open / high / low / close 四个矩阵，对齐现有 panel_ohlcv 的 dates 与 codes。

用法
----
    python tools/fetch_raw_panel.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\wkf\getrich\data_extract\was_stock_data_ext")
URL = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={sid}"
       "&klt=101&fqt=0&lmt=10000&end=20500101"
       "&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58")
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
       "Referer": "https://quote.eastmoney.com/"}


def secid(code: str) -> str:
    return ("1." if code[0] in "69" else "0.") + code


def fetch_one(code: str) -> tuple[str, dict]:
    try:
        req = urllib.request.Request(URL.format(sid=secid(code)), headers=HDR)
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
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    z = np.load(ROOT / "panel_ohlcv.npz", allow_pickle=True)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    if args.limit:
        codes = codes[: args.limit]

    dstr = [d.strftime("%Y-%m-%d") for d in dates]
    pos = {d: i for i, d in enumerate(dstr)}
    T, N = len(dates), len(codes)
    O = np.full((T, N), np.nan, dtype=np.float32)
    H = np.full((T, N), np.nan, dtype=np.float32)
    L = np.full((T, N), np.nan, dtype=np.float32)
    C = np.full((T, N), np.nan, dtype=np.float32)

    done = 0
    cidx = {c: i for i, c in enumerate(codes)}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for code, rec in ex.map(fetch_one, codes):
            j = cidx[code]
            for d, (o, c, h, l) in rec.items():
                i = pos.get(d)
                if i is None:
                    continue
                O[i, j], C[i, j], H[i, j], L[i, j] = o, c, h, l
            done += 1
            if done % 500 == 0:
                print("  %d/%d" % (done, len(codes)), flush=True)

    cov = np.isfinite(C).sum(axis=1)
    print("覆盖：最新交易日在市 %d 只（面板 %d）" % (int(cov[-1]), int(np.isfinite(z['close'][-1]).sum())))
    print("整体非空占比 %.1f%%" % (np.isfinite(C).mean() * 100))

    if args.limit:
        print("--limit 模式，不落盘")
        return
    np.savez_compressed(ROOT / "panel_raw_ohlc.npz", open=O, high=H, low=L, close=C)
    print("已写入", ROOT / "panel_raw_ohlc.npz")


if __name__ == "__main__":
    main()
