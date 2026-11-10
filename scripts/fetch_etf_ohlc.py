# -*- coding: utf-8 -*-
"""抓取 37 个 ETF 的完整 OHLC 日线（供箱体突破策略用）。

腾讯接口字段：[日期, 开, 收, 高, 低, 成交量, {}, 换手率, 成交额, {}]
注意顺序是 开/收/高/低（不是常见的 OHLC）。
"""
from __future__ import annotations

import json
import random
import socket
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\wkf\getrich\ashare_quant")
OUT = ROOT / "build" / "etf"
OUT.mkdir(parents=True, exist_ok=True)

socket.setdefaulttimeout(25)
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
URL = ("https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
       "?param={sym},day,{start},{end},{n},qfq")


def sym_of(code: str) -> str:
    c = str(code).zfill(6)
    return ("sh" if c.startswith(("5", "6")) else "sz") + c


def fetch(code: str, start="2013-01-01", end="2026-12-31",
          n=4000, tries=4):
    url = URL.format(sym=sym_of(code), start=start, end=end, n=n)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=25) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            data = d.get("data") or {}
            key = sym_of(code)
            node = data.get(key) or {}
            kl = node.get("qfqday") or node.get("day") or []
            return kl
        except Exception:
            time.sleep(0.6 * (2 ** k) + random.random() * 0.4)
    return None


def main():
    # 37 个 ETF
    u = json.load(open(OUT / "sector_universe.json", encoding="utf-8"))
    codes, names = [], []
    for name, v in u["sectors"].items():
        codes.append(str(v["code"]).zfill(6))
        names.append(name)
    # 补 3 只宽基（若不在板块池里，去重）
    for c, n in (("510300", "沪深300"), ("510500", "中证500"), ("512100", "中证1000")):
        if c not in codes:
            codes.append(c)
            names.append(n)

    print(f"共 {len(codes)} 只 ETF 待抓")

    cache = OUT / "etf_ohlc_raw.json"
    store = {}
    if cache.exists():
        try:
            store = json.load(open(cache, encoding="utf-8"))
            print(f"已有缓存 {len(store)} 只")
        except Exception:
            store = {}

    n_ok = n_fail = 0
    for i, (c, nm) in enumerate(zip(codes, names), 1):
        if c in store and store[c].get("rows"):
            continue
        kl = fetch(c)
        if kl:
            store[c] = {"name": nm, "rows": kl}
            n_ok += 1
            first = kl[0][0]
            last = kl[-1][0]
            print(f"  [{i}/{len(codes)}] {c} {nm}: {len(kl)} 条 "
                  f"({first} ~ {last})", flush=True)
        else:
            n_fail += 1
            print(f"  [{i}/{len(codes)}] {c} {nm}: ✗ 失败", flush=True)
        time.sleep(0.35)

    cache.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    print(f"\n完成：{n_ok} 新增 | {n_fail} 失败 | 共 {len(store)} 只")
    print(f"→ {cache}")

    # 转成 npz 面板
    build_panel(store)


def build_panel(store: dict):
    """把 JSON 转成统一日期的面板。"""
    if not store:
        return
    all_dates = set()
    for v in store.values():
        for row in v["rows"]:
            all_dates.add(row[0])
    dates = pd.DatetimeIndex(sorted(all_dates))
    codes = sorted(store.keys())
    T, N = len(dates), len(codes)
    pos = {d.normalize(): i for i, d in enumerate(dates)}

    O = np.full((T, N), np.nan, np.float32)
    C = np.full((T, N), np.nan, np.float32)
    H = np.full((T, N), np.nan, np.float32)
    L = np.full((T, N), np.nan, np.float32)
    V = np.full((T, N), np.nan, np.float32)
    A = np.full((T, N), np.nan, np.float32)

    for j, c in enumerate(codes):
        for row in store[c]["rows"]:
            try:
                di = pos[pd.Timestamp(row[0]).normalize()]
                o = float(row[1]); cl = float(row[2])
                hi = float(row[3]); lo = float(row[4])
                v = float(row[5]) if row[5] not in (None, "") else np.nan
                a = float(row[8]) if len(row) > 8 and row[8] not in (None, "") else np.nan
            except Exception:
                continue
            O[di, j] = o
            C[di, j] = cl
            H[di, j] = hi
            L[di, j] = lo
            V[di, j] = v
            A[di, j] = a

    np.savez_compressed(OUT / "etf_ohlc_panel.npz",
                        dates=dates.values, codes=np.array(codes),
                        open=O, close=C, high=H, low=L, volume=V, amount=A)
    cov = np.isfinite(C).mean()
    print(f"[panel] {T} 日 × {N} 只，覆盖 {cov:.1%}")

    # 元信息
    meta = {c: store[c]["name"] for c in codes}
    (OUT / "etf_ohlc_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT / 'etf_ohlc_panel.npz'}")


if __name__ == "__main__":
    main()
