# -*- coding: utf-8 -*-
"""抓取 ETF OHLC —— 分段版（腾讯接口单次最多 ~640 条）。

策略：按年分段请求，合并去重。
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
       "?param={sym},day,{start},{end},640,qfq")


def sym_of(code: str) -> str:
    c = str(code).zfill(6)
    return ("sh" if c.startswith(("5", "6")) else "sz") + c


def fetch_seg(code: str, start: str, end: str, tries: int = 3):
    url = URL.format(sym=sym_of(code), start=start, end=end)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=25) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            data = d.get("data") or {}
            node = data.get(sym_of(code)) or {}
            return node.get("qfqday") or node.get("day") or []
        except Exception:
            time.sleep(0.5 * (2 ** k) + random.random() * 0.3)
    return []


def fetch_all(code: str):
    """按年分段抓，合并去重。"""
    rows = {}
    for y in range(2013, 2027):
        seg = fetch_seg(code, f"{y}-01-01", f"{y}-12-31")
        for row in seg:
            rows[row[0]] = row
        time.sleep(0.18)
        if len(rows) and y > 2020:
            # 近期已覆盖，提前结束
            pass
    return [rows[d] for d in sorted(rows)]


def main():
    u = json.load(open(OUT / "sector_universe.json", encoding="utf-8"))
    codes, names = [], []
    for name, v in u["sectors"].items():
        codes.append(str(v["code"]).zfill(6))
        names.append(name)
    for c, n in (("510300", "沪深300"), ("510500", "中证500"), ("512100", "中证1000")):
        if c not in codes:
            codes.append(c)
            names.append(n)

    print(f"共 {len(codes)} 只 ETF（按年分段抓取）")

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
        if c in store and len(store[c].get("rows") or []) > 500:
            continue
        kl = fetch_all(c)
        if kl:
            store[c] = {"name": nm, "rows": kl}
            n_ok += 1
            print(f"  [{i}/{len(codes)}] {c} {nm}: {len(kl)} 条 "
                  f"({kl[0][0]} ~ {kl[-1][0]})", flush=True)
        else:
            n_fail += 1
            print(f"  [{i}/{len(codes)}] {c} {nm}: ✗", flush=True)
        # 增量保存
        if i % 5 == 0:
            cache.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")

    cache.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    print(f"\n完成：{n_ok} 新增 | {n_fail} 失败 | 共 {len(store)} 只")
    build_panel(store)


def build_panel(store: dict):
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
                O[di, j] = float(row[1])
                C[di, j] = float(row[2])
                H[di, j] = float(row[3])
                L[di, j] = float(row[4])
                if row[5] not in (None, ""):
                    V[di, j] = float(row[5])
                if len(row) > 8 and row[8] not in (None, ""):
                    A[di, j] = float(row[8])
            except Exception:
                continue

    np.savez_compressed(OUT / "etf_ohlc_panel.npz",
                        dates=dates.values, codes=np.array(codes),
                        open=O, close=C, high=H, low=L, volume=V, amount=A)
    meta = {c: store[c]["name"] for c in codes}
    (OUT / "etf_ohlc_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    cov = np.isfinite(C).mean()
    print(f"\n[panel] {T} 日 × {N} 只，覆盖 {cov:.1%}")
    print(f"  区间 {dates[0].date()} ~ {dates[-1].date()}")
    # 每只的有效天数
    ok = np.isfinite(C).sum(axis=0)
    print(f"  有效天数: 中位 {int(np.median(ok))}，最少 {int(ok.min())}，最多 {int(ok.max())}")
    print(f"→ {OUT / 'etf_ohlc_panel.npz'}")


if __name__ == "__main__":
    main()
