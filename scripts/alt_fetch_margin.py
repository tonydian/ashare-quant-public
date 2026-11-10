# -*- coding: utf-8 -*-
"""抓取全市场历史融资融券数据（东财，按日期迭代）。

策略：按交易日逐日拉全市场（每页 500，单日约 2000~4000 只 → 4~8 页）
关键字段：
  RZYE        融资余额
  RZMRE       融资买入额
  RZCHE       融资偿还额
  RZJME       融资净买入（= RZMRE - RZCHE）
  RQYE        融券余额
  RQYL        融券余量
  RZRQYE      融资融券余额
  SPJ         收盘价
  ZDF         涨跌幅
  以及 RZJME3D/5D/10D（多日累计净买入，东财已算好）
"""
from __future__ import annotations

import json
import os
import random
import socket
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
OUT = ROOT / "build" / "alt"
OUT.mkdir(parents=True, exist_ok=True)

socket.setdefaulttimeout(20)
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/rzrq/",
}
BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT = "RPTA_WEB_RZRQ_GGMX"
COLS = ("DATE,SCODE,SECNAME,RZYE,RZMRE,RZCHE,RZJME,RQYE,RQYL,RQYE,"
        "RZRQYE,SPJ,ZDF,FIN_BALANCE_GR")

FIELDS = ["DATE", "SCODE", "RZYE", "RZMRE", "RZCHE", "RZJME",
          "RQYE", "RQYL", "RZRQYE", "SPJ", "ZDF"]


def fetch_page(date: str, page: int, size: int = 500, tries: int = 4):
    p = {"reportName": REPORT, "columns": "ALL",
         "filter": f"(DATE='{date}')", "pageNumber": str(page),
         "pageSize": str(size), "sortTypes": "1", "sortColumns": "SCODE",
         "source": "WEB", "client": "WEB"}
    qs = urllib.parse.urlencode(p)
    for k in range(tries):
        try:
            req = urllib.request.Request(f"{BASE}?{qs}", headers=HDR)
            with urllib.request.urlopen(req, timeout=25) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            res = d.get("result")
            if not res:
                return 0, []
            return int(res.get("pages") or 1), res.get("data") or []
        except Exception:
            time.sleep(0.5 * (2 ** k) + random.random() * 0.3)
    return -1, []


def fetch_day(date: str):
    """拉某一天全市场。返回 (date, rows) 或 (date, None)。"""
    pages, first = fetch_page(date, 1)
    if pages <= 0:
        return date, None
    rows = list(first)
    for pg in range(2, pages + 1):
        _, more = fetch_page(date, pg)
        if more:
            rows.extend(more)
    return date, rows


def main():
    import os
    os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
    sys.path.insert(0, str(ROOT))
    from qf import data as qfdata

    md = qfdata.load_market(verbose=False)
    dates = [str(d.date()) for d in md.dates]
    print(f"交易日 {len(dates)} 天：{dates[0]} ~ {dates[-1]}")

    cache = OUT / "margin_raw.jsonl"
    done = set()
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["date"])
                except Exception:
                    pass
        print(f"已有缓存 {len(done)} 天，续抓")

    todo = [d for d in dates if d not in done]
    print(f"待抓 {len(todo)} 天")

    t0 = time.time()
    n_ok = n_fail = 0
    ln = open(cache, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(fetch_day, d): d for d in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                d, rows = fut.result()
                if rows is None:
                    n_fail += 1
                else:
                    n_ok += 1
                    slim = []
                    for r in rows:
                        slim.append({k: r.get(k) for k in FIELDS})
                    ln.write(json.dumps({"date": d, "rows": slim},
                                        ensure_ascii=False) + "\n")
                if i % 100 == 0:
                    ln.flush()
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)} | ok {n_ok} fail {n_fail} | "
                          f"{el:.0f}s，剩余 {el/i*(len(todo)-i)/60:.1f} 分钟",
                          flush=True)
    finally:
        ln.close()

    print(f"\n完成：ok {n_ok} | fail {n_fail}")
    print(f"→ {cache}")


if __name__ == "__main__":
    main()
