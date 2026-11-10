# -*- coding: utf-8 -*-
"""抓取龙虎榜数据（按日迭代）。

两个层次：
  1) 龙虎榜明细（RPT_DAILYBILLBOARD_DETAILSNEW）：每只上榜股票一行
     字段：BILLBOARD_DEAL_AMT 上榜成交额、DEAL_AMOUNT_RATIO 占比、
           EXPLAIN 上榜原因、CHANGE_RATE 涨跌幅、TURNOVERRATE 换手
  2) 席位明细（RPT_BILLBOARD_DAILYDETAILSBUY）：每个营业部一行
     字段：OPERATEDEPT_NAME 席位名、BUY/SELL/NET 买入卖出净额
     用途：识别"机构专用"席位 vs 游资

先抓明细（快），席位明细量大后抓。
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

ROOT = Path(r"D:\wkf\getrich\ashare_quant")
REPO = ROOT.parent
OUT = ROOT / "build" / "alt"
OUT.mkdir(parents=True, exist_ok=True)

socket.setdefaulttimeout(20)
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}
BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"

KEEP_DETAIL = ["TRADE_DATE", "SECURITY_CODE", "SECURITY_NAME_ABBR", "EXPLAIN",
               "CLOSE_PRICE", "CHANGE_RATE", "TURNOVERRATE",
               "BILLBOARD_DEAL_AMT", "DEAL_AMOUNT_RATIO",
               "BILLBOARD_NET_AMT", "BILLBOARD_BUY_AMT", "BILLBOARD_SELL_AMT",
               "FREE_MARKET_CAP", "TRADE_AMOUNT"]
KEEP_SEAT = ["TRADE_DATE", "SECURITY_CODE", "OPERATEDEPT_NAME",
             "BUY", "SELL", "NET", "ACCUM_AMOUNT", "ACCUM_VOLUME"]


def fetch(report, date, page=1, size=500, tries=4):
    p = {"reportName": report, "columns": "ALL",
         "filter": f"(TRADE_DATE='{date}')", "pageNumber": str(page),
         "pageSize": str(size), "source": "WEB", "client": "WEB"}
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


def fetch_day(date, report, keys):
    pages, first = fetch(report, date, 1)
    if pages <= 0:
        return date, None
    rows = list(first)
    for pg in range(2, pages + 1):
        _, more = fetch(report, date, pg)
        if more:
            rows.extend(more)
    slim = [{k: r.get(k) for k in keys} for r in rows]
    return date, slim


def crawl(report, out_name, keys):
    cache = OUT / out_name
    done = set()
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["date"])
                except Exception:
                    pass
        print(f"{out_name}: 已有 {len(done)} 天")

    os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
    sys.path.insert(0, str(ROOT))
    from qf import data as qfdata

    md = qfdata.load_market(verbose=False)
    dates = [str(d.date()) for d in md.dates]
    todo = [d for d in dates if d not in done]
    print(f"{out_name}: 待抓 {len(todo)} 天（总 {len(dates)}）")

    t0 = time.time()
    n_ok = n_empty = n_fail = 0
    ln = open(cache, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(fetch_day, d, report, keys): d for d in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                d, rows = fut.result()
                if rows is None:
                    n_fail += 1
                elif not rows:
                    n_empty += 1
                else:
                    n_ok += 1
                    ln.write(json.dumps({"date": d, "rows": rows},
                                        ensure_ascii=False) + "\n")
                if i % 150 == 0:
                    ln.flush()
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)} | 有数据 {n_ok} 空 {n_empty} "
                          f"失败 {n_fail} | {el:.0f}s，剩余 "
                          f"{el/i*(len(todo)-i)/60:.1f} 分钟", flush=True)
    finally:
        ln.close()
    print(f"{out_name} 完成：有数据 {n_ok} | 空 {n_empty} | 失败 {n_fail}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "detail"
    if which == "detail":
        crawl("RPT_DAILYBILLBOARD_DETAILSNEW", "billboard_detail.jsonl", KEEP_DETAIL)
    elif which == "seat":
        crawl("RPT_BILLBOARD_DAILYDETAILSBUY", "billboard_seat.jsonl", KEEP_SEAT)
