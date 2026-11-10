# -*- coding: utf-8 -*-
"""抓取分红明细，计算近 730 天股息率。

接口：东财 RPT_SHAREBONUS_DET（按报告期查）
字段：
  SECURITY_CODE      代码
  PRETAX_BONUS_RMB   每 10 股派息（税前，元）
  EX_DIVIDEND_DATE   除权除息日
  REPORT_DATE        报告期
  ASSIGN_PROGRESS    方案进度

股息率 = 近 730 天累计每股派息 / 当前股价
"""
from __future__ import annotations

import json
import random
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\wkf\getrich\ashare_quant")
OUT = ROOT / "build" / "alt"
OUT.mkdir(parents=True, exist_ok=True)

socket.setdefaulttimeout(25)
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}
BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"
KEEP = ["SECURITY_CODE", "PRETAX_BONUS_RMB", "EX_DIVIDEND_DATE",
        "REPORT_DATE", "ASSIGN_PROGRESS", "IMPL_PLAN_PROFILE"]


def fetch(report_date: str, page: int = 1, size: int = 500, tries: int = 4):
    p = {"reportName": "RPT_SHAREBONUS_DET", "columns": "ALL",
         "filter": f"(REPORT_DATE='{report_date}')",
         "pageNumber": str(page), "pageSize": str(size),
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
            time.sleep(0.6 * (2 ** k) + random.random() * 0.4)
    return -1, []


def fetch_all(report_date: str, max_pages: int = 60):
    rows = []
    pages, first = fetch(report_date, 1)
    if pages <= 0:
        return rows, pages
    rows.extend(first)
    for pg in range(2, min(pages, max_pages) + 1):
        _, more = fetch(report_date, pg)
        if more:
            rows.extend(more)
        time.sleep(0.25)
        if pg % 10 == 0:
            print(f"    {report_date} page {pg}/{pages} 已 {len(rows)} 行", flush=True)
    return rows, pages


def main():
    # 报告期：覆盖近 3 年（730 天 ≈ 2 年，多抓 1 年保险）
    periods = []
    for y in (2023, 2024, 2025, 2026):
        for m, day in ((6, 30), (12, 31)):
            dt = f"{y}-{m:02d}-{day:02d}"
            periods.append(dt)

    cache = OUT / "dividend_raw.jsonl"
    done = set()
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["report_date"])
                except Exception:
                    pass
        print(f"已有缓存 {len(done)} 期")

    todo = [p for p in periods if p not in done]
    print(f"待抓 {len(todo)} 个报告期: {todo}")

    with open(cache, "a", encoding="utf-8") as ln:
        for rd in todo:
            print(f"  抓 {rd} …", flush=True)
            rows, pages = fetch_all(rd)
            if rows:
                slim = [{k: r.get(k) for k in KEEP} for r in rows]
                ln.write(json.dumps({"report_date": rd, "rows": slim},
                                    ensure_ascii=False) + "\n")
                ln.flush()
                print(f"    ✓ {len(rows)} 行（{pages} 页）", flush=True)
            else:
                print(f"    ✗ 空/失败（{pages}）", flush=True)
            time.sleep(0.5)

    print(f"\n→ {cache}")


if __name__ == "__main__":
    main()
