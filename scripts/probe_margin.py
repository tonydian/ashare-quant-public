# -*- coding: utf-8 -*-
"""探测融资融券数据：历史深度、字段、批量可行性。

东财接口：
  RPTA_WEB_RZRQ_GGMX  个股融资融券明细（按日期查，返回该日全部股票）
字段（推测）：
  DATE 日期, SCODE 代码, SECNAME 名称,
  RZYE 融资余额, RZMRE 融资买入额, RZCHE 融资偿还额,
  RQYE 融券余额, RQYL 融券余量, RQMCL 融券卖出量, RQCHL 融券偿还量,
  RZRQYE 融资融券余额
关键：这个接口按**日期**返回全市场，所以每天一个请求 → 需要跑 4000+ 天
"""
from __future__ import annotations

import json
import socket
import time
import urllib.parse
import urllib.request

socket.setdefaulttimeout(20)
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/rzrq/",
}
BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def get(params: dict, tries: int = 3):
    qs = urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            req = urllib.request.Request(f"{BASE}?{qs}", headers=HDR)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception:
            time.sleep(0.8 * (2 ** k))
    return None


print("=" * 96)
print("1) 单个交易日：返回行数 + 字段结构")
print("=" * 96)
d = get({"reportName": "RPTA_WEB_RZRQ_GGMX", "columns": "ALL",
         "filter": "(DATE='2026-10-23')", "pageNumber": "1", "pageSize": "5",
         "sortTypes": "1", "sortColumns": "SCODE",
         "source": "WEB", "client": "WEB"})
if d and d.get("result"):
    res = d["result"]
    print(f"  pages={res.get('pages')} count={res.get('count')} "
          f"（≈当日股票数）")
    row = res["data"][0]
    print("\n  字段结构:")
    for k, v in row.items():
        print(f"    {k:<14} {v}")
else:
    print(f"  失败: {d}")

print("\n" + "=" * 96)
print("2) 单日全量拉取（pageSize=500）")
print("=" * 96)
t0 = time.time()
d2 = get({"reportName": "RPTA_WEB_RZRQ_GGMX", "columns": "ALL",
          "filter": "(DATE='2026-10-23')", "pageNumber": "1", "pageSize": "500",
          "sortTypes": "1", "sortColumns": "SCODE",
          "source": "WEB", "client": "WEB"})
el = time.time() - t0
if d2 and d2.get("result"):
    print(f"  返回 {len(d2['result']['data'])} 行，总 {d2['result']['count']} 行，"
          f"耗时 {el:.2f}s")
else:
    print(f"  失败")

print("\n" + "=" * 96)
print("3) 历史深度探测（不同年份的日期）")
print("=" * 96)
for dt in ("2010-04-01", "2015-06-01", "2018-06-01", "2020-06-01",
           "2022-06-01", "2024-06-03", "2026-10-23"):
    d3 = get({"reportName": "RPTA_WEB_RZRQ_GGMX", "columns": "ALL",
              "filter": f"(DATE='{dt}')", "pageNumber": "1", "pageSize": "1",
              "source": "WEB", "client": "WEB"})
    if d3 and d3.get("result"):
        print(f"  {dt}: ✅ count={d3['result'].get('count')}")
    else:
        print(f"  {dt}: ❌ {str(d3)[:80]}")
    time.sleep(0.3)
