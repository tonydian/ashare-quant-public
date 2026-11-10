# -*- coding: utf-8 -*-
"""一次性探测剩余所有可得的另类数据接口。

候选：
  1) 龙虎榜明细（每日）
  2) 龙虎榜营业部（机构/游资席位）
  3) 大宗交易（每日）
  4) 股东户数
  5) 分红送转
  6) 股票回购
  7) 限售解禁
  8) 业绩预告
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
    "Referer": "https://data.eastmoney.com/",
}
BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def probe(name, report, extra=None, filt=None, dt="2026-10-23"):
    p = {"reportName": report, "columns": "ALL",
         "pageNumber": "1", "pageSize": "2",
         "source": "WEB", "client": "WEB"}
    if filt is not None:
        p["filter"] = filt
    if extra:
        p.update(extra)
    qs = urllib.parse.urlencode(p)
    try:
        req = urllib.request.Request(f"{BASE}?{qs}", headers=HDR)
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        res = d.get("result")
        if res and res.get("data"):
            row = res["data"][0]
            keys = list(row.keys())
            print(f"  ✅ {name}")
            print(f"     count={res.get('count')} pages={res.get('pages')} "
                  f"字段 {len(keys)} 个")
            print(f"     关键字段: {keys[:14]}")
            return True, keys, row
        print(f"  ❌ {name}: {str(d)[:120]}")
        return False, [], None
    except Exception as e:
        print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:70]}")
        return False, [], None


print("=" * 100)
print("1) 龙虎榜明细")
print("=" * 100)
probe("龙虎榜-每日明细", "RPT_DAILYBILLBOARD_DETAILSNEW",
      filt=f"(TRADE_DATE='{ '2026-10-23' }')")

print("\n" + "=" * 100)
print("2) 龙虎榜机构/营业部席位")
print("=" * 100)
probe("龙虎榜-席位明细", "RPT_BILLBOARD_DAILYDETAILSBUY",
      filt="(TRADE_DATE='2026-10-23')")
probe("龙虎榜-营业部统计", "RPT_OPERATEDEPT_ACTIVE",
      extra={"sortTypes": "-1", "sortColumns": "TOTAL_BUYER_SALESTIMES"})

print("\n" + "=" * 100)
print("3) 大宗交易")
print("=" * 100)
probe("大宗交易-每日", "RPT_DATA_BLOCKTRADE",
      filt="(TRADE_DATE='2026-10-23')")

print("\n" + "=" * 100)
print("4) 股东户数")
print("=" * 100)
probe("股东户数", "RPT_HOLDERNUM_DET",
      filt="(END_DATE='2026-06-30')")
probe("股东户数-最新", "RPT_HOLDERNUMLATEST")

print("\n" + "=" * 100)
print("5) 分红送转")
print("=" * 100)
probe("分红送转", "RPT_SHAREBONUS_DET",
      filt="(REPORT_DATE='2025-12-31')")

print("\n" + "=" * 100)
print("6) 股票回购")
print("=" * 100)
probe("股票回购", "RPTA_WEB_GETHB")

print("\n" + "=" * 100)
print("7) 限售解禁")
print("=" * 100)
probe("限售解禁", "RPT_LIFT_STAGE",
      filt="(FREE_DATE>='2026-10-01')")

print("\n" + "=" * 100)
print("8) 业绩预告")
print("=" * 100)
probe("业绩预告", "RPT_PUBLIC_OP_NEWPREDICT",
      filt="(REPORT_DATE='2026-06-30')")
