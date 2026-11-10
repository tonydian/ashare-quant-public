# -*- coding: utf-8 -*-
"""探测另类数据可得性（先用标准库直连，不依赖 akshare）。

需要的数据（按价值排序）：
  1) 主力资金流（大单净流入）—— 最能解释"谁在买"
  2) 龙虎榜 —— 游资/机构席位
  3) 融资融券余额 —— 杠杆资金
  4) 股东户数变化 —— 筹码集中度
  5) 大宗交易

逐个试接口，记录哪些能拿、需要什么参数、返回结构。
"""
from __future__ import annotations

import json
import socket
import urllib.parse
import urllib.request

socket.setdefaulttimeout(12)
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
}


def try_get(name, url, params=None, note=""):
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        req = urllib.request.Request(full, headers=HDR)
        with urllib.request.urlopen(req, timeout=12) as r:
            raw = r.read()
            txt = raw.decode("utf-8", "replace")
            print(f"  ✅ {name}: {r.status} | {len(raw)} bytes")
            print(f"     {txt[:220]}")
            return txt
    except Exception as e:
        print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:80]}")
        return None


print("=" * 100)
print("另类数据接口探测")
print("=" * 100)

print("\n【1】东方财富 个股资金流（push2his）")
try_get("东财资金流-日频", "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get",
        {"secid": "0.000001", "fields1": "f1,f2,f3,f7",
         "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
         "klt": "101", "lmt": "30"})

print("\n【2】东方财富 资金流排行（全市场快照）")
try_get("东财资金流排行", "https://push2.eastmoney.com/api/qt/clist/get",
        {"pn": "1", "pz": "5", "po": "1", "np": "1",
         "fltt": "2", "invt": "2", "fid": "f62",
         "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
         "fields": "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87"})

print("\n【3】东方财富 龙虎榜")
try_get("东财龙虎榜", "https://datacenter-web.eastmoney.com/api/data/v1/get",
        {"reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
         "columns": "ALL", "filter": "(TRADE_DATE='2026-10-23')",
         "pageNumber": "1", "pageSize": "3", "sortTypes": "-1",
         "sortColumns": "NET_BUY_AMT", "source": "WEB", "client": "WEB"})

print("\n【4】东方财富 融资融券")
try_get("东财两融", "https://datacenter-web.eastmoney.com/api/data/v1/get",
        {"reportName": "RPTA_WEB_RZRQ_GGMX", "columns": "ALL",
         "filter": "(DATE='2026-10-23')", "pageNumber": "1", "pageSize": "3",
         "source": "WEB", "client": "WEB"})

print("\n【5】腾讯 资金流")
try_get("腾讯资金流", "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/fundflow/get",
        {"code": "sz000001", "type": "day"})

print("\n【6】新浪 资金流")
try_get("新浪资金流", "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssi_ssfx_flzjtj",
        {"daima": "sz000001"})

print("\n" + "=" * 100)
print("说明：接口探测结果直接决定 C 方案能不能做。")
print("=" * 100)
