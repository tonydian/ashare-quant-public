# -*- coding: utf-8 -*-
"""确认两个可用数据源的批量抓取能力与历史深度。

A) 新浪 ssl_qsfx_zjlrqs —— 多日资金流历史（netamount / r0_net 等）
B) 东财 fflow/kline/get —— 历史资金流（第二条路径，之前成功了）
"""
from __future__ import annotations

import json
import random
import socket
import time
import urllib.parse
import urllib.request

socket.setdefaulttimeout(15)
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn/",
}


def sina(code: str, num: int = 1000, tries: int = 3):
    c = str(code).zfill(6)
    pre = "sz" if c.startswith(("00", "30")) else "sh"
    url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"MoneyFlow.ssl_qsfx_zjlrqs?page=1&num={num}&sort=opendate&asc=0&daima={pre}{c}")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=15) as r:
                txt = r.read().decode("utf-8", "replace").strip()
            if not txt or txt == "null":
                return []
            return json.loads(txt)
        except Exception:
            time.sleep(0.5 * (2 ** k) + random.random() * 0.3)
    return None


def em(code: str, lmt: int = 6000, tries: int = 3):
    c = str(code).zfill(6)
    sec = f"1.{c}" if c.startswith(("60", "68", "9", "5")) else f"0.{c}"
    p = {"lmt": str(lmt), "klt": "101", "secid": sec,
         "fields1": "f1,f2,f3,f7",
         "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"}
    url = ("https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get?"
           + urllib.parse.urlencode(p))
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": HDR["User-Agent"],
                "Referer": "https://quote.eastmoney.com/"})
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            data = d.get("data") or {}
            return data.get("name"), (data.get("klines") or [])
        except Exception:
            time.sleep(0.6 * (2 ** k) + random.random() * 0.4)
    return None


print("=" * 96)
print("A) 新浪 多日资金流 —— 历史深度")
print("=" * 96)
for code in ("000001", "600519", "300750"):
    d = sina(code, num=1000)
    if d:
        print(f"  {code}: {len(d)} 条 | {d[-1]['opendate']} ~ {d[0]['opendate']}")
        print(f"    最新: netamount={d[0]['netamount']} r0_net={d[0]['r0_net']} "
              f"changeratio={d[0]['changeratio']}")
    else:
        print(f"  {code}: 失败/空")
    time.sleep(0.5)

print("\n" + "=" * 96)
print("B) 东财 历史资金流（kline 路径）—— 历史深度")
print("=" * 96)
for code in ("000001", "600519", "300750"):
    r = em(code, lmt=6000)
    if r and r[1]:
        name, kl = r
        print(f"  {code} {name}: {len(kl)} 条 | {kl[0].split(',')[0]} ~ {kl[-1].split(',')[0]}")
    else:
        print(f"  {code}: 失败/空")
    time.sleep(0.5)

print("\n" + "=" * 96)
print("C) 批量稳定性：各 15 只，计时 + 成功率")
print("=" * 96)
codes = ["000001", "000002", "000063", "000100", "000157", "000333", "000538",
         "000568", "000625", "000651", "000725", "000858", "002415", "002594", "600000"]
for tag, fn in (("新浪", lambda c: sina(c, num=1000)),
                ("东财", lambda c: em(c, lmt=6000))):
    t0 = time.time()
    ok = 0
    for c in codes:
        r = fn(c)
        ok += 1 if r else 0
        time.sleep(0.3)
    el = time.time() - t0
    print(f"  {tag}: 成功 {ok}/{len(codes)}，{el:.1f}s → 单只 {el/len(codes):.2f}s"
          f" | 5281只串行 {el/len(codes)*5281/60:.0f}分钟，6线程 {el/len(codes)*5281/60/6*1.4:.0f}分钟")

print("\n" + "=" * 96)
print("D) 字段说明（新浪 ssl_qsfx_zjlrqs）")
print("=" * 96)
d = sina("000001", num=3)
if d:
    for k, v in d[0].items():
        print(f"  {k:<16} {v}")
