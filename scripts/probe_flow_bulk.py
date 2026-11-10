# -*- coding: utf-8 -*-
"""验证资金流数据能否批量获取 + 历史覆盖多长。

东财资金流日频返回字段（f51~f65）：
  f51 日期, f52 主力净流入额, f53 小单净流入, f54 中单, f55 大单,
  f56 超大单, f57 主力净流入占比, f58 小单占比, ...
"""
from __future__ import annotations

import json
import socket
import time
import urllib.parse
import urllib.request

socket.setdefaulttimeout(15)
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://quote.eastmoney.com/",
}


def secid(code: str) -> str:
    """代码 → 东财 secid（0=深，1=沪）。"""
    c = str(code).zfill(6)
    if c.startswith(("60", "68", "9", "5")):
        return f"1.{c}"
    return f"0.{c}"


def fetch_flow(code: str, lmt: int = 10000):
    p = {
        "secid": secid(code),
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "klt": "101", "lmt": str(lmt),
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?" + \
          urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    data = d.get("data") or {}
    kl = data.get("klines") or []
    return data.get("name"), kl


print("=" * 96)
print("资金流数据验证")
print("=" * 96)

# ---------- 1) 单只股票：历史覆盖
for code in ("000001", "600000", "300750"):
    try:
        name, kl = fetch_flow(code)
        if kl:
            first = kl[0].split(",")[0]
            last = kl[-1].split(",")[0]
            print(f"\n  {code} {name}: {len(kl)} 条 | {first} ~ {last}")
            print(f"    最新一条: {kl[-1][:150]}")
            print(f"    字段数: {len(kl[-1].split(','))}")
        else:
            print(f"  {code}: 无数据")
    except Exception as e:
        print(f"  {code}: 失败 {type(e).__name__}: {str(e)[:60]}")

# ---------- 2) 批量可行性（10 只计时）
print("\n" + "=" * 96)
print("批量可行性：10 只股票耗时")
print("=" * 96)
codes = ["000001", "000002", "600000", "600036", "300750",
         "002594", "601318", "000858", "600519", "002415"]
t0 = time.time()
ok = 0
for c in codes:
    try:
        _, kl = fetch_flow(c, lmt=2500)
        if kl:
            ok += 1
    except Exception:
        pass
el = time.time() - t0
print(f"  {ok}/10 成功，耗时 {el:.1f}s → 单只 {el/10:.2f}s")
print(f"  推算 5281 只：{el/10*5281/60:.1f} 分钟（串行）")
print(f"  8 线程并发：约 {el/10*5281/60/8:.1f} 分钟")

# ---------- 3) 字段含义验证：主力净流入 与 当日涨跌 的关系
print("\n" + "=" * 96)
print("字段质量：主力净流入 vs 当日涨跌（应正相关）")
print("=" * 96)
name, kl = fetch_flow("000001", lmt=2500)
if kl:
    rows = [x.split(",") for x in kl]
    print(f"  {name}: {len(rows)} 条")
    print(f"  日期: {rows[0][0]} ~ {rows[-1][0]}")
    # f52=主力净流入额, f57=主力占比
    sample = rows[-1]
    print(f"\n  最新一条解析:")
    labels = ["日期", "主力净流入", "小单净流入", "中单净流入", "大单净流入",
              "超大单净流入", "主力占比%", "小单%", "中单%", "大单%",
              "超大单%", "收盘价?", "涨跌幅?", "?", "?"]
    for i, v in enumerate(sample):
        lab = labels[i] if i < len(labels) else f"f{51+i}"
        print(f"    {lab:<14} {v}")
