# -*- coding: utf-8 -*-
"""资金流数据批量验证（带重试/限速/多域名轮换）。

发现：东财 push2his 短时间高频请求会 RemoteDisconnected。
对策：不同域名轮换 + 请求间隔 + 指数退避重试。
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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Connection": "close",
}
HOSTS = ["push2his.eastmoney.com", "push2his.eastmoney.com",
         "42.push2his.eastmoney.com", "push2his.eastmoney.com"]


def secid(code: str) -> str:
    c = str(code).zfill(6)
    return f"1.{c}" if c.startswith(("60", "68", "9", "5")) else f"0.{c}"


def fetch_flow(code: str, lmt: int = 6000, tries: int = 4):
    p = {
        "secid": secid(code),
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "klt": "101", "lmt": str(lmt),
    }
    qs = urllib.parse.urlencode(p)
    last = None
    for k in range(tries):
        host = HOSTS[k % len(HOSTS)]
        try:
            req = urllib.request.Request(f"https://{host}/api/qt/stock/fflow/daykline/get?{qs}",
                                         headers=HDR)
            with urllib.request.urlopen(req, timeout=15) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            data = d.get("data") or {}
            return data.get("name"), (data.get("klines") or [])
        except Exception as e:
            last = e
            time.sleep(0.6 * (2 ** k) + random.random() * 0.4)
    raise last


print("=" * 96)
print("资金流数据批量验证（带重试）")
print("=" * 96)

# ---------- 1) 历史覆盖
for code in ("000001", "600519", "300750"):
    try:
        name, kl = fetch_flow(code)
        if kl:
            print(f"\n  {code} {name}: {len(kl)} 条 | {kl[0].split(',')[0]} ~ {kl[-1].split(',')[0]}")
            if code == "000001":
                print(f"    字段数 {len(kl[-1].split(','))}")
                print(f"    最新: {kl[-1]}")
        else:
            print(f"  {code}: 空")
    except Exception as e:
        print(f"  {code}: 失败 {type(e).__name__}: {str(e)[:70]}")
    time.sleep(1.0)

# ---------- 2) 批量计时（20 只，串行 + 间隔）
print("\n" + "=" * 96)
print("批量可行性：20 只（串行，间隔 0.35s）")
print("=" * 96)
codes = ["000001", "000002", "000063", "000100", "000157", "000333", "000538",
         "000568", "000625", "000651", "000725", "000768", "000858", "002027",
         "002049", "002142", "002230", "002304", "002352", "002415"]
t0 = time.time()
ok, fail = 0, 0
for c in codes:
    try:
        _, kl = fetch_flow(c, lmt=6000)
        ok += 1 if kl else 0
    except Exception:
        fail += 1
    time.sleep(0.35)
el = time.time() - t0
print(f"  成功 {ok}/{len(codes)}（失败 {fail}），耗时 {el:.1f}s → 单只 {el/len(codes):.2f}s")
print(f"  全市场 5281 只串行: {el/len(codes)*5281/60:.1f} 分钟")
print(f"  6 线程并发（含限速）: 约 {el/len(codes)*5281/60/6*1.5:.1f} 分钟")

# ---------- 3) 字段质量：主力净流入 vs 涨跌
print("\n" + "=" * 96)
print("字段质量检验")
print("=" * 96)
try:
    name, kl = fetch_flow("000001")
    if kl:
        rows = [x.split(",") for x in kl]
        print(f"  {name} 共 {len(rows)} 条，最新一条：")
        for i, v in enumerate(rows[-1]):
            print(f"    f{51+i}: {v}")
        # 检查 f57（主力占比）与 f52（主力净额）符号一致性
        import numpy as np
        a = np.array([float(r[1]) if r[1] not in ("-", "") else np.nan for r in rows])
        b = np.array([float(r[6]) if len(r) > 6 and r[6] not in ("-", "") else np.nan
                      for r in rows])
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 10:
            print(f"\n  f52(主力净额) 与 f57(主力占比) 相关系数: "
                  f"{np.corrcoef(a[m], b[m])[0,1]:.4f}（应接近 1.0 或 -1.0）")
            print(f"  f52 范围: {np.nanmin(a[m]):.0f} ~ {np.nanmax(a[m]):.0f}")
except Exception as e:
    print(f"  失败: {e}")
