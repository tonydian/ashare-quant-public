# -*- coding: utf-8 -*-
"""抓取快照类另类数据（一次拉多期，效率高）。

三类：
  1) 股东户数（RPT_HOLDERNUM_DET）：按 END_DATE 期查询，反映筹码集中度
  2) 限售解禁（RPT_LIFT_STAGE）：按 FREE_DATE 查询，解禁前后有压力
  3) 业绩预告（RPT_PUBLIC_OP_NEWPREDICT）：按 REPORT_DATE 查询，盈利惊喜

关键：这些都是"事件型"数据，需要映射到时间轴上（公告日/发生日之后才可知）。
"""
from __future__ import annotations

import json
import random
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"D:\wkf\getrich\ashare_quant")
OUT = ROOT / "build" / "alt"
OUT.mkdir(parents=True, exist_ok=True)

socket.setdefaulttimeout(25)
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}
BASE = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def fetch_all(report, filt, keys, size=500, tries=4, max_pages=200):
    """拉取某 filter 下的全部行（分页）。"""
    rows = []
    page = 1
    while page <= max_pages:
        p = {"reportName": report, "columns": "ALL", "filter": filt,
             "pageNumber": str(page), "pageSize": str(size),
             "source": "WEB", "client": "WEB"}
        qs = urllib.parse.urlencode(p)
        ok = False
        for k in range(tries):
            try:
                req = urllib.request.Request(f"{BASE}?{qs}", headers=HDR)
                with urllib.request.urlopen(req, timeout=25) as r:
                    d = json.loads(r.read().decode("utf-8", "replace"))
                res = d.get("result")
                if not res:
                    return rows, 0
                data = res.get("data") or []
                pages = int(res.get("pages") or 1)
                for rr in data:
                    rows.append({k2: rr.get(k2) for k2 in keys})
                ok = True
                if page >= pages:
                    return rows, pages
                break
            except Exception:
                time.sleep(0.6 * (2 ** k) + random.random() * 0.4)
        if not ok:
            return rows, -1
        page += 1
        time.sleep(0.25)
    return rows, page


# ============================================================
# 1) 股东户数：按季度 END_DATE
HOLDER_KEYS = ["SECURITY_CODE", "END_DATE", "HOLDER_NUM", "PRE_HOLDER_NUM",
               "HOLDER_NUM_CHANGE", "HOLDER_NUM_RATIO", "AVG_MARKET_CAP",
               "AVG_HOLD_NUM", "TOTAL_MARKET_CAP"]
QUARTERS = [f"{y}-{m:02d}-{(31 if m in (3, 12) else 30):02d}"
            for y in range(2016, 2027) for m in (3, 6, 9, 12)
            if not (y == 2026 and m > 9)]


def crawl_holder():
    cache = OUT / "holder_num.jsonl"
    done = set()
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["end_date"])
                except Exception:
                    pass
    todo = [q for q in QUARTERS if q not in done]
    print(f"股东户数：{len(QUARTERS)} 期，待抓 {len(todo)}")
    with open(cache, "a", encoding="utf-8") as ln:
        for q in todo:
            rows, pages = fetch_all("RPT_HOLDERNUM_DET", f"(END_DATE='{q}')", HOLDER_KEYS)
            if rows:
                ln.write(json.dumps({"end_date": q, "rows": rows},
                                    ensure_ascii=False) + "\n")
                ln.flush()
                print(f"  {q}: {len(rows)} 行（{pages} 页）", flush=True)
            else:
                print(f"  {q}: 空/失败 ({pages})", flush=True)
            time.sleep(0.4)


# ============================================================
# 2) 限售解禁：按月拉（FREE_DATE 范围）
LIFT_KEYS = ["SECURITY_CODE", "FREE_DATE", "LIFT_MARKET_CAP",
             "CURRENT_FREE_SHARES", "FREE_SHARES", "FREE_RATIO",
             "BATCH_HOLDER_NUM", "FREE_SHARES_TYPE"]


def crawl_lift():
    cache = OUT / "lift.jsonl"
    done = set()
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["month"])
                except Exception:
                    pass
    months = []
    for y in range(2016, 2027):
        for m in range(1, 13):
            if y == 2026 and m > 10:
                continue
            months.append(f"{y}-{m:02d}")
    todo = [m for m in months if m not in done]
    print(f"限售解禁：{len(months)} 月，待抓 {len(todo)}")
    with open(cache, "a", encoding="utf-8") as ln:
        for mm in todo:
            y, mo = mm.split("-")
            last = f"{y}-{mo}-31" if mo in ("01", "03", "05", "07", "08", "10", "12") \
                else (f"{y}-{mo}-30" if mo != "02" else f"{y}-02-28")
            filt = f"(FREE_DATE>='{mm}-01')(FREE_DATE<='{last}')"
            rows, pages = fetch_all("RPT_LIFT_STAGE", filt, LIFT_KEYS)
            if rows:
                ln.write(json.dumps({"month": mm, "rows": rows},
                                    ensure_ascii=False) + "\n")
                ln.flush()
                print(f"  {mm}: {len(rows)} 行", flush=True)
            else:
                print(f"  {mm}: 空 ({pages})", flush=True)
            time.sleep(0.4)


# ============================================================
# 3) 业绩预告：按报告期
FORECAST_KEYS = ["SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE",
                 "PREDICT_FINANCE_CODE", "PREDICT_AMT_LOWER", "PREDICT_AMT_UPPER",
                 "ADD_AMP_LOWER", "ADD_AMP_UPPER", "PREDICT_TYPE"]
REPORTS = [f"{y}-{m:02d}-{(31 if m in (3, 12) else 30):02d}"
           for y in range(2016, 2027) for m in (3, 6, 9, 12)
           if not (y == 2026 and m > 6)]


def crawl_forecast():
    cache = OUT / "forecast.jsonl"
    done = set()
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["report_date"])
                except Exception:
                    pass
    todo = [r for r in REPORTS if r not in done]
    print(f"业绩预告：{len(REPORTS)} 期，待抓 {len(todo)}")
    with open(cache, "a", encoding="utf-8") as ln:
        for rp in todo:
            rows, pages = fetch_all("RPT_PUBLIC_OP_NEWPREDICT",
                                    f"(REPORT_DATE='{rp}')", FORECAST_KEYS)
            if rows:
                ln.write(json.dumps({"report_date": rp, "rows": rows},
                                    ensure_ascii=False) + "\n")
                ln.flush()
                print(f"  {rp}: {len(rows)} 行", flush=True)
            else:
                print(f"  {rp}: 空 ({pages})", flush=True)
            time.sleep(0.4)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "holder"):
        print("=" * 90)
        print("1) 股东户数")
        print("=" * 90)
        crawl_holder()
    if which in ("all", "lift"):
        print("\n" + "=" * 90)
        print("2) 限售解禁")
        print("=" * 90)
        crawl_lift()
    if which in ("all", "forecast"):
        print("\n" + "=" * 90)
        print("3) 业绩预告")
        print("=" * 90)
        crawl_forecast()
