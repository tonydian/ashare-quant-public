# -*- coding: utf-8 -*-
"""C 方案第一步：抓取全市场历史资金流（新浪）。

字段：
  opendate      日期
  netamount     净流入额（全单）
  r0_net        超大单净额
  r0_ratio      超大单净占比
  ratioamount   净流入占比
  changeratio   涨跌幅
  turnover      换手率

产出：build/alt/flow_panel.npz
  dates, codes, netamount(净流入占成交额%), r0_net(超大单%), turnover
"""
from __future__ import annotations

import json
import random
import socket
import sys
import time
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
    "Referer": "https://finance.sina.com.cn/",
}
URL = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
       "MoneyFlow.ssl_qsfx_zjlrqs?page=1&num={num}&sort=opendate&asc=0&daima={sym}")


def sina_sym(code: str) -> str:
    c = str(code).zfill(6)
    return ("sz" if c.startswith(("00", "30")) else "sh") + c


def fetch(code: str, num: int = 6000, tries: int = 4):
    """抓一只股票的资金流历史。返回 list[dict] 或 None。"""
    url = URL.format(num=num, sym=sina_sym(code))
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=20) as r:
                txt = r.read().decode("utf-8", "replace").strip()
            if not txt or txt == "null":
                return []
            return json.loads(txt)
        except Exception:
            time.sleep(0.4 * (2 ** k) + random.random() * 0.3)
    return None


def main():
    # 股票池：从行情面板取（含退市，避免幸存者偏差）
    import os
    os.environ.setdefault("QF_DATA_RAW", str(REPO / "data_extract" / "was_stock_data_ext"))
    sys.path.insert(0, str(ROOT))
    from qf import data as qfdata

    md = qfdata.load_market(verbose=False)
    codes = [str(c) for c in md.codes]
    print(f"股票池 {len(codes)} 只（含退市 {sum(1 for c in codes if c not in set(md.codes))}）")

    cache = OUT / "flow_raw.jsonl"
    done = set()
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["code"])
                except Exception:
                    pass
        print(f"已有缓存 {len(done)} 只，续抓")

    todo = [c for c in codes if c not in done]
    print(f"待抓 {len(todo)} 只")

    n_ok = n_empty = n_fail = 0
    t0 = time.time()
    lock_w = open(cache, "a", encoding="utf-8")

    def task(code):
        return code, fetch(code)

    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(task, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                code, recs = fut.result()
                if recs is None:
                    n_fail += 1
                elif not recs:
                    n_empty += 1
                else:
                    n_ok += 1
                    lock_w.write(json.dumps({"code": code, "rows": recs},
                                            ensure_ascii=False) + "\n")
                if i % 200 == 0:
                    lock_w.flush()
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)} | ok {n_ok} empty {n_empty} fail {n_fail} "
                          f"| {el:.0f}s，预计剩余 {el/i*(len(todo)-i)/60:.1f} 分钟",
                          flush=True)
    finally:
        lock_w.close()

    print(f"\n抓取完成：ok {n_ok} | empty {n_empty} | fail {n_fail}")
    print(f"→ {cache}")


if __name__ == "__main__":
    main()
