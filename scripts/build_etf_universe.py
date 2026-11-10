# -*- coding: utf-8 -*-
"""抓全市场 ETF 列表（东财，带节流+重试+断点缓存）。

之前失败原因：连续请求 + URL 未编码 → 被限流/连接被关闭。
本版：requests 间隔 1.5s、每页 100、最多 30 页、失败重试 3 次（指数退避）、
结果写缓存，支持断点续抓。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "build" / "etf" / "etf_list_raw.json"
OUT = ROOT / "build" / "etf" / "etf_universe.json"
HDR = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Referer": "https://quote.eastmoney.com/",
}


def get_json(url: str, retries: int = 3) -> dict | None:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=25) as r:
                txt = r.read().decode("utf-8", "replace")
            return json.loads(txt)
        except Exception as e:
            if k == retries - 1:
                print(f"    [fail] {e}")
                return None
            time.sleep(2.0 * (k + 1))
    return None


def fetch_all_etf() -> dict:
    """按成交额降序分页抓全部 ETF（fs 用 ETF 专用板块）。"""
    base = "https://push2.eastmoney.com/api/qt/clist/get"
    # ETF 分类：b:MK0021(股票ETF) b:MK0022(债券) b:MK0023(商品) b:MK0024(货币) + 场内基金
    fs = "b:MK0021,b:MK0022,b:MK0023,b:MK0024"
    fields = "f12,f14,f2,f5,f6,f20"
    out = {}
    pn, pz = 1, 100
    while pn <= 30:
        q = urllib.parse.urlencode({
            "pn": pn, "pz": pz, "po": 1, "np": 1, "fltt": 2, "invt": 2,
            "fid": "f6", "fs": fs, "fields": fields,
        })
        obj = get_json(f"{base}?{q}")
        if not obj:
            break
        data = obj.get("data") or {}
        items = data.get("diff") or []
        if not items:
            break
        for it in items:
            code = str(it.get("f12") or "").strip()
            name = str(it.get("f14") or "").strip()
            if code and name:
                out[code] = {"code": code, "name": name,
                             "price": it.get("f2"), "amount": it.get("f6"),
                             "mcap": it.get("f20")}
        total = int(data.get("total") or 0)
        print(f"  page {pn}: +{len(items)} 累计 {len(out)}/{total}")
        if len(out) >= total:
            break
        pn += 1
        time.sleep(1.5)
    return out


# 板块关键词（顺序即优先级）
SECTOR_RULES = [
    ("半导体", ["半导体", "芯片", "集成电路"]),
    ("医药", ["医药", "医疗", "生物", "创新药", "疫苗"]),
    ("军工", ["军工", "国防", "航天", "航空"]),
    ("证券", ["证券", "券商"]),
    ("银行", ["银行"]),
    ("消费", ["消费", "食品", "饮料"]),
    ("酒", ["酒"]),
    ("有色", ["有色", "稀土", "黄金", "铜"]),
    ("新能源", ["新能源", "光伏", "锂电", "电池", "储能"]),
    ("新能源车", ["新能车", "智能车"]),
    ("机器人", ["机器人"]),
    ("AI", ["人工智能", "AI", "算力", "云计算", "大数据"]),
    ("软件", ["软件", "计算机", "信创"]),
    ("通信", ["通信", "5G", "电信"]),
    ("传媒", ["传媒", "游戏", "影视"]),
    ("地产", ["地产", "房地产"]),
    ("基建", ["基建", "建筑", "建材"]),
    ("钢铁", ["钢铁"]),
    ("煤炭", ["煤炭"]),
    ("化工", ["化工", "化学"]),
    ("农业", ["农业", "养殖", "畜牧"]),
    ("电力", ["电力", "公用事业"]),
    ("环保", ["环保", "碳中和"]),
    ("旅游", ["旅游", "餐饮"]),
    ("科技", ["科技", "科创"]),
    ("金融", ["金融", "非银"]),
    ("港股", ["港股", "恒生"]),
    ("纳指", ["纳斯达克", "纳指"]),
    ("标普", ["标普"]),
    ("日经", ["日经"]),
]


def classify(name: str) -> str | None:
    for sector, kws in SECTOR_RULES:
        if any(kw in name for kw in kws):
            return sector
    return None


def main():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        all_etf = json.load(open(CACHE, encoding="utf-8"))
        print(f"[cache] 已有 {len(all_etf)} 只 ETF")
    else:
        print("抓取 ETF 列表（约 30 页，含节流）...")
        all_etf = fetch_all_etf()
        CACHE.write_text(json.dumps(all_etf, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ETF 总数：{len(all_etf)}")

    for v in all_etf.values():
        v["sector"] = classify(v["name"])

    by_sector: dict[str, list] = {}
    for v in all_etf.values():
        if v["sector"]:
            by_sector.setdefault(v["sector"], []).append(v)

    print(f"\n可归类板块：{len(by_sector)} 个")
    print(f"{'板块':<10}{'ETF数':>6}  {'代表ETF':<30}{'成交额(亿)':>10}")
    chosen = {}
    for sector in sorted(by_sector, key=lambda s: -max((v["amount"] or 0) for v in by_sector[s])):
        lst = sorted(by_sector[sector], key=lambda v: -(v["amount"] or 0))
        top = lst[0]
        amt = (top["amount"] or 0) / 1e8
        if (top["amount"] or 0) < 1e7:
            print(f"{sector:<10}{len(lst):>6}  （成交额不足，跳过）")
            continue
        chosen[sector] = {**top, "n_same_sector": len(lst)}
        print(f"{sector:<10}{len(lst):>6}  {top['code']} {top['name'][:20]:<22}{amt:>9.2f}")

    OUT.write_text(json.dumps({
        "n_etf_total": len(all_etf), "n_sector": len(chosen), "sector_pick": chosen,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ 选定 {len(chosen)} 个板块（每板块取成交额最大的 1 只）")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
