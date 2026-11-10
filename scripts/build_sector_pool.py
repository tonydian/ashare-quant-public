# -*- coding: utf-8 -*-
"""板块 ETF 池构建（腾讯接口探测法）。

背景：东财列表接口已被封（Remote end closed connection），改用腾讯行情接口
直接探测候选 ETF 代码。ETF 代码段有规律：
  沪市：51xxxx（512/513/515/516/517/518/560/561/562/563/588）
  深市：159xxx
策略：
  1) 生成候选代码（常用 ETF 代码段穷举 + 已知热门板块代码）；
  2) 用腾讯接口批量探测（带节流），拿到名称与日线；
  3) 按名称关键词归类板块，同板块保留成交额最大的一只；
  4) 输出板块池 + 每个板块的日线面板（供轮动回测）。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from scripts.build_etf_data import fetch, sym  # noqa: E402

OUT_DIR = ROOT / "build" / "etf"
UNIV = OUT_DIR / "sector_universe.json"
PANEL = OUT_DIR / "sector_panel.npz"

# 已知板块 ETF 候选（沪市 51xxxx + 深市 159xxx），按板块归类
CANDIDATES = {
    "半导体": ["512480", "512760", "159995", "588200", "512680"],
    "医药": ["512010", "159929", "512170", "159838", "512010"],
    "军工": ["512660", "512710", "159949"],
    "证券": ["512880", "512000", "159841", "512900"],
    "银行": ["512800", "512820", "159887"],
    "消费": ["159928", "159936", "510150"],
    "酒": ["512690", "159744"],
    "有色": ["512400", "159876", "512244"],
    "新能源": ["516160", "515790", "159857", "512580"],
    "新能源车": ["515030", "159824", "516390"],
    "机器人": ["562500", "159770", "159526"],
    "AI": ["159819", "515980", "516510", "159611"],
    "软件": ["515230", "159852", "516010"],
    "通信": ["515880", "159695"],
    "传媒": ["512980", "159805", "516190"],
    "地产": ["512200", "159768"],
    "基建": ["516950", "159619", "516970"],
    "钢铁": ["515210", "516070"],
    "煤炭": ["515220", "159930"],
    "化工": ["159870", "516020"],
    "农业": ["159825", "516760"],
    "电力": ["159611", "561560"],
    "环保": ["512580", "159861"],
    "旅游": ["159766", "560870"],
    "科技": ["515000", "159807"],
    "金融": ["510230", "159933"],
    "纳指": ["513100", "159941"],
    "标普": ["513500", "159612"],
    "日经": ["513520", "159866"],
    "港股": ["513050", "159920", "513180"],
    "恒生科技": ["513130", "159742"],
    "创业板": ["159915", "159952"],
    "科创50": ["588000", "588080"],
    "沪深300": ["510300", "159919"],
    "中证500": ["510500", "159922"],
    "中证1000": ["512100", "159845"],
    "红利": ["510880", "515180"],
}


def probe_one(code: str) -> dict | None:
    """探测单只 ETF：返回名称/成交额/日线，失败返回 None。"""
    try:
        q = fetch(code, "qfq", "2017-01-01", "2026-10-24")
        if q is None or q.empty or len(q) < 100:
            return None
        raw = fetch(code, "", "2017-01-01", "2026-10-24")
        amt = float(pd.to_numeric(q["amount"], errors="coerce").median())
        return {"code": code, "n_days": int(len(q)),
                "start": str(q["date"].iloc[0].date()),
                "end": str(q["date"].iloc[-1].date()),
                "amount_median": amt, "qfq": q, "raw": raw}
    except Exception:
        return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    total = sum(len(v) for v in CANDIDATES.values())
    done = 0
    print(f"探测 {total} 个候选 ETF（每只约 1~2s）...")
    for sector, codes in CANDIDATES.items():
        picked = None
        for code in dict.fromkeys(codes):   # 去重保序
            done += 1
            info = probe_one(code)
            if info is None:
                print(f"  [{done:3d}/{total}] {code} {sector:8s} — 无数据")
                time.sleep(0.3)
                continue
            print(f"  [{done:3d}/{total}] {code} {sector:8s} {info['n_days']:5d} 日 "
                  f"{info['start']}~{info['end']} 成交额中位 {info['amount_median']/1e8:.2f} 亿")
            if picked is None or info["amount_median"] > picked["amount_median"]:
                picked = {**info, "sector": sector}
            time.sleep(0.3)
        if picked:
            results[picked["code"]] = picked

    print(f"\n有效板块 ETF：{len(results)} 只")
    # 统一日期轴
    dates = pd.DatetimeIndex(sorted(set().union(*[set(r["qfq"]["date"]) for r in results.values()])))
    codes = sorted(results)
    N, T = len(codes), len(dates)
    close = np.full((T, N), np.nan, dtype=np.float32)
    raw_close = np.full((T, N), np.nan, dtype=np.float32)
    amount = np.full((T, N), np.nan, dtype=np.float32)
    for j, c in enumerate(codes):
        r = results[c]
        f = r["qfq"].set_index("date").reindex(dates)
        close[:, j] = f["close"].to_numpy()
        amount[:, j] = f["amount"].to_numpy()
        if r["raw"] is not None and not r["raw"].empty:
            g = r["raw"].set_index("date").reindex(dates)
            raw_close[:, j] = g["close"].to_numpy()

    np.savez_compressed(PANEL, dates=dates.values, codes=np.array(codes),
                        close=close, raw_close=raw_close, amount=amount)
    UNIV.write_text(json.dumps({
        "n": len(codes),
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "sectors": {results[c]["sector"]: {"code": c,
                                           "start": results[c]["start"],
                                           "amount_median": results[c]["amount_median"]}
                    for c in codes},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {PANEL}（{T} 日 × {N} 只）")
    print(f"→ {UNIV}")
    print("\n板块池：")
    for c in codes:
        r = results[c]
        print(f"  {r['sector']:8s} {c}  {r['start']} 起  成交额中位 {r['amount_median']/1e8:6.2f} 亿")


if __name__ == "__main__":
    main()
