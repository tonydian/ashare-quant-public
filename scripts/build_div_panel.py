# -*- coding: utf-8 -*-
"""红利指数 / 红利 ETF 面板：低换手策略的对照基准。

为什么需要
----------
低换手策略本质是「红利/低波替代」：持仓常年四大行、中石油、中国电信。
只跟中证全指比是**错的基准**（不做红利筛选、不含股息再投资）。
真正该比的：
  * 510880 上证红利ETF（2007 起，最长样本）
  * 515080 中证红利ETF（2019-12 起）
  * 512890 红利低波ETF（2019-01 起）
  * 000922 中证红利、H30269 中证红利低波（**价格指数，不含股息**，仅作参考）

价格口径（踩过的坑）
--------------------
* 腾讯 `newfqkline` 的 `qfq` 是「以最新价为基准」的前复权：分段抓取每段基准不同，
  拼起来有阶梯跳变（实测 510880 拼出 vol 454%、最大回撤 -107%），不能用于收益计算。
* 改用东方财富 `push2his` 一次返回全历史 + `fqt=2`（后复权，含分红再投资），
  数据连续、无需分段拼接。ETF 后复权收益 ≈ 全收益，才与策略口径可比。
* 指数没有免费的全收益版本（H00922 取不到），价格指数直接比会**低估基准 4~6pp**
  （红利指数股息率），所以只作参考列、不作主判据。

输出：build/etf/div_panel.parquet
  长表：date, code, name, kind(etf/index), close_adj, close_raw, amount
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "etf" / "div_panel.parquet"

URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# secid -> (名称, kind)；1.=沪 2.=深(含中证指数)
TARGETS = {
    "1.510880": ("上证红利ETF", "etf"),
    "1.515080": ("中证红利ETF", "etf"),
    "1.512890": ("红利低波ETF", "etf"),
    "1.512800": ("银行ETF", "etf"),
    "1.000922": ("中证红利(价格)", "index"),
    "2.H30269": ("红利低波(价格)", "index"),
    "2.399986": ("中证银行(价格)", "index"),
}


def fetch(secid: str, fqt: int = 2) -> pd.DataFrame:
    q = (f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
         f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
         f"&klt=101&fqt={fqt}&beg=0&end=20500101&lmt=100000")
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{URL}?{q}", headers=HDR)
            with urllib.request.urlopen(req, timeout=25) as r:
                o = json.loads(r.read().decode("utf-8", "replace"))
            d = o.get("data")
            if not d or not d.get("klines"):
                return pd.DataFrame()
            rows = [k.split(",") for k in d["klines"]]
            df = pd.DataFrame(rows, columns=[
                "date", "open", "close", "high", "low", "volume",
                "amount", "amplitude", "pct", "chg", "turn"])
            df["date"] = pd.to_datetime(df["date"])
            for c in ("open", "close", "high", "low", "volume", "amount"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


def main() -> None:
    frames = []
    for secid, (name, kind) in TARGETS.items():
        # fqt=2 后复权；指数本身就是价格指数，fqt 无实质影响
        adj = fetch(secid, 2)
        if adj.empty:
            print(f"  [WARN] {secid} {name} 抓取失败")
            continue
        raw = fetch(secid, 0) if kind == "etf" else pd.DataFrame()
        m = adj[["date", "close", "amount"]].rename(columns={"close": "close_adj"})
        if not raw.empty:
            m = m.merge(raw[["date", "close"]].rename(columns={"close": "close_raw"}),
                        on="date", how="left")
        else:
            m["close_raw"] = np.nan
        m = m.sort_values("date").reset_index(drop=True)

        s = m.set_index("date")["close_adj"].astype(float)
        r = s.pct_change().dropna()
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
        jumps = r[r.abs() > 0.15]
        flag = "" if not len(jumps) else f"  ⚠{len(jumps)} 处 >15% 跳动"
        print(f"  {name:16s} {len(m):5d} 日 {m['date'].iloc[0].date()} ~ "
              f"{m['date'].iloc[-1].date()} | CAGR {cagr:6.2%} "
              f"vol {r.std()*np.sqrt(252):6.2%} maxDD {(s/s.cummax()-1).min():7.1%}{flag}")
        m["code"], m["name"], m["kind"] = secid.split(".")[1], name, kind
        frames.append(m)
        time.sleep(0.3)

    if not frames:
        raise SystemExit("全部抓取失败")
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[out] {OUT}  {len(out)} 行 / {out['code'].nunique()} 个标的")


if __name__ == "__main__":
    main()
