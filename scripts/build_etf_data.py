# -*- coding: utf-8 -*-
"""ETF 日线数据管道：抓取 3 只宽基 ETF 的前复权/后复权日线，供右侧趋势模块使用。

设计要点（针对 ETF 特有问题）：
  * 用腾讯 newfqkline 接口，同时取 **qfq（前复权）** 与 **raw（不复权）**，
    二者比例可得每日复权因子 → 用后复权价算收益（含分红再投资），用不复权价算真实成交价/整手；
  * 成交额/成交量用于流动性过滤（512100 早期成交额低）；
  * 保存 build/etf/etf_panel.npz + etf_meta.json；
  * 增量更新：已有数据则只补最后日期之后的部分（--start 可控）。

用法：
    python scripts/build_etf_data.py                # 全量（2013 起）
    python scripts/build_etf_data.py --update       # 增量：重抓 30 天，按重叠复权系数接续

面板布局：`dates` + `codes` + 各字段 2D 矩阵（close[日期, 标的]），
**没有** close_510300 这种按标的的键——增量的读取逻辑必须按列取。
--update 会用重叠日估出「老 qfq 基准 → 新 qfq 基准」的缩放系数（qfq 以最新价为基准，
除息后整条历史等比缩放），只缩放价格列；系数不稳或重叠不足则自动回退全量重抓。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "build" / "etf"
PANEL = OUT_DIR / "etf_panel.npz"
META = OUT_DIR / "etf_meta.json"

# ETF 清单：代码 -> (名称, 跟踪指数, 上市日约)
ETFS = {
    "510300": ("沪深300ETF", "hs300", "2012-05-28"),
    "510500": ("中证500ETF", "csi500", "2013-03-15"),
    "512100": ("中证1000ETF", "csi1000", "2016-11-04"),
}
URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def sym(code: str) -> str:
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def _fetch_once(code: str, fq: str, start: str, end: str, count: int = 320) -> pd.DataFrame:
    """单次请求（接口每次最多约 320 行）。"""
    params = {"param": f"{sym(code)},day,{start},{end},{count}," + fq}
    url = URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=20) as r:
                txt = r.read().decode("utf-8", "replace")
            obj = json.loads(txt)
            data = obj.get("data", {}).get(sym(code), {})
            key = (fq + "day") if fq else "day"
            rows = data.get(key) or data.get("day") or []
            if not rows:
                return pd.DataFrame()
            recs = []
            for it in rows:
                recs.append({
                    "date": it[0],
                    "open": float(it[1]), "close": float(it[2]),
                    "high": float(it[3]), "low": float(it[4]),
                    "volume": float(it[5]) if len(it) > 5 and it[5] else np.nan,
                    "amount": float(it[8]) * 1e4 if len(it) > 8 and it[8] else np.nan,
                })
            df = pd.DataFrame(recs)
            df["date"] = pd.to_datetime(df["date"])
            return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame()


def fetch(code: str, fq: str, start: str, end: str) -> pd.DataFrame:
    """分段拉取（按 ~400 自然日窗口），覆盖 start~end 全区间。"""
    s0 = pd.Timestamp(start)
    e0 = pd.Timestamp(end)
    chunks = []
    cur = s0
    while cur <= e0:
        seg_end = min(cur + pd.Timedelta(days=400), e0)
        df = _fetch_once(code, fq, cur.strftime("%Y-%m-%d"), seg_end.strftime("%Y-%m-%d"))
        if not df.empty:
            chunks.append(df[df["date"] >= cur])
        cur = seg_end + pd.Timedelta(days=1)
        time.sleep(0.15)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


KEEP = ("open", "high", "low", "close", "raw_close", "volume", "amount")


def _build_m(adj: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """qfq 与不复权合并，得到日线表（含 raw_close）。"""
    m = pd.merge(adj[["date", "open", "close", "high", "low", "volume", "amount"]],
                 raw[["date", "close"]].rename(columns={"close": "raw_close"}),
                 on="date", how="left")
    return m.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def _merge_on_basis(old: pd.DataFrame, new: pd.DataFrame, code: str,
                    tol: float = 2e-3):
    """把老面板（旧 qfq 基准）接到新抓窗口（新 qfq 基准）之前。

    qfq 以最新价为基准，除息/分红后**整条历史会等比缩放**。直接 concat 老历史与新
    窗口会在接缝处产生阶梯跳变（假收益）。做法：用重叠日估出老→新的缩放系数 c，
    只缩放价格列（raw_close/volume/amount 不动），再接上新窗口。
    重叠不足或 c 不稳定 → 返回 None，调用方回退全量重抓。

    2026-11-02 注：原 --update 分支算了 existing 却从未使用，等于把历史全清成 NaN。
    """
    ov = old.index.intersection(pd.DatetimeIndex(new["date"]))
    if len(ov) < 3:
        return None
    o = old.loc[ov, "close"].to_numpy(dtype=np.float64)
    n = new.set_index("date").loc[ov, "close"].to_numpy(dtype=np.float64)
    ok = np.isfinite(o) & np.isfinite(n) & (o > 0) & (n > 0)
    if ok.sum() < 3:
        return None
    ratio = n[ok] / o[ok]
    c = float(np.median(ratio))
    resid = float(np.max(np.abs(ratio / c - 1.0)))
    if not np.isfinite(c) or c <= 0 or resid > tol:
        print(f"  [warn] {code} 复权基准漂移不稳（c={c:.6f}, 残差={resid:.2e}）")
        return None
    scaled = old.copy()
    for k in ("open", "high", "low", "close"):
        scaled[k] = scaled[k] * c
    merged = pd.concat([scaled, new.set_index("date")])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    print(f"  {code} 增量合并：重叠 {len(ov)} 日，复权系数 {c:.6f}，"
          f"残差 {resid:.2e}，历史 {len(old)} → {len(merged)} 日")
    return merged.reset_index().rename(columns={"index": "date"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2013-01-01")
    ap.add_argument("--update", action="store_true", help="只抓最新 30 天并接在旧面板前")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    old: dict[str, pd.DataFrame] = {}
    if args.update and PANEL.exists():
        z = np.load(PANEL, allow_pickle=True)
        old_dates = pd.DatetimeIndex(z["dates"])
        # 面板是「2D 矩阵 + codes」布局（close[日期, 标的]），不是 close_510300 这种键
        old_codes = [str(c) for c in z["codes"]]
        for j, c in enumerate(old_codes):
            cols = {k: z[k][:, j] for k in KEEP if k in z.files}
            if "close" in cols:
                old[c] = pd.DataFrame(cols, index=old_dates)
        start = (old_dates[-1] - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        got = ", ".join(f"{c}({len(old[c])}日)" for c in old) or "无"
        print(f"[增量] 已有 {len(old_dates)} 日（至 {old_dates[-1].date()}）：{got}")
        print(f"       重抓 {start} 起做重叠校验，按复权系数接续")
    else:
        start = args.start
        print(f"[全量] 从 {start} 开始")

    frames = {}
    for code, (name, index_name, list_date) in ETFS.items():
        s = max(start, list_date)
        adj = fetch(code, "qfq", s, end)
        raw = fetch(code, "", s, end)
        if adj.empty or raw.empty:
            print(f"  [WARN] {code} {name} 抓取失败（adj={len(adj)} raw={len(raw)}）")
            continue
        m = _build_m(adj, raw)
        if code in old:
            merged = _merge_on_basis(old[code], m, code)
            if merged is None:
                print(f"  [warn] {code} 回退全量重抓（重叠校验不通过）")
                s_full = max(args.start, list_date)
                adj = fetch(code, "qfq", s_full, end)
                raw = fetch(code, "", s_full, end)
                if adj.empty or raw.empty:
                    print(f"  [WARN] {code} {name} 全量重抓失败")
                    continue
                m = _build_m(adj, raw)
            else:
                m = merged
        # 复权因子：qfq 相对不复权的比例（qfq 是"以最新价为基准"的前复权，比例逐日变化）
        m["adj_ratio"] = m["raw_close"] / m["close"]
        frames[code] = m
        print(f"  {code} {name}: {len(m)} 日 {m['date'].iloc[0].date()} ~ {m['date'].iloc[-1].date()}"
              f" | 成交额中位 {m['amount'].median()/1e8:.2f} 亿")

    if not frames:
        raise SystemExit("所有 ETF 抓取失败")

    # 统一日期轴（并集；未上市保持 NaN）
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(f["date"]) for f in frames.values()])))
    T, N = len(all_dates), len(ETFS)
    codes = list(ETFS)
    arr = {k: np.full((T, N), np.nan, dtype=np.float64) for k in KEEP}
    for j, c in enumerate(codes):
        if c not in frames:
            continue
        f = frames[c].set_index("date").reindex(all_dates)
        for k in KEEP:
            arr[k][:, j] = f[k].to_numpy()

    payload = {"dates": all_dates.values, "codes": np.array(codes)}
    for k, v in arr.items():
        payload[k] = v.astype(np.float32)
    np.savez_compressed(PANEL, **payload)
    META.write_text(json.dumps({
        "etfs": {c: {"name": ETFS[c][0], "index": ETFS[c][1], "list_date": ETFS[c][2]} for c in codes},
        "rows": int(T), "start": str(all_dates[0].date()), "end": str(all_dates[-1].date()),
        "price_note": "close=qfq(前复权,含分红), raw_close=不复权, 收益率用 close 计算",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {PANEL}（{T} 日 × {N} 只，{all_dates[0].date()} ~ {all_dates[-1].date()}）")
    print(f"→ {META}")


if __name__ == "__main__":
    main()
