# -*- coding: utf-8 -*-
"""从零构建行情数据包（腾讯行情公开接口，免 token、免费）。

原始数据包 `was_stock_data`（263 MB）不在代码包里，因为它压完仍有 260 MB，
无法随邮件或网页传输。若你拿不到那个包，可以用本脚本自己把**行情部分**拉下来。

    能拉：open/high/low/close(后复权) + volume + amount + 不复权收盘 + 4 条基准指数
    拉不到：逐日总股本(市值)、42 期财报、行业历史归属、退市名单
            —— 见《数据获取指南.md》，这四样必须由数据提供方给，缺失时系统跑不起来

用法::

    # 1. 先把轻数据包（约 1.2 MB，含 meta.csv / industry.npy / 退市名单）解压到位
    #    目录结构：  ../data_extract/was_stock_data/meta.csv

    # 2. 小样本试跑，确认网络通（拉 200 只，约 1 分钟）
    python tools/fetch_data_full.py --limit 200 --out ../data_extract/was_stock_data

    # 3. 全量（5281 只，16 线程约 15-25 分钟）
    python tools/fetch_data_full.py --workers 16

支持断点续传：每只股票拉完立刻落盘到 `_cache/`，中断后重跑只补缺的。

产物：
    panel_ohlcv.npz   dates/codes/open/high/low/close/volume/amount（后复权）
    raw_close.npz     rawclose（不复权收盘，实际成交价）
    indices.csv       hs300/csi500/csi1000/csi_all
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from qf import config as C  # noqa: E402

URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
INDEX_SYMBOLS = {"hs300": "sh000300", "csi500": "sh000905",
                 "csi1000": "sh000852", "csi_all": "sh000985"}
PAGE = 640          # 接口单次上限，超过要分段
AMOUNT_UNIT = 1e4   # 接口成交额单位是万元


def sym(code: str) -> str:
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def vol_mult(code: str) -> float:
    """成交量单位归一（复刻 akshare stock_zh_a_hist_tx 的规则）。"""
    s = sym(code)
    if s.startswith(("sh688", "sz399", "sh000", "sz000")):
        return 1.0
    return 100.0


def _get(params: dict, timeout: int = 12, tries: int = 3):
    full = URL + "?" + urllib.parse.urlencode(params)
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(full, headers=HDR)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                t = r.read().decode("utf-8", "replace")
            return json.loads(t[t.find("={"):].lstrip("="))
        except Exception as e:
            last = e
            time.sleep(0.25 * (k + 1) + np.random.rand() * 0.1)
    raise RuntimeError(str(last)[:120])


def _rows(code: str, start: str, end: str, adjust: bool):
    """取一段 K 线。adjust=True 取后复权，False 取不复权。

    注意：从未除权过的股票（如 688981），腾讯不返回 hfqday 而直接返回 day，
    两者等价。只读 hfqday 会把这些股票整只漏掉 —— 科创板会缺一大片。
    """
    s = code if code[:2] in ("sh", "sz") else sym(code)
    if adjust:
        j = _get({"_var": "kline_dayhfq", "r": "0.82",
                  "param": f"{s},day,{start},{end},{PAGE},hfq"})
    else:
        j = _get({"_var": "kline_day", "r": "0.82",
                  "param": f"{s},day,{start},{end},{PAGE},"})
    d = j.get("data")
    if isinstance(d, list):                  # 个别标的 / 极短区间会返回 list
        d = next((x for x in d if isinstance(x, dict) and s in x), {})
    d = (d or {}).get(s) if isinstance(d, dict) else {}
    if not isinstance(d, dict):
        return []
    return d.get("hfqday") or d.get("day") or []


def fetch_all(code: str, start: str, end: str, adjust: bool):
    """分页把 [start, end] 全区间拉完。

    接口单次最多 640 行且优先保留靠后的数据，所以从 end 往前推：
    每轮拿返回中最早的日期，减一天作为下一轮的 end，直到越过 start。
    """
    got: dict[str, list] = {}
    cur = end
    for _ in range(40):                      # 40 页 × 640 = 25600 天，足够十年
        rows = _rows(code, start, cur, adjust)
        if not rows:
            break
        for x in rows:
            if len(x) >= 6 and x[0] >= start:
                got[x[0]] = x
        earliest = min(x[0] for x in rows)
        if earliest <= start or len(rows) < PAGE:
            break
        cur = (pd.Timestamp(earliest) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return [got[k] for k in sorted(got)]


def parse(rows, code: str) -> tuple[list, list]:
    """把接口行拆成 (日期, [open,close,high,low,volume,amount])。"""
    mult = vol_mult(code)
    out = []
    for r in rows:
        if len(r) < 6:
            continue
        try:
            d = r[0]
            o, c, h, lo = float(r[1]), float(r[2]), float(r[3]), float(r[4])
            v = float(r[5]) * mult if r[5] not in ("", None) else np.nan
            a = np.nan
            if len(r) > 8 and r[8] not in ("", None):
                a = float(r[8]) * AMOUNT_UNIT
            elif np.isfinite(v):
                a = c * v
            # 腾讯对退市整理期个别股票返回负价，属脏值，丢弃
            if not (o > 0 and c > 0 and h > 0 and lo > 0):
                continue
            out.append((d[:10], (o, c, h, lo, v, a)))
        except (TypeError, ValueError):
            continue
    return out


def worker(code: str, start: str, end: str, cache: pathlib.Path) -> str:
    f = cache / f"{code}.npz"
    if f.exists():
        return "skip"
    try:
        hfq = parse(fetch_all(code, start, end, True), code)
        raw = parse(fetch_all(code, start, end, False), code)
        if not hfq:
            return "empty"
        np.savez(f, hfq=np.array([x[0] for x in hfq]),
                    hfqv=np.array([x[1] for x in hfq], dtype=np.float32),
                    raw=np.array([x[0] for x in raw]),
                    rawv=np.array([x[1] for x in raw], dtype=np.float32))
        return "ok"
    except Exception as e:
        return f"err:{type(e).__name__}:{str(e)[:60]}"


def load_codes(meta: pathlib.Path) -> list[str]:
    m = pd.read_csv(meta, dtype={"code": str})
    codes = m["code"].str.zfill(6).tolist()
    return sorted(set(codes))


def fetch_indices(start: str, end: str) -> pd.DataFrame:
    cols = {}
    for name, s in INDEX_SYMBOLS.items():
        rows = fetch_all(s, start, end, False)
        if not rows:
            print(f"  [WARN] 指数 {name} 无数据")
            continue
        ser = {r[0][:10]: float(r[2]) for r in rows if len(r) >= 6}
        cols[name] = ser
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df.reset_index()


def build(cache: pathlib.Path, codes: list[str], start: str, end: str, out: pathlib.Path):
    print("\n组装面板 ...", flush=True)
    dates, fields, raws = [], [], []
    used = []
    for code in codes:
        f = cache / f"{code}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        if len(z["hfq"]) == 0:
            continue
        dates.append(pd.to_datetime(z["hfq"]).values)
        fields.append(z["hfqv"])
        raws.append((pd.to_datetime(z["raw"]).values, z["rawv"]))
        used.append(code)
    if not used:
        raise SystemExit("没有任何股票数据，检查网络或 meta.csv 路径")

    all_dates = np.unique(np.concatenate(dates))
    # 剔除只有极少数股票有数据的非交易日（退市整理期的异常日期等）
    cnt = np.zeros(len(all_dates), dtype=np.int32)
    pos = {d: i for i, d in enumerate(all_dates)}
    for d in dates:
        cnt[[pos[x] for x in d]] += 1
    keep = cnt >= max(10, int(0.02 * len(used)))
    all_dates = all_dates[keep]
    dpos = {d: i for i, d in enumerate(all_dates)}
    print(f"  交易日 {len(all_dates)} 天  {str(all_dates[0])[:10]} ~ {str(all_dates[-1])[:10]}")

    T, N = len(all_dates), len(used)
    arr = {k: np.full((T, N), np.nan, dtype=np.float32) for k in
           ("open", "close", "high", "low", "volume", "amount")}
    rawc = np.full((T, N), np.nan, dtype=np.float32)
    for j, (code, d, v) in enumerate(zip(used, dates, fields)):
        ii = [dpos[x] for x in d if x in dpos]
        k = len(ii)
        arr["open"][ii, j] = v[:k, 0]
        arr["close"][ii, j] = v[:k, 1]
        arr["high"][ii, j] = v[:k, 2]
        arr["low"][ii, j] = v[:k, 3]
        arr["volume"][ii, j] = v[:k, 4]
        arr["amount"][ii, j] = v[:k, 5]
    for j, (d, v) in enumerate(raws):
        ii = [dpos[x] for x in d if x in dpos]
        k = len(ii)
        rawc[ii, j] = v[:k, 1]

    out.mkdir(parents=True, exist_ok=True)
    codes_arr = np.array([str(c).zfill(6) for c in used])
    np.savez(out / "panel_ohlcv.npz",
             dates=all_dates.astype("datetime64[D]"), codes=codes_arr, **arr)
    np.savez(out / "raw_close.npz", rawclose=rawc)
    cov = np.isfinite(arr["close"]).mean()
    print(f"  panel_ohlcv.npz  {T}×{N}  收盘覆盖 {cov:.1%}")
    print(f"  raw_close.npz    覆盖 {np.isfinite(rawc).mean():.1%}")
    return used


def build_mcap(cache: pathlib.Path, codes: list[str], meta: pathlib.Path,
               out: pathlib.Path):
    """用「不复权收盘 × 静态总股本」近似市值。

    ⚠️ 不是 PIT 市值：原始包用的是逐日总股本，这里只能用 meta.csv 里当前那一期。
       后果 —— 增发/回购当年市值会偏，市值中性化与 MIN/MAX_MCAP 过滤会有偏差。
       只建议在没有原始 market_cap.npz 时兜底，别用它复现回测数字。
    """
    m = pd.read_csv(meta, dtype={"code": str})
    m["code"] = m["code"].str.zfill(6)
    shares = dict(zip(m["code"], pd.to_numeric(m["shares"], errors="coerce")))
    z = np.load(out / "panel_ohlcv.npz", allow_pickle=True)
    raw = np.load(out / "raw_close.npz")["rawclose"]
    sh = np.array([shares.get(str(c), np.nan) for c in z["codes"]], dtype=np.float32)
    mcap = (raw * sh[None, :]).astype(np.float32)
    np.savez(out / "market_cap.npz", mcap=mcap)
    print(f"  market_cap.npz  覆盖 {np.isfinite(mcap).mean():.1%}（静态股本近似）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出目录，默认 $QF_ROOT/data_extract/was_stock_data")
    ap.add_argument("--start", default="2016-10-14")
    ap.add_argument("--end", default=None, help="默认今天")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只拉前 N 只，用于试跑")
    ap.add_argument("--mcap", action="store_true", help="额外生成近似市值（非 PIT）")
    ap.add_argument("--rebuild", action="store_true", help="清空缓存重拉")
    a = ap.parse_args()

    out = pathlib.Path(a.out) if a.out else (C.ROOT / "data_extract" / "was_stock_data")
    meta = out / "meta.csv"
    if not meta.exists():
        print(f"[FAIL] 找不到 {meta}")
        print("       meta.csv 在「轻数据包」里（约 1.2 MB），必须先从数据提供方拿到。")
        return 1
    end = a.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    codes = load_codes(meta)
    if a.limit:
        codes = codes[: a.limit]
    print(f"股票 {len(codes)} 只  区间 {a.start} ~ {end}  线程 {a.workers}")
    print(f"输出 {out}")

    cache = out / "_cache"
    if a.rebuild and cache.exists():
        for f in cache.glob("*.npz"):
            f.unlink()
    cache.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stat = {"ok": 0, "skip": 0, "empty": 0, "err": 0}
    errs = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(worker, c, a.start, end, cache): c for c in codes}
        for n, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            key = "err" if r.startswith("err") else r
            stat[key] = stat.get(key, 0) + 1
            if key == "err":
                errs.append((futs[fu], r))
            if n % 200 == 0 or n == len(codes):
                el = time.time() - t0
                eta = el / n * (len(codes) - n)
                print(f"  {n}/{len(codes)}  ok={stat['ok']} skip={stat['skip']} "
                      f"empty={stat['empty']} err={stat['err']}  "
                      f"已用 {el/60:.0f}分 预计剩 {eta/60:.0f}分", flush=True)

    if errs:
        print(f"\n失败 {len(errs)} 只（前 5 条）：")
        for c, e in errs[:5]:
            print(f"  {c}  {e}")
        print("  重跑本脚本会自动补这些（已成功的有缓存）")

    used = build(cache, codes, a.start, end, out)
    print("\n拉取指数 ...", flush=True)
    idx = fetch_indices(a.start, end)
    if len(idx):
        # 指数接口比个股行情早一步可查（当天收盘前就有当日行情），
        # 对齐到面板最后一个交易日，否则两者长度差一天，回测对齐会错位
        last = str(np.load(out / "panel_ohlcv.npz")["dates"][-1])
        idx = idx[idx["date"] <= last]
        idx.to_csv(out / "indices.csv", index=False)
        print(f"  indices.csv  {len(idx)} 行  {idx['date'].iloc[0]} ~ {idx['date'].iloc[-1]}")

    if a.mcap:
        print("\n生成近似市值 ...", flush=True)
        build_mcap(cache, used, meta, out)

    print(f"\n完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
    print("下一步： python tools/check_env.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
