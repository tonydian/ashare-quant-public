# -*- coding: utf-8 -*-
"""行情数据增量更新（腾讯行情源，与原始数据包同源同口径）。

原始数据包 `data_extract/was_stock_data` 截止 2026-10-14。本脚本从腾讯行情接口
（与数据包构建同源：后复权价/成交量/成交额逐点一致，已用重叠区间校验）拉取最新
交易日数据，appending 到面板末尾：

    源 = 原始包 → 输出 data_extract/was_stock_data_ext（首建）
    源 = ext    → 原地更新 ext（此后常态，见 default_out_for）

> 2026-11-02 修复：原实现固定把输出派生到 `_ext`，而 `C.DATA_RAW` 会自动选中 ext，
> 于是 SRC == out，第 5 步 `shutil.copy2(SRC/f, out/f)` 自我拷贝抛
> `PermissionError[WinError 32]`，进程非 0 退出，`indices.csv` 与 `UPDATE_INFO.txt`
> 未落盘 —— 个股面板已到最新、指数停在上周（bench 被 ffill 成"零涨跌"）。

更新内容：
    panel_ohlcv.npz   open/high/low/close(后复权) + volume/amount
    raw_close.npz     不复权收盘（用缩放价近似，口径见代码注释）
    market_cap.npz    总市值（用最近一期总股本近似）
    industry.npy      行业矩阵（新行沿用最后一行）
    indices.csv       hs300/csi500/csi1000/csi_all 基准

用法:
    python scripts/update_data.py [--out data_extract/was_stock_data_ext] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qf import config as C  # noqa: E402

SRC = C.DATA_RAW
URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

INDEX_SYMBOLS = {"hs300": "sh000300", "csi500": "sh000905",
                 "csi1000": "sh000852", "csi_all": "sh000985"}


def sym(code: str) -> str:
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _get(params, timeout=12, url=None):
    """调用腾讯行情 JSONP 接口（标准库 urllib，线程安全，无第三方依赖）。"""
    last = None
    full = (url or URL) + "?" + urllib.parse.urlencode(params)
    for k in range(3):
        try:
            req = urllib.request.Request(full, headers=HDR)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                t = r.read().decode("utf-8", "replace")
            pos = t.find("={")
            return json.loads(t[pos + 1:])
        except Exception as e:
            last = e
            time.sleep(0.25 * (k + 1) + np.random.rand() * 0.1)
    raise RuntimeError(str(last)[:120])


def fetch_stock(code: str, start: str, end: str):
    s = sym(code)
    j = _get({"_var": "kline_dayhfq", "r": "0.82",
              "param": f"{s},day,{start},{end},640,hfq"})
    d = j.get("data", {}).get(s, {})
    return d.get("hfqday") or []


def fetch_index(s: str, start: str, end: str):
    try:
        j = _get({"_var": "kline_day", "r": "0.82",
                  "param": f"{s},day,{start},{end},640,"})
        d = j.get("data", {}).get(s, {})
        rows = d.get("day") or []
        if rows:
            return rows
    except Exception:
        pass
    j = _get({"r": "0.82", "param": f"{s},day,{start},{end},320,"},
             url="https://web.ifzq.gtimg.cn/appstock/app/fqkline/get")
    d = j.get("data", {}).get(s, {})
    return d.get("day") or []


def default_out_for(src: pathlib.Path) -> pathlib.Path:
    """输出目录：源是原始包 → 派生 _ext；源本身就是 _ext → 原地更新。

    C.DATA_RAW 会自动选行数更多的目录，一旦 ext 建好，它就等于 ext；此时若仍把
    输出派生到 ext，源与目标重合，第 5 步的 shutil.copy2/copytree 会自我拷贝，
    Windows 抛 PermissionError[WinError 32]（2026-10-29、10-30 两次更新失败即此）。
    """
    return src if src.name.endswith("_ext") else src.parent / ("was_stock_data_ext")


def merge_indices(idx_old: pd.DataFrame, idx_new: dict, extra_dates) -> tuple:
    """把 missing 日期的 4 条指数收盘价并入旧表；返回 (合并后表, 实际补上的日期)。

    只补 4 条指数都有值的日期，缺任何一条就跳过（宁缺勿造）。
    """
    add = {}
    for d in extra_dates:
        d = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
        row, ok = {}, True
        for name in INDEX_SYMBOLS:
            ser = idx_new.get(name)
            if ser is None or d not in ser.index:
                ok = False
                break
            v = float(ser.loc[d])
            if not np.isfinite(v):
                ok = False
                break
            row[name] = v
        if ok:
            add[d] = row
    if not add:
        return idx_old.copy(), []
    add_df = pd.DataFrame(add).T.reindex(columns=idx_old.columns)
    all_df = pd.concat([idx_old, add_df])
    all_df.index = pd.DatetimeIndex(all_df.index)
    all_df = all_df[~all_df.index.duplicated(keep="last")].sort_index()
    return all_df, sorted(add)


def sync_bond_gold(out: pathlib.Path, src: pathlib.Path, window_days: int = 20) -> dict:
    """增量更新 国债指数(000012) / 黄金ETF(518880) 日线。

    口径与原始数据包逐点对齐（重叠日 close 相对误差 <1e-6 才写）：
      * bond_000012.parquet：open/high/low/close = 上证国债指数原始点位；
        volume = 腾讯 vol × M，M 由重叠日实测（历史文件是 ×100）；
      * gold_518880.parquet：close = **不复权**收盘（与历史文件一致）。
    """
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    res = {}
    # ---- 国债指数
    p = out / "bond_000012.parquet"
    try:
        old = pd.read_parquet(p if p.exists() else src / "bond_000012.parquet")
        old["date"] = pd.to_datetime(old["date"])
        old = old.sort_values("date").reset_index(drop=True)
        start = (old["date"].iloc[-1] - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
        rows = fetch_index("sh000012", start, end)
        if not rows:
            raise RuntimeError("sh000012 无返回")
        new = pd.DataFrame([{
            "date": pd.Timestamp(r[0][:10]),
            "open": float(r[1]), "close": float(r[2]),
            "high": float(r[3]), "low": float(r[4]),
            "volume": float(r[5]) if len(r) > 5 and r[5] else np.nan,
        } for r in rows if len(r) >= 6])
        ov = new.merge(old, on="date", suffixes=("_n", "_o"))
        if len(ov) >= 3:
            # 腾讯指数只给 2 位小数，历史文件保留 3 位 → 相对误差上限 ~2.4e-5（即 0.0024%）
            diff = np.nanmax(np.abs(ov["close_n"] / ov["close_o"] - 1))
            assert diff < 5e-5, f"国债指数与历史数据不一致：{diff:.2e}"
            res["bond_diff"] = f"{diff:.1e}"
            m = float(np.nanmedian(ov["volume_o"] / ov["volume_n"]))
            if not np.isfinite(m) or m <= 0:
                m = 100.0
        else:
            m = 100.0
        new["volume"] = new["volume"] * m
        add = new[~new["date"].isin(old["date"])]
        if len(add):
            merged = pd.concat([old, add[old.columns]], ignore_index=True) \
                .sort_values("date").drop_duplicates("date", keep="last")
            merged.to_parquet(p, index=False)
            res["bond"] = f"+{len(add)} 日 → {merged['date'].iloc[-1].date()}"
        else:
            res["bond"] = f"已最新（{old['date'].iloc[-1].date()}）"
    except Exception as e:
        res["bond"] = f"[失败] {type(e).__name__}: {str(e)[:80]}"
    # ---- 黄金 ETF（不复权）
    p = out / "gold_518880.parquet"
    try:
        old = pd.read_parquet(p if p.exists() else src / "gold_518880.parquet")
        old["date"] = pd.to_datetime(old["date"])
        old = old.sort_values("date").reset_index(drop=True)
        start = (old["date"].iloc[-1] - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
        rows = fetch_index("sh518880", start, end)
        if not rows:
            raise RuntimeError("sh518880 无返回")
        new = pd.DataFrame([{"date": pd.Timestamp(r[0][:10]), "close": float(r[2])}
                            for r in rows if len(r) >= 6])
        ov = new.merge(old, on="date", suffixes=("_n", "_o"))
        if len(ov) >= 3:
            diff = np.nanmax(np.abs(ov["close_n"] / ov["close_o"] - 1))
            assert diff < 1e-6, f"黄金 ETF 与历史数据不一致：{diff:.2e}"
        add = new[~new["date"].isin(old["date"])]
        if len(add):
            merged = pd.concat([old, add[old.columns]], ignore_index=True) \
                .sort_values("date").drop_duplicates("date", keep="last")
            merged.to_parquet(p, index=False)
            res["gold"] = f"+{len(add)} 日 → {merged['date'].iloc[-1].date()}"
        else:
            res["gold"] = f"已最新（{old['date'].iloc[-1].date()}）"
    except Exception as e:
        res["gold"] = f"[失败] {type(e).__name__}: {str(e)[:80]}"
    return res


def sync_etf_panel(target_last: pd.Timestamp) -> str:
    """ETF 面板（3 只宽基，供右侧趋势模块）增量刷新到 target_last。"""
    panel = C.BUILD / "etf" / "etf_panel.npz"
    try:
        if panel.exists():
            last = pd.Timestamp(np.load(panel, allow_pickle=True)["dates"][-1])
            if last >= target_last:
                return f"已最新（{last.date()}）"
    except Exception:
        pass
    script = pathlib.Path(__file__).resolve().parent / "build_etf_data.py"
    try:
        cp = subprocess.run([sys.executable, "-u", str(script), "--update"],
                            cwd=str(script.parent.parent), capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=900)
    except Exception as e:
        return f"[失败] {type(e).__name__}: {str(e)[:80]}"
    if cp.returncode != 0:
        tail = (cp.stderr or cp.stdout or "").strip().splitlines()[-3:]
        return f"[失败] 退出码 {cp.returncode}：{' / '.join(tail)[:160]}"
    lines = [l for l in (cp.stdout or "").splitlines() if "→ " in l]
    return lines[-1].replace("→ ", "").strip() if lines else "完成"


def vol_mult(code: str) -> float:
    """复刻 akshare stock_zh_a_hist_tx 的成交量单位归一规则。"""
    s = sym(code)
    if s.startswith(("sh688", "sz399", "sh000", "sz000")):
        return 1.0
    return 100.0


def _mem_mb() -> str:
    """当前进程内存占用（MB）。psutil 没装就用 Win32 API，都没有就返回空串。"""
    try:
        import psutil
        return f"{psutil.Process().memory_info().rss / 1e6:.0f}MB"
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            # 0x0400 = PROCESS_QUERY_INFORMATION，0x1000 = LIMITED
            h = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x1000, False, os.getpid())
            if h:
                try:
                    if ctypes.windll.psapi.GetProcessMemoryInfo(
                            h, ctypes.byref(pmc), pmc.cb):
                        return f"{pmc.WorkingSetSize / 1e6:.0f}MB"
                finally:
                    ctypes.windll.kernel32.CloseHandle(h)
        except Exception:
            pass
    return ""


def _replace_with_retry(tmp: pathlib.Path, path: pathlib.Path, tries: int = 6) -> None:
    """os.replace 带重试：Windows 上被别的进程短时占用（杀软扫描/服务重载）会失败。"""
    last = None
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last = e
            time.sleep(0.5 * (k + 1))
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    raise PermissionError(
        f"无法替换 {path.name}：目标文件被其他进程占用（通常是正在运行的网页服务）。"
        f"请先停止网页服务再更新，或改用 --out 指定另一个目录。原始异常：{last}")


def _atomic_savez(path: pathlib.Path, **arrays) -> None:
    """先写 .tmp 再原子替换。

    `np.savez(目标)` 会以 "wb" 打开目标文件，一旦在写 300MB 面板的过程中被中断
    （系统杀进程 / 断电 / 磁盘满），原文件已经被截断，等于把可用数据写坏。
    2026-11-03 那次卡死就停在保存前一步，只是运气好没踩到。
    """
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    np.savez(tmp, **arrays)
    _replace_with_retry(tmp, path)


def _atomic_save_npy(path: pathlib.Path, arr) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    np.save(tmp, arr)
    _replace_with_retry(tmp, path)


def _atomic_to_csv(df, path: pathlib.Path) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    df.to_csv(tmp)
    _replace_with_retry(tmp, path)


def _repair_extract(t0_day: pd.Timestamp, out: pathlib.Path) -> None:
    """用东财前复权重算增量段的 OHLC。

    为什么需要（2026-11-10 血泪）
    ----------------------------
    腾讯 hfq 的日收益率与真实成交价背离，且幅度因股而异（实测逐股斜率中位 0.94、
    p10 0.75、极端 2.6）。最直观的反例：000635 在 2026-11-05 被记成 +12.10%，
    而深市主板涨停上限是 10%，物理上不可能。

    本函数拉东财前复权（已吸收送股/转增/分红，除权日不跳空），以 T0 的后复权收盘
    为锚点重算增量段。失败只告警，不中断更新。

    注意：原先的 ret_diffs 校验只比对**重叠日**，等于只检查了昨天；新增段永远没有
    参照物，所以这种失真可以无声无息地累积。前复权重算就是补上这个参照物。
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    py = sys.executable
    print("[7/7] 用东财前复权重算增量段（腾讯 hfq 收益与真实成交价背离）...")
    try:
        r = subprocess.run([py, str(root / "tools" / "fetch_fix_panel.py")], cwd=str(root))
        if r.returncode != 0:
            raise RuntimeError("抓取前复权失败")
        r = subprocess.run([py, str(root / "tools" / "repair_ext_panel.py"),
                            "--no-backup",
                            "--t0", pd.Timestamp(t0_day).strftime("%Y-%m-%d")],
                           cwd=str(root))
        if r.returncode != 0:
            raise RuntimeError("重算失败")
    except Exception as e:                                    # noqa: BLE001
        print(f"    [warn] 增量段修正未完成：{e}")
        print("    [warn] 请手动执行 tools/repair_ext_panel.py，否则新增段收益不可信")


def main():
    ap = argparse.ArgumentParser()
    # 输出目录：见 default_out_for（源是 ext 时原地更新，否则派生 _ext；
    # 不通地固定派生 _ext 是为了避免 was_stock_data_ext_ext 这种反复累后缀）
    ap.add_argument("--out", default=str(default_out_for(SRC)))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--start", default=None,
                    help="抓取窗口起点（缺省=原始数据最后交易日往前 60 天，需覆盖重叠校验日）")
    ap.add_argument("--no-repair", action="store_true",
                    help="跳过增量段的收益口径修正（默认会用东财前复权重算，见 _repair_extract）")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inplace = out.resolve() == pathlib.Path(SRC).resolve()
    if inplace:
        print(f"    原地更新模式：{out}（面板已读入内存，跳过固定文件复制）")

    def sync_aux(last_day: pd.Timestamp):
        """[6/6] 国债/黄金/ETF 面板：与个股面板分开的数据源，各自增量。"""
        print("[6/6] 同步国债指数/黄金ETF/ETF面板 ...")
        for k, v in sync_bond_gold(out, SRC).items():
            print(f"    {k}: {v}")
        print(f"    etf_panel: {sync_etf_panel(last_day)}")

    print("[1/6] 读取原始面板 ...")
    z = np.load(SRC / "panel_ohlcv.npz")
    dates = pd.DatetimeIndex(z["dates"])
    codes = np.array([str(c) for c in z["codes"]])
    panels = {k: z[k] for k in ["open", "high", "low", "close", "volume", "amount"]}
    # 必须显式关掉：NpzFile 会一直持有 panel_ohlcv.npz 的文件句柄，而原地更新时
    # 输出路径就是这个文件。Windows 上仍被打开的句柄没有 FILE_SHARE_DELETE 权限，
    # 第 5 步 os.replace 会抛 PermissionError[WinError 5]（2026-11-04 实测）。
    try:
        z.close()
    except Exception:
        pass
    del z
    raw = np.load(SRC / "raw_close.npz")["rawclose"]
    mcap = np.load(SRC / "market_cap.npz")["mcap"]
    industry = np.load(SRC / "industry.npy")
    print(f"    {len(dates)} 交易日 × {len(codes)} 只，最后 {dates[-1].date()}")
    start = args.start or (dates[-1] - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    print(f"    抓取窗口起点: {start}")

    # ---------------- 指数：确定新交易日
    print("[2/6] 拉取基准指数 ...")
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    idx_old = pd.read_csv(SRC / "indices.csv", index_col=0, parse_dates=True)
    idx_old.index = pd.DatetimeIndex(idx_old.index)
    idx_new = {}
    new_dates = None
    for name, s in INDEX_SYMBOLS.items():
        rows = fetch_index(s, start, end)
        df = pd.DataFrame(rows)
        if df.empty:
            raise SystemExit(f"指数 {s} 拉取失败")
        df = df.iloc[:, :6]
        df.columns = ["date", "open", "close", "high", "low", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.sort_values("date")
        # 与已有数据校验重叠日
        old = idx_old[name]
        ov = df[df["date"].isin(old.index)]
        if len(ov):
            diff = np.nanmax(np.abs(ov["close"].to_numpy() / old.reindex(ov["date"]).to_numpy() - 1))
            assert diff < 1e-6, f"指数 {name} 与原始数据不一致：{diff}"
        idx_new[name] = df.set_index("date")["close"]
        cand = df[df["date"] > dates[-1]]["date"]
        if new_dates is None:
            new_dates = list(cand)
        else:
            assert new_dates == list(cand), "指数交易日历不一致"
    print(f"    新增交易日 {len(new_dates)} 个：{[str(d.date()) for d in new_dates]}")
    # 面板已有、indices.csv 缺失的交易日（历史失败留下的缺口，必须补齐）
    idx_gap = [d for d in dates if d not in idx_old.index]
    if idx_gap:
        print(f"    [warn] indices.csv 缺 {len(idx_gap)} 个面板交易日："
              f"{[str(pd.Timestamp(d).date()) for d in idx_gap]}")
    if not new_dates:
        if not idx_gap:
            # 数据已是最新不是失败：正常退出（exit 0），否则管道会把"无需更新"判成 error 并中止
            print("没有新增交易日（数据已是最新），跳过行情更新")
            sync_aux(dates[-1])          # 个股不动，但附属数据可能落后
            return
        # 只补指数缺口，不动个股面板
        idx_fix, filled = merge_indices(idx_old, idx_new, idx_gap)
        if not filled:
            raise SystemExit(f"面板领先 indices.csv {len(idx_gap)} 天，但抓取窗口内取不到对应指数收盘价")
        _atomic_to_csv(idx_fix, out / "indices.csv")
        (out / "UPDATE_INFO.txt").write_text(
            "由 scripts/update_data.py 修复 indices.csv 缺口\n"
            f"数据目录：{SRC}\n补齐交易日：{', '.join(str(d.date()) for d in filled)}\n"
            f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n",
            encoding="utf-8")
        left = [d for d in idx_gap if d not in idx_fix.index]
        print(f"    已补 {len(filled)} 天指数：{[str(d.date()) for d in filled]}"
              f"{'，仍缺 ' + str(len(left)) + ' 天' if left else ''}")
        sync_aux(dates[-1])
        return
    nd = pd.DatetimeIndex(new_dates)

    # ---------------- 个股
    print(f"[3/6] 拉取 {len(codes)} 只个股（workers={args.workers}）...")
    t0 = time.time()
    results: dict[int, dict] = {}
    overlaps: dict[int, dict] = {}
    new_set = {d.strftime("%Y-%m-%d") for d in nd}
    panel_set = {d.strftime("%Y-%m-%d") for d in dates}

    def task(j):
        rows = fetch_stock(str(codes[j]), start, end)
        keep, ov = {}, {}
        for r in rows:
            if len(r) < 6:
                continue
            d = str(r[0])[:10]
            try:
                rec = {
                    "open": float(r[1]), "close": float(r[2]),
                    "high": float(r[3]), "low": float(r[4]),
                    "volume": float(r[5]),
                    "amount": float(r[8]) * 10000.0 if len(r) > 8 and str(r[8]) else np.nan,
                }
            except (ValueError, TypeError):
                continue
            if d in new_set:
                keep[d] = rec
            elif d in panel_set:
                ov[d] = rec
        return j, keep, ov

    done = 0
    n_err = 0
    aborted = False
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, j): j for j in range(len(codes))}
        for fut in as_completed(futs):
            try:
                j, keep, ov = fut.result()
                results[j] = keep
                overlaps[j] = ov
            except Exception:
                n_err += 1
            done += 1
            if done % 1000 == 0:
                print(f"    {done}/{len(codes)}，{time.time()-t0:.0f}s，"
                      f"有数据 {len(results)}，网络失败 {n_err}", flush=True)
            # 熔断：接口整体不可用时不要一路重试到底。
            # 每只股票最多 3 次重试 × 12s 超时，5281 只 / 16 线程 ≈ 3.7 小时，
            # 期间页面只会显示"拉取中"，和卡死无法区分。
            if done >= 400 and n_err / done > 0.5:
                aborted = True
                print(f"    [熔断] 前 {done} 只有 {n_err} 只网络失败（{n_err/done:.0%}），"
                      f"判定行情接口不可用，提前终止。行情源可能异常或本机网络不通。",
                      flush=True)
                for f in futs:
                    f.cancel()
                break
    if aborted:
        raise SystemExit("行情接口不可用（失败率 > 50%），本次更新已中止，数据未改动。")
    n_ok = sum(1 for v in results.values() if v)
    print(f"    完成：{n_ok}/{len(codes)} 只有新增数据，用时 {time.time()-t0:.0f}s")

    # ---------------- 构建扩展数组
    # 内存：6 个面板 × 2441×5281 × 4B ≈ 310MB；新旧两套同时驻留会翻倍到 620MB，
    # 再加上 raw/mcap/industry 前后各一份，峰值能到 900MB+。16GB 机器上网页服务
    # 自己也占了几百 MB，容易触发换页甚至被 OOM 杀掉（2026-11-03 卡在 [4/6] 即此处）。
    # 做法：拷完一个面板就立刻释放旧的，只保留 close 供下面逐股校验使用。
    print(f"[4/6] 构建扩展面板 ...（当前内存 {_mem_mb() or 'n/a'}）")
    T0, N = dates.shape[0], len(codes)
    Tn = T0 + len(nd)
    ext = {k: np.full((Tn, N), np.nan, dtype=np.float32) for k in panels}
    for k in list(panels):
        ext[k][:T0] = panels[k]
        if k != "close":
            del panels[k]          # open/high/low/volume/amount 拷完即可释放
    raw_ext = np.full((Tn, N), np.nan, dtype=np.float32)
    raw_ext[:T0] = raw
    mcap_ext = np.full((Tn, N), np.nan, dtype=np.float32)
    mcap_ext[:T0] = mcap
    ind_ext = np.full((Tn, N), -1, dtype=np.int16)
    ind_ext[:T0] = industry
    print(f"    扩展数组就绪 {Tn}×{N}（当前内存 {_mem_mb() or 'n/a'}）")

    # 每只股票：最近有效日、缩放系数 C（TX hfq 与面板一致，仍做校验性缩放）、
    # 不复权/复权比、总股本
    # 注意：这里必须整块扫描。原来写的是 `for j: np.flatnonzero(isfinite(close[:, j]))`，
    # 等于按列遍历一个 51MB 的 C 序数组（2441 步长跳读），5281 次全列扫描在内存紧张时
    # 全是缺页，实测把 [4/6] 这一段拖到 130 秒 / 1000 只。改成一次性 reverse-argmax。
    close = panels["close"]
    _fin = np.isfinite(close)
    _rev = _fin[::-1]
    _argm = _rev.argmax(axis=0)
    _has = _rev[_argm, np.arange(N)]
    last_i = np.where(_has, T0 - 1 - _argm, -1).astype(np.int64)
    del _fin, _rev, _has, _argm

    # 日期→行号 只建一次。原来每只股票、每个重叠日都 `pd.Timestamp(dstr)` + `dates.get_loc`，
    # 5281 × ~40 次 ≈ 21 万次串行调用，是无谓开销。
    date_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    mults = np.array([vol_mult(str(c)) for c in codes], dtype=np.float64)
    print(f"    预处理完成（当前内存 {_mem_mb() or 'n/a'}）", flush=True)

    n_missing = 0
    n_scale_ok = n_scale_bad = 0
    ret_diffs = []
    t_loop = time.time()
    for j in range(N):
        if j and j % 1000 == 0:
            print(f"    {j}/{N}，{time.time()-t_loop:.0f}s，"
                  f"内存 {_mem_mb() or 'n/a'}", flush=True)
        keep = results.get(j) or {}
        if not keep:
            n_missing += 1
            continue
        li = last_i[j]
        Cj = 1.0
        rshare = np.nan
        r_ratio = np.nan
        if li >= 0:
            # 用重叠日校验并估计缩放（数据包与腾讯同源时应≈1）
            ov = overlaps.get(j) or {}
            for dstr in sorted(ov.keys()):
                i = date_idx.get(dstr)
                if i is None:
                    continue
                v = float(close[i, j])
                if not np.isfinite(v):
                    continue
                tx = ov[dstr]["close"]
                if tx > 0:
                    Cj = v / tx
                    if abs(Cj - 1.0) < 1e-3:
                        n_scale_ok += 1
                    else:
                        n_scale_bad += 1
            # 重叠收益率一致性
            ov_dates = sorted(d for d in ov.keys() if d in panel_set)
            if len(ov_dates) >= 2:
                ii = [date_idx[d] for d in ov_dates]
                pa = close[ii, j].astype(np.float64)
                tx = np.array([ov[d]["close"] for d in ov_dates], dtype=np.float64)
                m = np.isfinite(pa) & (pa > 0) & (tx > 0)
                if m.sum() >= 3:
                    d1 = np.diff(np.log(pa[m]))
                    d2 = np.diff(np.log(tx[m]))
                    ret_diffs.append(float(np.max(np.abs(d1 - d2))))
            _c = float(close[li, j])
            r_ratio = (raw[li, j] / _c
                       if np.isfinite(raw[li, j]) and _c > 0 else np.nan)
            if np.isfinite(mcap[li, j]) and np.isfinite(raw[li, j]) and raw[li, j] > 0:
                rshare = float(mcap[li, j] / raw[li, j])
        else:
            r_ratio = np.nan
        for k, d in enumerate(nd):
            row = keep.get(d.strftime("%Y-%m-%d"))
            if row is None:
                continue
            t = T0 + k
            mult = mults[j]
            for fld in ("open", "high", "low", "close"):
                ext[fld][t, j] = row[fld] * Cj
            ext["volume"][t, j] = row["volume"] * mult
            ext["amount"][t, j] = row["amount"]
            if np.isfinite(r_ratio):
                adj_factor = r_ratio
            else:
                adj_factor = 1.0
            raw_est = row["close"] * Cj * adj_factor
            raw_ext[t, j] = raw_est
            if np.isfinite(rshare) and rshare > 0:
                mcap_ext[t, j] = raw_est * rshare
    # 行业：新行沿用最后一行
    ind_ext[T0:] = industry[-1:]
    # 保存前先释放旧数组，给写 npz 腾出内存
    del close, raw, mcap, industry, panels, results, overlaps
    print(f"    新增行填充完成；无新增数据的股票 {n_missing} 只（退市/停牌/接口缺失），"
          f"内存 {_mem_mb() or 'n/a'}", flush=True)
    if ret_diffs:
        rd = np.array(ret_diffs)
        print(f"    重叠区间收益率一致性：n={len(rd)}，p50={np.median(rd):.2e}，"
              f"p99={np.percentile(rd, 99):.2e}，max={rd.max():.2e}")
        if np.percentile(rd, 99) > 1e-3:
            print("    [warn] 重叠收益率偏差偏大，请检查数据源口径")
    tot = n_scale_ok + n_scale_bad
    if tot:
        print(f"    重叠日缩放系数≈1 的比例：{n_scale_ok/tot:.4%}")

    # ---------------- 保存
    print(f"[5/6] 保存新数据目录 ...（写入中，约 30~60 秒）", flush=True)
    _atomic_savez(out / "panel_ohlcv.npz",
                  dates=np.array(list(dates) + list(nd), dtype="datetime64[ns]"),
                  codes=codes.astype("<U6"),
                  **{k: v for k, v in ext.items()})
    _atomic_savez(out / "raw_close.npz", rawclose=raw_ext)
    _atomic_savez(out / "market_cap.npz", mcap=mcap_ext)
    _atomic_save_npy(out / "industry.npy", ind_ext)
    print(f"    面板已落盘（内存 {_mem_mb() or 'n/a'}）", flush=True)
    for fn in ("industry_cats.txt", "meta.csv", "delisted_universe.csv",
               "bond_000012.parquet", "gold_518880.parquet"):
        s, d = SRC / fn, out / fn
        if not s.exists():
            continue
        if s.resolve() == d.resolve():      # 原地更新：源=目标，跳过
            continue
        shutil.copy2(s, d)
    # 财报目录（42 期 parquet）一并复制，使扩展目录可独立用于基本面因子
    fund_s, fund_d = SRC / "fund", out / "fund"
    if fund_s.exists() and fund_s.resolve() != fund_d.resolve():
        shutil.copytree(fund_s, fund_d, dirs_exist_ok=True)
    # 指数：原有 + 新行 + 历史缺口（原文件可能已含部分新日期，merge 内去重）
    idx_all, filled_new = merge_indices(idx_old, idx_new, list(nd) + idx_gap)
    _atomic_to_csv(idx_all, out / "indices.csv")
    lost = [d for d in list(nd) + idx_gap if d not in idx_all.index]
    if lost:
        print(f"    [warn] 以下日期指数缺失，未能写入：{[str(pd.Timestamp(d).date()) for d in lost]}")
    # 自检：面板最后一日必须有指数
    assert pd.Timestamp(nd[-1]) in idx_all.index, f"面板尾部 {nd[-1].date()} 无指数数据"
    (out / "UPDATE_INFO.txt").write_text(
        "由 scripts/update_data.py 从腾讯行情增量更新\n"
        f"数据目录：{SRC}{'（原地更新）' if inplace else ''}\n"
        f"新增交易日：{', '.join(str(d.date()) for d in nd)}\n"
        f"补齐历史缺口：{', '.join(str(pd.Timestamp(d).date()) for d in idx_gap if d in idx_all.index) or '无'}\n"
        f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}\n"
        "口径：后复权价/成交量/成交额为腾讯行情源，与原始数据包重叠区间逐点校验一致；\n"
        "raw_close 为缩放价近似（误差仅来自除权除息日），market_cap 用最近一期总股本近似。\n",
        encoding="utf-8")
    print(f"完成 → {out}")
    print(f"  指数：{idx_all.index[-1].date()}（补 {len(filled_new)} 天）")
    print("  新增日期：", ", ".join(str(d.date()) for d in nd))
    # 简单质量检查
    for k, d in enumerate(nd):
        t = T0 + k
        fin = np.isfinite(ext["close"][t])
        print(f"  {d.date()}: 有收盘价股票 {int(fin.sum())} 只，"
              f"成交额中位 {np.nanmedian(ext['amount'][t])/1e8:.2f} 亿")
    sync_aux(nd[-1])
    if not args.no_repair:
        _repair_extract(pd.Timestamp(dates[-1]), out)


if __name__ == "__main__":
    main()
