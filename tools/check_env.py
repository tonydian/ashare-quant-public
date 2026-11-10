# -*- coding: utf-8 -*-
"""环境自检：一条命令确认这台电脑能不能跑这套系统。

用法（必须在本文件所在目录的上一级，也就是 ashare_quant/ 下执行）::

    python tools/check_env.py          # 完整检查（会加载 450MB 数据，约 20-40 秒）
    python tools/check_env.py --fast   # 只查依赖与文件，不加载数据

退出码 0 = 可以跑；1 = 有 FAIL，按提示修。
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                      # ashare_quant/
sys.path.insert(0, str(PKG))

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
_results: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str = "") -> None:
    _results.append((level, name, detail))
    print(f"{level} {name}{('  — ' + detail) if detail else ''}", flush=True)


def check_python() -> None:
    v = sys.version_info
    detail = f"{platform.python_version()} / {platform.system()} {platform.machine()}"
    if v < (3, 10):
        record(FAIL, "Python 版本", detail + " 需要 >= 3.10")
    elif v >= (3, 14):
        record(WARN, "Python 版本", detail + " 未在 3.14+ 上验证过，建议 3.12/3.13")
    else:
        record(OK, "Python 版本", detail)
    if sys.version_info[:2] == (3, 13):
        print("       提示：Windows 下若中文显示为乱码，先执行 chcp 65001 或设 PYTHONIOENCODING=utf-8")


def check_packages() -> None:
    import importlib

    need = {
        "numpy": "1.26",
        "pandas": "2.1",
        "pyarrow": "14",
        "matplotlib": "3.7",
        "scipy": "1.11",
    }
    for mod, minver in need.items():
        try:
            m = importlib.import_module(mod)
        except ImportError:
            record(FAIL, f"依赖 {mod}", "未安装 → pip install -r requirements.txt")
            continue
        ver = getattr(m, "__version__", "?")
        record(OK, f"依赖 {mod}", ver)
    try:
        import pytest  # noqa: F401
        record(OK, "依赖 pytest", getattr(pytest, "__version__", "?"))
    except ImportError:
        record(WARN, "依赖 pytest", "未安装，单元测试无法运行（不影响出信号）")


def check_paths(C) -> None:
    print()
    print(f"   QF_ROOT  -> {C.ROOT}")
    print(f"   数据目录  -> {C.DATA_RAW}")
    print(f"   因子库索引 -> {C.VENDOR}")
    print(f"   代码目录  -> {C.PKG}")
    print()
    if not C.ROOT.exists():
        record(FAIL, "QF_ROOT 目录", f"{C.ROOT} 不存在")
    else:
        record(OK, "QF_ROOT 目录", str(C.ROOT))

    if not C.DATA_RAW.exists():
        record(FAIL, "数据目录", f"{C.DATA_RAW} 不存在 —— 数据包还没放进来")
        return
    record(OK, "数据目录", str(C.DATA_RAW))


def check_data_files(C) -> bool:
    """检查数据文件是否齐全，并核对面板形状。"""
    import numpy as np
    import pandas as pd

    files = {
        "panel_ohlcv.npz": "行情面板（必需）",
        "raw_close.npz": "不复权收盘（必需）",
        "market_cap.npz": "逐日总市值（必需）",
        "industry.npy": "行业矩阵（必需）",
        "industry_cats.txt": "行业名称（必需）",
        "meta.csv": "股票基本信息（必需）",
        "indices.csv": "基准指数（必需）",
        "bond_000012.parquet": "国债指数（现金端，必需）",
        "gold_518880.parquet": "黄金 ETF（可选）",
    }
    missing = []
    for fn, desc in files.items():
        p = C.DATA_RAW / fn
        if p.exists():
            record(OK, f"数据文件 {fn}", f"{p.stat().st_size / 1048576:.1f} MB  {desc}")
        else:
            lvl = WARN if "可选" in desc else FAIL
            record(lvl, f"数据文件 {fn}", f"缺失（{desc}）")
            missing.append(fn)

    if "panel_ohlcv.npz" in missing:
        return False

    # 形状与日期
    try:
        z = np.load(C.DATA_RAW / "panel_ohlcv.npz", allow_pickle=True)
        T, N = z["close"].shape
        dates = pd.DatetimeIndex(z["dates"])
        record(OK, "面板形状", f"{T} 交易日 × {N} 只股票")
        record(OK, "数据区间", f"{dates[0]:%Y-%m-%d} ~ {dates[-1]:%Y-%m-%d}")
        # 交易日历是否连续到最近
        gap = (pd.Timestamp.today().normalize() - dates[-1]).days
        if gap > 10:
            record(WARN, "数据新鲜度", f"最后一根 K 线在 {dates[-1]:%Y-%m-%d}，距今 {gap} 天 → 跑 scripts/update_data.py")
        else:
            record(OK, "数据新鲜度", f"距今 {gap} 天")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "面板读取", f"{type(e).__name__}: {e}")
        return False

    # 对齐性：几个面板的 T/N 必须一致
    try:
        rc = np.load(C.DATA_RAW / "raw_close.npz")["rawclose"]
        mc = np.load(C.DATA_RAW / "market_cap.npz")["mcap"]
        ind = np.load(C.DATA_RAW / "industry.npy")
        shapes = {"panel": (T, N), "raw_close": rc.shape, "market_cap": mc.shape, "industry": ind.shape}
        bad = {k: v for k, v in shapes.items() if v != (T, N)}
        if bad:
            record(FAIL, "面板对齐", f"形状不一致 {bad} vs panel {(T, N)}")
        else:
            record(OK, "面板对齐", "panel / raw_close / market_cap / industry 全部一致")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "面板对齐", f"{type(e).__name__}: {e}")

    # 基金财报目录
    fund = C.DATA_RAW / "fund"
    n = len(list(fund.glob("*.parquet"))) if fund.exists() else 0
    if n >= 40:
        record(OK, "财报目录 fund/", f"{n} 期")
    elif n > 0:
        record(WARN, "财报目录 fund/", f"只有 {n} 期（预期 42 期），基本面因子会退化")
    else:
        record(WARN, "财报目录 fund/", "缺失，价值/基本面因子不可用")

    return True


def check_factor_catalog(C) -> None:
    try:
        from qf import factors as qff

        cat = qff.load_catalog()
        record(OK, "因子索引 vendor/", f"{len(cat)} 个因子")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "因子索引 vendor/", f"{type(e).__name__}: {e} —— vendor/ 没放对位置？")


def check_model(C) -> None:
    try:
        from qf import model as qfm

        res = qfm.load_result(C.BUILD_MODELS / "wf_h10")
        record(OK, "生产模型 wf_h10", f"池内 {len(res.pool)} 个因子 / {len(res.weights)} 行权重")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "生产模型 wf_h10", f"{type(e).__name__}: {e}")


def check_load_market(C) -> None:
    """真实加载数据并算一个因子——这一步过了，出信号就没问题。"""
    try:
        t0 = time.time()
        from qf import data as qfdata
        from qf import factors as qff

        md = qfdata.load_market(verbose=False)
        dt = time.time() - t0
        record(OK, "加载行情面板", f"{md.T}×{md.N}，耗时 {dt:.1f}s")

        t0 = time.time()
        inp = qff.FactorInputs(md)
        cat = qff.load_catalog()
        entry = cat.iloc[0].to_dict()
        panel = qff.compute_factor(inp, entry)
        dt = time.time() - t0
        if panel.ndim == 1:
            panel = panel[None, :]
        import numpy as np

        valid = int(np.isfinite(panel[-1]).sum())
        if valid < 100:
            record(WARN, "因子可用性", f"截面只有 {valid} 只有效值，数据可能不完整")
        else:
            record(OK, "因子可用性", f"示例因子截面 {valid} 只有效值，耗时 {dt:.1f}s")
        record(OK, "最新交易日", f"{md.dates[-1]:%Y-%m-%d}")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "加载行情面板", f"{type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过数据加载（只查依赖与文件）")
    args = ap.parse_args()

    print("=" * 68)
    print(" A 股多因子量化系统 —— 环境自检")
    print("=" * 68)
    print()

    check_python()
    check_packages()

    try:
        from qf import config as C
    except Exception as e:  # noqa: BLE001
        record(FAIL, "导入 qf.config", f"{type(e).__name__}: {e}")
        return 1

    check_paths(C)
    has_data = check_data_files(C)
    check_factor_catalog(C)
    check_model(C)

    if has_data and not args.fast:
        print()
        print("-- 加载数据（约 20-40 秒）--")
        check_load_market(C)
    elif args.fast:
        print()
        print("-- 已跳过数据加载（--fast）--")

    n_fail = sum(1 for lv, _, _ in _results if lv == FAIL)
    n_warn = sum(1 for lv, _, _ in _results if lv == WARN)
    print()
    print("=" * 68)
    if n_fail == 0:
        print(f" 结论：可以运行。{len(_results) - n_warn} 项通过，{n_warn} 项警告。")
        print(" 下一步：python scripts/daily_signal.py --four")
    else:
        print(f" 结论：有 {n_fail} 项失败，{n_warn} 项警告。按上面的 [FAIL] 逐条修。")
        print(" 常见原因：① 数据包没解压到 data_extract/was_stock_data_ext/（见 数据获取指南.md）")
        print("           ② 用了系统 python 而不是虚拟环境里的 python")
        print("           ③ vendor/ 缺失 —— 仓库自带索引，若报错请检查是否完整 clone")
    print("=" * 68)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
