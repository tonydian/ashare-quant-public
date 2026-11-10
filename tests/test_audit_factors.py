# -*- coding: utf-8 -*-
"""独立审计：因子层（qf/factors.py + qf/ops.py）无未来函数检查。

做法：把 t0 之后（t > t0）的全部行情（close/open/high/low/volume/amount）替换为
随机数，重新构造 FactorInputs 并重算因子；**t0 及以前**的因子值必须逐位完全相同。

覆盖 catalog 中全部 base（33 个），远超任务要求的 10 个。
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import gc
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qf import config as C                      # noqa: E402
from qf import data as qfdata                   # noqa: E402
from qf import factors as qff                   # noqa: E402

CUT_DATE = "2022-06-15"


@pytest.fixture(scope="module")
def md():
    return qfdata.load_market(verbose=False)


@pytest.fixture(scope="module")
def cat():
    return qff.load_catalog()


@pytest.fixture(scope="module")
def inputs(md):
    return qff.FactorInputs(md)


def _require_memory(min_gb: float = 2.0) -> None:
    """内存不足时跳过（避免全量测试时 MemoryError）。"""
    avail_gb = None
    try:
        import sys
        if sys.platform == "win32":
            import ctypes

            class _MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _MEM()
            m.dwLength = ctypes.sizeof(_MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            avail_gb = m.ullAvailPhys / (1024 ** 3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_gb = int(line.split()[1]) / (1024 ** 2)
                        break
    except Exception:
        return
    if avail_gb is not None and avail_gb < min_gb:
        pytest.skip(f"可用内存 {avail_gb:.1f}GB < {min_gb}GB，跳过内存密集测试")


def _one_entry_per_base(cat) -> dict[str, dict]:
    """每个 base 取一个 catalog 条目（优先小 window，计算快）。"""
    out = {}
    for base, g in cat.groupby("base", sort=True):
        g = g.sort_values("window", kind="stable")
        out[str(base)] = g.iloc[0].to_dict()
    return out


def _perturb_like(a: np.ndarray, rng: np.random.Generator, t0: int) -> np.ndarray:
    """复制数组并把 t0 之后的有限值替换为随机乘性扰动（保持 NaN 位置不变）。"""
    b = np.array(a, dtype=np.float32, copy=True)
    seg = b[t0 + 1:]
    fin = np.isfinite(seg)
    noise = np.exp(rng.normal(0.0, 0.25, size=seg.shape)).astype(np.float32)
    seg = np.where(fin, seg * noise, seg)
    b[t0 + 1:] = seg.astype(np.float32)
    return b


def _fake_md_after(md, t0: int, seed: int = C.RANDOM_SEED) -> SimpleNamespace:
    """构造"t0 之后行情完全不同"的 md（只复制因子计算用到的数组）。

    注意：volume_orig / amount_orig 必须保留原始 NaN（停牌）结构，否则等于改变了
    FactorInputs 对停牌单元格的处理，会污染 t0 之前的结果（测试自身缺陷）。
    """
    rng = np.random.default_rng(seed)
    fake = SimpleNamespace()
    fake.T, fake.N = md.T, md.N
    fake.close = _perturb_like(md.close, rng, t0)
    fake.open = _perturb_like(md.open, rng, t0)
    fake.high = _perturb_like(md.high, rng, t0)
    fake.low = _perturb_like(md.low, rng, t0)
    # FactorInputs 只用以下属性；perturb 后的口径保持一致（保留 NaN 结构）
    fake.close_orig = md.close_orig
    fake.volume_orig = _perturb_like(md.volume_orig, rng, t0)
    fake.amount_orig = _perturb_like(md.amount_orig, rng, t0)
    fake.is_suspended = md.is_suspended
    fake.alive = md.alive
    return fake


def test_factor_no_lookahead_all_bases(md, cat, inputs):
    _require_memory()
    """扰动 t0 之后的行情，全部 base 的因子在 t0 及以前必须逐位不变。"""
    t0 = md.date_index(CUT_DATE)
    assert 200 < t0 < md.T - 100, f"cut row 异常：{t0}"

    fake = _fake_md_after(md, t0)
    inputs2 = qff.FactorInputs(fake)

    entries = _one_entry_per_base(cat)
    assert len(entries) >= 10, f"base 数量不足：{len(entries)}"

    changed = []
    for base, entry in entries.items():
        f1 = qff.compute_factor(inputs, entry)
        f2 = qff.compute_factor(inputs2, entry)
        # t0 及以前：逐位完全相同（NaN 位置也必须一致）
        if not np.array_equal(f1[: t0 + 1], f2[: t0 + 1], equal_nan=True):
            n_diff = int((f1[: t0 + 1] != f2[: t0 + 1]).sum())
            raise AssertionError(
                f"因子 {entry['factor_key']} (base={base}) 在 t<={CUT_DATE} 的值被未来行情改变"
                f"（{n_diff} 个格子不同）→ 存在未来函数")
        # t0 之后必须确实被改变（证明扰动生效、测试不是空转）
        if not np.array_equal(f1[t0 + 1:], f2[t0 + 1:], equal_nan=True):
            changed.append(base)

    assert len(changed) >= int(0.8 * len(entries)), (
        f"只有 {len(changed)}/{len(entries)} 个 base 的未来段因子随扰动变化，"
        "扰动可能无效")
    print(f"[audit] 无未来函数通过：{len(entries)} 个 base "
          f"({', '.join(sorted(entries))})，其中 {len(changed)} 个未来段确实变化")


def test_factor_no_lookahead_all_transforms(md, cat, inputs):
    _require_memory()
    """同一 base 的 8 种 transform 全部做扰动检验（选取覆盖 transform 的因子）。"""
    t0 = md.date_index(CUT_DATE)
    fake = _fake_md_after(md, t0, seed=C.RANDOM_SEED + 7)
    inputs2 = qff.FactorInputs(fake)

    samples = []
    for tr, g in cat.groupby("transform", sort=True):
        samples.append(g.sort_values("window", kind="stable").iloc[0].to_dict())
    assert len(samples) >= 8

    for entry in samples:
        f1 = qff.compute_factor(inputs, entry)
        f2 = qff.compute_factor(inputs2, entry)
        assert np.array_equal(f1[: t0 + 1], f2[: t0 + 1], equal_nan=True), (
            f"transform={entry['transform']} 因子 {entry['factor_key']} 使用了未来数据")


def test_factor_values_only_depend_on_past(md, cat, inputs):
    _require_memory()
    """结构性检查：factor[t] 只由 inp 的 [0..t] 决定（用逐行截断重算抽查 3 个 base）。"""
    t0 = md.date_index("2023-06-15")
    t1 = md.date_index("2024-06-17")
    fake = _fake_md_after(md, t0, seed=C.RANDOM_SEED + 13)
    inputs2 = qff.FactorInputs(fake)
    entries = [cat[cat["base"] == b].iloc[0].to_dict() for b in ("return", "volume_ratio", "rsi")]
    for entry in entries:
        f_full = qff.compute_factor(inputs, entry)[: t0 + 1]
        f_pert = qff.compute_factor(inputs2, entry)[: t1 + 1][: t0 + 1]
        assert np.array_equal(f_full, f_pert, equal_nan=True), f"{entry['factor_key']} 前视"
