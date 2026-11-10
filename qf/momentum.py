# -*- coding: utf-8 -*-
"""额外因子族：中期动量（momentum-extra）。

动机（DEV 期证据，2017-2019）：QuantSkills 因子库的窗口只有 5/7/10/14 日，处于
A 股短周期反转区；而 20/40/60/120 日收益在 DEV 期呈现稳定的**正向**横截面预测力
（DEV 期 top50 多头组合相对选股域等权有明显正超额，见报告）。因此补充一个中期
动量族：只用 t 及以前价格，无未来函数。

因子定义（在周频行上计算，值 = 过去 k 日收益，可跳过最近 skip 日）：
    mom{k}:        price[t] / price[t-k] - 1
    mom{k}s{s}:    price[t-s] / price[t-k-s] - 1   （跳过最近 s 日，避开短期反转）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# (factor_id, k, skip, name_cn)
DEFS = [
    ("M001", 20, 0, "20日动量"),
    ("M002", 40, 0, "40日动量"),
    ("M003", 60, 0, "60日动量"),
    ("M004", 120, 0, "120日动量"),
    ("M005", 20, 5, "20日动量(跳过近5日)"),
    ("M006", 60, 5, "60日动量(跳过近5日)"),
    ("M007", 120, 5, "120日动量(跳过近5日)"),
    ("M008", 10, 5, "10日动量(跳过近5日)"),
]
LIB = "momentum-extra"


def _factor_panel(md, k: int, skip: int) -> np.ndarray:
    p = md.price.astype(np.float64)
    if skip > 0:
        cur = pd.DataFrame(p).shift(skip).to_numpy(dtype=np.float64)
        prev = pd.DataFrame(p).shift(k + skip).to_numpy(dtype=np.float64)
    else:
        cur = p
        prev = pd.DataFrame(p).shift(k).to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = cur / prev - 1.0
    return np.where(md.alive & np.isfinite(v), v, np.nan).astype(np.float32)


def build_panels(md, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """返回 {factor_key: (len(rows), N) 或 (T,N)}。"""
    out = {}
    for fid, k, skip, _ in DEFS:
        key = f"{LIB}:{fid}"
        full = _factor_panel(md, k, skip)
        out[key] = full if rows is None else full[rows]
    return out


def catalog() -> pd.DataFrame:
    rows = []
    for fid, k, skip, name in DEFS:
        rows.append({
            "lib": LIB, "factor_id": fid, "slug": f"mom{k}_skip{skip}",
            "name_cn": name, "name_en": f"{k}D momentum skip {skip}",
            "category_cn": "动量", "category_en": "Momentum-extra",
            "base": f"mom{k}_skip{skip}", "transform": "raw",
            "window": int(k), "norm": 20, "params": {"window": int(k), "skip": int(skip)},
            "path": "", "factor_key": f"{LIB}:{fid}",
        })
    return pd.DataFrame(rows)


def build_live(md, key: str) -> np.ndarray:
    """单因子全面板（供实盘信号使用）。"""
    fid = key.split(":")[1]
    for f_id, k, skip, _ in DEFS:
        if f_id == fid:
            return _factor_panel(md, k, skip)
    raise KeyError(key)
