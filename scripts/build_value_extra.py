# -*- coding: utf-8 -*-
"""从 fundamental_extra.npz 抽出纯价值腿使用的 bp 单因子面板。

背景
----
生产模型的 ``--extra`` 列表里有 ``build/factors/value_extra.npz``，它只有
一个 key：``fundamental-extra:F002``（账面市值比 bp）。这份文件此前是
**手工**从 ``fundamental_extra.npz`` 切出来的，没有可复现的生成入口，
后果是：

  1. 每次 ``build_factors.py`` 重建因子面板后，它都比 ``weekly_meta`` 少一周；
  2. 模型打分时该因子在最后一行是 NaN，``np.tensordot`` 让整行复合分变 NaN；
  3. ``composite_score.npz`` 的最后有效行停在上一周；
  4. ``check_data_source.py`` 报「模型滞后 6 日」→ 整条管道标红失败。

而前三步都是静默的，只有第 4 步会喊，且喊的是「模型滞后」这种指向性错误的
诊断（真正的病灶是过期的 extra 面板）。本脚本把这一步固化成可复现入口。

用法::

    python scripts/build_value_extra.py
输出::

    build/factors/value_extra.npz          fundamental-extra:F002 -> (n_weeks, N)
    build/factors/value_catalog.parquet    对应 catalog 单行
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C

KEY = "fundamental-extra:F002"
SRC = "fundamental_extra.npz"
OUT = "value_extra.npz"


def build(verbose: bool = True) -> pathlib.Path:
    src = C.BUILD_FACTORS / SRC
    if not src.exists():
        raise SystemExit(f"缺少源面板 {src}，请先跑 scripts/build_fundamental_extra.py")
    z = np.load(src)
    if KEY not in z.files:
        raise SystemExit(f"{SRC} 中找不到 {KEY}，可选：{list(z.files)[:8]}")
    panel = z[KEY].astype(np.float32)

    dst = C.BUILD_FACTORS / OUT
    # tmp 必须以 .npz 结尾：np.savez_compressed 会给非 .npz 结尾的路径自动补后缀，
    # 写成 "value_extra.npz.tmp" 会实际落到 "value_extra.npz.tmp.npz"，随后的 replace 就找不到文件
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{KEY: panel})
    tmp.replace(dst)

    # catalog：只保留这一行，供 run_research 的 extra 路径读取
    cat_src = C.BUILD_FACTORS / "fundamental_catalog.parquet"
    cat_dst = C.BUILD_FACTORS / "value_catalog.parquet"
    if cat_src.exists():
        cat = pd.read_parquet(cat_src)
        row = cat[cat["factor_key"] == KEY]
        row.to_parquet(cat_dst, index=False)
    else:
        row = pd.DataFrame()

    if verbose:
        cov = float(np.isfinite(panel).mean())
        print(f"  {KEY}: {panel.shape[0]} 周 × {panel.shape[1]} 只，覆盖 {cov:.1%}")
        print(f"\n→ {dst}\n→ {cat_dst}（{len(row)} 行）")
    return dst


def main() -> None:
    build(verbose=True)


if __name__ == "__main__":
    main()
