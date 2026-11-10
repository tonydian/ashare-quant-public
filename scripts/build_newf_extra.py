# -*- coding: utf-8 -*-
"""构造「换手/流动性/量价形态/隔夜」四个新因子族，落到 ``newf_extra.npz``（全 12 个）。

> 2026-11-03 起本脚本只剩一个入口壳：算式已集中到 ``qf.newf.build_panels``，
> 落盘逻辑在 ``scripts/build_newf_family.py``。这样「历史面板」与
> 「最新一日 live 行」必然是同一份代码——否则实盘打分与训练打分的口径会悄悄分叉。

产物（与 ``--family all`` 相同）::

    build/factors/newf_extra.npz      factor_key -> (477, N) 周频面板
    build/factors/newf_catalog.parquet

按族单独落盘（生产用的换手族走这里）::

    python scripts/build_newf_family.py --family turnover

设计原则
--------
- 每个 base 是现有 33 个 base 之外的新信息维度（现有库无 turnover/amihud 类）
- 全部只用「当期及历史」数据，无未来信息
- 缺失一律 NaN，交给下游 mask 处理，不填充
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_newf_family import build_and_save  # noqa: E402

# 兼容旧调用方：SPEC 仍可从这里取（格式 = qf.newf.SPEC）
from qf.newf import SPEC  # noqa: E402,F401


if __name__ == "__main__":
    build_and_save("all")
