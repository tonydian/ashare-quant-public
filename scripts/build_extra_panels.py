# -*- coding: utf-8 -*-
"""重建生产模型 ``--extra`` 引用到的全部额外因子族面板。

为什么要有这一步
----------------
``scripts/build_factors.py`` 只重建主因子面板（``weekly_top.npz`` /
``weekly_meta.npz``），而生产模型还通过 ``--extra`` 挂了几个**独立生成**的
因子族面板。这些面板此前没有任何一环在管道里重建它们，于是：

    数据更新 → weekly_meta 推进到本周
             → momentum_extra / newf_turnover_extra / value_extra 仍停在上一周
             → 模型打分时这些因子在最后一行是 NaN
             → np.tensordot 让整行复合分变 NaN
             → composite_score.npz 最后有效行停在上一周
             → check_data_source 报「模型滞后 6 日」，整条管道标红

2026-11-04 22:56 那次「更新失败」就是这么来的：17 个阶段全部 rc=0，
唯独最后的一致性检测报模型滞后，而根因与模型毫无关系。

本脚本把「--extra 引用的面板必须跟 weekly_meta 同长」固化成管道里的一个阶段。

单一真源
--------
要重建哪些面板，**不在这里写死**。脚本直接读 ``scripts/run_pipeline.py`` 里
model 阶段的 ``--extra`` 参数，因此改了模型命令就自动跟进，不会出现两处
列表各自漂移的老问题。

用法::

    python scripts/build_extra_panels.py            # 重建 --extra 引用的全部面板
    python scripts/build_extra_panels.py --list     # 只看会重建哪些，不执行
    python scripts/build_extra_panels.py --only momentum_extra
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
FACTORS = ROOT / "build" / "factors"
PIPELINE = ROOT / "scripts" / "run_pipeline.py"

PY = sys.executable

# 面板 stem -> 重建命令（相对 ROOT，不含解释器）
# 注意 value_extra 依赖 fundamental_extra，顺序由 DEPENDS 保证
REGEN: dict[str, list[str]] = {
    "momentum_extra": ["scripts/build_momentum_extra.py"],
    "fundamental_extra": ["scripts/build_fundamental_extra.py"],
    "value_extra": ["scripts/build_value_extra.py"],
    "newf_extra": ["scripts/build_newf_family.py", "--family", "all"],
    "newf_turnover_extra": ["scripts/build_newf_family.py", "--family", "turnover"],
    "newf_liquidity_extra": ["scripts/build_newf_family.py", "--family", "liquidity"],
    "newf_pricevolume_extra": ["scripts/build_newf_family.py", "--family", "pricevolume"],
    "newf_overnight_extra": ["scripts/build_newf_family.py", "--family", "overnight"],
}

# 上游依赖：value_extra 是从 fundamental_extra 切出来的
DEPENDS: dict[str, list[str]] = {"value_extra": ["fundamental_extra"]}

# 生成顺序：被依赖者先跑
ORDER_HINT = ["momentum_extra", "fundamental_extra", "value_extra", "newf_extra",
              "newf_turnover_extra", "newf_liquidity_extra",
              "newf_pricevolume_extra", "newf_overnight_extra"]


def production_extras() -> list[str]:
    """从 run_pipeline.py 的 model 阶段解析 --extra 路径，返回面板 stem 列表。"""
    if not PIPELINE.exists():
        raise SystemExit(f"找不到 {PIPELINE}")
    src = PIPELINE.read_text(encoding="utf-8")
    m = re.search(r'\("model",.*?\n    \),', src, re.S)
    block = m.group(0) if m else src
    paths = re.findall(r'"--extra",\s*"([^"]+)"', block)
    stems: list[str] = []
    for p in paths:
        stem = pathlib.Path(p).stem
        if stem in stems:
            continue
        stems.append(stem)
    if not stems:
        raise SystemExit("model 阶段没有解析到任何 --extra，请检查 run_pipeline.py")
    # 自动补上 run_research 会**自动加载**的面板（不需 --extra 也在用）
    auto = "momentum_extra"          # extra_sets 默认项
    if auto not in stems:
        stems.insert(0, auto)
    return stems


def with_deps(stems: list[str]) -> list[str]:
    out: list[str] = []
    for s in stems:
        for d in DEPENDS.get(s, []):
            if d not in out:
                out.append(d)
        if s not in out:
            out.append(s)
    # 按 ORDER_HINT 稳定排序，未登记的排后面、保持原序
    known = [k for k in ORDER_HINT if k in out]
    unknown = [k for k in out if k not in ORDER_HINT]
    return known + unknown


def panel_weeks(path: pathlib.Path) -> int | None:
    import numpy as np
    if not path.exists():
        return None
    z = np.load(path)
    if not z.files:
        return None
    return int(z[z.files[0]].shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列出将重建的面板")
    ap.add_argument("--only", default=None, help="逗号分隔的 stem，只重建这些")
    ap.add_argument("--force", action="store_true",
                    help="即使周数已对齐也重建（默认已对齐则跳过，省 3 分钟）")
    args = ap.parse_args()

    if args.only:
        # --only 允许点名不在生产 --extra 列表里的面板（实验分支、单测依赖的族）。
        # 依赖（如 value_extra → fundamental_extra）由 with_deps 自动补齐。
        want = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = [s for s in want if s not in REGEN]
        if unknown:
            raise SystemExit(f"未知面板 {unknown}，可选：{sorted(REGEN)}")
        stems = with_deps(want)
    else:
        stems = with_deps(production_extras())

    # 目标周数 = weekly_meta 的周数
    import numpy as np
    wm = FACTORS / "weekly_meta.npz"
    if not wm.exists():
        raise SystemExit(f"缺少 {wm}，请先跑 scripts/build_factors.py")
    target = int(len(np.load(wm)["rows"]))
    print(f"目标：{target} 周（对齐 weekly_meta）")
    print(f"待查面板：{', '.join(stems)}\n")

    todo: list[str] = []
    for s in stems:
        p = FACTORS / f"{s}.npz"
        w = panel_weeks(p)
        if w is None:
            print(f"  {s:24s} 不存在 → 需重建")
            todo.append(s)
        elif w != target:
            print(f"  {s:24s} {w} 周，缺 {target - w} 周 → 需重建")
            todo.append(s)
        else:
            print(f"  {s:24s} {w} 周 ✓")

    if args.list:
        print(f"\n将重建 {len(todo)} 个：{todo}")
        return
    if not todo and not args.force:
        print("\n全部已对齐，无需重建。")
        return
    if args.force:
        todo = stems

    t0 = time.time()
    for s in todo:
        cmd = REGEN.get(s)
        if not cmd:
            raise SystemExit(f"面板 {s} 没有登记重建命令，请补到 REGEN 映射")
        print(f"\n[extra] 重建 {s} …", flush=True)
        t1 = time.time()
        r = subprocess.run([PY, "-u"] + cmd, cwd=str(ROOT))
        if r.returncode != 0:
            raise SystemExit(f"[extra] {s} 重建失败 rc={r.returncode}（cmd: {' '.join(cmd)}）")
        w = panel_weeks(FACTORS / f"{s}.npz")
        print(f"[extra] ✓ {s} {time.time()-t1:.1f}s → {w} 周", flush=True)

    # 收尾复核：任何一个没对齐就直接失败，不留半成品
    bad = []
    for s in stems:
        w = panel_weeks(FACTORS / f"{s}.npz")
        if w != target:
            bad.append((s, w))
    print(f"\n[extra] 完成，用时 {time.time()-t0:.1f}s")
    if bad:
        for s, w in bad:
            print(f"  ✗ {s}: {w} 周（应为 {target}）")
        raise SystemExit(1)
    print(f"  ✓ {len(stems)} 个面板全部对齐到 {target} 周")


if __name__ == "__main__":
    main()
