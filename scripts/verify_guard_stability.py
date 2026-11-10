# -*- coding: utf-8 -*-
"""干净验证：连续跑 3 次重训，看 guard 是否让池稳定下来。

步骤：
  1. 清理 guard 状态
  2. 第 1 次：无历史 → 建池（记录）
  3. 第 2 次：数据不变 → 应换手 0
  4. 第 3 次：数据不变 → 应换手 0

对比：无 guard 时，同样连续跑 3 次，每次池都可能有变化。
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")
PY = sys.executable
STATE = ROOT / "build" / "models" / "_pool_guard_state.json"
META = ROOT / "build" / "models" / "wf_h10" / "model_meta.json"

env = {**os.environ}
env.pop("QF_DATA_RAW", None)
env["PYTHONUTF8"] = "1"
env["PYTHONUNBUFFERED"] = "1"

CMD = [PY, "-u", "scripts/run_research.py", "--out", "build/models/wf_h10",
       "--method", "blend", "--max-factors", "34", "--horizon", "10",
       "--no-fundamental", "--pool-guard", "--pool-streak", "2"]


def read_pool():
    if not META.exists():
        return None
    return list(json.load(open(META, encoding="utf-8")).get("pool") or [])


def run_once(tag):
    print(f"\n{'='*80}\n{tag}\n{'='*80}")
    r = subprocess.run(CMD, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    for line in (r.stdout or "").splitlines():
        if "pool-guard" in line or "pool=" in line:
            print("  " + line.strip())
    if r.returncode != 0:
        print(f"  ❌ 失败: {(r.stderr or '')[-300:]}")
    return r.returncode == 0


def main():
    # 1) 清理状态
    if STATE.exists():
        STATE.unlink()
        print("已清理 guard 状态")
    # 备份原模型
    bak = ROOT / "build" / "models" / "_wf_h10_bak_test"
    if (ROOT / "build" / "models" / "wf_h10").exists():
        if bak.exists():
            shutil.rmtree(bak)
        shutil.copytree(ROOT / "build" / "models" / "wf_h10", bak)
        print("已备份原模型")

    pools = []
    for i in range(1, 4):
        ok = run_once(f"第 {i} 次重训")
        p = read_pool()
        pools.append(p)
        if p:
            print(f"  池大小 {len(p)}")
        if i > 1 and pools[-2] is not None and p is not None:
            added = set(p) - set(pools[-2])
            removed = set(pools[-2]) - set(p)
            print(f"  相对上次：+{len(added)} -{len(removed)}")
            if added:
                print(f"    + {sorted(added)[:8]}")
            if removed:
                print(f"    - {sorted(removed)[:8]}")

    print(f"\n{'='*80}\n结论\n{'='*80}")
    if len(pools) >= 3 and all(p is not None for p in pools[:3]):
        d12 = len(set(pools[1]) - set(pools[0]))
        d23 = len(set(pools[2]) - set(pools[1]))
        print(f"  第1→2 次变化: {d12} 个")
        print(f"  第2→3 次变化: {d23} 个")
        if d12 == 0 and d23 == 0:
            print("\n  ✅ 池完全稳定（数据不变时零换手）")
        elif d23 == 0:
            print(f"\n  ✅ 第 2 次后稳定（首次建立状态，之后零换手）")
        else:
            print(f"\n  ⚠️ 仍在变化，需检查逻辑")

    # 恢复
    if bak.exists():
        shutil.rmtree(ROOT / "build" / "models" / "wf_h10")
        shutil.move(str(bak), str(ROOT / "build" / "models" / "wf_h10"))
        print("\n已恢复原模型")


if __name__ == "__main__":
    main()
