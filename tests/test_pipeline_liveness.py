# -*- coding: utf-8 -*-
"""管道状态存活检测的单元测试。

背景：2026-11-03 21:44 启动的更新，`update_data.py` 在「[4/6] 构建扩展面板」
被系统杀掉，`run_pipeline.py` 也随之消失，但 `build/pipeline/status.json` 留在
`running=true / finished_at=null`，网页因此一直显示「运行中·数据增量更新」空转
24 小时。修复的核心是：状态文件里必须留下"我是否还活着"的证据（pid + 心跳），
读取方据此把死掉的状态改判为「已中断」。

不变式（与具体数据无关）：
 1. `running=false` 的旧状态永远不算 stale（否则每次都会误报中断）；
 2. pid 进程不存在 → 必判 stale；
 3. pid 活着但心跳超过阈值 → 判 stale（覆盖"进程卡死不推进"）；
 4. pid 活着且心跳新鲜 → 不判 stale（不能误杀健康任务）；
 5. 老格式（无 pid 无心跳）按 updated_at 兜底，且宁可漏判不误判；
 6. 两条实现（scripts 与 web）判断必须一致，否则页面和脚本会各说各话。
"""
from __future__ import annotations

import datetime
import pathlib
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import pid_alive, status_is_stale  # noqa: E402


def _iso(sec_ago: float) -> str:
    t = datetime.datetime.now() - datetime.timedelta(seconds=sec_ago)
    return t.isoformat(timespec="seconds")


def _dead_pid() -> int:
    """几乎不可能是活进程的 pid。"""
    return 0x7FFFFFF0


# --------------------------------------------------------------- pid_alive
def test_pid_alive_self():
    import os

    assert pid_alive(os.getpid()) is True


def test_pid_alive_bad_input():
    for bad in (None, "", "abc", -1, 0):
        assert pid_alive(bad) is False


def test_pid_alive_dead():
    assert pid_alive(_dead_pid()) is False


# ---------------------------------------------------------- status_is_stale
def test_not_running_is_never_stale():
    assert status_is_stale({"running": False, "pid": _dead_pid()}) is False
    assert status_is_stale({}) is False
    assert status_is_stale(None) is False


def test_dead_pid_is_stale():
    # 进程已不在，即便心跳时间戳很新，也必须判死
    st = {"running": True, "pid": _dead_pid(), "heartbeat_ts": time.time()}
    assert status_is_stale(st) is True


def test_alive_pid_fresh_heartbeat_is_healthy():
    import os

    st = {"running": True, "pid": os.getpid(), "heartbeat_ts": time.time()}
    assert status_is_stale(st) is False


def test_alive_pid_stale_heartbeat_is_stale():
    """进程还在但 24 小时没心跳 → 卡死，必须判 stale。"""
    import os

    st = {"running": True, "pid": os.getpid(),
          "heartbeat_ts": time.time() - 86400}
    assert status_is_stale(st) is True


def test_heartbeat_threshold_boundary():
    import os

    now = time.time()
    # 179 秒：还没超 180 秒阈值，视为健康
    assert status_is_stale({"running": True, "pid": os.getpid(),
                            "heartbeat_ts": now - 179}, now=now) is False
    # 181 秒：已超阈值
    assert status_is_stale({"running": True, "pid": os.getpid(),
                            "heartbeat_ts": now - 181}, now=now) is True


def test_legacy_no_pid_uses_updated_at():
    """老格式：没有 pid、没有心跳，只能看 updated_at。"""
    fresh = {"running": True, "updated_at": _iso(30)}
    old = {"running": True, "updated_at": _iso(86400)}
    assert status_is_stale(fresh) is False
    assert status_is_stale(old) is True


def test_legacy_garbage_updated_at_keeps_running():
    """updated_at 解析不出来时宁可漏判，也不能把健康任务标成中断。"""
    assert status_is_stale({"running": True, "updated_at": "???"}) is False
    assert status_is_stale({"running": True}) is False


# ---------------------------------------------------- 两条实现必须一致
def test_web_and_script_agree():
    import os

    from web.services import _pipe_is_stale

    cases = [
        {"running": False, "pid": _dead_pid()},
        {"running": True, "pid": _dead_pid(), "heartbeat_ts": time.time()},
        {"running": True, "pid": os.getpid(), "heartbeat_ts": time.time()},
        {"running": True, "pid": os.getpid(), "heartbeat_ts": time.time() - 86400},
        {"running": True, "updated_at": _iso(86400)},
        {"running": True},
    ]
    for st in cases:
        assert status_is_stale(st) == _pipe_is_stale(st), f"判断不一致: {st}"
