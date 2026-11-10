# -*- coding: utf-8 -*-
"""管道状态文件的健壮性测试（2026-11-07 事故回归）。

事故回顾：20 个阶段全部成功（含 verify 一致），最后一步
`status.tmp -> status.json` 抛 `PermissionError: [WinError 5]`，
因为网页每 2~3 秒轮询这个文件，而 Python 的 open() 在 Windows 上
不带 FILE_SHARE_DELETE，`os.replace` 就会撞上。

后果不是"少写一次进度"，而是：
  1. 异常没人接 → 整条管道非零退出；
  2. 网页的僵尸检测把这次运行判成 interrupted / ok=False；
  3. 界面显示"更新失败" → 用户再点一次 → factors 白重跑 850 秒。

所以这里锁三条底线：
  * 写状态**永不抛异常**（撞锁就重试，再不行退化成原地写）；
  * 已成功的运行不会因为写状态失败而被判失败；
  * 已有实例在跑时不再启动第二个。
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_pipeline_mod",
                                               ROOT / "scripts" / "run_pipeline.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)


@pytest.fixture
def status_path(tmp_path, monkeypatch):
    """把 STATUS 指到临时目录，并把退避时间压到毫秒级。"""
    p = tmp_path / "status.json"
    monkeypatch.setattr(rp, "STATUS", p)
    monkeypatch.setattr(rp, "STATUS_WRITE_BACKOFF", 0.001)
    return p


# ---------------------------------------------------------------------------
# 一、写状态：撞锁要能扛过去
# ---------------------------------------------------------------------------
def test_写状态_短暂占用后重试成功(status_path, monkeypatch):
    """模拟网页正在读：前 3 次 replace 被拒，第 4 次成功。"""
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(5, "拒绝访问")
        return real_replace(src, dst)

    monkeypatch.setattr(rp.os, "replace", flaky)
    rp.write_status({"running": True, "ok": True})

    assert calls["n"] == 4, "应当重试到成功为止"
    assert json.loads(status_path.read_text(encoding="utf-8"))["ok"] is True


def test_写状态_始终换不动时退化成原地写(status_path, monkeypatch):
    """文件被长时间持有：replace 永远失败，但状态必须落盘。"""
    def always_locked(src, dst):
        raise PermissionError(5, "拒绝访问")

    monkeypatch.setattr(rp.os, "replace", always_locked)
    monkeypatch.setattr(rp, "STATUS_WRITE_ATTEMPTS", 2)
    rp.write_status({"running": False, "ok": True, "note": "兜底"})

    got = json.loads(status_path.read_text(encoding="utf-8"))
    assert got["ok"] is True and got["note"] == "兜底"


def test_写状态_彻底写不进去也不抛(tmp_path, monkeypatch, capsys):
    """连兜底都失败时：只打警告，不改变任何阶段结果。"""
    monkeypatch.setattr(rp, "STATUS", tmp_path)       # 目标是目录 → 必然写不进去
    monkeypatch.setattr(rp, "STATUS_WRITE_BACKOFF", 0.001)
    monkeypatch.setattr(rp, "STATUS_WRITE_ATTEMPTS", 2)

    rp.write_status({"running": True})                # 关键：不抛异常

    assert "status.json 写入失败" in capsys.readouterr().err


def test_写状态_正常路径是原子替换(status_path):
    rp.write_status({"running": True, "stages": [{"key": "data", "status": "ok"}]})
    got = json.loads(status_path.read_text(encoding="utf-8"))
    assert got["stages"][0]["key"] == "data"
    assert "updated_at" in got
    assert not status_path.with_suffix(".tmp").exists(), "临时文件应已被替换掉"


# ---------------------------------------------------------------------------
# 二、重入保护：不许在同一时刻跑两遍
# ---------------------------------------------------------------------------
def _write(status_path, **kw):
    status_path.write_text(json.dumps(kw), encoding="utf-8")


def test_有实例在跑就拒绝再启动(status_path, monkeypatch):
    _write(status_path, running=True, pid=4242, heartbeat_ts=time.time())
    monkeypatch.setattr(rp, "pid_alive", lambda p: True)
    assert rp.another_instance_running() == 4242


def test_进程已死则允许启动(status_path, monkeypatch):
    """上一条命令崩了不该把后面的更新永久锁死。"""
    _write(status_path, running=True, pid=4242, heartbeat_ts=time.time())
    monkeypatch.setattr(rp, "pid_alive", lambda p: False)
    assert rp.another_instance_running() is None


def test_心跳停了也算死(status_path, monkeypatch):
    _write(status_path, running=True, pid=4242, heartbeat_ts=time.time() - 999)
    monkeypatch.setattr(rp, "pid_alive", lambda p: True)
    assert rp.another_instance_running() is None


def test_已完成不阻塞(status_path):
    _write(status_path, running=False, ok=True, pid=4242)
    assert rp.another_instance_running() is None


def test_自己不算另一个实例(status_path, monkeypatch):
    _write(status_path, running=True, pid=os.getpid(), heartbeat_ts=time.time())
    monkeypatch.setattr(rp, "pid_alive", lambda p: True)
    assert rp.another_instance_running() is None


def test_状态文件坏掉时不阻塞(status_path):
    status_path.write_text("{ 这不是 json", encoding="utf-8")
    assert rp.another_instance_running() is None


# ---------------------------------------------------------------------------
# 三、收尾失败不该把成功的运行判成失败
# ---------------------------------------------------------------------------
def test_计划内阶段全完成时判成功(monkeypatch):
    """复刻 11-07 现场：plan 内 20 个阶段全 ok，进程死在收尾。"""
    from web import services as svc

    monkeypatch.setattr(svc, "_pipe_is_stale", lambda *a, **k: True)
    st = {"running": True, "pid": 999999, "heartbeat_ts": time.time() - 999,
          "ok": None, "plan": ["data", "factors", "verify"],
          "stages": [{"key": "data", "status": "ok"},
                     {"key": "factors", "status": "ok"},
                     {"key": "verify", "status": "ok"}]}
    # _mark_stale 是 staticmethod，直接调用
    got = svc.DataService._mark_stale(st)
    assert got["running"] is False
    assert got["ok"] is True, "阶段全跑完却判失败，会导致无谓重跑"
    assert got["interrupted"] is False
    assert "全部完成" in got["stale_reason"]


def test_中途被杀仍判失败(monkeypatch):
    from web import services as svc

    monkeypatch.setattr(svc, "_pipe_is_stale", lambda *a, **k: True)
    st = {"running": True, "pid": 999999, "heartbeat_ts": time.time() - 999,
          "ok": None, "plan": ["data", "factors", "verify"],
          "stages": [{"key": "data", "status": "ok"},
                     {"key": "factors", "status": "ok"}],
          "current": {"key": "verify", "name": "数据源一致性检测",
                      "started_at": "2026-11-07T06:39:39"}}
    got = svc.DataService._mark_stale(st)
    assert got["ok"] is False
    assert got["interrupted"] is True
    assert "中断" in got["stale_reason"]
