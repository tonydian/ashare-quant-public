# -*- coding: utf-8 -*-
"""「更新完提示还在」的回归测试（2026-11-11 事故）。

事故经过：用户点「开始更新」（走 /api/pipeline/run，一个完全脱离的子进程），
管道 26.7 分钟跑完，磁盘面板、因子、模型、信号全部更新到 11-11，
但 web 服务进程内存里的 `self.md` 还停在启动那天的 11-09。
`_stale_diag` 只看内存 → 一直报「已 2 个交易日未更新，请点开始更新」，
用户再点一次、再等半小时、还是同一句。这个循环重复了多次。

修法有两层：
1. `_probe_panel_changed()`：拿磁盘面板指纹（mtime+size）跟内存快照比对，
   变了就后台重载 —— 不管更新是谁触发的（UI / 命令行 / 另一个终端）。
2. `_stale_diag()`：磁盘比内存新时返回 kind=server_stale / action=reload，
   而不是 update_data。提示语从「请再更新一次」改成「正在自动重载」。

这里锁住这两层，避免退化回死循环。
"""
from __future__ import annotations

import sys
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.services import DataService  # noqa: E402


class _FakeMD:
    """只提供 _stale_diag 用到的字段，避免加载 260 MB 行情。"""

    def __init__(self, days):
        self.dates = pd.DatetimeIndex(days)
        self.T = len(self.dates)

    def date_index(self, s):
        return int(self.dates.get_loc(pd.Timestamp(s)))


def _svc(mem_days, sig=(1, 1)):
    """造一个「内存快照 = mem_days」的服务实例。

    loading=True 是为了抑制 _probe_panel_changed 真起重载线程（测试里不加载行情）。
    """
    svc = DataService.__new__(DataService)
    svc.md = _FakeMD(mem_days)
    svc.loading = True
    svc.reloading = False
    svc._panel_sig = sig
    svc._panel_probe_ts = 0.0
    svc._panel_reload_ts = 0.0
    return svc


@pytest.fixture
def disk_last(monkeypatch):
    """把「磁盘末交易日」固定成指定值，不依赖本机数据包的新旧。"""
    def _set(date_str):
        monkeypatch.setattr(DataService, "_last_date_of",
                            staticmethod(lambda path: date_str))
    return _set


def test_disk_ahead_mem_triggers_reload_not_update(disk_last):
    """磁盘 11-11、内存 11-09 → 必须让服务自己重载，不能叫用户再点更新。"""
    disk_last("2026-11-11")
    svc = _svc(["2026-11-06", "2026-11-09"])

    out = svc._stale_diag("2026-11-09")

    assert out["kind"] == "server_stale"
    assert out["action"] == "reload"
    assert out["data_last"] == "2026-11-11"      # 对外报磁盘真相
    assert out["mem_last"] == "2026-11-09"
    assert "不用再点更新" in out["message"]
    assert "请点「开始更新」" not in out["message"]


def test_mem_catches_up_then_quiet(disk_last):
    """重载完成后内存追上磁盘，提示应消失。"""
    disk_last("2026-11-11")
    svc = _svc(["2026-11-09", "2026-11-10", "2026-11-11"])

    out = svc._stale_diag("2026-11-11")

    assert out["level"] == "ok"
    assert out["action"] == "none"


def test_disk_also_old_still_asks_update(disk_last):
    """磁盘自己也旧 → 老提示要保留，别为了修 bug 把真问题藏掉。"""
    disk_last("2026-11-09")
    svc = _svc(["2026-11-06", "2026-11-09"])

    out = svc._stale_diag("2026-11-09")

    assert out["kind"] == "data_stale"
    assert out["action"] == "update_data"
    assert out["data_last"] == "2026-11-09"


def test_probe_detects_fingerprint_change(monkeypatch):
    """指纹变化要被识别成「磁盘已更新」；相同则不动。"""
    svc = _svc(["2026-11-09"], sig=(100, 200))

    monkeypatch.setattr(DataService, "_panel_fingerprint",
                        staticmethod(lambda: (100, 200)))
    assert svc._probe_panel_changed() is False, "指纹没变不该触发重载"

    svc._panel_probe_ts = 0.0                     # 绕过 8 秒节流
    monkeypatch.setattr(DataService, "_panel_fingerprint",
                        staticmethod(lambda: (999, 200)))
    assert svc._probe_panel_changed() is True, "指纹变了必须触发重载"
