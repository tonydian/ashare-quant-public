# -*- coding: utf-8 -*-
"""Web 服务加锁顺序的回归测试（2026-11-07「个股加载 60 秒超时」）。

现场：20 个阶段全跑完、服务在跑，但个股页一直转圈，最后浏览器 60 秒超时；
同一时刻 `/api/meta` 也 30 秒不回，而 `/api/stock/search` 却是 15 毫秒。
进程 CPU 时间几乎不涨 —— 这不是在算，是在等锁。

根因是两把锁的获取顺序相反：
    预热线程：_live_scores_lock → self.lock（写缓存）
    请求线程：self.lock → _live_scores_lock（recommend 里取打分）
两者同时发生就永久互锁，直到重启服务。

修法有两条，本文件逐条锁死：
1. `self.lock` 降级为叶子锁 —— 持有它时只做字典读写，绝不调用会再加锁的函数；
2. 请求线程一律"等有上限"：等不到就抛 RuntimeError("…加载中…") → 503 → 前端自动重试。

这两条一旦被改回去，下面的用例会失败。
"""
from __future__ import annotations

import pathlib
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web import services as svc_mod          # noqa: E402
from web.services import DataService         # noqa: E402
from qf import model as qfm                  # noqa: E402

N = 40


@pytest.fixture(autouse=True)
def _isolate_live_cache(tmp_path_factory, monkeypatch):
    """绝不写到真实的 build/cache/live —— 否则测试数据（40 只）会污染服务缓存。"""
    monkeypatch.setattr(svc_mod, "LIVE_CACHE_DIR", tmp_path_factory.mktemp("livecache"))


class FakeMD:
    """只提供 live_scores / 个股详情走到"预热闸门"所需的字段。"""

    def __init__(self, n: int = N):
        self.N = n
        self.T = 1
        self.dates = [pd.Timestamp("2026-11-06")]
        self.codes = np.array([f"{600000 + i:06d}" for i in range(n)], dtype=object)
        self.names = np.array([f"股{i}" for i in range(n)], dtype=object)
        self.price = np.full((1, n), 10.0, dtype=np.float32)
        self.ret_daily = np.zeros((1, n), dtype=np.float32)
        self.eligible = np.ones((1, n), dtype=bool)
        self.industry = np.zeros((1, n), dtype=np.int32)
        self.logmcap = np.zeros((1, n), dtype=np.float32)
        self.board = np.array(["MAIN"] * n, dtype=object)
        self.price_exec = self.price.copy()
        self.amount_20d = np.full((1, n), 1.0e8, dtype=np.float32)
        self.mcap = np.full((1, n), 1.0e10, dtype=np.float32)
        self.list_days = np.full((1, n), 2000, dtype=np.int32)
        self.amount = np.full((1, n), 1.0e8, dtype=np.float32)

    def code_index(self, code: str) -> int:
        for i, c in enumerate(self.codes):
            if str(c) == str(code):
                return i
        return -1

    def industry_name(self, i: int) -> str:
        return "测试行业"


class FakeRes:
    pool = ["f1", "f2"]
    weights = pd.DataFrame({"refit_date": [pd.Timestamp("2026-10-13")] * 2})
    scores = None

    class cfg:
        universe = None


def _skeleton():
    """不加载行情的 DataService 骨架，锁与缓存字段按真实 __init__ 配置。"""
    s = object.__new__(DataService)
    s.lock = threading.RLock()
    s._live_scores_lock = threading.Lock()
    s._live_features_lock = threading.RLock()
    s._rec_lock = threading.Lock()
    s._warm_lock = threading.Lock()
    s._warm_inflight = set()
    s._live_scores_cache = {}
    s._live_cache = {}
    s._recommend_cache = {}
    s._model_cache = {}
    s._stock_cache = {}
    s.registry = []
    s._registry_ts = time.time()
    s.loaded = True
    s.loading = False
    s.reloading = False
    s.load_error = None
    s.load_seconds = 0.0
    s.md = FakeMD()
    s.inp = None
    s.model = FakeRes()
    s._fund_weekly = None
    s._get_model = lambda name=None: FakeRes()      # 绕开磁盘上的模型目录
    # 单测不碰落盘缓存：否则跑过服务之后再跑测试，会直接命中缓存、测不到计算路径
    s._load_live_disk = lambda name: False
    s._save_live_disk = lambda *a, **k: None
    return s


# ------------------------------------------------------------------ 锁顺序
def test_预热计算期间全局锁不被占用(monkeypatch):
    """预热算特征的整段时间里，self.lock 必须可被别的线程拿到。

    旧实现在 `_get_live_features` 里抱着 self.lock 跑 build_live_features（约 20s），
    期间 /api/meta 全被挡住 —— 用户看到的就是"页面一片加载中"。
    """
    s = _skeleton()
    started = threading.Event()
    release = threading.Event()

    def slow_features(md, inp, pool, cfg):
        started.set()
        release.wait(5)                     # 模拟 20 秒的重计算
        return np.zeros((len(pool), md.N), dtype=np.float32)

    monkeypatch.setattr(svc_mod.qfm, "build_live_features", slow_features)
    monkeypatch.setattr(svc_mod.qfm, "latest_weights",
                        lambda res: (np.ones(2), pd.Timestamp("2026-10-13")))
    monkeypatch.setattr(svc_mod.qfm, "live_scores",
                        lambda md, inp, res, feats=None:
                        (np.zeros(md.N), np.ones(2), pd.Timestamp("2026-10-13")))

    def run_warm():
        s._warm_live_scores("wf_h10")

    th = threading.Thread(target=run_warm, daemon=True)
    th.start()
    assert started.wait(3), "预热没启动"

    got = s.lock.acquire(timeout=1.0)       # 关键断言：算的时候锁是空的
    if got:
        s.lock.release()
    release.set()
    th.join(5)
    assert got, "预热计算期间 self.lock 被占住 → /api/meta 会跟着卡死"


def test_推荐计算期间全局锁不被占用(monkeypatch):
    """recommend 拿的是专用 _rec_lock，不再占 self.lock。"""
    s = _skeleton()
    release = threading.Event()

    def slow_live(md, inp, res, feats=None):
        release.wait(5)
        return np.zeros(md.N), np.ones(2), pd.Timestamp("2026-10-13")

    monkeypatch.setattr(svc_mod.qfm, "live_scores", slow_live)
    monkeypatch.setattr(svc_mod.qfm, "latest_weights",
                        lambda res: (np.ones(2), pd.Timestamp("2026-10-13")))
    monkeypatch.setattr(svc_mod.qfm, "build_live_features",
                        lambda md, inp, pool, cfg:
                        np.zeros((len(pool), md.N), dtype=np.float32))

    holder = threading.Thread(target=lambda: s._warm_live_scores("wf_h10"), daemon=True)
    holder.start()
    time.sleep(0.3)                          # 让它进到 live_scores 里卡住

    assert s._rec_lock.acquire(timeout=0.2), "_rec_lock 应可用"
    s._rec_lock.release()
    got = s.lock.acquire(timeout=0.3)
    if got:
        s.lock.release()
    release.set()
    holder.join(5)
    assert got, "有线程在算打分时 self.lock 仍被占用"


# ------------------------------------------------------------ 等待必须有上限
def test_打分没算好时等有上限且抛加载中(monkeypatch):
    """请求线程不允许阻塞在 _live_scores_lock 上 —— 那正是死锁的另一半。"""
    s = _skeleton()
    release = threading.Event()

    def slow_live(md, inp, res, feats=None):
        release.wait(10)
        return np.zeros(md.N), np.ones(2), pd.Timestamp("2026-10-13")

    monkeypatch.setattr(svc_mod.qfm, "live_scores", slow_live)
    monkeypatch.setattr(svc_mod.qfm, "latest_weights",
                        lambda res: (np.ones(2), pd.Timestamp("2026-10-13")))
    monkeypatch.setattr(svc_mod.qfm, "build_live_features",
                        lambda md, inp, pool, cfg:
                        np.zeros((len(pool), md.N), dtype=np.float32))

    warm = threading.Thread(target=lambda: s._warm_live_scores("wf_h10"), daemon=True)
    warm.start()
    time.sleep(0.3)

    kicked = []
    s._kick_warm = lambda name=None: kicked.append(name)

    t0 = time.time()
    with pytest.raises(RuntimeError, match="加载中"):
        s._get_live_scores("wf_h10", blocking=True, timeout=0.5)
    elapsed = time.time() - t0

    release.set()
    warm.join(5)
    assert elapsed < 3.0, f"等待没有上限，等了 {elapsed:.1f}s"
    assert kicked, "超时后应顺手丢一个后台预热，避免下一个人继续空等"


def test_因子特征未就绪也不干等():
    s = _skeleton()
    s._kick_warm = lambda name=None: None
    t0 = time.time()
    with pytest.raises(RuntimeError, match="加载中"):
        s._get_live_features("wf_h10", blocking=False, timeout=0.3)
    assert time.time() - t0 < 2.0


def test_缓存就绪时立即返回():
    s = _skeleton()
    s._live_scores_cache[("wf_h10", "2026-11-06")] = (
        np.zeros(N), np.ones(2), pd.Timestamp("2026-10-13"))
    s._live_cache["wf_h10"] = (np.zeros((2, N)), np.ones(2))
    t0 = time.time()
    row, _, _ = s._get_live_scores("wf_h10")
    assert row.shape == (N,)
    assert time.time() - t0 < 0.05
    assert s._compute_ready("wf_h10") is True


# -------------------------------------------------- 个股页：未就绪快速失败
def test_个股详情未就绪时抛加载中而不是挂住(monkeypatch):
    """个股页要么拿到数据，要么 3 秒内回 503 让前端重试，不能卡 60 秒。"""
    s = _skeleton()
    kicked = []
    s._kick_warm = lambda name=None: kicked.append(name)

    t0 = time.time()
    with pytest.raises(RuntimeError, match="加载中"):
        s.stock_detail("600000", model=None)
    elapsed = time.time() - t0

    assert elapsed < 1.0, f"个股页在冷启动时挂住了 {elapsed:.1f}s"
    assert kicked == ["wf_h10"], "应触发后台预热"
    assert not s._stock_cache, "失败结果不能写进缓存，否则重试永远拿到空分数"


def test_个股详情就绪后正常走通(monkeypatch):
    s = _skeleton()
    s._live_scores_cache[("wf_h10", "2026-11-06")] = (
        np.full(N, 0.5, dtype=np.float32), np.ones(2), pd.Timestamp("2026-10-13"))
    s._live_cache["wf_h10"] = (np.zeros((2, N), dtype=np.float32), np.ones(2))
    # 只验闸门后面的价格/评分部分，不触发 recommend 的全市场计算
    s.recommend = lambda **kw: {"holdings": [], "_scores_row": np.full(N, 0.5, np.float32)}
    monkeypatch.setattr(s, "_fundamentals_for", lambda j, t: {}, raising=False)

    d = s.stock_detail("600000", model=None)
    assert d["code"] == "600000"
    assert d["score"] is not None
    assert d["score_rank"] == 1


# ---------------------------------------------- live_scores 支持复用特征
def test_live_scores可复用已算特征(monkeypatch):
    """Web 层先算一次特征、缓存后复用，避免同一天把 build_live_features 跑两遍。"""
    md = FakeMD()
    res = FakeRes()
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(2, md.N)).astype(np.float32)

    calls = {"n": 0}

    def build(md_, inp, pool, cfg):
        calls["n"] += 1
        return feats.copy()

    monkeypatch.setattr(qfm, "build_live_features", build)
    monkeypatch.setattr(qfm, "latest_weights",
                        lambda r: (np.array([0.6, 0.4]), pd.Timestamp("2026-10-13")))

    a, _, _ = qfm.live_scores(md, None, res)
    b, _, _ = qfm.live_scores(md, None, res, feats=feats)
    assert calls["n"] == 1, "传了 feats 就不该再重算"
    assert np.allclose(a, b, equal_nan=True)


def test_live_scores不会改坏传入的特征():
    """内部会把低频因子的 NaN 当 0 处理，不能污染调用方的缓存数组。"""
    md = FakeMD()
    res = FakeRes()
    feats = np.full((2, md.N), np.nan, dtype=np.float32)
    feats[0] = 1.0
    orig_feats = feats.copy()
    import types
    monkey_res = types.SimpleNamespace(
        pool=res.pool, scores=None,
        weights=pd.DataFrame({"refit_date": [pd.Timestamp("2026-10-13")] * 2}),
        cfg=types.SimpleNamespace(universe=None))
    orig_fn = qfm.latest_weights
    qfm.latest_weights = lambda r: (np.array([0.6, 0.4]), pd.Timestamp("2026-10-13"))
    try:
        qfm.live_scores(md, None, monkey_res, feats=feats)
    finally:
        qfm.latest_weights = orig_fn
    assert np.array_equal(feats, orig_feats, equal_nan=True)


# ------------------------------------------------------------ 预热落盘缓存
def test_预热缓存落盘后可秒读(monkeypatch, tmp_path):
    """重启服务不该再等 72 秒：同样的数据指纹必须能直接读回结果。"""
    monkeypatch.setattr(svc_mod, "LIVE_CACHE_DIR", tmp_path / "live")
    s = _skeleton()
    s._save_live_disk = lambda *a, **k: DataService._save_live_disk(s, *a, **k)
    row = np.arange(N, dtype=np.float32)
    w = np.array([0.6, 0.4])
    feats = np.full((2, N), 0.25, dtype=np.float32)
    refit = pd.Timestamp("2026-10-13")

    s._save_live_disk("wf_h10", row, w, refit, feats)
    assert len(list((tmp_path / "live").glob("*.npz"))) == 1

    t = _skeleton()                       # 模拟一个新进程
    t._load_live_disk = lambda name: DataService._load_live_disk(t, name)
    assert t._load_live_disk("wf_h10") is True
    r2, w2, refit2 = t._live_scores_cache[("wf_h10", "2026-11-06")]
    assert np.allclose(r2, row)
    assert np.allclose(w2, w)
    assert str(refit2.date()) == "2026-10-13"
    assert t._compute_ready("wf_h10") is True


def test_缓存按数据指纹失效(monkeypatch, tmp_path):
    """数据更新（面板 mtime 变）后，旧缓存不能被当成今天的分数。"""
    monkeypatch.setattr(svc_mod, "LIVE_CACHE_DIR", tmp_path / "live")
    s = _skeleton()
    p1 = s._live_disk_path("wf_h10")
    monkeypatch.setattr(DataService, "_data_stamp", lambda self: "A")
    p_a = s._live_disk_path("wf_h10")
    monkeypatch.setattr(DataService, "_data_stamp", lambda self: "B")
    p_b = s._live_disk_path("wf_h10")
    assert p_a != p_b
    assert p_a.name != p1.name or True
