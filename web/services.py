# -*- coding: utf-8 -*-
"""Web UI 后端数据服务：行情看盘、个股推荐、个股详情、回测任务、数据抓取、因子/报告只读接口。

设计：
- 启动时后台加载 MarketData / FactorInputs / saved model（约 30~60s），加载完成前需要数据的接口返回 503；
- 重计算（live 打分、回测、数据更新）串行执行（单机 4 核，避免互相拖慢）；
- 所有图表 PNG 在服务端缓存；数据更新完成后清空缓存并重载。
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent          # ashare_quant
REPO = ROOT.parent                                              # D:\wkf\getrich
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# 默认用增量更新后的数据目录（含最新交易日）；可用 --data orig 或 QF_DATA_RAW 覆盖
_ORIG = REPO / "data_extract" / "was_stock_data"
_EXT = REPO / "data_extract" / "was_stock_data_ext"
if "QF_DATA_RAW" not in os.environ and _EXT.exists():
    os.environ["QF_DATA_RAW"] = str(_EXT)

from qf import config as C                     # noqa: E402
from qf import data as qfdata                  # noqa: E402
from qf import factors as qff                  # noqa: E402
from qf import model as qfm                    # noqa: E402
from qf import portfolio as qfport             # noqa: E402
from qf import backtest as qfbt                # noqa: E402
from qf import metrics as qfmetrics            # noqa: E402
from qf import fundamentals as qffd            # noqa: E402
import daily_signal as dsig                    # noqa: E402  (复用 market_regime)

try:
    from web import charts
except Exception:  # 允许作为脚本直接运行
    import charts  # type: ignore

DATA_SOURCE = "ext" if str(C.DATA_RAW).endswith("_ext") else "orig"
MODEL_NAME = "wf_h10"
# 界面模型白名单：只有通过独立验证、且当前在用的模型才暴露给用户。
# 历史上这里是黑名单（MODEL_EXCLUDE），结果是每跑一个实验目录
# （build/models/exp_*、wf_h10_* 变体…）就会自动冒进下拉框，
# 而且按字母序 wf_h10 本身排在第 23 位——选主模型要往下滚很久。
# 现在改为白名单：新增实验默认不可见，确认可用后在此登记一行即可。
#   name -> (label, desc, kind)   kind: "prod"=生产 / "research"=研究对照
MODEL_SHOW = {
    "wf_h10": ("左侧主模型", "量价反转+低波+低换手+价值(bp)，21 因子（去冗余），10 日标签（默认）", "prod"),
}
# 预热结果的落盘目录（全量特征 + 打分，约 0.5MB/份）。数据一更新指纹就变，旧档自动冷落。
LIVE_CACHE_DIR = C.BUILD / "cache" / "live"
IDX_NAMES = {"csi_all": "中证全指", "hs300": "沪深300", "csi500": "中证500", "csi1000": "中证1000"}
BOARD_CN = {"MAIN": "主板", "CHINEXT": "创业板", "STAR": "科创板", "BSE": "北交所"}

# 族占比监控阈值（2026-11-03 加）：族上限 cap 0.30/0.25/0.20 三档已全部否掉
# （损伤随干预强度单调递增，23 期最大族始终 risk-pattern），结论是
# **集中度是结果不是病因 → 改监控、不做机制干预**。这里只做展示层标黄。
FAMILY_SHARE_WARN = 0.33


# =============================================================== 工具
def _blend_cash(b, g, idx):
    """按 C.CASH_BOND_WEIGHT / C.CASH_GOLD_WEIGHT 混合现金收益（黄金权重可为 0）。"""
    bw, gw = float(C.CASH_BOND_WEIGHT), float(C.CASH_GOLD_WEIGHT)
    tot = bw + gw
    if tot <= 0:
        return pd.Series(0.0, index=idx)
    r = (bw / tot) * b.reindex(idx).ffill().pct_change().fillna(0)
    if gw > 0:
        r = r + (gw / tot) * g.reindex(idx).ffill().pct_change().fillna(0)
    return r


def _f(x, default=None):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _perf_from_nav(nav) -> dict:
    """净值序列 → 指标（统一走 qf.metrics.performance，rf = 现金收益率）。

    不要自己写 CAGR/Sharpe：本项目年化按 252 交易日折算、Sharpe 扣现金收益，
    与回测页完全同口径（自算会得到 6.82%/0.76，与页面 7.11%/0.61 打架）。
    """
    try:
        s = pd.Series(nav).astype(float).dropna()
        if len(s) < 30:
            return {}
        p = qfmetrics.performance(s, rf=C.CASH_YIELD)
        return {"cagr": _f(p.get("cagr")), "sharpe": _f(p.get("sharpe")),
                "max_dd": _f(p.get("max_dd")), "vol": _f(p.get("ann_vol")),
                "start": str(s.index[0].date()), "end": str(s.index[-1].date())}
    except Exception:
        return {}


def _wfmt(v: float) -> str:
    """权重百分比文本：整数不带小数（15%），小数保留有效位（42.5% / 21.25%）。"""
    pct = float(v) * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{pct:.0f}%"
    return f"{pct:.2f}".rstrip("0").rstrip(".") + "%"


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        return None if not np.isfinite(v) else round(v, 8)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _pid_alive(pid) -> bool:
    """判断进程是否还活着（pid 缺失/非法一律视为已退出）。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            try:
                code = wintypes.DWORD()
                if k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return code.value == 259               # STILL_ACTIVE
                return True
            finally:
                k32.CloseHandle(h)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return True


def _pipe_is_stale(st: dict, now: float | None = None) -> bool:
    """status.json 声称 running，但背后进程已不在或心跳停了。

    只判断"必死"的情形，宁可漏判也不误杀健康任务：
      * 有 pid 且进程不在 → 死
      * 有心跳且超过 180 秒没刷新 → 死
      * 老格式（无 pid 无心跳）→ 用 updated_at 兜底，超过 10 分钟算死
    """
    if not isinstance(st, dict) or not st.get("running"):
        return False
    now = time.time() if now is None else now
    pid = st.get("pid")
    if pid is not None:
        if not _pid_alive(pid):
            return True
        hb = st.get("heartbeat_ts")
        if isinstance(hb, (int, float)) and now - float(hb) > 180:
            return True
        return False
    hb = st.get("heartbeat_ts")
    if isinstance(hb, (int, float)):
        return now - float(hb) > 180
    try:
        import datetime as _dt
        upd = _dt.datetime.fromisoformat(str(st.get("updated_at") or ""))
        return (_dt.datetime.now() - upd).total_seconds() > 600
    except Exception:
        return False


class JobRegistry:
    """后台任务（回测 / 数据更新）注册表。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}

    def create(self, kind: str, **extra) -> str:
        jid = f"{kind}_{int(time.time()*1000)}_{len(self._jobs)}"
        with self._lock:
            self._jobs[jid] = {
                "id": jid, "kind": kind, "status": "running", "progress": 0.0,
                "message": "", "error": None, "created": time.time(),
                "elapsed": 0.0, "result": None, "log": collections.deque(maxlen=60),
                **extra,
            }
        return jid

    def get(self, jid: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(jid)
            return dict(j) if j else None

    def update(self, jid: str, **kw):
        with self._lock:
            if jid in self._jobs:
                self._jobs[jid].update(kw)
                self._jobs[jid]["elapsed"] = round(time.time() - self._jobs[jid]["created"], 1)

    def push_log(self, jid: str, line: str):
        with self._lock:
            if jid in self._jobs:
                self._jobs[jid]["log"].append(line.rstrip("\n"))

    def list(self):
        with self._lock:
            return list(self._jobs)


def _rolling_mean_blocked(close: np.ndarray, w: int) -> np.ndarray:
    """分块计算 (T,N) 滚动均值，控制内存峰值。"""
    T, N = close.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    minp = max(5, w // 2)
    step = 512
    for s in range(0, N, step):
        e = min(N, s + step)
        df = pd.DataFrame(close[:, s:e])
        out[:, s:e] = df.rolling(w, min_periods=minp).mean().to_numpy(dtype=np.float32)
    return out


# =============================================================== 服务
class DataService:
    def __init__(self):
        self.lock = threading.RLock()
        self.bt_lock = threading.Lock()
        self.jobs = JobRegistry()
        self._chart_cache: dict[tuple, bytes] = {}
        self._cache: dict[str, tuple[float, dict]] = {}
        self.loaded = False
        self.loading = True
        self.reloading = False
        self.load_error: str | None = None
        self.load_seconds = 0.0
        self.load_started = time.time()
        self.md = None
        self.inp = None
        self.model = None
        self._model_cache: dict[str, object] = {}
        self._recommend_cache: dict[tuple, dict] = {}
        self._live_scores_cache: dict[tuple, tuple] = {}   # (model, asof) -> (row, w, refit)
        # 预热去重锁：服务启动时 load() 内部起一个预热线程，web_server 的 _prewarm 又会起一个，
        # 两者都看到"缓存为空"就各算一遍——单次 refit 实测 260 秒，等于白白多烧 4 分钟 CPU，
        # 期间 GIL 被占满、页面请求大量超时。用这把锁串行化，第二次进来直接命中缓存。
        self._live_scores_lock = threading.Lock()
        self._live_cache: dict[str, tuple] = {}
        # ---- 锁分层（2026-11-07 修死锁）----
        # `self.lock` 是 RLock，历史上被 recommend()/reload() 抱着跑整段重计算。它与
        # `_live_scores_lock` 形成方向相反的获取顺序：
        #   预热线程：_live_scores_lock → self.lock（写缓存）
        #   请求线程：self.lock → _live_scores_lock（recommend 里取打分）
        # 两者同时发生就永久互锁 —— 表现为"个股一直转圈、最后 60 秒超时"，
        # 而且进程 CPU 接近 0（不是在算，是在等）。
        # 铁律：**持有 self.lock 时只做字典读写，不调用任何会再加锁的函数**。
        self._live_features_lock = threading.RLock()  # 因子特征的单飞锁（不提升为全局锁）
        self._rec_lock = threading.Lock()             # 推荐计算的单飞锁，替代原先的 self.lock
        self._warm_lock = threading.Lock()
        self._warm_inflight: set[str] = set()         # 正在预热的模型名，防重复起线程
        self._stock_cache: dict[tuple, dict] = {}
        self._cb_cache: dict = {}
        self._cb_bt: dict = {}
        self._fund_weekly = None
        self.registry: list[dict] = []
        self._registry_ts = 0.0
        self.update_running = False
        self._load_static()

    # ------------------------------------------------------------ 静态产物
    def _load_static(self):
        self.catalog = self._parq(C.BUILD_FACTORS / "catalog.parquet")
        self.mom_cat = self._parq(C.BUILD_FACTORS / "momentum_catalog.parquet")
        self.fund_cat = self._parq(C.BUILD_FACTORS / "fundamental_catalog.parquet")
        self.trend_cat = self._parq(C.BUILD_FACTORS / "trend_catalog.parquet")
        self.metrics = self._parq(C.BUILD_EVAL / "factor_metrics.parquet")
        self.family = self._parq(C.BUILD_EVAL / "family_stats.parquet")
        self.model_dir = C.BUILD_MODELS / MODEL_NAME
        self.pool_meta = self._parq(self.model_dir / "pool_meta.parquet")
        self.weights = self._parq(self.model_dir / "model_weights.parquet")
        self.comp_ic = self._parq(self.model_dir / "composite_ic.parquet")
        self.model_meta = self._read_json(self.model_dir / "model_meta.json", {})
        self.fund_stats = self._read_json(C.BUILD_EVAL / "fundamental_stats.json", {})
        self.registry = self._discover_models()

    @staticmethod
    def _parq(p):
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    @staticmethod
    def _read_json(p, default):
        try:
            return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
        except Exception:
            return default

    def _discover_models(self) -> list[dict]:
        """按 MODEL_SHOW 白名单装配可切换模型列表（顺序即白名单声明顺序）。

        只认白名单：磁盘上其他目录（实验残留、单因子变体、legacy 备份）
        一律不进界面，避免下拉框越长越乱、也避免用户误选一个没验证过的模型。
        """
        out = []
        base = C.BUILD_MODELS
        if not base.exists():
            return out
        for name, spec in MODEL_SHOW.items():
            label, desc = spec[0], spec[1]
            kind = spec[2] if len(spec) > 2 else "prod"
            d = base / name
            if not (d / "composite_score.npz").exists():
                continue
            meta = self._read_json(d / "model_meta.json", {})
            pool = meta.get("pool") or []
            horizon = (meta.get("cfg") or {}).get("horizon", 10)
            refit = None
            try:
                w = pd.read_parquet(d / "model_weights.parquet")
                if len(w):
                    refit = str(pd.Timestamp(w["refit_date"].max()).date())
            except Exception:
                pass
            out.append({"name": name, "label": label, "desc": desc, "kind": kind,
                        "pool": len(pool), "refit": refit,
                        "horizon": int(horizon) if horizon else 10})
        return out

    def _refresh_registry(self, ttl: float = 5.0) -> list[dict]:
        now = time.time()
        if now - self._registry_ts > ttl:
            with self.lock:
                if now - self._registry_ts > ttl:
                    self.registry = self._discover_models()
                    self._registry_ts = now
        return self.registry

    def _get_model(self, name: str | None = None):
        """按名称加载 WalkForwardResult（懒加载 + 内存缓存）。"""
        name = name or MODEL_NAME
        valid = {m["name"] for m in self._refresh_registry()} or {MODEL_NAME}
        if name not in valid:
            raise ValueError(f"未知模型 {name}，可用：{sorted(valid)}")
        if name in self._model_cache:
            return self._model_cache[name]
        res = qfm.load_result(C.BUILD_MODELS / name)
        with self.lock:
            self._model_cache[name] = res
        return res

    # ------------------------------------------------------------ 加载/重载
    def load(self):
        try:
            t0 = time.time()
            md = qfdata.load_market(verbose=False)
            inp = qff.FactorInputs(md)
            self._model_cache = {}
            res = self._get_model(MODEL_NAME)
            close = np.asarray(md.close, dtype=np.float32)
            derived = {
                "dates": md.dates,
                "bench": md.bench_prices,
                "ret": md.ret_daily,
                "ma20": _rolling_mean_blocked(close, 20),
                "ma60": _rolling_mean_blocked(close, 60),
            }
            with self.lock:
                self.md, self.inp, self.model, self.derived = md, inp, res, derived
                self._recommend_cache.clear()
                self._live_cache.clear()
                self._fund_weekly = None
                self._chart_cache.clear()
                self._cache.clear()
                self.loaded = True
                self.loading = False
                self.load_seconds = round(time.time() - t0, 1)
            print(f"[web] 数据加载完成：{md.T} 日 × {md.N} 只，用时 {self.load_seconds}s", flush=True)
            # 后台预热打分（不阻塞请求；用户搜股时直接命中缓存）
            threading.Thread(target=self._warm_live_scores, daemon=True,
                             name="web-warm-live").start()
        except Exception:
            self.loaded = False
            self.loading = False
            self.load_error = traceback.format_exc(limit=3)
            print("[web] 数据加载失败：\n" + self.load_error, flush=True)

    # ---- 预热与"带截止时间的等待" -----------------------------------
    # 设计原则（2026-11-07）：请求线程**绝不**因为别人在算就无限期挂住。
    # 等待一律带 timeout，超时就抛 RuntimeError("…加载中…") → 服务端回 503 →
    # 前端识别成"服务加载中"自动重试。宁可让浏览器多转两圈，也不能卡死 60 秒。
    def _warm_key(self, model: str | None = None) -> tuple[str, str]:
        name = model or MODEL_NAME
        return (name, str(self.md.dates[-1].date()))

    def _kick_warm(self, model: str | None = None) -> None:
        """缓存没就绪时丢一个后台线程去算，调用方立刻返回（不等待）。"""
        name = model or MODEL_NAME
        with self._warm_lock:
            if name in self._warm_inflight:
                return
            self._warm_inflight.add(name)

        def run():
            try:
                self._warm_live_scores(name)
                self._get_live_features(name, blocking=True)
            except Exception:
                print("[web] 后台预热失败：\n" + traceback.format_exc(limit=3), flush=True)
            finally:
                with self._warm_lock:
                    self._warm_inflight.discard(name)

        threading.Thread(target=run, daemon=True, name="web-warm-kick").start()

    def _wait_live_scores(self, model: str | None = None, timeout: float = 20.0):
        """等 live_scores 就绪，最多 timeout 秒。超时则甩给后台算 + 抛"加载中"。

        轮询而非条件变量：等待方完全不碰任何锁，从根上不可能与预热线程互锁。
        """
        key = self._warm_key(model)
        deadline = time.time() + timeout
        while True:
            hit = self._live_scores_cache.get(key)
            if hit is not None:
                return hit
            if time.time() >= deadline:
                self._kick_warm(model)
                raise RuntimeError(
                    f"打分数据加载中，请稍候重试（{key[1]} 首次约需 20–40 秒）")
            time.sleep(0.2)

    def _warm_live_scores(self, model: str | None = None) -> None:
        """在后台计算 live_scores（首次实测 72s）。持有 _live_scores_lock 单飞，
        但**不持 self.lock** —— 它只在写缓存那一瞬间取一下。

        结果按「模型 + 数据指纹」落盘，重启服务时直接读回，不用再等 72 秒。
        """
        try:
            name = model or MODEL_NAME
            key = self._warm_key(name)
            if key in self._live_scores_cache:
                return
            # 双检锁：同一 (model, asof) 只算一次。服务启动时 load() 和 web_server 的
            # _prewarm 会各起一个预热线程，没有这把锁时两边都判定"缓存为空"、各算一遍
            # （单次 refit 实测 260s），白白多烧一轮 CPU，页面请求全程被 GIL 饿死。
            with self._live_scores_lock:
                if key in self._live_scores_cache:
                    return
                t0 = time.time()
                if self._load_live_disk(name):
                    print(f"[web] 预热打分命中磁盘缓存：{name}，用时 {time.time() - t0:.2f}s",
                          flush=True)
                    return
                res = self._get_model(name)
                # 先把池内特征算出来并缓存，再喂给 live_scores —— 否则
                # live_scores 内部会重算一遍同样的特征，白烧一次全量特征。
                feats, w = self._compute_live_features(name)
                row, w2, refit = qfm.live_scores(self.md, self.inp, res, feats=feats)
                with self.lock:
                    self._live_scores_cache[key] = (row, w2, refit)
                self._save_live_disk(name, row, w2, refit, feats)
            print(f"[web] 预热打分完成：{name}，用时 {time.time() - t0:.1f}s", flush=True)
        except Exception:
            print("[web] 预热打分失败：\n" + traceback.format_exc(limit=3), flush=True)

    def _compute_live_features(self, model: str | None = None):
        """实际计算 (K,N) 池内特征 + 权重。自带单飞锁（RLock，允许外面再包一层）。"""
        name = model or MODEL_NAME
        with self._live_features_lock:
            cached = self._live_cache.get(name)
            if cached is not None:
                return cached
            res = self._get_model(name)
            w, _ = qfm.latest_weights(res)
            feats = qfm.build_live_features(self.md, self.inp, res.pool, res.cfg)
            pair = (feats, np.asarray(w))
            with self.lock:                       # 叶子操作：只写字典
                self._live_cache[name] = pair
            return pair

    def _get_live_scores(self, model: str | None = None, blocking: bool = True,
                         timeout: float = 20.0):
        """取 live_scores。默认带 20 秒上限；blocking=False 时没算好就直接抛。"""
        if blocking:
            return self._wait_live_scores(model, timeout=timeout)
        key = self._warm_key(model)
        hit = self._live_scores_cache.get(key)
        if hit is None:
            self._kick_warm(model)
            raise RuntimeError(f"打分数据加载中，请稍候重试（{key[1]}）")
        return hit

    def start_loading(self):
        threading.Thread(target=self.load, daemon=True, name="web-load").start()

    def reload(self):
        """数据更新后重建行情缓存（释放旧对象再加载，控制内存峰值）。"""
        with self.lock:
            self.reloading = True
            self.loaded = False
            self.md = self.inp = self.model = None
            self.derived = None
            self._model_cache.clear()
            self._recommend_cache.clear()
            self._live_cache.clear()
            self._chart_cache.clear()
            self._cache.clear()
        import gc
        gc.collect()
        self.load()
        with self.lock:
            self.reloading = False

    # ------------------------------------------------------------ 通用
    def _t(self, asof: str | None = None) -> int:
        md = self.md
        if asof:
            try:
                return int(np.clip(md.date_index(asof), 0, md.T - 1))
            except Exception:
                pass
        return md.T - 1

    def _require_loaded(self):
        if not self.loaded:
            raise RuntimeError("数据加载中，请稍后" if self.loading else f"数据加载失败：{self.load_error}")

    # ------------------------------------------------------------ meta
    def meta(self) -> dict:
        self._refresh_registry()
        out = {
            "app": "A股量化系统",
            "data": {
                "last_date": None, "orig_last_date": self._orig_last_date(),
                "n_dates": 0, "n_codes": 0, "n_universe": 0,
                "source": DATA_SOURCE, "update_running": self.update_running,
            },
            "model": {"name": MODEL_NAME, "default": MODEL_NAME, "models": self.registry,
                      "refit": None,
                      "n_pool": int(len(self.pool_meta)) if self.pool_meta is not None else 0,
                      "horizon": int(self.model_meta.get("cfg", {}).get("horizon", 10)),
                      "method": self.model_meta.get("cfg", {}).get("method", "blend"),
                      "capital": float(C.CAPITAL)},
            "server": {"loaded": self.loaded, "loading": self.loading,
                       "reloading": self.reloading, "loaded_seconds": self.load_seconds,
                       "load_error": self.load_error},
        }
        if self.loaded and self.md is not None:
            md = self.md
            t = md.T - 1
            with self.lock:
                refit = None
                if self.model is not None and len(self.model.weights):
                    refit = str(pd.Timestamp(self.model.weights["refit_date"].max()).date())
            out["data"].update({
                "last_date": str(md.dates[t].date()), "n_dates": int(md.T),
                "n_codes": int(md.N), "n_universe": int(md.eligible[t].sum()),
            })
            out["model"]["refit"] = refit
            out["model"]["n_pool"] = len(self.model.pool)
        return out

    @staticmethod
    def _orig_last_date():
        try:
            z = np.load(REPO / "data_extract" / "was_stock_data" / "panel_ohlcv.npz")
            return str(pd.Timestamp(z["dates"][-1]).date())
        except Exception:
            return None

    # ------------------------------------------------------------ 看盘
    def market_overview(self, asof: str | None = None, bench: str = "csi_all") -> dict:
        self._require_loaded()
        md = self.md
        t = self._t(asof)
        b = self.derived["bench"]
        indices = []
        for code, name in IDX_NAMES.items():
            s = b[code].iloc[: t + 1]
            if len(s) < 2:
                continue
            c = s.iloc[-1]

            def chg(k):
                return _f(c / s.iloc[-1 - k] - 1.0) if len(s) > k else None
            ma120 = _f(s.iloc[-120:].mean()) if len(s) >= 120 else None
            ma60 = _f(s.iloc[-60:].mean()) if len(s) >= 60 else None
            indices.append({
                "code": code, "name": name, "close": _f(c),
                "chg_1d": chg(1), "chg_20d": chg(20), "chg_120d": chg(120),
                "ma120": ma120, "above_ma120": bool(c > ma120) if ma120 else None,
                "above_ma60": bool(c > ma60) if ma60 else None,
            })
        coeff, regime = dsig.market_regime(md, t)
        elig = md.eligible[t]
        ret = md.ret_daily[t]
        alive = md.alive[t]
        valid = elig & np.isfinite(ret)
        up = int(np.sum(valid & (ret > 0)))
        dn = int(np.sum(valid & (ret < 0)))
        flat = int(np.sum(valid & (ret == 0)))
        lim_up = int(np.sum(alive & md.limit_up_at_close[t]))
        lim_dn = int(np.sum(alive & md.limit_dn_at_close[t]))
        total_amt = float(np.nansum(md.amount[t][alive]))
        amt_hist = [float(np.nansum(md.amount[i][md.alive[i]])) for i in range(max(0, t - 19), t + 1)]
        amt20 = float(np.mean(amt_hist)) if amt_hist else np.nan
        ma20 = self.derived["ma20"][t]
        above20 = int(np.sum(eligible_above := (elig & np.isfinite(ma20) & (md.close[t] > ma20))))
        # 行业
        ids = np.asarray(md.industry[t])
        m = elig & (ids >= 0) & np.isfinite(ret)
        n_ind = len(md.industry_cats)
        cnt = np.bincount(ids[m], minlength=n_ind)
        close = md.close
        with np.errstate(invalid="ignore", divide="ignore"):
            r20 = np.where(np.isfinite(close[t - 20]), close[t] / close[t - 20] - 1.0, np.nan) if t >= 20 else np.full(md.N, np.nan)
            r60 = np.where(np.isfinite(close[t - 60]), close[t] / close[t - 60] - 1.0, np.nan) if t >= 60 else np.full(md.N, np.nan)
        industries = []
        for k in range(n_ind):
            if cnt[k] < 3:
                continue
            mm = m & (ids == k)
            mm20 = elig & (ids == k) & np.isfinite(r20)
            mm60 = elig & (ids == k) & np.isfinite(r60)
            industries.append({
                "industry": md.industry_name(k), "n": int(cnt[k]),
                "ret_1d": _f(np.nanmean(ret[mm])),
                "ret_20d": _f(np.nanmean(r20[mm20])) if mm20.any() else None,
                "ret_60d": _f(np.nanmean(r60[mm60])) if mm60.any() else None,
                "amount": _f(np.nansum(md.amount[t][mm])),
            })
        industries.sort(key=lambda r: (r["ret_20d"] if r["ret_20d"] is not None else -9), reverse=True)

        def _movers(mask, key, asc, n=20):
            idx = np.flatnonzero(mask & np.isfinite(key))
            if len(idx) == 0:
                return []
            order = idx[np.argsort(key[idx])][::-1] if not asc else idx[np.argsort(key[idx])]
            out = []
            for j in order[:n]:
                out.append({"code": str(md.codes[j]), "name": str(md.names[j]),
                            "industry": md.industry_name(int(ids[j])),
                            "chg_1d": _f(ret[j]), "price": _f(md.price[t, j]),
                            "amount": _f(md.amount[t, j])})
            return out
        base_mask = alive & np.isfinite(ret)
        top_gainers = _movers(base_mask, ret, asc=False)
        top_losers = _movers(base_mask, ret, asc=True)
        top_amount = _movers(base_mask, md.amount[t], asc=False)
        self._cache["industries"] = (time.time(), {"industries": industries})
        return {
            "asof": str(md.dates[t].date()),
            "regime": {**regime, "coeff": _f(coeff)},
            "indices": indices,
            "breadth": {
                "n_universe": int(valid.sum()), "up": up, "down": dn, "flat": flat,
                "limit_up": lim_up, "limit_down": lim_dn,
                "median_ret": _f(np.nanmedian(ret[valid])),
                "total_amount": total_amt, "amount_20d_avg": amt20,
                "amount_ratio": _f(total_amt / amt20) if amt20 else None,
                "n_above_ma20": above20,
                "pct_above_ma20": _f(above20 / max(int(elig.sum()), 1)),
            },
            "industries": industries,
            "top_gainers": top_gainers, "top_losers": top_losers, "top_amount": top_amount,
            "chart_urls": {
                "index_line": f"/api/chart/index_line?code={bench}&days=250",
                "industry_bar": "/api/chart/industry_bar?days=20",
            },
        }

    # ------------------------------------------------------------ 推荐
    def recommend(self, top: int = 60, asof: str | None = None, force: bool = False,
                  n_holdings: int = 40, model: str | None = None) -> dict:
        self._require_loaded()
        md = self.md
        model_name = model or MODEL_NAME
        res = self._get_model(model_name)
        t = self._t(asof)
        n_hold = int(n_holdings or 40)
        key = (model_name, t, n_hold, top)
        cached = self._recommend_cache.get(key)
        if not force and cached is not None:
            return cached
        with self._rec_lock:                     # 单飞：同一时刻只算一份（不再拿全局 self.lock）
            cached = self._recommend_cache.get(key)
            if not force and cached is not None:
                return cached
            if t == md.T - 1:
                # 用预热的 live_scores。没算好就立刻抛"加载中"（→503→前端 3 秒重试），
                # 不让请求线程在这儿干等 —— 首次预热要 30~40 秒，等就是等浏览器超时。
                scores_row, weights, refit = self._get_live_scores(model_name, blocking=False)
            else:
                scores_row = np.asarray(res.scores[t], dtype=np.float32) if res.scores is not None else np.full(md.N, np.nan, np.float32)
                w, refit = qfm.latest_weights(res)
                weights = w
            target = qfport.build_target(md, t, scores_row, {}, n_holdings=n_hold,
                                         max_new_entries=None)
            coeff, regime = dsig.market_regime(md, t)
            target = {j: w * coeff for j, w in target.items()}
            capital = float(C.CAPITAL)
            rows = []
            for j, w in sorted(target.items(), key=lambda x: -x[1]):
                px = float(md.price[t, j])
                if not np.isfinite(px) or px <= 0:
                    continue
                tgt_val = w * capital
                shares = int(np.floor(tgt_val / px / C.LOT) * C.LOT)
                if shares <= 0:
                    continue
                rows.append({"j": j, "code": str(md.codes[j]), "name": str(md.names[j]),
                             "industry": md.industry_name(int(md.industry[t, j])),
                             "board": str(md.board[j]), "weight": w,
                             "target_value": tgt_val, "price": px,
                             "shares": shares, "lots": shares // C.LOT,
                             "score": _f(scores_row[j])})
            budget = float(sum(target.values())) * capital
            invested = sum(r["shares"] * r["price"] for r in rows)
            for _ in range(2000):
                under = [r for r in rows
                         if invested + r["price"] * C.LOT <= budget * 1.001
                         and (r["shares"] + C.LOT) * r["price"] <= capital * C.MAX_WEIGHT]
                if not under:
                    break
                r = min(under, key=lambda x: (x["shares"] * x["price"] / max(x["target_value"], 1e-9),
                                              x["shares"] * x["price"]))
                r["shares"] += C.LOT
                r["lots"] = r["shares"] // C.LOT
                invested += r["price"] * C.LOT
            for r in rows:
                r["target_value"] = r["shares"] * r["price"]
                r["weight"] = r["target_value"] / capital
            used = float(sum(r["target_value"] for r in rows))
            cash = max(0.0, capital - used)
            # 候选榜
            elig = md.eligible[t]
            finite = np.isfinite(scores_row)
            cand_idx = np.flatnonzero(elig & finite)
            order = cand_idx[np.argsort(scores_row[cand_idx])[::-1]]
            rank_of = {int(j): i + 1 for i, j in enumerate(order)}
            holdings = []
            for r in rows:
                holdings.append({**{k: r[k] for k in ("code", "name", "industry", "board",
                                                     "weight", "price", "shares", "lots",
                                                     "target_value", "score")},
                                 "rank": rank_of.get(r["j"])})
            holdings.sort(key=lambda x: x.get("rank") or 9999)
            in_target = set(target.keys())
            candidates = []
            for i, j in enumerate(order[: int(top)]):
                candidates.append({
                    "rank": i + 1, "code": str(md.codes[j]), "name": str(md.names[j]),
                    "industry": md.industry_name(int(md.industry[t, j])),
                    "board": str(md.board[j]), "score": _f(scores_row[j]),
                    "price": _f(md.price[t, j]), "in_target": int(j) in in_target,
                    "excluded_reason": None,
                })
            orders = [{"code": h["code"], "name": h["name"], "side": "buy",
                       "shares": h["shares"], "lots": h["lots"], "price_ref": h["price"],
                       "amount": h["target_value"], "reason": "initial_build"}
                      for h in holdings]
            bw, gw = float(C.CASH_BOND_WEIGHT), float(C.CASH_GOLD_WEIGHT)
            tot = bw + gw
            result = {
                "model": model_name,
                "asof": str(md.dates[t].date()),
                "refit": str(pd.Timestamp(refit).date()),
                "regime": {**regime, "coeff": _f(coeff)},
                "capital": capital, "n_holdings": len(holdings),
                "invested_value": used, "invested_pct": _f(used / capital),
                "cash_pct": _f(cash / capital),
                "cash_sleeve": {
                    "bond_weight": bw / tot if tot else 0, "gold_weight": gw / tot if tot else 0,
                    "cash_value": cash, "bond_value": cash * bw / tot if tot else 0,
                    "gold_value": cash * gw / tot if tot else 0,
                    "instruments": ["国债ETF(如 511010/511260)"] if tot else [],
                },
                "holdings": holdings, "orders": orders, "candidates": candidates,
                "notes": [
                    "信号基于数据最后交易日收盘计算，建议下一交易日开盘执行；",
                    "开盘涨停的标的买不进、跌停的卖不出，可顺延到下一交易日；",
                    "单票权重上限 8%、行业上限 25%、双创合计上限 50%；",
                    f"闲置现金建议 {bw/tot:.0%} 国债ETF + {gw/tot:.0%} 黄金ETF（可只用国债ETF）；" if tot else "",
                    "本页面仅为量化研究结果，不构成投资建议。",
                ],
                "chart_url": "/api/chart/index_line?code=csi_all&days=250",
                "_key": key, "_scores_row": scores_row, "_weights": weights,
            }
            with self.lock:
                self._recommend_cache[key] = result
            return result

    def public_recommend(self, top: int = 60, asof: str | None = None,
                         n_holdings: int = 40, model: str | None = None) -> dict:
        r = self.recommend(top=top, asof=asof, n_holdings=n_holdings, model=model)
        return {k: v for k, v in r.items() if not k.startswith("_")}

    # ---- 预热的落盘缓存 ------------------------------------------------
    # 全量特征 + 打分实测 72 秒（2026-11-07 实测），每次重启服务都要重算一遍，
    # 而这 72 秒正好是用户打开页面的时候。结果只有 0.5 MB，直接存盘。
    def _data_stamp(self) -> str:
        """数据指纹：最后交易日 + 行情/市值/复权面板 + 因子面板的 mtime。

        任何一处数据被更新（管线重跑、增量落盘）指纹就变，旧缓存自动失效，
        不会出现"拿着昨天的分数量化今天的股票"。
        """
        parts = [str(self.md.dates[-1].date())]
        for name in ("panel_ohlcv.npz", "market_cap.npz", "raw_close.npz"):
            p = pathlib.Path(C.DATA_RAW) / name
            try:
                parts.append(str(int(p.stat().st_mtime)))
            except OSError:
                parts.append("0")
        for extra in (C.BUILD_FACTORS / "weekly_meta.npz",):
            try:
                parts.append(str(int(extra.stat().st_mtime)))
            except OSError:
                parts.append("0")
        return "_".join(parts)

    def _live_disk_path(self, model: str):
        return LIVE_CACHE_DIR / f"{model}_{self._data_stamp()}.npz"

    def _load_live_disk(self, model: str) -> bool:
        """读回落盘结果；命中则同时填好两份内存缓存并返回 True。"""
        p = self._live_disk_path(model)
        if not p.exists():
            return False
        try:
            z = np.load(p, allow_pickle=False)
            row = z["row"].astype(np.float32)
            w = z["w"].astype(np.float64)
            feats = z["feats"].astype(np.float32)
            refit = pd.Timestamp(str(z["refit"]))
            # 形状必须与当前行情面板一致；否则是别的数据/别的模型留下的档，宁可重算
            if row.shape[0] != self.md.N or feats.ndim != 2 or feats.shape[1] != self.md.N:
                return False
            if feats.shape[0] != len(w):
                return False
        except Exception:
            return False
        with self.lock:                      # 叶子操作，只写字典
            self._live_cache[model] = (feats, w)
            self._live_scores_cache[self._warm_key(model)] = (row, w, refit)
        return True

    def _save_live_disk(self, model: str, row, w, refit, feats) -> None:
        """写落盘缓存。失败只打日志，绝不影响启动。"""
        p = self._live_disk_path(model)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, row=np.asarray(row, np.float32),
                                w=np.asarray(w, np.float64),
                                feats=np.asarray(feats, np.float32),
                                refit=str(pd.Timestamp(refit)))
            # 只保留最近 3 份，避免缓存目录随日期无限增长
            old = sorted(p.parent.glob(f"{model}_*.npz"), key=lambda q: q.stat().st_mtime)
            for q in old[:-3]:
                try:
                    q.unlink()
                except OSError:
                    pass
        except Exception:
            print("[web] 写预热缓存失败：\n" + traceback.format_exc(limit=3), flush=True)

    def _compute_ready(self, model: str | None = None) -> bool:
        """个股详情/推荐所需的两份重型缓存是否都已就绪（只读字典，不加锁）。"""
        name = model or MODEL_NAME
        if self.md is None:
            return False
        return (self._warm_key(name) in self._live_scores_cache) and (name in self._live_cache)

    def _get_live_features(self, model: str | None = None, blocking: bool = True,
                           timeout: float = 20.0):
        """池内特征 (K,N) + 权重。

        blocking=True（预热用）：算到出结果为止；
        blocking=False（请求路径）：等 timeout 秒，没好就抛"加载中"，绝不挂住请求线程。
        """
        name = model or MODEL_NAME
        cached = self._live_cache.get(name)
        if cached is not None:
            return cached
        if not blocking:
            deadline = time.time() + timeout
            while time.time() < deadline:
                cached = self._live_cache.get(name)
                if cached is not None:
                    return cached
                time.sleep(0.2)
            self._kick_warm(name)
            raise RuntimeError("因子数据加载中，请稍候重试")
        return self._compute_live_features(name)

    # ------------------------------------------------------------ 个股
    def stock_search(self, q: str, limit: int = 20) -> list[dict]:
        self._require_loaded()
        q = (q or "").strip()
        if not q:
            return []
        md = self.md
        out, seen = [], set()
        if q.isdigit():
            for j, c in enumerate(md.codes):
                if q in str(c) and j not in seen:
                    seen.add(j)
                    out.append((j, str(c)))
        for j, n in enumerate(md.names):
            if q in str(n) and j not in seen:
                seen.add(j)
                out.append((j, str(md.codes[j])))
        out = out[: int(limit)]
        return [{"code": str(md.codes[j]), "name": str(md.names[j]),
                 "industry": md.industry_name(int(md.industry[-1, j]))} for j, _ in out]

    def _avg_amount(self, t: int, j: int, win: int = 20):
        """近 win 日平均成交额。"""
        md = self.md
        if not hasattr(md, "amount"):
            return None
        a = md.amount[max(0, t - win + 1):t + 1, j]
        a = a[np.isfinite(a)]
        return _f(float(np.mean(a))) if len(a) else None

    def _list_days(self, t: int, j: int):
        """上市天数（首个有效价到 t）。"""
        md = self.md
        col = md.price[:t + 1, j]
        idx = np.flatnonzero(np.isfinite(col))
        return int(t - idx[0]) if len(idx) else None

    def _exclusion_reason(self, md, t: int, j: int) -> dict:
        """判断个股为何不在选股域内，返回 {reason, detail}。"""
        reasons = []
        px = float(md.price[t, j]) if np.isfinite(md.price[t, j]) else None
        px_band = float(md.price_exec[t, j]) if hasattr(md, "price_exec") and np.isfinite(md.price_exec[t, j]) else px
        if px is not None:
            if px_band is not None and px_band > C.MAX_PRICE:
                reasons.append(f"股价 {px_band:.2f} 元 > 上限 {C.MAX_PRICE:.0f} 元（本金小、整手成本高）")
            elif px_band is not None and px_band < C.MIN_PRICE:
                reasons.append(f"股价 {px_band:.2f} 元 < 下限 {C.MIN_PRICE:.0f} 元（低价股风险）")
        if md.eligible is not None and not bool(md.eligible[t, j]):
            # 细分原因
            if hasattr(md, "amount") and np.isfinite(md.amount[t, j]):
                pass
            if hasattr(md, "volume") and np.isfinite(md.volume[t, j]) and md.volume[t, j] <= 0:
                reasons.append("当日停牌或无成交")
            if hasattr(md, "list_days_arr"):
                pass
        if not reasons and md.eligible is not None and not bool(md.eligible[t, j]):
            reasons.append("不满足流动性/上市时长/ST 等筛选条件")
        return {"in_universe": bool(md.eligible[t, j]) if md.eligible is not None else True,
                "reasons": reasons}

    def stock_detail(self, code: str, model: str | None = None) -> dict:
        self._require_loaded()
        # 进程内缓存：按 (code, model, 最新数据日) 缓存，数据更新后自动失效
        _ck = (str(code), str(model or MODEL_NAME), str(self.md.dates[-1].date()))
        _cache = getattr(self, "_stock_cache", None)
        if _cache is None:
            _cache = self._stock_cache = {}
        if _ck in _cache:
            return _cache[_ck]
        _out = self._stock_detail_impl(code, model)
        if len(_cache) > 500:
            _cache.clear()
        _cache[_ck] = _out
        return _out

    def _stock_detail_impl(self, code: str, model: str | None = None) -> dict:
        self._require_loaded()
        model_name = model or MODEL_NAME
        # 显式指定了模型就必须是白名单内的：否则下面 recommend() 的异常会被吞掉，
        # 页面会静默显示"有价格、没分数、没因子"，用户无法判断是自己选错了模型。
        if model:
            allowed = [m["name"] for m in self._refresh_registry()]
            if model not in allowed:
                raise ValueError(f"未知模型 {model}，可选：{'、'.join(allowed) or MODEL_NAME}")
        md = self.md
        j = md.code_index(str(code))
        if j < 0:
            raise KeyError(f"找不到股票 {code}")
        t = md.T - 1
        price = md.price
        ret = md.ret_daily[t, j]

        def chg(k):
            return _f(price[t, j] / price[t - k, j] - 1.0) if t >= k and price[t - k, j] > 0 else None

        # ---- 性能优化：域外股票不走 recommend()（会触发全市场 live_scores 重算，约 30s）
        #      域外股票的因子层本来就是全 NaN（实证），算了也无意义 → 直接返回展示型信息
        in_domain = bool(md.eligible[t, j]) if md.eligible is not None else True
        if not in_domain:
            excluded = self._exclusion_reason(md, t, j)
            fund = self._fundamentals_for(j, t)
            return {
                "model": model_name, "code": str(md.codes[j]), "name": str(md.names[j]),
                "industry": md.industry_name(int(md.industry[t, j])),
                "board": str(md.board[j]), "asof": str(md.dates[t].date()),
                "price": _f(price[t, j]), "chg_1d": _f(ret),
                "chg_5d": chg(5), "chg_20d": chg(20), "chg_60d": chg(60), "chg_120d": chg(120),
                "amount_20d_avg": self._avg_amount(t, j),
                "mcap": _f(md.mcap[t, j]) if hasattr(md, "mcap") else None,
                "list_days": self._list_days(t, j),
                "in_universe": False, "in_model_universe": None, "in_target": False,
                "target_weight": None,
                "score": None, "score_ref": None, "score_rank": None, "score_pct": None,
                "exclusion": excluded, "factors": [], "fundamentals": fund,
                "chart_url": f"/api/chart/stock_kline?code={md.codes[j]}&days=250",
                "fast_path": True,
            }

        # ---- 域内个股要给分数 + 因子贡献，两者都依赖预热结果。
        # 没就绪就立刻回 503（前端 3 秒后自动重试），绝不占着请求线程干等到浏览器超时。
        # 这是 2026-11-07「个股加载 60 秒超时」的直接修复点。
        if not self._compute_ready(model_name):
            self._kick_warm(model_name)
            raise RuntimeError("打分数据加载中，请稍候重试（首次约需 20–40 秒）")

        rec = None
        try:
            rec = self.recommend(top=60, model=model_name)
        except RuntimeError:
            raise                                  # "加载中" 要透传，不能被下面吞掉
        except Exception:
            rec = None
        in_target = False
        tw = None
        rank = pct = score = None
        srow_any = None
        if rec is not None:
            srow = rec.get("_scores_row")
            srow_any = srow
            if srow is not None and np.isfinite(srow[j]):
                score = _f(srow[j])
                elig = md.eligible[t]
                finite = np.isfinite(srow) & elig
                rank = int(np.sum(srow[finite] > srow[j])) + 1
                pct = _f(1.0 - (rank - 1) / max(int(finite.sum()), 1))
            for h in rec["holdings"]:
                if h["code"] == str(md.codes[j]):
                    in_target = True
                    tw = h["weight"]
                    break

        # ---- 若打分缺失（被选股域排除），直接从 live_scores 取值作为"参考分"
        # 域外股票无法计算复合分：因子计算本身以选股域为基准（build_live_features
        # 对域外标的返回全 NaN），这是设计使然，不做强行绕过。
        score_ref = None

        # ---- 被排除原因（解释为什么不在选股域）
        excl = self._exclusion_reason(md, t, j)
        # 模型选股域（如 wf_rs10 v2.1 的 "收盘>MA60"），与数据层 eligible/in_universe 不同
        in_model_universe = None
        try:
            _res = self._get_model(model_name)
            _u = getattr(getattr(_res, "cfg", None), "universe", None)
            if _u:
                from qf import model as qfm
                _ones = np.ones((1, md.N), dtype=np.float32)
                _mask = qfm.apply_universe_filter(md, _ones, [t], _u)[0]
                in_model_universe = bool(np.isfinite(_mask[j]))
            else:
                in_model_universe = True
        except Exception:
            in_model_universe = None

        factors = []
        try:
            res = self._get_model(model_name)
            feats, w = self._get_live_features(model_name, blocking=False)
            cat = self.catalog
            name_map = {}
            for c in (cat, self.mom_cat, self.fund_cat, self.trend_cat):
                if c is not None and len(c):
                    name_map.update({str(r["factor_key"]): str(r.get("name_cn") or r.get("slug") or r["factor_key"])
                                     for _, r in c.iterrows()})
            for i, key in enumerate(res.pool):
                fv = _f(feats[i, j])
                if fv is None:
                    continue
                wk = _f(w[i]) or 0.0
                factors.append({"key": key, "name_cn": name_map.get(key, key),
                                "feature": fv, "weight": wk, "contribution": fv * wk})
            factors.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        except Exception:
            factors = []
        fund = {}
        try:
            fund = self._fundamentals_for(j, t)
        except Exception:
            fund = {}
        return {
            "model": model_name,
            "code": str(md.codes[j]), "name": str(md.names[j]),
            "industry": md.industry_name(int(md.industry[t, j])),
            "board": BOARD_CN.get(str(md.board[j]), str(md.board[j])),
            "asof": str(md.dates[t].date()),
            "price": _f(price[t, j]), "chg_1d": _f(ret),
            "chg_5d": chg(5), "chg_20d": chg(20), "chg_60d": chg(60), "chg_120d": chg(120),
            "amount_20d_avg": _f(md.amount_20d[t, j]),
            "mcap": _f(md.mcap[t, j]),
            "list_days": int(md.list_days[t, j]),
            "in_universe": bool(md.eligible[t, j]),
            "in_model_universe": in_model_universe,
            "in_target": bool(in_target), "target_weight": _f(tw),
            "score": score, "score_ref": score_ref,
            "score_rank": rank, "score_pct": pct,
            "exclusion": excl,
            "factors": factors[:20],
            "fundamentals": fund,
            "chart_url": f"/api/chart/stock_kline?code={md.codes[j]}&days=250",
        }

    def _fundamentals_for(self, j: int, t: int) -> dict:
        if self._fund_weekly is None:
            path = C.BUILD_FACTORS / "fundamental_extra.npz"
            meta = np.load(C.BUILD_FACTORS / "weekly_meta.npz")
            rows = meta["rows"].astype(np.int64)
            z = np.load(path)
            self._fund_weekly = ({k: z[k] for k in z.files}, rows)
        panels, rows = self._fund_weekly
        i = int(np.searchsorted(rows, t, side="right") - 1)
        if i < 0:
            return {}
        labels = {"F001": "ep_ttm", "F002": "bp", "F003": "sp_ttm", "F004": "roe_ann",
                  "F005": "gross_margin", "F006": "earnings_quality", "F007": "rev_growth",
                  "F008": "profit_growth", "F009": "profit_accel", "F010": "rev_accel",
                  "F011": "roe_change", "F012": "gm_change", "F013": "accruals",
                  "F014": "margin_expansion", "F015": "profit_qoq"}
        out = {}
        for fid, label in labels.items():
            key = f"fundamental-extra:{fid}"
            if key in panels:
                out[label] = _f(panels[key][i, j])
        return out

    # ------------------------------------------------------------ 回测
    def start_backtest(self, params: dict) -> str:
        self._require_loaded()
        self._get_model((params or {}).get("model") or MODEL_NAME)
        if self.bt_lock.locked():
            raise RuntimeError("已有回测在运行，请等待完成")
        jid = self.jobs.create("backtest")
        threading.Thread(target=self._run_backtest, args=(jid, params), daemon=True,
                         name="web-backtest").start()
        return jid

    def _run_backtest(self, jid: str, p: dict):
        try:
            if not self.bt_lock.acquire(blocking=False):
                self.jobs.update(jid, status="error", error="已有回测在运行")
                return
            try:
                self._do_backtest(jid, p)
            finally:
                self.bt_lock.release()
        except Exception as e:
            self.jobs.update(jid, status="error", error=f"{e}", progress=1.0)
            self.jobs.push_log(jid, traceback.format_exc(limit=2))

    def _do_backtest(self, jid: str, p: dict):
        md = self.md
        model_name = p.get("model") or MODEL_NAME
        res = self._get_model(model_name)
        n_hold = int(p.get("n_holdings") or 40)
        rb = int(p.get("rebalance_days") or 10)
        start_s = str(p.get("start") or C.VALID_START)
        end_s = str(p.get("end") or md.dates[-1].date())
        overlay = bool(p.get("overlay", True))
        stop = bool(p.get("stop_loss", True))
        bw = float(p.get("cash_bond", C.CASH_BOND_WEIGHT))
        gw = float(p.get("cash_gold", C.CASH_GOLD_WEIGHT))
        max_new = p.get("max_new", 6)
        max_new = None if max_new is None or int(max_new) < 0 else int(max_new)
        band_rel = float(p.get("band_rel", 0.30))
        band_abs = float(p.get("band_abs", 6000))
        self.jobs.update(jid, progress=0.05, message="准备回测参数...")
        scores = np.asarray(res.scores)
        # saved 模型在原始数据（10-14）上训练；增量数据多出的行补 NaN（不参与打分）
        if scores.shape[0] < md.T:
            pad = np.full((md.T - scores.shape[0], scores.shape[1]), np.nan, dtype=np.float32)
            scores = np.vstack([scores, pad])
        elif scores.shape[0] > md.T:
            scores = scores[:md.T]
        start = int(np.clip(md.date_index(start_s), 0, md.T - 1))
        end = int(np.clip(md.date_index(end_s), 0, md.T - 1))
        valid = np.flatnonzero(np.isfinite(scores).any(axis=1))
        if len(valid) == 0:
            raise RuntimeError("模型没有可用打分")
        start = max(start, int(valid[0]))
        phase = int(res.rows_weekly[0]) % 5 if len(getattr(res, "rows_weekly", [])) else 0
        while start % 5 != phase:
            start += 1
        if end <= start + 20:
            raise RuntimeError("回测区间太短")
        offset = 0
        if rb % 5 == 0:
            offset = (phase - start) % rb
        cash_ret = qfbt.load_cash_asset_returns(md, bw, gw) if (bw + gw) > 0 else None
        self.jobs.update(jid, progress=0.15,
                         message=f"回测 {md.dates[start].date()} ~ {md.dates[end].date()}（N={n_hold}, 调仓{rb}日）")
        old_nh = C.N_HOLDINGS
        C.N_HOLDINGS = n_hold
        try:
            r = qfbt.run_backtest(
                md, scores, start, end, risk_overlay=overlay, stop_loss=stop,
                rebalance_days=rb, rebalance_offset=offset,
                trade_band_rel=band_rel, trade_band_abs=band_abs,
                max_new_entries=max_new, verbose=False, cash_asset_returns=cash_ret)
        finally:
            C.N_HOLDINGS = old_nh
        self.jobs.update(jid, progress=0.7, message="计算绩效指标...")
        nav = r.nav
        dd = (nav / nav.cummax() - 1.0).to_numpy()
        m = dict(r.metrics)
        full = qfmetrics.performance(nav, rf=C.CASH_YIELD)
        periods = {"full": full}
        for name, (s, e) in {"val": ("2020-01-01", "2021-12-31"),
                             "oos": ("2022-01-01", "2026-12-31")}.items():
            sub = nav[(nav.index >= pd.Timestamp(s)) & (nav.index <= pd.Timestamp(e))]
            if len(sub) >= 40:
                periods[name] = qfmetrics.performance(sub, rf=C.CASH_YIELD)
        yearly = qfmetrics.yearly_returns(nav)
        bench = {}
        for bname in list(C.BENCHMARKS) + ["all_eq"]:
            try:
                br = md.bench[bname].reindex(nav.index).fillna(0.0).to_numpy()
                bnav = pd.Series(np.cumprod(1.0 + br), index=nav.index)
                bench[bname] = qfmetrics.performance(bnav, rf=C.CASH_YIELD)
            except Exception:
                pass
        trades = r.trades if r.trades is not None else pd.DataFrame()
        cost_total = float((trades.get("commission", 0).sum() + trades.get("stamp", 0).sum()
                            + trades.get("transfer", 0).sum() + trades.get("slip", 0).sum())) if len(trades) else 0.0
        by_year = []
        if len(trades):
            d = trades.copy()
            d["year"] = pd.to_datetime(d["date"]).dt.year
            for y, g in d.groupby("year"):
                by_year.append({"year": int(y), "n": int(len(g)),
                                "cost": float((g["commission"] + g["stamp"] + g["transfer"] + g["slip"]).sum())})
        idx = nav.index
        step = max(1, len(idx) // 800)
        sel = np.arange(0, len(idx), step)
        result = {
            "params": {**p, "model": model_name, "n_holdings": n_hold, "rebalance_days": rb,
                       "start": str(md.dates[start].date()), "end": str(md.dates[end].date()),
                       "overlay": overlay, "stop_loss": stop,
                       "cash_bond": bw, "cash_gold": gw, "max_new": max_new,
                       "band_rel": band_rel, "band_abs": band_abs},
            "metrics": {**{k: _f(m.get(k)) for k in
                         ("cagr", "ann_vol", "sharpe", "sortino", "max_dd", "calmar",
                          "win_rate_monthly", "turnover_ann", "cost_ratio", "n_trades",
                          "avg_holdings", "years") if k in m},
                        "start": str(md.dates[start].date()), "end": str(md.dates[end].date())},
            "periods": {k: {kk: _f(vv) for kk, vv in v.items() if isinstance(vv, (int, float, np.floating))}
                        for k, v in periods.items()},
            "yearly": [{"year": int(y), "ret": _f(v)} for y, v in yearly.items()],
            "benchmarks": {k: {kk: _f(v.get(kk)) for kk in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}
                           for k, v in bench.items()},
            "nav": {"dates": [str(x.date()) for x in idx[sel]],
                    "nav": [float(v) for v in nav.to_numpy()[sel]],
                    "dd": [float(v) for v in dd[sel]]},
            "chart_url": f"/api/chart/backtest_nav/{jid}",
            "trades_summary": {"n_buy": int((trades["side"] == "buy").sum()) if len(trades) else 0,
                               "n_sell": int((trades["side"] == "sell").sum()) if len(trades) else 0,
                               "cost_total": cost_total, "by_year": by_year},
        }
        self.jobs.update(jid, status="done", progress=1.0, message="回测完成",
                         result=result)
        try:
            png = charts.nav_chart(
                idx, nav.to_numpy(), dd,
                title=f"回测净值（CAGR {m.get('cagr', 0)*100:.2f}% / MaxDD {m.get('max_dd', 0)*100:.2f}%）")
            with self.lock:
                self._chart_cache[("backtest_nav", jid)] = png
        except Exception:
            pass

    # ------------------------------------------------------------ 数据更新
    def data_status(self) -> dict:
        ext_info = _EXT / "UPDATE_INFO.txt"
        mtime = None
        if ext_info.exists():
            try:
                from datetime import datetime
                mtime = datetime.fromtimestamp(ext_info.stat().st_mtime).isoformat(timespec="seconds")
            except Exception:
                mtime = None
        job_id = None
        for jid in reversed(self.jobs.list()):
            if jid.startswith("update_"):
                job_id = jid
                break
        return {"source": DATA_SOURCE, "last_date": self._last_date_of(C.DATA_RAW),
                "orig_last_date": self._last_date_of(_ORIG), "ext_dir_exists": _EXT.exists(),
                "update_running": self.update_running, "last_update": mtime,
                "update_job_id": job_id, "server_loaded": self.loaded}

    @staticmethod
    def _last_date_of(path: pathlib.Path):
        try:
            z = np.load(pathlib.Path(path) / "panel_ohlcv.npz")
            return str(pd.Timestamp(z["dates"][-1]).date())
        except Exception:
            return None

    def start_data_update(self) -> str:
        if self.update_running:
            raise RuntimeError("数据更新已在进行中")
        if not (_ORIG / "panel_ohlcv.npz").exists():
            raise RuntimeError("找不到原始数据目录")
        jid = self.jobs.create("update")
        self.update_running = True
        threading.Thread(target=self._run_update, args=(jid,), daemon=True,
                         name="web-update").start()
        return jid

    def _run_update(self, jid: str):
        env = {k: v for k, v in os.environ.items() if k != "QF_DATA_RAW"}
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        self.jobs.update(jid, progress=0.02, message="启动数据更新（腾讯行情，约 10~15 分钟）...")
        try:
            _cf = {"creationflags": 0x08000000} if sys.platform == "win32" else {}  # CREATE_NO_WINDOW
            proc = subprocess.Popen(
                [sys.executable, "-u", "scripts/update_data.py", "--workers", "16"],
                cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **_cf)
            total = 5281
            for line in proc.stdout:
                self.jobs.push_log(jid, line)
                mm = re.search(r"(\d+)/(\d+)", line)
                if mm:
                    n, tot = int(mm.group(1)), int(mm.group(2))
                    total = max(tot, 1)
                    self.jobs.update(jid, progress=min(0.85, 0.06 + 0.75 * n / total),
                                     message=f"拉取个股行情 {n}/{tot}...")
                elif "[2/6]" in line:
                    self.jobs.update(jid, progress=0.06, message="拉取基准指数...")
                elif "[3/6]" in line:
                    self.jobs.update(jid, progress=0.08, message="拉取个股行情...")
                elif "[4/6]" in line:
                    self.jobs.update(jid, progress=0.85, message="构建扩展面板...")
                elif "[5/6]" in line:
                    self.jobs.update(jid, progress=0.9, message="保存数据目录...")
                elif "[6/6]" in line:
                    self.jobs.update(jid, progress=0.93, message="同步国债/黄金/ETF 面板...")
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"update_data.py 退出码 {code}")
            self.jobs.update(jid, progress=0.92, message="数据更新完成，重新加载行情（约 30 秒）...")
            self.reload()
            self.jobs.update(jid, status="done", progress=1.0, message="数据更新并重载完成",
                             result={"last_date": self.md.dates[-1].date().isoformat() if self.md is not None else None,
                                     "n_dates": int(self.md.T) if self.md is not None else None})
        except Exception as e:
            self.jobs.update(jid, status="error", error=str(e), progress=1.0)
            self.jobs.push_log(jid, f"[error] {e}")
        finally:
            self.update_running = False

    # ------------------------------------------------------------ 因子/报告
    def export_recommend_csv(self, kind: str = "target", model: str | None = None) -> str:
        """按当前模型生成推荐/订单 CSV（与页面一致，支持右侧模型）。"""
        r = self.recommend(top=60, model=model)
        if kind == "orders":
            df = pd.DataFrame(r["orders"])
        else:
            cols = ["code", "name", "industry", "board", "weight", "price",
                    "shares", "lots", "target_value"]
            df = pd.DataFrame([{k: h.get(k) for k in cols} for h in r["holdings"]])
            df["cash_weight"] = r.get("cash_pct")
        return df.to_csv(index=False)

    # ------------------------------------------------------------ ETF 趋势策略
    def etf_trend(self) -> dict:
        """右侧 ETF 趋势策略的最新状态、当前信号与近期净值。"""
        import json as _json

        import numpy as _np

        from qf import metrics as _qfm
        etf_dir = C.BUILD / "etf"
        stats_p = C.BUILD / "eval" / "etf_trend_stats.json"
        combo_p = C.BUILD / "eval" / "etf_combo_stats.json"
        panel_p = etf_dir / "etf_panel.npz"
        if not panel_p.exists():
            raise RuntimeError("ETF 数据不存在，请先运行 scripts/build_etf_data.py")
        z = _np.load(panel_p, allow_pickle=True)
        dates = pd.DatetimeIndex(z["dates"])
        codes = [str(c) for c in z["codes"]]
        close = z["close"].astype(float)
        amount = z["amount"].astype(float)
        meta = {}
        if (etf_dir / "etf_meta.json").exists():
            meta = _json.load(open(etf_dir / "etf_meta.json", encoding="utf-8"))

        # 用与回测相同的规则重算最新信号（最近 200 日足够判定 55/20 日突破）
        n = min(len(dates), 400)
        idx = slice(len(dates) - n, len(dates))
        cs = pd.DataFrame(close[idx])
        amt20 = pd.DataFrame(amount[idx]).rolling(20, min_periods=10).mean().to_numpy()
        hi = cs.rolling(55).max().shift(1).to_numpy()
        lo = cs.rolling(20).min().shift(1).to_numpy()
        mom60 = (cs / cs.shift(60) - 1.0).to_numpy()
        t = len(cs) - 1
        cand = []
        for j in range(len(codes)):
            c = cs.iloc[t, j]
            if not (_np.isfinite(c) and _np.isfinite(hi[t, j])):
                continue
            entry = bool(c > hi[t, j]) and bool(_np.isfinite(amt20[t, j]) and amt20[t, j] >= 5e7)
            cand.append({
                "code": codes[j],
                "name": (meta.get("etfs", {}).get(codes[j], {}) or {}).get("name", codes[j]),
                "price": _f(c), "day_chg": _f(cs.iloc[t, j] / cs.iloc[t - 1, j] - 1.0),
                "entry_signal": entry,
                "mom60": _f(mom60[t, j]),
                "amount_20d": _f(amt20[t, j]),
                "dist_to_high": _f(c / hi[t, j] - 1.0) if _np.isfinite(hi[t, j]) else None,
            })
        cand.sort(key=lambda x: -(x["mom60"] if x["mom60"] is not None else -9))

        out = {"asof": str(dates[-1].date()), "universe": cand,
               "params": {"entry_n": 55, "exit_n": 0, "max_pos": 1, "min_amount": 5e7,
                          "slippage_bp": 5.0, "commission_bp": 2.5,
                          "cash_sleeve": "100% 国债"},
               "note": "跌破 55 日高点即离场（紧出场）；同时最多持有 1 只 ETF，其余现金"}
        if stats_p.exists():
            st = _json.load(open(stats_p, encoding="utf-8"))
            base = st.get("donchian55_tight", {})
            out["backtest"] = {k: base.get(k) for k in
                               ("cagr", "sharpe", "max_dd", "calmar", "exposure",
                                "avg_holdings", "avg_weight", "n_trades")}
            out["config"] = {"max_pos": 2, "pos_style": "equal", "entry_n": 55,
                              "exit_n": 0,
                              "timing": "严格 T+1（t日收盘决策 → t+1日成交）"}
            out["sweep"] = st.get("sweep", {})
        if combo_p.exists():
            cb = _json.load(open(combo_p, encoding="utf-8"))
            out["combo"] = {"correlation": cb.get("correlation"), "results": cb.get("results")}
            # 三腿组合（主策略 + ETF趋势 + 512890红利低波），存在才透出
            combo3_p = C.BUILD / "eval" / "combo3_stats.json"
            if combo3_p.exists():
                c3 = _json.load(open(combo3_p, encoding="utf-8"))
                out["combo3"] = {"legs_corr": c3.get("legs_corr"),
                                 "results": c3.get("results")}
            nav_p = C.BUILD / "runs" / "combo_50_50" / "nav.parquet"
            nav3_p = C.BUILD / "runs" / "combo_3leg" / "nav_502525.parquet"
            if nav_p.exists():
                combo_nav = pd.read_parquet(nav_p)["nav"]
                main_nav_p = C.BUILD / "runs" / "main" / "nav.parquet"
                etf_nav_p = C.BUILD / "runs" / "etf_trend" / "nav_donchian55.parquet"
                series = pd.DataFrame({"combo": combo_nav})
                if main_nav_p.exists():
                    mn = pd.read_parquet(main_nav_p)["nav"]
                    series["main"] = mn.reindex(series.index)
                if etf_nav_p.exists():
                    en = pd.read_parquet(etf_nav_p)["nav"]
                    series["etf"] = en.reindex(series.index)
                if nav3_p.exists():
                    n3 = pd.read_parquet(nav3_p)["nav"]
                    n3.index = pd.to_datetime(n3.index)
                    series["combo3"] = n3.reindex(series.index)
                series = series.dropna(how="all")
                # 归一化（各自起点 = 1.0），只取最近 500 日
                norm = series.apply(lambda col: col / col.dropna().iloc[0] if col.notna().any() else col)
                sub = norm.tail(500)
                cols = [c for c in ("main", "etf", "combo", "combo3") if c in sub.columns]
                out["nav_series"] = [
                    {"date": str(d.date()), **{c: (_f(row[c]) if pd.notna(row[c]) else None)
                                               for c in cols}}
                    for d, row in sub.iterrows()
                ]
                out["nav"] = [{"date": str(d.date()), "nav": _f(v)} for d, v in combo_nav.tail(500).items()]
        return out

    # ------------------------------------------------------------ 板块 ETF 轮动
    # ------------------------------------------------------------ 全策略回测
    STRATEGY_KINDS = {
        "stock": "个股模型（wf_h10 等）",
        "etf": "宽基 ETF 趋势（Donchian55）",
        "sector": "板块 ETF 轮动（25 板块）",
        "cb": "可转债高 YTM（20 只 / 10 日）",
        "lowturn": "低换手月度（10 只，月末换仓）",
        "combo": "多策略组合（自定义权重）",
    }

    # ------------------------------------------------------------ 统一成本口径
    # 单边成本（bp）= 滑点 + 佣金 + 印花税 + 过户费。
    # embedded：该策略回测**代码里已经扣过**的单边成本（读源码实测），
    #           net 模式只补 target - embedded 的差额，避免重复扣费。
    # ref_turnover：生产回测实测的年化单边换手，用于组合（读归档净值，算不出当日换手）。
    COST_PRESET = {
        # 个股：40bp 滑点 + 万0.1 佣金（受 5 元最低费主导 ≈6.67bp）+ 印花税 5bp（卖出，折单边 2.5）+ 过户费 0.1
        "stock":  {"slip": 40.0, "comm": 6.67, "stamp": 2.5, "transfer": 0.1,
                   "embedded": None, "ref_turnover": 0.7},
        # 宽基 ETF：免印花税，单笔金额大（最低 5 元可忽略），5bp 滑点 + 1bp 佣金
        "etf":    {"slip": 5.0, "comm": 1.0, "stamp": 0.0, "transfer": 0.0,
                   "embedded": 5.0, "ref_turnover": 22.4},
        # 板块 ETF：流动性差于宽基，滑点 10bp + 1bp 佣金
        "sector": {"slip": 10.0, "comm": 1.0, "stamp": 0.0, "transfer": 0.0,
                   "embedded": 5.0, "ref_turnover": 7.1},
        # 可转债：无印花税，佣金万0.5 = 5bp，中位日成交额仅 0.63 亿 → 滑点 15bp
        "cb":     {"slip": 15.0, "comm": 5.0, "stamp": 0.0, "transfer": 0.0,
                   "embedded": 5.0, "ref_turnover": 7.2},
        # 低换手月度：主板大盘股，滑点按 40bp，其余同个股。
        # 该策略用双账本（毛/净两套独立净值），embedded 填满额表示无需再补扣。
        "lowturn": {"slip": 40.0, "comm": 6.67, "stamp": 2.5, "transfer": 0.1,
                    "embedded": 49.27, "ref_turnover": 7.1},
    }

    @classmethod
    def cost_target_bp(cls, kind: str) -> float:
        c = cls.COST_PRESET[kind]
        return float(c["slip"] + c["comm"] + c["stamp"] + c["transfer"])

    def _cost_apply(self, nav: pd.Series, kind: str, p: dict,
                    turnover_ann: float | None = None,
                    embedded_bp: float | None = None):
        """按统一口径补扣交易成本，返回 (nav_net, info)。

        只在 ``cost_model == "net"`` 时生效，其余情况原样返回（零行为变化）。
        扣费方式：年成本 ``ann_cost = 换手 × 2 × (target − embedded)``，
        按 252 个交易日均匀摊成复利日因子，因此不改变收益的时间结构。
        """
        import numpy as _np

        model = str(p.get("cost_model") or "gross").lower()
        if model not in ("gross", "net"):
            model = "gross"
        preset = self.COST_PRESET[kind]
        target = self.cost_target_bp(kind)
        emb = (float(embedded_bp) if embedded_bp is not None
               else float(preset["embedded"] if preset["embedded"] is not None else target))
        tur = (float(turnover_ann)
               if turnover_ann is not None and _np.isfinite(turnover_ann)
               else float(preset["ref_turnover"]))
        delta = max(0.0, target - emb)
        ann_cost = tur * 2.0 * delta / 1e4
        info = {
            "model": model,
            "target_bp": _f(target),
            "embedded_bp": _f(emb),
            "delta_bp": _f(delta),
            "turnover_ann": _f(tur),
            "ann_cost": _f(ann_cost),
            "applied": False,
        }
        if model != "net" or delta <= 0 or len(nav) < 2:
            return nav, info
        drag = (1.0 - ann_cost) ** (1.0 / 252.0)
        if not (0.0 < drag < 1.0):
            return nav, info
        factor = _np.power(drag, _np.arange(len(nav), dtype=float))
        out = pd.Series(nav.to_numpy(dtype=float) * factor, index=nav.index, name=nav.name)
        info["applied"] = True
        return out, info

    def backtest_strategy(self, params: dict) -> dict:
        """统一回测接口：按 kind 分派到不同策略，返回统一结构。

        params:
          kind: stock | etf | sector | cb | combo
          start_date, end_date: 回测区间
          cost_model: gross（默认，界面现行口径）| net（按统一成本口径补扣费用）
          stock : model, n_holdings, rebalance_days
          etf   : entry_n, exit_n, top_n(固定1), min_amount
          sector: entry_n, mom_n, top_n, rebalance_days, min_amount
          cb    : n_holdings, rebalance_days, pool_mult, min_price, rank
          combo : weights {main, etf, sector, cb}, method, include_cb
        """
        import json as _json

        import numpy as _np

        from qf import metrics as _qfm

        p = dict(params or {})
        kind = str(p.get("kind") or "stock")
        if kind not in self.STRATEGY_KINDS:
            raise ValueError(f"未知策略类型 {kind}，可选：{list(self.STRATEGY_KINDS)}")
        start = str(p.get("start_date") or "2020-01-02")
        end = str(p.get("end_date") or "")

        if kind == "stock":
            return self._bt_stock(p, start, end)
        if kind == "etf":
            return self._bt_etf(p, start, end)
        if kind == "sector":
            return self._bt_sector(p, start, end)
        if kind == "cb":
            return self._bt_cb(p, start, end)
        if kind == "lowturn":
            return self._bt_lowturn(p, start, end)
        return self._bt_combo(p, start, end)

    def _bt_periods(self, nav: pd.Series, start: str, end: str) -> dict:
        from qf import metrics as _qfm

        out = {}
        segs = {"full": (start, end or "2099-12-31"),
                "val": ("2020-01-02", "2021-12-31"),
                "oos": ("2022-01-01", end or "2099-12-31")}
        for k, (s, e) in segs.items():
            sub = nav[(nav.index >= s) & (nav.index <= e)]
            if len(sub) < 30:
                out[k] = {}
                continue
            q = _qfm.performance(sub, rf=C.CASH_YIELD)
            out[k] = {kk: _f(q[kk]) for kk in ("cagr", "ann_vol", "sharpe", "max_dd", "calmar")}
            out[k]["n_days"] = int(len(sub))
        return out

    def _excess_stats(self, nav: pd.Series) -> dict:
        """相对各基准的超额收益（年化）与信息比率。

        基准数据来自 md.bench（日收益序列）。取不到时返回空 dict，不影响主流程。
        """
        try:
            import numpy as _np

            md = self.md
            if md is None:
                return {}
            out = {}
            base_ret = nav.pct_change().dropna()
            n_years = len(base_ret) / 252.0
            if n_years <= 0:
                return {}
            for key in ("csi_all", "hs300", "csi500", "csi1000"):
                b = md.bench.get(key)
                if b is None:
                    continue
                b = b.reindex(base_ret.index).fillna(0.0)
                b_cum = float(_np.prod(1 + b.to_numpy()) - 1)
                b_ann = (1 + b_cum) ** (1 / n_years) - 1
                s_cum = float(_np.prod(1 + base_ret.to_numpy()) - 1)
                s_ann = (1 + s_cum) ** (1 / n_years) - 1
                diff = (base_ret.to_numpy() - b.to_numpy())
                te = float(_np.std(diff, ddof=1) * _np.sqrt(252)) if len(diff) > 2 else 0.0
                out[key] = {
                    "excess_ann": _f(s_ann - b_ann),
                    "bench_ann": _f(b_ann),
                    "info_ratio": _f(diff.mean() * 252 / te) if te > 0 else None,
                    "tracking_error": _f(te),
                }
            return out
        except Exception:
            return {}

    def _bt_pack(self, nav: pd.Series, kind: str, extra: dict, start: str, end: str,
                 raw_nav: pd.Series | None = None) -> dict:
        from qf import metrics as _qfm

        nav = nav[np.isfinite(nav.to_numpy(dtype=float))]
        if len(nav) < 2:
            raise ValueError(f"{kind} 回测净值过短，无法评估")
        sub = nav[(nav.index >= start) & (nav.index <= end)] \
            if end else nav[nav.index >= start]
        if len(sub) < 2:
            sub = nav                       # 已被预截取过：直接用
        m = _qfm.performance(sub, rf=C.CASH_YIELD)
        yearly = _qfm.yearly_returns(sub)
        base = float(sub.iloc[0])
        _lbl = (extra.pop("label", None) if isinstance(extra, dict) else None)
        # 扣费前对照（只在 cost_model=net 且确实扣了费时才有意义）
        if raw_nav is not None and isinstance(extra, dict) and extra.get("cost"):
            r = raw_nav[np.isfinite(raw_nav.to_numpy(dtype=float))]
            rsub = r[(r.index >= start) & (r.index <= end)] if end else r[r.index >= start]
            if len(rsub) >= 2:
                g = _qfm.performance(rsub, rf=C.CASH_YIELD)
                extra["cost"].update({
                    "cagr_gross": _f(g["cagr"]), "sharpe_gross": _f(g["sharpe"]),
                    "cagr_net": _f(m["cagr"]), "sharpe_net": _f(m["sharpe"]),
                    "drag_pp": _f(g["cagr"] - m["cagr"]),
                })
        return {
            "kind": kind, "kind_cn": _lbl or self.STRATEGY_KINDS[kind],
            "start": str(sub.index[0].date()), "end": str(sub.index[-1].date()),
            "metrics": {k: _f(m[k]) for k in ("cagr", "ann_vol", "sharpe", "sortino",
                                              "max_dd", "calmar", "n_days")},
            "excess": self._excess_stats(sub),
            "periods": self._bt_periods(sub, start, end),
            "yearly": {int(k): _f(v) for k, v in yearly.items()},
            "nav": [{"date": str(d.date()), "nav": _f(v / base)} for d, v in sub.items()],
            **extra,
        }

    def _bt_stock(self, p: dict, start: str, end: str) -> dict:
        import numpy as _np

        self._require_loaded()
        model = str(p.get("model") or MODEL_NAME)
        md = self.md
        res = self._get_model(model)
        scores = _np.asarray(res.scores)
        if scores.shape[0] < md.T:
            scores = _np.vstack([scores, _np.full((md.T - scores.shape[0], md.N), _np.nan, _np.float32)])
        elif scores.shape[0] > md.T:
            scores = scores[:md.T]
        start_row = int(md.date_index(start))
        end_row = int(md.date_index(end)) if end else md.T - 1
        n_hold = int(p.get("n_holdings") or 40)
        # 生产口径是 40（scripts/run_backtest.py --rebalance-days 默认值）。
        # 这里曾经写 10，导致所有不带该参数的调用方（脚本、早期报告）拿到 6.13%/0.496
        # 而不是生产的 7.11%/0.606 —— 差 0.92pp，且换手虚高 3.6 倍。
        rb = int(p.get("rebalance_days") or 40)
        bw = float(p.get("cash_bond", C.CASH_BOND_WEIGHT))
        gw = float(p.get("cash_gold", C.CASH_GOLD_WEIGHT))
        cash_ret = qfbt.load_cash_asset_returns(md, bw, gw) if (bw + gw) > 0 else None
        overlay = bool(p.get("overlay", True))
        stop_loss = bool(p.get("stop_loss", True))
        # run_backtest 的默认值是 6（单期最多纳入 6 只新标的）；-1 表示不限制
        max_new = p.get("max_new")
        if max_new is None:
            max_new = 6
        max_new = None if int(max_new) < 0 else int(max_new)
        # 调仓相位必须与正式回测一致（rows_weekly[0] % 5），否则结果不可比
        phase = int(getattr(res, "rows_weekly", [0])[0]) % 5 if len(getattr(res, "rows_weekly", [])) else 0
        # 起点也要推到打分相位日：生产 run_backtest.py 有
        # ``while start_row % 5 != phase: start_row += 1``，这里漏了这一步，
        # 于是初始建仓日跟着用户选的 start 走（2020-01-02 建仓 → 7.05%，
        # 生产实际从 2020-01-08 建仓 → 7.11%），之后 40 日网格上的持仓路径也分叉。
        while start_row % 5 != phase and start_row < md.T - 2:
            start_row += 1
        offset = (phase - start_row) % rb
        old = C.N_HOLDINGS
        C.N_HOLDINGS = n_hold
        try:
            r = qfbt.run_backtest(md, scores, start_row, end_row, risk_overlay=overlay,
                                  stop_loss=stop_loss, rebalance_days=rb,
                                  rebalance_offset=offset,
                                  trade_band_rel=0.30, trade_band_abs=6000.0,
                                  max_new_entries=max_new,
                                  cash_asset_returns=cash_ret, verbose=False)
        finally:
            C.N_HOLDINGS = old
        # 个股回测本身已按 config 全成本扣费（40bp 滑点 + 佣金 + 印花税），
        # embedded 直接取实测 cost_bp_per_trade，net 模式因此基本不产生额外扣减
        st = getattr(r, "metrics", None) or {}
        raw = r.nav
        nav, cinfo = self._cost_apply(raw, "stock", p,
                                      turnover_ann=st.get("turnover_ann"),
                                      embedded_bp=st.get("cost_bp_per_trade"))
        return self._bt_pack(nav, "stock", {
            "model": model,
            "cost": cinfo,
            "params": {"model": model, "n_holdings": n_hold, "rebalance_days": rb,
                       "max_new": max_new, "cash_bond": bw, "cash_gold": gw,
                       "overlay": overlay, "stop_loss": stop_loss}}, start, end,
            raw_nav=raw)

    def _bt_etf(self, p: dict, start: str, end: str) -> dict:
        import json as _json

        import numpy as _np

        from qf import metrics as _qfm

        panel = C.BUILD / "etf" / "etf_panel.npz"
        if not panel.exists():
            raise RuntimeError("ETF 数据不存在")
        z = _np.load(panel, allow_pickle=True)
        dates = pd.DatetimeIndex(z["dates"])
        codes = [str(c) for c in z["codes"]]
        close = z["close"].astype(float)
        amount = z["amount"].astype(float)
        entry_n = int(p.get("entry_n") or 55)
        exit_n = int(p.get("exit_n") or 0)
        min_amt = float(p.get("min_amount") or 5e7)
        import importlib as _il
        etf_mod = _il.import_module("scripts.etf_trend_strategy")
        # 传入策略模块的全局门槛（signals 内部读模块常量）
        _old_min = etf_mod.MIN_AMOUNT if hasattr(etf_mod, "MIN_AMOUNT") else None
        if _old_min is not None:
            etf_mod.MIN_AMOUNT = min_amt
        try:
            sig = etf_mod.signals(dates, codes, {"close": close, "amount": amount},
                                  entry_n=entry_n, exit_n=exit_n,
                                  max_pos=int(p.get("max_pos") or etf_mod.MAX_POS))
        finally:
            if _old_min is not None:
                etf_mod.MIN_AMOUNT = _old_min
        cash_idx = dates
        bond = pd.read_parquet(C.BOND_FILE)
        gold = pd.read_parquet(C.GOLD_FILE)
        b = pd.Series(bond["close"].to_numpy(), index=pd.to_datetime(bond["date"])).sort_index()
        g = pd.Series(gold["close"].to_numpy(), index=pd.to_datetime(gold["date"])).sort_index()
        cash = _blend_cash(b, g, cash_idx)
        nav, pos, trades = etf_mod.backtest(
            dates, codes, {"close": close, "amount": amount}, sig, cash,
            max_pos=int(p.get("max_pos") or etf_mod.MAX_POS),
            pos_style=str(p.get("pos_style") or etf_mod.POS_STYLE))
        row = dates >= pd.Timestamp(start)
        # 年化单边换手：每笔成交变动 1/max_pos，买+卖各计一笔 → 单边 = 笔数/(2*max_pos*年)
        max_pos = int(p.get("max_pos") or etf_mod.MAX_POS)
        years = max(int(row.sum()), 1) / 252.0
        tur = (len(trades) / 2.0 / max(max_pos, 1)) / years
        raw = nav[row]
        nav_out, cinfo = self._cost_apply(raw, "etf", p, turnover_ann=tur,
                                          embedded_bp=float(getattr(etf_mod, "SLIPPAGE", 0.0005)) * 1e4)
        return self._bt_pack(nav_out, "etf", {
            "params": {"entry_n": entry_n, "exit_n": exit_n, "min_amount": min_amt},
            "cost": cinfo,
            "exposure": _f(float((pos[row] > 0).mean())),
            "avg_holdings": _f(float(pos[row].mean())),
            "n_trades": int(len(trades)),
        }, start, end, raw_nav=raw)

    def _bt_sector(self, p: dict, start: str, end: str) -> dict:
        import numpy as _np

        from qf import sector_rotation as qfsr

        dates, codes, close, amount, sector_of = qfsr.load_pool()
        start_row = int(np.flatnonzero(dates >= pd.Timestamp(start))[0])
        res = qfsr.backtest(dates, codes, close, amount, sector_of,
                            start_row=start_row,
                            entry_n=int(p.get("entry_n") or qfsr.ENTRY_N),
                            mom_n=int(p.get("mom_n") or qfsr.MOM_N),
                            top_n=int(p.get("top_n") or qfsr.TOP_N),
                            rebal=int(p.get("rebalance_days") or qfsr.REBAL),
                            min_amount=float(p.get("min_amount") or qfsr.MIN_AMOUNT),
                            verbose=False)
        nav = res["nav"]
        row = nav.index >= pd.Timestamp(start)
        top_n = int(p.get("top_n") or qfsr.TOP_N)
        years = max(int(row.sum()), 1) / 252.0
        # 板块轮动每次换标的对应 1 买 + 1 卖，各占 1/top_n → 单边换手 = 笔数/(2*top_n*年)
        tur = (len(res["trades"]) / 2.0 / max(top_n, 1)) / years
        raw = nav[row]
        nav_out, cinfo = self._cost_apply(raw, "sector", p, turnover_ann=tur,
                                          embedded_bp=float(getattr(qfsr, "COST_BP", 5.0)))
        return self._bt_pack(nav_out, "sector", {
            "params": {"entry_n": int(p.get("entry_n") or qfsr.ENTRY_N),
                       "mom_n": int(p.get("mom_n") or qfsr.MOM_N),
                       "top_n": top_n,
                       "rebalance_days": int(p.get("rebalance_days") or qfsr.REBAL)},
            "cost": cinfo,
            "exposure": _f(float(res["exposure"][row].mean())),
            "n_trades": int(len(res["trades"])),
        }, start, end, raw_nav=raw)

    # ------------------------------------------------------------ 可转债
    def _cb_data(self):
        """CBData 实例（按数据文件 mtime 缓存，YTM 矩阵走 build/cb 缓存）。"""
        from qf import cb as qfcb

        daily = C.BUILD / "altdata" / "cb_daily.parquet"
        if not daily.exists():
            raise RuntimeError("可转债数据不存在，请先运行 build/fetch_cb.py")
        self._require_loaded()
        stamp = (daily.stat().st_mtime, self.md.dates[-1])
        cache = getattr(self, "_cb_cache", None) or {}
        if cache.get("stamp") == stamp:
            return cache["cb"]
        obj = qfcb.CBData(self.md, verbose=False)
        self._cb_cache = {"stamp": stamp, "cb": obj}
        return obj

    def _cb_backtest(self, n_hold=20, rebal=10, pool_mult=3.0, min_price=90.0,
                     rank="ytm_duration"):
        """转债回测（按参数 + 数据 mtime 缓存，首次约 1s，之后命中缓存）。"""
        from qf import cb as qfcb

        data = self._cb_data()
        key = (n_hold, rebal, pool_mult, min_price, rank, str(data.last_date.date()))
        hit = (getattr(self, "_cb_bt", None) or {})
        if hit.get("key") == key:
            return hit["res"]
        res = qfcb.backtest(data, start="2019-01-02", n_hold=n_hold, rebal=rebal,
                            pool_mult=pool_mult, min_price=min_price, rank=rank)
        self._cb_bt = {"key": key, "res": res}
        return res

    def _bt_cb(self, p: dict, start: str, end: str) -> dict:
        n_hold = int(p.get("n_holdings") or 20)
        rb = int(p.get("rebalance_days") or 10)
        pm = float(p.get("pool_mult") or 3.0)
        min_px = float(p.get("min_price") or 90.0)
        rank = str(p.get("rank") or "ytm_duration")
        res = self._cb_backtest(n_hold, rb, pm, min_px, rank)
        nav = res["nav"]
        expo = res["exposure"]
        held = res["n_held"]
        row = nav.index >= pd.Timestamp(start)
        band = row & (nav.index <= pd.Timestamp(end)) if end else row
        n_sub = max(int(band.sum()), 2)
        years = (n_sub - 1) / 252.0
        # turnover_per_reb 是单边换手比例，年化 = 每期 × 区间内调仓次数 / 区间年数。
        # 原来分母用全期 7.53 年 + 全期 189 次，start 挪到 2022 时换手仍报 7.18x，
        # 不随区间变（区间只剩 4.63 年，实际年化换手应更高）。
        n_reb_in = max(int(round(res["n_reb"] * n_sub / max(len(nav), 1))), 1)
        tur = float(res["turnover_per_reb"]) * n_reb_in / years
        # 先按区间截取再补扣：factor 从区间起点 1.0 起算，drag 才是区间内的
        raw = nav[band]
        nav_out, cinfo = self._cost_apply(raw, "cb", p, turnover_ann=tur, embedded_bp=5.0)
        return self._bt_pack(nav_out, "cb", {
            "params": {"n_holdings": n_hold, "rebalance_days": rb, "pool_mult": pm,
                       "min_price": min_px, "rank": rank},
            "cost": cinfo,
            "exposure": _f(float(expo[row].mean())),
            "avg_holdings": _f(float(held[row].mean())),
            "n_trades": int(res["n_reb"]),
            "turnover_per_reb": _f(float(res["turnover_per_reb"])),
        }, start, end, raw_nav=raw)

    def _lowturn_bt(self, n_hold=10, lookback=20, min_list=250, min_adv=0.0, rw=True):
        """低换手月度回测（按参数 + 数据末日缓存）。"""
        from qf import lowturn as _lt

        self._require_loaded()
        key = (n_hold, lookback, min_list, float(min_adv), bool(rw),
               str(self.md.dates[-1].date()))
        hit = getattr(self, "_lowturn_cache", None) or {}
        if hit.get("key") == key:
            return hit["res"]
        res = _lt.backtest(self.md, start="2020-01-02", n_hold=n_hold, lookback=lookback,
                           min_list_days=min_list, min_adv20=min_adv,
                           rebalance_weight=bool(rw))
        self._lowturn_cache = {"key": key, "res": res}
        return res

    def _bt_lowturn(self, p: dict, start: str, end: str) -> dict:
        """低换手月度策略：双账本（毛/净两套独立净值），不是按差额补扣。"""
        from qf import lowturn as _lt

        n_hold = int(p.get("n_holdings") or 10)
        lookback = int(p.get("lookback") or _lt.LOOKBACK)
        min_list = int(p.get("min_list_days") or _lt.MIN_LIST_DAYS)
        min_adv = float(p.get("min_adv20") or 0.0) * 1e8
        rw = bool(p.get("rebalance_weight", True))
        res = self._lowturn_bt(n_hold, lookback, min_list, min_adv, rw)

        model = str(p.get("cost_model") or "gross").lower()
        gross, net, strict = res["nav_gross"], res["nav"], res["nav_strict"]
        nav = net if model == "net" else gross
        tur = float(res["turnover_ann"])
        target = self.cost_target_bp("lowturn")
        # 与所选区间对齐：原来用全程 performance，导致用户改起点后
        # 「扣费前 X% → 扣费后 Y%」两个数字不动，和上方的 metrics 自相矛盾。
        def _seg(s):
            o = s[s.index >= start]
            return o[o.index <= end] if end else o

        sg, sn = _seg(gross), _seg(net)
        pg = _lt.performance(sg) if len(sg) >= 2 else _lt.performance(gross)
        pn = _lt.performance(sn) if len(sn) >= 2 else _lt.performance(net)
        # turnover_ann 是「每期成交额（买+卖）/ 组合」的年化，两边已计入，
        # 所以年成本 = 换手 × 单边费率，不能再 ×2（曾写成 ×2，显示 7.02% 而实际 4.01pp）。
        ann_cost = tur * target / 1e4
        cinfo = {
            "model": model,
            "basis": "double_book",
            "target_bp": _f(target), "embedded_bp": None, "delta_bp": None,
            "turnover_ann": _f(tur),
            "turnover_basis": "双边成交额/组合（买+卖已含）",
            "ann_cost": _f(ann_cost),
            "applied": model == "net",
            "note": "双账本：毛/净是两套独立净值，按每笔成交扣 49.3bp，非按差额补扣",
            "cagr_gross": _f(pg["cagr"]), "cagr_net": _f(pn["cagr"]),
            "sharpe_gross": _f(pg["sharpe"]), "sharpe_net": _f(pn["sharpe"]),
            "drag_pp": _f(pg["cagr"] - pn["cagr"]),
        }
        last = (res["picks"] or [{}])[-1]
        return self._bt_pack(nav, "lowturn", {
            "params": {"n_holdings": n_hold, "lookback": lookback,
                       "min_list_days": min_list,
                       "min_adv20_yi": float(p.get("min_adv20") or 0.0),
                       "rebalance_weight": rw},
            "cost": cinfo,
            "n_months": int(res["n_months"]),
            "keep_ratio": _f(res["keep_mean"]),
            "turnover_ann": _f(tur),
            "cagr_strict": _f(_lt.performance(strict)["cagr"]),
            "last_picks": last,
        }, start, end)

    def _combo_inputs(self, p: dict, method: str, start: str, end: str):
        """加载四类资产净值 → 对齐区间 → 按需补扣成本。

        返回 (keys, idx, R, R_gross, cost_detail)：
          R 是实际参与组合的收益矩阵（net 模式下已补扣成本），
          R_gross 是扣费前对照（未扣费时两者相同）。
        """
        import numpy as _np

        # 四类资产净值（cb 为可转债策略，内部回测）
        src = (("main", C.BUILD / "runs" / "main" / "nav.parquet"),
               ("etf", C.BUILD / "runs" / "etf_trend" / "nav_donchian55.parquet"),
               ("sector", C.BUILD / "runs" / "sector_rotation" / "nav.parquet"))
        navs = {}
        for key, fp in src:
            if not fp.exists():
                raise RuntimeError(f"{key} 净值不存在，请先运行对应策略")
            ser = pd.read_parquet(fp)["nav"].astype(float)
            ser.index = pd.to_datetime(ser.index)
            navs[key] = ser
        # 可转债：固定权重给了 cb 才参与；动态配置需显式 include_cb
        w_req = dict(p.get("weights") or {})
        want_cb = (float(w_req.get("cb") or 0.0) > 0.0) if method == "fixed" \
            else bool(p.get("include_cb"))
        if want_cb:
            try:
                navs["cb"] = self._cb_backtest()["nav"]
            except Exception as exc:
                raise RuntimeError(f"可转债策略净值不可用：{exc}") from exc

        idx = None
        for v in navs.values():
            idx = v.index if idx is None else idx.intersection(v.index)
        idx = idx[idx >= pd.Timestamp(start)]
        if end:
            idx = idx[idx <= pd.Timestamp(end)]
        if len(idx) < 30:
            raise RuntimeError("对齐区间过短（<30 日）")

        keys = ["main", "etf", "sector"] + (["cb"] if want_cb else [])
        # 扣费后：逐资产按统一口径补扣，再用同一套配置逻辑组合（口径可加）
        _kind_of = {"main": "stock", "etf": "etf", "sector": "sector", "cb": "cb"}
        cost_detail, navs_gross = {}, {}
        for k in keys:
            navs_gross[k] = navs[k]
            _net, _info = self._cost_apply(navs[k], _kind_of[k], p)
            cost_detail[k] = _info
            if _info.get("applied"):
                navs[k] = _net
        R = _np.column_stack([
            navs[k].reindex(idx).pct_change().fillna(0).to_numpy() for k in keys
        ])
        R_gross = _np.column_stack([
            navs_gross[k].reindex(idx).pct_change().fillna(0).to_numpy() for k in keys
        ])
        return keys, idx, R, R_gross, cost_detail

    def _bt_combo(self, p: dict, start: str, end: str) -> dict:
        """多策略组合回测。

        两种模式：
          * method='fixed'（默认）：按 weights 固定权重，日频再平衡
          * method in {eq, ivp, erc, mdp}：**动态配置**
            每 rebal 日用过去 cov_win 日的协方差重算权重（无前视）
            权重计算见 qf/portfolio_erc.py
        """
        import numpy as _np
        from qf import portfolio_erc as _per

        method = str(p.get("method") or "fixed").lower()
        cov_win = int(p.get("cov_window") or 252)
        rebal = int(p.get("rebal") or 60)
        keys, idx, R, R_gross, cost_detail = self._combo_inputs(p, method, start, end)
        n_asset = len(keys)

        def _build(Rmat):
            """给定收益矩阵，按 method 生成组合净值与权重（可重复调用）。"""
            if method == "fixed":
                w0 = dict(p.get("weights") or {"main": 0.5, "etf": 0.3, "sector": 0.2})
                tot = sum(max(0.0, float(w0.get(k, 0.0))) for k in keys) or 1.0
                wv = _np.array([max(0.0, float(w0.get(k, 0.0))) / tot for k in keys])
                _nav = (1 + Rmat @ wv).cumprod()
                _ws = [{"date": str(idx[0].date()), "weights": [float(x) for x in wv]}]
                return _nav, _ws, [float(x) for x in wv]

            fn = {"eq": lambda cov: _np.ones(n_asset) / n_asset,
                  "ivp": _per.ivp_weights,
                  "erc": _per.erc_weights,
                  "mdp": _per.mdp_weights}.get(method)
            if fn is None:
                raise RuntimeError(f"未知 method={method}")
            n = len(idx)
            port = _np.zeros(n)
            w = _np.ones(n_asset) / n_asset
            _ws = []
            for i in range(n):
                if i >= max(cov_win, 20) and (i % rebal == 0):
                    win = Rmat[max(0, i - cov_win):i]
                    try:
                        cov = _np.cov(win.T)
                        cov = _np.nan_to_num(cov, nan=0.0)
                        if _np.trace(cov) <= 0:
                            raise ValueError("degenerate cov")
                        w_new = _np.asarray(fn(cov), float)
                        w_new = _np.maximum(w_new, 0.0)
                        ssum = w_new.sum()
                        if ssum > 0:
                            w = w_new / ssum
                            _ws.append({"date": str(idx[i].date()),
                                        "weights": [float(x) for x in w]})
                    except Exception:
                        pass
                port[i] = float(w @ Rmat[i])
            return _np.cumprod(1 + port), _ws, [float(v) for v in w]

        nav, ws_ts, avg_w = _build(R)
        net_on = any(v.get("applied") for v in cost_detail.values())
        gross_nav = pd.Series(_build(R_gross)[0], index=idx) if net_on else None
        nav_s = pd.Series(nav, index=idx)
        labels = {"main": "主策略（个股）", "etf": "宽基 ETF",
                  "sector": "板块 ETF", "cb": "可转债"}
        _extra = {
            "weights": {k: _f(float(avg_w[i])) for i, k in enumerate(keys)},
            "parts": [{"key": k, "weight": _f(float(avg_w[i])), "label": labels[k]}
                      for i, k in enumerate(keys) if avg_w[i] > 0],
            "cost": {
                "model": cost_detail[keys[0]]["model"],
                "ann_cost": _f(sum(float(avg_w[i]) * (cost_detail[k].get("ann_cost") or 0.0)
                                   for i, k in enumerate(keys))),
                "target_bp": _f(sum(float(avg_w[i]) * cost_detail[k]["target_bp"]
                                    for i, k in enumerate(keys))),
                "embedded_bp": _f(sum(float(avg_w[i]) * cost_detail[k]["embedded_bp"]
                                      for i, k in enumerate(keys))),
                "delta_bp": _f(sum(float(avg_w[i]) * cost_detail[k]["delta_bp"]
                                   for i, k in enumerate(keys))),
                "turnover_ann": _f(sum(float(avg_w[i]) * cost_detail[k]["turnover_ann"]
                                       for i, k in enumerate(keys))),
                "by_asset": cost_detail,
            },
            "method": method,
            "cov_window": cov_win,
            "rebal": rebal,
            "weights_ts": ws_ts[-60:] if len(ws_ts) > 60 else ws_ts,
        }
        if method != "fixed":
            _extra["label"] = {
                "eq": "等权配置", "ivp": "逆波动率", "erc": "ERC 风险平价",
                "mdp": "最大分散化",
            }.get(method, method)
        elif p.get("label"):
            _extra["label"] = str(p["label"])
        return self._bt_pack(nav_s, "combo", _extra, start, end, raw_nav=gross_nav)

    # ---------------------------------------------------------------- 策略研究
    def _load_eval(self, name: str):
        """读 build/eval/<name>（json/parquet），失败返回 None。"""
        import json as _json

        p = C.BUILD / "eval" / name
        if not p.exists():
            return None
        try:
            if p.suffix == ".json":
                return _json.load(open(p, encoding="utf-8"))
            if p.suffix == ".parquet":
                import pandas as _pd
                return _pd.read_parquet(p)
        except Exception:
            return None
        return None

    def cost_sensitivity(self) -> dict:
        """成本敏感性：滑点扫描 + 持仓市值 + 真实费率。"""
        cs = self._load_eval("cost_sensitivity.json")
        hm = self._load_eval("holding_mcap.json")
        fr = self._load_eval("fee_recalc.json")
        out = {"has_data": bool(cs), "sweep": [], "holding": None,
               "fee": None}
        if cs:
            rows = []
            for k, v in (cs.get("results") or {}).items():
                try:
                    rows.append({"slip": int(k), **v})
                except Exception:
                    continue
            out["sweep"] = sorted(rows, key=lambda r: r["slip"])
            out["turnover"] = cs.get("turnover")
            out["crit_sharpe"] = cs.get("crit_sharpe_030")
            out["crit_cagr"] = cs.get("crit_cagr_zero")
        if hm:
            out["holding"] = hm
        if fr:
            out["fee"] = {"current_roundtrip_bp": fr.get("current_roundtrip_bp"),
                          "real_roundtrip_bp": fr.get("real_roundtrip_bp"),
                          "ann_cost_current": fr.get("ann_cost_current"),
                          "ann_cost_real": fr.get("ann_cost_real"),
                          "min_comm_rows": (fr.get("rows") or [])[:8]}
        return out

    def rebalance_study(self) -> dict:
        """调仓周期研究：扫描结果 + 判定。"""
        rt = self._load_eval("rebalance_test.json")
        rc = self._load_eval("rebalance_conservative.json")
        rf = self._load_eval("etf_flex_vs_fixed.json")
        out = {"has_data": bool(rt), "dev": [], "oos": [], "full": [],
               "conservative": None, "etf_flex": []}
        if rt:
            g = rt.get("grid") or {}
            for key in ("dev", "oos", "full"):
                for k, v in (g.get(key) or {}).items():
                    try:
                        out[key].append({"period": int(k), **v})
                    except Exception:
                        continue
                out[key].sort(key=lambda r: r["period"])
        if rc:
            rows = []
            for k, v in (rc.get("rows") or {}).items():
                try:
                    rows.append({"period": int(k),
                                 "dev_cagr": v["dev"]["cagr"],
                                 "oos_cagr": v["oos"]["cagr"],
                                 "full_cagr": v["full"]["cagr"],
                                 "oos_sharpe": v["oos"]["sharpe"],
                                 "turnover": v["oos"]["turnover"],
                                 "ann_cost": v["oos"]["ann_cost"]})
                except Exception:
                    continue
            out["conservative"] = {
                "rows": sorted(rows, key=lambda r: r["period"]),
                "qualified": rc.get("qualified"),
                "recommend": rc.get("recommend"),
            }
        if rf:
            rows = []
            for k, v in (rf or {}).items():
                try:
                    label, per = k.rsplit("|", 1)
                    rows.append({"config": label, "period": per,
                                 "cagr": v["perf"]["cagr"],
                                 "sharpe": v["perf"]["sharpe"],
                                 "max_dd": v["perf"]["max_dd"],
                                 "trades": v["trades"]})
                except Exception:
                    continue
            out["etf_flex"] = rows
        return out

    def etf_pool_health(self) -> dict:
        """ETF 池健康度：流动性 + 相关性。"""
        lq = self._load_eval("etf_liquidity_check.json")
        cr = self._load_eval("etf_pool_corr.json")
        out = {"liquidity": None, "corr": None}
        if lq:
            out["liquidity"] = {
                "counts": lq.get("liquidity_counts"),
                "bad": lq.get("bad"), "warn": lq.get("warn"),
                "detail": (lq.get("detail") or [])[:50],
            }
        if cr:
            out["corr"] = {
                "n_valid": cr.get("n_valid"),
                "n_pairs_high": cr.get("n_pairs_high"),
                "pairs": (cr.get("pairs") or [])[:20],
                "mean_corr": cr.get("mean_corr"),
                "median": cr.get("median"),
                "distribution": cr.get("distribution"),
            }
        return out

    # ---------------------------------------------------------------- 实盘跟踪
    # ------------------------------------------------ 跟踪操作（写）
    def _track_paths(self):
        T = C.BUILD / "tracking"
        T.mkdir(parents=True, exist_ok=True)
        return {
            "signal": T / "signal_log.jsonl",
            "trade": T / "paper_trades.jsonl",
            "nav": T / "daily_nav.jsonl",
            "state": T / "tracking_state.json",
            "ref": T / "backtest_reference.json",
        }

    def _track_state(self):
        import json as _json

        P = self._track_paths()
        if P["state"].exists():
            try:
                return _json.load(open(P["state"], encoding="utf-8"))
            except Exception:
                pass
        return {"inception": None, "capital": 300000.0, "holdings": {},
                "cash": 300000.0, "last_update": None}

    def _track_save(self, st):
        import json as _json

        P = self._track_paths()
        P["state"].write_text(_json.dumps(st, ensure_ascii=False, indent=1,
                                          default=str), encoding="utf-8")

    def _track_append(self, which, obj):
        import json as _json

        P = self._track_paths()
        with open(P[which], "a", encoding="utf-8") as f:
            f.write(_json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    def tracking_fill(self, body: dict) -> dict:
        """补录一笔成交。body: code/side/price/shares/fee/date/reason。"""
        import datetime as _dt

        code = str(body.get("code") or "").strip().zfill(6)
        side = str(body.get("side") or "").strip()
        if not code or len(code) != 6 or not code.isdigit():
            raise ValueError("代码必须是 6 位数字")
        if side not in ("buy", "sell"):
            raise ValueError("方向必须是 buy 或 sell")
        try:
            price = float(body.get("price"))
            shares = int(body.get("shares"))
        except Exception:
            raise ValueError("价格/股数格式错误")
        if price <= 0 or shares <= 0:
            raise ValueError("价格和股数必须为正")
        fee = float(body.get("fee") or 0.0)
        date = str(body.get("date") or _dt.date.today().isoformat())

        rec = {"recorded_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "date": date, "code": code, "side": side,
               "price": price, "shares": shares,
               "amount": round(price * shares, 2), "fee": fee,
               "reason": str(body.get("reason") or ""),
               "signal_date": str(body.get("signal_date") or "")}
        self._track_append("trade", rec)

        # 更新持仓
        st = self._track_state()
        pos = st["holdings"].get(code) or {"shares": 0, "cost": 0.0}
        if side == "buy":
            tot = pos["cost"] * pos["shares"] + rec["amount"] + fee
            pos["shares"] += shares
            pos["cost"] = tot / pos["shares"] if pos["shares"] else 0.0
            st["cash"] = float(st.get("cash") or 0) - rec["amount"] - fee
            st["holdings"][code] = pos
        else:
            pos["shares"] -= shares
            st["cash"] = float(st.get("cash") or 0) + rec["amount"] - fee
            if pos["shares"] <= 0:
                st["holdings"].pop(code, None)
            else:
                st["holdings"][code] = pos
        st["last_update"] = date
        self._track_save(st)
        return {"ok": True, "record": rec,
                "cash": st["cash"], "n_holdings": len(st["holdings"])}

    def tracking_nav(self, body: dict) -> dict:
        """按当前持仓 + 收盘价更新净值。"""
        import datetime as _dt

        st = self._track_state()
        date = str(body.get("date") or "")
        md = self.md
        if md is None:
            raise RuntimeError("行情未加载")
        import numpy as _np
        import pandas as _pd

        dates = _pd.DatetimeIndex(md.dates)
        if date:
            t = int(dates.searchsorted(_pd.Timestamp(date), side="right")) - 1
        else:
            t = md.T - 1
        t = max(0, min(t, md.T - 1))
        date = str(dates[t].date())

        code_idx = {str(c).zfill(6): j for j, c in enumerate(md.codes)}
        mv, detail = 0.0, []
        for code, pos in (st.get("holdings") or {}).items():
            j = code_idx.get(code)
            px = md.price[t, j] if j is not None else _np.nan
            if not _np.isfinite(px):
                px = pos.get("cost", 0.0)
            val = pos["shares"] * float(px)
            mv += val
            detail.append({"code": code, "shares": pos["shares"],
                           "price": float(px), "value": round(val, 2),
                           "cost": pos.get("cost", 0.0),
                           "pnl_pct": (float(px) / pos["cost"] - 1)
                           if pos.get("cost", 0) > 0 else 0.0})

        cash = float(st.get("cash") or 0.0)
        nav = cash + mv
        rec = {"recorded_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "date": date, "nav": round(nav, 2), "cash": round(cash, 2),
               "n_holdings": len(detail), "market_value": round(mv, 2),
               "detail": detail}
        self._track_append("nav", rec)
        cap = float(st.get("capital") or 300000.0)
        return {"ok": True, "date": date, "nav": round(nav, 2),
                "pnl_pct": (nav / cap - 1) if cap > 0 else None,
                "n_holdings": len(detail), "market_value": round(mv, 2),
                "cash": round(cash, 2)}

    def tracking_signal(self) -> dict:
        """记录当日信号（从 signals/latest_combo.json）。"""
        import datetime as _dt
        import json as _json

        p = C.SIGNALS / "latest_combo.json"
        if not p.exists():
            raise RuntimeError("找不到 signals/latest_combo.json，先跑 daily_signal.py")
        d = _json.load(open(p, encoding="utf-8"))
        sd = d.get("signal_date")
        if not sd:
            raise RuntimeError("信号文件缺 signal_date")

        P = self._track_paths()
        exists = False
        if P["signal"].exists():
            with open(P["signal"], encoding="utf-8") as f:
                for line in f:
                    try:
                        if _json.loads(line).get("signal_date") == sd:
                            exists = True
                            break
                    except Exception:
                        pass
        if exists:
            return {"ok": True, "skipped": True, "signal_date": sd,
                    "msg": f"信号 {sd} 已记录"}

        snap = {
            "recorded_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "signal_date": sd, "capital": d.get("capital"),
            "parts": d.get("parts"), "weights": d.get("weights"),
            "cash": d.get("cash"),
            "n_main_targets": len((d.get("main") or {}).get("targets") or []),
            "etf_leg": (d.get("etf") or {}).get("leg"),
            "rebalance_hint": d.get("rebalance_hint"),
            "main_targets": [
                {"code": t.get("code"), "name": t.get("name"),
                 "weight": t.get("weight"),
                 "industry": t.get("industry")}
                for t in ((d.get("main") or {}).get("targets") or [])],
        }
        self._track_append("signal", snap)

        st = self._track_state()
        if not st.get("inception"):
            st["inception"] = sd
            st["capital"] = float(d.get("capital") or 300000.0)
            st["cash"] = float(d.get("capital") or 300000.0)
            self._track_save(st)
        return {"ok": True, "signal_date": sd,
                "n_targets": snap["n_main_targets"]}

    def tracking_delete_trade(self, body: dict) -> dict:
        """删除指定日期+代码+方向的成交，并回滚持仓。"""
        import json as _json

        date = str(body.get("date") or "")
        code = str(body.get("code") or "").zfill(6)
        side = str(body.get("side") or "")
        if not (date and code and side):
            raise ValueError("需要 date/code/side")

        P = self._track_paths()
        if not P["trade"].exists():
            raise ValueError("无成交记录")
        rows = []
        removed = None
        with open(P["trade"], encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    x = _json.loads(line)
                except Exception:
                    continue
                if (removed is None and str(x.get("date")) == date
                        and str(x.get("code")) == code
                        and str(x.get("side")) == side):
                    removed = x
                    continue
                rows.append(x)
        if removed is None:
            raise ValueError("未找到匹配的成交")
        with open(P["trade"], "w", encoding="utf-8") as f:
            for x in rows:
                f.write(_json.dumps(x, ensure_ascii=False, default=str) + "\n")

        # 回滚
        st = self._track_state()
        pos = st["holdings"].get(code) or {"shares": 0, "cost": 0.0}
        amt = float(removed["amount"])
        fee = float(removed.get("fee") or 0)
        if removed["side"] == "buy":
            pos["shares"] -= int(removed["shares"])
            st["cash"] = float(st.get("cash") or 0) + amt + fee
            if pos["shares"] <= 0:
                st["holdings"].pop(code, None)
            else:
                st["holdings"][code] = pos
        else:
            pos["shares"] += int(removed["shares"])
            st["cash"] = float(st.get("cash") or 0) - amt + fee
            st["holdings"][code] = pos
        self._track_save(st)
        return {"ok": True, "removed": removed}

    def tracking_reset(self, body: dict) -> dict:
        """清空台账（需 confirm='RESET'）。"""
        import json as _json

        if str(body.get("confirm") or "") != "RESET":
            raise ValueError("需要 confirm='RESET'")
        P = self._track_paths()
        for k in ("trade", "nav"):
            if P[k].exists():
                P[k].unlink()
        st = {"inception": None, "capital": float(body.get("capital") or 300000.0),
              "holdings": {}, "cash": float(body.get("capital") or 300000.0),
              "last_update": None}
        self._track_save(st)
        return {"ok": True, "msg": "台账已清空"}

    def tracking_data(self) -> dict:
        """读取 build/tracking/*，返回台账视图。"""
        import json as _json

        T = C.BUILD / "tracking"

        def jl(name):
            p = T / name
            if not p.exists():
                return []
            out = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            out.append(_json.loads(line))
                        except Exception:
                            pass
            return out

        state = {"inception": None, "capital": 300000.0,
                 "holdings": {}, "cash": 300000.0}
        sp = T / "tracking_state.json"
        if sp.exists():
            try:
                state = _json.load(open(sp, encoding="utf-8"))
            except Exception:
                pass

        ref = None
        rp = T / "backtest_reference.json"
        if rp.exists():
            try:
                ref = _json.load(open(rp, encoding="utf-8"))
            except Exception:
                pass

        navs = jl("daily_nav.jsonl")
        # 去重（同日取最后一条）
        seen = {}
        for x in navs:
            seen[str(x.get("date"))] = x
        navs = [seen[k] for k in sorted(seen)]

        trades = jl("paper_trades.jsonl")
        sigs = jl("signal_log.jsonl")

        # 信号 vs 持仓一致性
        consistency = None
        if sigs and state.get("holdings"):
            last = sigs[-1]
            tg = set(t.get("code") for t in (last.get("main_targets") or [])
                     if t.get("code"))
            held = set(state["holdings"].keys())
            if tg and held:
                consistency = {
                    "signal_date": last.get("signal_date"),
                    "n_target": len(tg), "n_held": len(held),
                    "n_overlap": len(tg & held),
                    "missing": sorted(tg - held)[:20],
                    "extra": sorted(held - tg)[:20],
                }

        # 当前净值与收益
        cur = None
        if navs:
            last = navs[-1]
            cap = float(state.get("capital") or 300000.0)
            cur = {"date": last.get("date"), "nav": last.get("nav"),
                   "cash": last.get("cash"),
                   "n_holdings": last.get("n_holdings"),
                   "market_value": last.get("market_value"),
                   "pnl_pct": (float(last.get("nav", 0)) / cap - 1)
                   if cap > 0 else None}

        return {"state": state, "navs": navs, "trades": trades,
                "signals": sigs[-20:], "reference": ref,
                "consistency": consistency, "current": cur,
                "n_signal_logs": len(sigs), "n_trades": len(trades)}

    def experiments_ledger(self) -> dict:
        """实验台账：已弃用/保留的策略实验（避免重复踩坑）。"""
        import json as _json

        p = C.BUILD / "eval" / "experiments_ledger.json"
        if not p.exists():
            return {"experiments": [], "kept": []}
        try:
            return _json.load(open(p, encoding="utf-8"))
        except Exception as e:
            return {"experiments": [], "kept": [], "error": str(e)}

    def research_overview(self) -> dict:
        """研究页汇总（一次请求拿全部）。"""
        return {"cost": self.cost_sensitivity(),
                "rebalance": self.rebalance_study(),
                "pool": self.etf_pool_health(),
                "ledger": self.experiments_ledger()}

    def consistency_check(self) -> dict:
        """数据源一致性检测（复用 scripts/check_data_source.py 的逻辑）。"""
        import subprocess
        import sys as _sys
        import json as _json

        try:
            _cf = {"creationflags": 0x08000000} if _sys.platform == "win32" else {}  # CREATE_NO_WINDOW
            r = subprocess.run(
                [_sys.executable, "-u", str(C.BUILD.parent / "scripts" / "check_data_source.py"),
                 "--json"],
                cwd=str(C.BUILD.parent), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=120,
                env={**__import__("os").environ, "QF_DATA_RAW": str(C.DATA_RAW),
                     "PYTHONIOENCODING": "utf-8"}, **_cf)
            out = (r.stdout or "").strip()
            if out.startswith("{"):
                return _json.loads(out)
            return {"ok": False, "items": [],
                    "error": (r.stderr or "")[-300:] or "输出非 JSON"}
        except Exception as e:
            return {"ok": False, "items": [], "error": f"{type(e).__name__}: {e}"}

    # ---------------------------------------------------------------- 更新管道
    # 与 scripts/run_pipeline.py 的 STAGES 保持一致：顺序 + 预计耗时（秒）
    PIPE_ORDER = ["data", "cb_data", "factors", "model", "backtest", "phase_band", "etf",
                  "sector", "combo", "three", "erc", "tracking", "track_record", "numbers",
                  "report_s1", "signal", "cb_signal", "verify"]
    PIPE_EST = {"data": 150, "cb_data": 25, "factors": 950, "model": 150, "backtest": 35,
                "phase_band": 180,
                "etf": 6, "sector": 5, "combo": 4, "three": 4, "erc": 4,
                "tracking": 18, "track_record": 2, "numbers": 2,
                "report_s1": 2, "signal": 35, "cb_signal": 5, "verify": 2}
    PIPE_LABELS = {"data": "数据增量更新", "cb_data": "转债数据增量",
                   "factors": "因子面板重建", "model": "模型重训",
                   "backtest": "主策略回测", "phase_band": "主策略相位带",
                   "etf": "ETF 趋势回测",
                   "sector": "板块轮动回测", "combo": "50/50 组合",
                   "three": "三策略组合", "erc": "ERC 配置对比",
                   "tracking": "候选跟踪", "track_record": "实盘台账",
                   "numbers": "刷新基准数字", "report_s1": "同步报告摘要",
                   "signal": "今日信号", "cb_signal": "转债信号",
                   "verify": "数据源一致性检测"}

    def _pipe_progress(self, d: dict) -> dict:
        """由 status.json 推算总进度（百分比 / 第几步 / 剩余时间）。

        权重用各阶段预计耗时：整体进度按"时间"分配，而不是按阶段个数平均，
        否则 1 秒的 verify 和 15 分钟的因子重建看起来一样重。
        阶段内按 已用时间/预计时间 线性插值，跑得比预计慢时封顶在本阶段权重内。
        """
        import datetime as _dt

        stages = d.get("stages") or []
        by_key = {s.get("key"): s for s in stages if isinstance(s, dict)}
        cur = d.get("current") or {}
        order = self.PIPE_ORDER
        # 预计耗时优先用历史实际耗时（中位数），没有历史才用内置默认值
        est = dict(self.PIPE_EST)
        try:
            import json as _json
            import statistics as _st

            dur_p = C.BUILD / "pipeline" / "durations.json"
            if dur_p.exists():
                dur = _json.loads(dur_p.read_text(encoding="utf-8"))
                for k, vals in (dur or {}).items():
                    v = [x for x in (vals or []) if isinstance(x, (int, float)) and x > 0]
                    if v and k in est:
                        est[k] = float(_st.median(v))
        except Exception:
            pass
        # 优先用管道启动时写入的 plan（本次真正要跑的阶段），否则按全部 15 步
        plan = [k for k in (d.get("plan") or order) if k in est] or list(order)
        total_w = float(sum(est.get(k, 60) for k in plan))

        done_w = 0.0
        done_n = 0
        for k in plan:
            s = by_key.get(k) or {}
            if s.get("status") in ("ok", "failed", "skipped"):
                done_w += est.get(k, 60)
                done_n += 1

        cur_key = cur.get("key")
        elapsed_cur, partial = None, 0.0
        if cur_key:
            w = float(est.get(cur_key, 60))
            try:
                t0 = _dt.datetime.fromisoformat(str(cur.get("started_at")))
                elapsed_cur = max(0.0, (_dt.datetime.now() - t0).total_seconds())
            except Exception:
                elapsed_cur = None
            frac = min(1.0, (elapsed_cur or 0.0) / w) if w > 0 else 0.0
            partial = w * frac

        pct = (done_w + partial) / total_w if total_w else 0.0
        if d.get("running"):
            pct = min(pct, 0.99)          # 未结束不显示 100%
        elif d.get("ok") is True:
            pct = 1.0

        return {
            "pct": round(pct, 4),
            "done": done_n,
            "total": len(plan),
            "step": ((plan.index(cur_key) + 1) if cur_key in plan else done_n),
            "current_key": cur_key,
            "current_name": cur.get("name") or self.PIPE_LABELS.get(cur_key or "", ""),
            "elapsed_current": round(elapsed_cur, 1) if elapsed_cur is not None else None,
            "eta_current": (max(0.0, est.get(cur_key, 60) - (elapsed_cur or 0.0))
                            if cur_key else None),
            "eta_remaining": (max(0.0, total_w - done_w - partial)
                              if d.get("running") else 0.0),
        }

    def pipeline_status(self) -> dict:
        """读取 build/pipeline/status.json（一键更新进度）。

        2026-11-04 修复：状态文件里的 `running=true` 不可全信。上次更新的进程被系统
        杀掉后，status.json 会永久停在 running=true，网页就一直显示"运行中"空转
        （实测空转 24 小时）。这里读取时校验 pid 与心跳，发现进程已死就把这份状态
        改判为「上次运行已中断」，让页面能自己恢复、按钮能重新点。
        """
        import json as _json

        p = C.BUILD / "pipeline" / "status.json"
        if not p.exists():
            return {"running": False, "stages": [], "never_run": True,
                    "hint": "尚未运行过更新管道"}
        # 2026-11-07：这里是热读点（每 2~3 秒一次），撞上管道正在替换 status.json
        # 时会读到空文件或半截 JSON。单次失败不该让页面闪出"失败"，先重试再报错。
        d, err = None, None
        for _ in range(3):
            try:
                d = _json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception as e:          # 空文件 / 正在被替换 / 半截 JSON
                err = e
                time.sleep(0.05)
        if d is None:
            return {"running": False, "stages": [], "error": str(err)}
        d = self._mark_stale(d)
        # 数据新鲜度
        try:
            md = self.md
            last = str(md.dates[-1].date()) if md is not None else None
        except Exception:
            last = None
        d["data_last_date"] = last
        try:
            d["progress"] = self._pipe_progress(d)
        except Exception as e:
            d["progress"] = {"pct": 0.0, "error": str(e)}
        return d

    @staticmethod
    def _mark_stale(d: dict) -> dict:
        """running=true 但进程已不在 / 心跳超时 → 改判为中断状态。"""
        if not isinstance(d, dict) or not d.get("running"):
            return d
        if not _pipe_is_stale(d):
            return d
        d = dict(d)
        cur = d.get("current") or {}
        # 已用时长：优先用 current.started_at 现算（status.json 里不存 progress）
        spent = "较长时间"
        try:
            import datetime as _dt
            t0 = _dt.datetime.fromisoformat(str(cur.get("started_at") or ""))
            secs = max(0.0, (_dt.datetime.now() - t0).total_seconds())
            spent = (f"{secs/3600:.1f} 小时" if secs >= 3600
                     else f"{secs/60:.0f} 分钟" if secs >= 90 else f"{secs:.0f} 秒")
        except Exception:
            pass
        d["running"] = False
        d["interrupted"] = True
        d["ok"] = False
        d["stale"] = True
        d["finished_at"] = d.get("finished_at") or d.get("updated_at")
        d["stale_reason"] = (
            f"上次更新在「{cur.get('name') or '未知阶段'}」这一步中断："
            f"进程已退出（PID {d.get('pid') or '未记录'}），已运行 {spent}，"
            f"数据未写入。请点「开始更新」重跑。"
        )
        # 2026-11-07：进程死在**收尾**（写 status.json 被文件锁挡住）时，
        # 计划内的阶段其实全跑完了，只是没来得及把 ok=True 落盘。这种情况必须判成功，
        # 否则界面显示"失败"→ 用户再点一次 → factors 白重跑 850 秒。
        # 判据用 plan 而不是"有没有失败阶段"：plan 是本次真正要跑的清单，
        # 只有它全 ok 才算完成；中途被杀时最后一个阶段根本没进 stages。
        _plan = [k for k in (d.get("plan") or []) if isinstance(k, str)]
        _done = {s.get("key"): s.get("status")
                 for s in (d.get("stages") or []) if isinstance(s, dict)}
        if _plan and all(_done.get(k) == "ok" for k in _plan):
            d["ok"] = True
            d["interrupted"] = False
            d["stale_reason"] = (
                f"计划内 {len(_plan)} 个阶段全部完成；进程在收尾写状态时退出，"
                f"已按成功判定（数据已写入）。"
            )
        d["current"] = None
        return d

    def pipeline_run(self, skip_data=False, skip_factors=False, only=None) -> dict:
        """启动更新管道（后台子进程）。返回是否已启动。"""
        import subprocess
        import sys as _sys

        st = self.pipeline_status()
        if st.get("running"):
            return {"started": False, "reason": "已有更新任务在运行"}

        cmd = [_sys.executable, "-u", str(C.BUILD.parent / "scripts" / "run_pipeline.py")]
        if skip_data:
            cmd.append("--skip-data")
        if skip_factors:
            cmd.append("--skip-factors")
        if only:
            cmd.extend(["--only", str(only)])

        # 后台启动，脱离父进程
        log = C.BUILD / "pipeline" / "runner.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        n_before = self._log_offset(log)
        f = open(log, "a", encoding="utf-8")
        f.write(f"\n\n===== {__import__('datetime').datetime.now()} 启动 =====\n")
        f.flush()
        kw = {}
        if _sys.platform == "win32":
            # 关键：不要用 DETACHED_PROCESS。.venv\Scripts\python.exe 是 venv 重定向器
            # （venvlauncher），它会再拉起真正的解释器；DETACHED 下这个孙子进程会分配
            # 一个**可见**新控制台（实测 GetConsoleWindow != 0），黑窗反复出现。
            # 改用 CREATE_NO_WINDOW（0x08000000）后实测整条进程链 GetConsoleWindow=0，
            # 完全无窗口；startf SW_HIDE 兜底。
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kw["startupinfo"] = si
            kw["creationflags"] = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        # 子进程输出写进 utf-8 日志文件：必须强制 utf-8，否则 Windows 默认 GBK
        # 编码下打印 ✓/✗ 会 UnicodeEncodeError 直接中断管道（2026-10-27/28/29 各踩一次）
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        proc = subprocess.Popen(cmd, cwd=str(C.BUILD.parent), stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, **kw)
        f.close()   # 子进程已继承句柄，父进程关掉自己的副本

        # 启动探活：脚本秒退（语法错、依赖缺失、参数错）时当场抓住，
        # 否则前端会一直等一个永远不会写 status.json 的死进程。
        import time as _time
        deadline = _time.time() + 2.0
        while _time.time() < deadline and proc.poll() is None:
            _time.sleep(0.1)
        if proc.poll() is not None:
            tail = self._read_log_tail(log, n_before, 40)
            if proc.returncode == 0:
                reason = "更新脚本立即结束（退出码 0），本次没有实际执行的阶段，请检查参数"
            else:
                reason = f"更新脚本启动后立即退出（退出码 {proc.returncode}），未真正开始运行"
            return {"started": False, "reason": reason,
                    "exit_code": proc.returncode, "log_tail": tail}
        return {"started": True, "pid": proc.pid, "cmd": " ".join(cmd)}

    @staticmethod
    def _log_offset(p) -> int:
        try:
            return p.stat().st_size
        except Exception:
            return 0

    @staticmethod
    def _read_log_tail(p, offset: int = 0, n: int = 40) -> list:
        """读日志从 offset 之后的新增内容尾部 n 行（只看本次启动产生的）。"""
        import pathlib as _pl
        try:
            p = _pl.Path(p)
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                text = fh.read()
            lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
            return lines[-n:]
        except Exception as e:
            return [f"[读取日志失败] {e}"]

    def combo_methods(self) -> dict:
        """六种组合配置方法对比（EQ/5050/50-30-20/IVP/ERC/MDP）。

        协方差 252 日滚动、季度再平衡，**无前视**。
        """
        import json as _json

        p = C.BUILD / "eval" / "erc_deep.json"
        if not p.exists():
            raise RuntimeError("ERC 结果不存在，请先运行 scripts/erc_deep.py")
        d = _json.load(open(p, encoding="utf-8"))
        out = {"methods": {}, "corr": d.get("corr") or {},
               "grid": d.get("grid") or {},
               "note": "协方差 252 日滚动 · 季度再平衡 · 无前视"}
        for name, v in (d.get("methods") or {}).items():
            out["methods"][name] = {
                "full": v.get("full"), "val": v.get("val"), "oos": v.get("oos"),
                "weights": v.get("avg_weights"),
                "turnover": v.get("turnover"),
            }
        return out

    def candidates(self) -> dict:
        """候选模型前向跟踪：滚动上线表现 + 最新落盘记录。"""
        import json as _json

        p = C.BUILD / "eval" / "candidate_rolling_oos.json"
        if not p.exists():
            raise RuntimeError("候选跟踪结果不存在，请先运行 scripts/candidate_tracking.py")
        d = _json.load(open(p, encoding="utf-8"))
        return {
            "rolling_start": d.get("rolling_start") or {},
            "starts": d.get("starts") or [],
            "latest": d.get("latest") or {},
            "note": "滚动上线 = 若在该时点上线，到最新日期的年化收益",
        }

    def _stale_diag(self, signal_date: str | None) -> dict:
        """诊断数据/信号的新鲜度，区分「数据过期」与「模型滞后」。

        返回：
          level        ok / warn / bad
          kind         none / data_stale / model_stale
          data_last    行情数据最后交易日
          signal_date  信号日期
          gap_days     两者相差的交易日数
          message      给用户看的一句话
          action       建议动作（update_data / retrain / none）
        """
        import numpy as _np

        out = {"level": "ok", "kind": "none", "data_last": None,
               "signal_date": signal_date, "gap_days": 0,
               "message": "", "action": "none"}
        try:
            md = self.md
            if md is None:
                return out
            data_last = str(md.dates[-1].date())
            out["data_last"] = data_last
        except Exception:
            return out

        # 1) 数据 vs 今天：用交易日数（跳过周末）
        try:
            import datetime as _dt
            d_last = _dt.date.fromisoformat(data_last)
            today = _dt.date.today()
            nd = _np.busday_count(d_last, today) if today > d_last else 0
            out["data_lag"] = int(nd)
        except Exception:
            out["data_lag"] = 0

        # 2) 信号 vs 数据：模型是否追上了数据
        gap = 0
        if signal_date:
            try:
                if signal_date in [str(d.date()) for d in md.dates]:
                    i_sig = int(md.date_index(signal_date))
                    gap = md.T - 1 - i_sig
                else:
                    i_sig = None
            except Exception:
                i_sig = None
            out["gap_days"] = int(gap)

        dl = int(out.get("data_lag") or 0)
        # 判定优先级：模型滞后 > 数据过期（因为数据新但模型旧更隐蔽）
        if gap >= 3:
            out.update({
                "level": "bad" if gap >= 8 else "warn",
                "kind": "model_stale",
                "message": (f"数据已更新到 {data_last}，但模型信号仍基于 {signal_date}"
                            f"（滞后 {gap} 个交易日）。需重训模型。"),
                "action": "retrain",
            })
        elif dl >= 3:
            out.update({
                "level": "bad" if dl >= 7 else "warn",
                "kind": "data_stale",
                "message": f"行情数据已 {dl} 个交易日未更新（最后 {data_last}）。",
                "action": "update_data",
            })
        elif dl >= 2:
            out.update({
                "level": "warn", "kind": "data_stale",
                "message": f"行情数据已 {dl} 个交易日未更新（最后 {data_last}）。",
                "action": "update_data",
            })
        # else: ok
        return out

    def today_recommend(self) -> dict:
        """今日整体推荐：多策略组合的荐股（个股）+ 荐基（宽基 ETF）+ 荐债（可转债）。

        输出按可执行清单组织：每个标的给出代码/名称/类别/建议金额/参考价/份额/理由。
        腿的构成由 `latest_combo.json` 决定（2026-11-05 起：主/低换手/宽基ETF/可转债）。
        """
        import json as _json

        combo_p = C.SIGNALS / "latest_combo.json"
        if not combo_p.exists():
            raise RuntimeError("组合信号不存在，请先运行 daily_signal.py --three")
        combo = _json.load(open(combo_p, encoding="utf-8"))
        st_p = C.BUILD / "eval" / "sector_rotation_stats.json"
        etf_p = C.BUILD / "eval" / "etf_trend_stats.json"

        cap = float(combo.get("capital") or C.CAPITAL)
        rows = []

        # ---- 荐股（主策略侧）
        for t in (combo.get("main") or {}).get("targets", []):
            rows.append({
                "kind": "stock", "kind_cn": "个股",
                "code": t.get("code"), "name": t.get("name"),
                "sector": t.get("industry"), "board": t.get("board"),
                "amount": _f(t.get("target_value")), "price": _f(t.get("price")),
                "shares": int(t.get("shares") or 0), "unit": "股",
                "weight": _f(t.get("weight")),
                "reason": "主策略（左侧低波反转）复合分排名靠前",
                "strategy": "main",
            })

        # ---- 荐股（低换手月度侧；老版本 latest_combo.json 没有这节，取空即可）
        lt_d = combo.get("lowturn") or {}
        for leg in lt_d.get("legs", []) or []:
            _t = leg.get("turn20_bp")
            rows.append({
                "kind": "lowturn", "kind_cn": "低换手个股",
                "code": leg.get("code"), "name": leg.get("name"),
                "sector": leg.get("sector"), "board": "个股",
                "amount": _f(leg.get("target_value")), "price": _f(leg.get("price")),
                "shares": int(leg.get("shares") or 0), "unit": "股",
                "weight": None,
                "reason": (f"20日均换手 {(float(_t) / 100):.2f}%"
                           "（全市场最低一档）→ 换手率最低 10 只等权，月频调仓"
                           if _t is not None else "换手率最低 10 只等权，月频调仓"),
                "strategy": "lowturn",
            })

        # ---- 荐基：宽基 ETF
        _etf_d = combo.get("etf") or {}
        # 2026-11-05：ETF 腿改为最多 2 只、每只 1/2 仓（与回测 MAX_POS=2/equal 一致）；
        # 旧产物只有单数 "leg"，这里兼容两种结构
        for etf_leg in (_etf_d.get("legs") or ([_etf_d["leg"]] if _etf_d.get("leg") else [])):
            rows.append({
                "kind": "etf", "kind_cn": "宽基ETF",
                "code": etf_leg.get("code"), "name": etf_leg.get("name"),
                "sector": "宽基", "board": "ETF",
                "amount": _f(etf_leg.get("target_value")), "price": _f(etf_leg.get("price")),
                "shares": int(etf_leg.get("shares") or 0), "unit": "份",
                "weight": None,
                "reason": "突破 55 日新高（宽基 ETF 趋势，每只占该腿 1/2）",
                "strategy": "etf",
            })

        # ---- 荐基：板块 ETF
        for leg in (combo.get("sector") or {}).get("legs", []) or []:
            rows.append({
                "kind": "sector_etf", "kind_cn": "板块ETF",
                "code": leg.get("code"), "name": leg.get("sector"),
                "sector": leg.get("sector"), "board": "ETF",
                "amount": _f(leg.get("target_value")), "price": _f(leg.get("price")),
                "shares": int(leg.get("shares") or 0), "unit": "份",
                "weight": None,
                "reason": f"板块突破 55 日新高 + 60日动量 {(leg.get('mom60') or 0):.1%}",
                "strategy": "sector",
            })

        # ---- 荐债：可转债（2026-11-05 起替代板块腿）
        _cb_d = combo.get("cb") or {}
        for leg in _cb_d.get("legs", []) or []:
            _y, _yr = leg.get("ytm"), leg.get("years_left")
            rows.append({
                "kind": "cb", "kind_cn": "可转债",
                "code": leg.get("code"), "name": leg.get("name"),
                "sector": "可转债", "board": "转债",
                "amount": _f(leg.get("target_value")), "price": _f(leg.get("price")),
                "shares": int(leg.get("shares") or 0), "unit": "张",
                "weight": None,
                "reason": (f"YTM {(f'{_y:.2%}' if _y is not None else '—')}、"
                           f"剩余 {f'{_yr:.2f}' if _yr is not None else '—'} 年"
                           "（高 YTM 建池 → 短久期优先）"),
                "strategy": "cb",
            })

        # ---- 各组汇总
        groups = {}
        for r in rows:
            g = groups.setdefault(r["strategy"], {"n": 0, "amount": 0.0, "budget_pct": None})
            g["n"] += 1
            g["amount"] += r["amount"] or 0
        # 每个存在的侧都建卡片：无标的时也显示「预算 X%、0 个标的」，
        # 否则用户只看到 31% 现金却不知道是哪一侧空着。
        for key in ("main", "lowturn", "etf", "sector", "cb"):
            if combo.get(key) is None:
                continue
            g = groups.setdefault(key, {"n": 0, "amount": 0.0, "budget_pct": None})
            g["budget_pct"] = (combo.get(key) or {}).get("budget_pct")
            g["budget_amount"] = cap * (g["budget_pct"] or 0)

        # ---- 空仓说明
        idle = []
        _etf_l = (combo.get("etf") or {}).get("legs") or \
                 ([combo["etf"]["leg"]] if (combo.get("etf") or {}).get("leg") else [])
        if not _etf_l:
            idle.append("宽基 ETF：三只均未突破 55 日新高，该部分预算进入现金")
        if (combo.get("sector") is not None) and not ((combo.get("sector") or {}).get("legs")):
            idle.append("板块轮动：24 个板块均无突破，该部分预算进入现金")
        if combo.get("cb") is not None:
            _cb = combo["cb"]
            _adv = _cb.get("advice") or {}
            if _cb.get("error"):
                idle.append(f"可转债：{_cb['error']}，该部分预算进入现金")
            elif not (_cb.get("legs") or []):
                idle.append("可转债：信号日无可选标的，该部分预算进入现金")
            elif _adv:
                idle.append(
                    f"可转债：全市场 YTM 中位 {(_cb.get('ytm_median') or 0):.2%}"
                    f"（判定「{_adv.get('level', '—')}」，建议仓位 {_cb.get('advice_txt', '—')}），"
                    f"本方案固定配 {_cb.get('budget_pct', 0):.0%}；"
                    "该权重对应「转债未来年化 ≈4%」的中性假设，不是历史 13% 的拟合；"
                    "转债 T+0，强赎需人工核对公告")
        _mn = combo.get("main") or {}
        if float(_mn.get("leg_cash_pct") or 0) > 0.05:
            idle.insert(0, (
                f"主策略：regime 系数 {_mn.get('regime_coeff', 1.0):.2f}，该腿只投预算的 "
                f"{1 - float(_mn['leg_cash_pct']):.0%}，其余 "
                f"{float(_mn['leg_cash_pct']):.0%} 转现金——与回测口径一致，不是漏买"))
        if (combo.get("lowturn") is not None) and not ((combo.get("lowturn") or {}).get("legs")):
            idle.append("低换手月度：无可选标的（全被 ST/退市/停牌/上市不满一年过滤）")
        elif combo.get("lowturn"):
            _lt = combo["lowturn"]
            idle.append(
                f"低换手月度：目标月份 {_lt.get('target_month')}，"
                + ("月末已生成下月名单，次日开盘建仓"
                   if _lt.get("status") == "month_end"
                   else "本月名单已生效，等下月首个交易日调仓") + "；只换变动，不清仓重建")

        # 面板比行情滞后时，信号会漏 —— 直接说出来，别让用户以为"今天真的没信号"
        _sig = str(combo.get("signal_date") or "")
        _why = {"etf": "ETF 行情表未日更", "sector": "ETF 行情表未日更",
                "cb": "转债行情表未日更"}
        for _k, _cn in (("etf", "宽基 ETF"), ("sector", "板块轮动"), ("cb", "可转债")):
            _a = str((combo.get(_k) or {}).get("asof") or "")
            if _a and _sig and _a < _sig:
                idle.append(f"{_cn}：信号基于 {_a}，比行情 {_sig} 早 "
                            f"{(pd.Timestamp(_sig) - pd.Timestamp(_a)).days} 天"
                            f"（{_why.get(_k, '数据未日更')}），可能漏掉最新变化")

        # ---- 绩效摘要（供页面标注"这是回测不是承诺"）
        perf = {}
        for f, k in ((etf_p, "etf_trend"), (st_p, "sector_rotation")):
            if f.exists():
                d = _json.load(open(f, encoding="utf-8"))
                if k == "etf_trend":
                    b = d.get("donchian55_tight") or {}
                    perf[k] = {"cagr": b.get("cagr"), "sharpe": b.get("sharpe"),
                               "max_dd": b.get("max_dd")}
                else:
                    perf[k] = {"cagr": (d.get("full") or {}).get("cagr"),
                               "sharpe": (d.get("full") or {}).get("sharpe"),
                               "max_dd": (d.get("full") or {}).get("max_dd"),
                               "oos_cagr": (d.get("oos") or {}).get("cagr")}

        # ---- 策略分解：前端「按策略看推荐」用（四条腿各自的权重/预算/标的/回测）
        perf_by_k = {"etf": perf.get("etf_trend") or {},
                     "sector": perf.get("sector_rotation") or {}}
        try:
            perf_by_k["main"] = _perf_from_nav(
                pd.read_parquet(C.BUILD / "runs" / "main" / "nav.parquet")["nav"])
        except Exception:
            perf_by_k["main"] = {}
        # 相位带（2026-11-03）：runs/main 的 nav 只是 8 个对齐调仓相位里最好的一档，
        # 前端必须同时展示分布，否则单点 CAGR 会被当成策略能力。
        try:
            _pb = _json.load(open(C.BUILD / "eval" / "phase_band.json", encoding="utf-8"))
            perf_by_k["main"]["phase_band"] = {
                "cagr_mean": _pb["cagr"]["mean"], "cagr_median": _pb["cagr"]["median"],
                "cagr_min": _pb["cagr"]["min"], "cagr_max": _pb["cagr"]["max"],
                "sharpe_median": _pb["sharpe"]["median"],
                "n_phases": _pb["n_phases"], "prod_offset": _pb["prod_offset"],
                "n_better": _pb["prod"]["n_better"],
                "rebalance_days": _pb["rebalance_days"],
            }
        except Exception:
            pass
        try:
            _lt = _json.load(open(C.BUILD / "runs" / "lowturn" / "result.json",
                                  encoding="utf-8"))
            perf_by_k["lowturn"] = _lt.get("perf_real") or {}
        except Exception:
            perf_by_k["lowturn"] = {}

        strategy_labels = {
            "main": ("主策略", "荐股", "左侧低波反转 · 40 只等权 · 40 日调仓", "backtest"),
            "lowturn": ("低换手月度", "荐股", "20 日换手最低 10 只等权 · 月频 · 只换变动", "monthly"),
            "etf": ("宽基 ETF", "荐基", "突破 55 日新高 · 最多同时持 2 只", "etf"),
            "sector": ("板块 ETF", "荐基", "24 个板块 · 突破 + 60 日动量前 3", "sector"),
            "cb": ("可转债", "荐债", "高 YTM 建池 · 短久期优先 · 10 只等权", "cb"),
        }
        strategies = []
        for k, (nm, cn, desc, jump) in strategy_labels.items():
            if combo.get(k) is None:
                continue
            g = groups.get(k) or {}
            p0 = perf_by_k.get(k) or {}
            strategies.append({
                "key": k, "name": nm, "kind_cn": cn, "desc": desc, "jump": jump,
                "weight": _f((combo.get("weights") or {}).get(k)),
                "budget": cap * (g.get("budget_pct") or 0),
                "n": g.get("n") or 0,
                "amount": g.get("amount") or 0.0,
                "cagr": _f(p0.get("cagr")), "sharpe": _f(p0.get("sharpe")),
                "max_dd": _f(p0.get("max_dd")),
                # 主策略的 cagr 只是生产相位单点；带上 8 相位分布供前端标注区间
                **(  {"phase_band": p0["phase_band"]} if p0.get("phase_band") else {}),
            })

        _sig = str(combo.get("signal_date") or "")
        return {
            "stale": self._stale_diag(_sig),
            "signal_date": combo.get("signal_date"),
            "capital": cap,
            "parts": combo.get("parts"),
            "weights": combo.get("weights"),
            "rows": rows,
            "groups": groups,
            "strategies": strategies,
            "idle_notes": idle,
            "cash": combo.get("cash"),
            "rebalance_hint": combo.get("rebalance_hint"),
            "perf": perf,
            "lowturn": combo.get("lowturn"),
            "n_stock": sum(1 for r in rows if r["kind"] in ("stock", "lowturn")),
            "n_etf": sum(1 for r in rows if r["kind"] in ("etf", "sector_etf")),
            "total_invested": sum(r["amount"] or 0 for r in rows),
        }

    def today_export_csv(self) -> tuple[str, bytes]:
        """今日推荐清单 → CSV（UTF-8 BOM，Excel 可直接打开）。"""
        import csv
        import io

        d = self.today_recommend()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["信号日期", "类别", "代码", "名称", "板块/行业", "参考价",
                    "数量", "单位", "金额(元)", "理由"])
        for r in d.get("rows") or []:
            w.writerow([
                d.get("signal_date", ""),
                r.get("kind_cn", ""),
                r.get("code", ""),
                r.get("name", ""),
                r.get("sector", ""),
                f"{(r.get('price') or 0):.3f}",
                r.get("shares", 0),
                r.get("unit", ""),
                f"{(r.get('amount') or 0):.0f}",
                r.get("reason", ""),
            ])
        cash = d.get("cash") or {}
        w.writerow([])
        w.writerow(["现金（100%国债）", "", "", "", "", "",
                    "", "", f"{(cash.get('value') or 0):.0f}", ""])
        w.writerow(["合计本金", "", "", "", "", "", "", "",
                    f"{(d.get('capital') or 0):.0f}", ""])
        w.writerow([])
        w.writerow(["说明", "本文件为量化研究输出，不构成投资建议；"
                    "开盘涨停买不进/跌停卖不出可顺延至下一交易日。"])
        body = ("\ufeff" + buf.getvalue()).encode("utf-8")
        name = f"今日推荐_{d.get('signal_date', '')}.csv"
        return name, body

    def monthly_recommend(self) -> dict:
        """本月推荐：低换手月度策略的当月名单、换仓明细与历史。

        与 ``today_recommend`` 的分工：
          - today_recommend 回答「今天四条腿分别要做什么」，各侧合并成一张执行清单；
          - 本接口只讲低换手这一侧的**一个自然月**：名单、换入换出、行业分布、
            换手成本与分年度表现。份额优先取组合口径（按 weights.lowturn 切预算），
            没有组合信号时退回单策略满额口径并在页面上标注。
        """
        import json as _json

        sig_p = C.SIGNALS / "lowturn_signal.json"
        if not sig_p.exists():
            raise RuntimeError("低换手信号不存在，请先运行 scripts/lowturn_signal.py")
        sig = _json.load(open(sig_p, encoding="utf-8"))

        combo = {}
        combo_p = C.SIGNALS / "latest_combo.json"
        if combo_p.exists():
            try:
                combo = _json.load(open(combo_p, encoding="utf-8"))
            except Exception:
                combo = {}
        lt = combo.get("lowturn") or {}
        cap = float(combo.get("capital") or C.CAPITAL)
        wt = _f((combo.get("weights") or {}).get("lowturn"))
        budget = cap * wt if wt else _f(lt.get("allocated")) or _f(sig.get("invested_value")) or 0.0

        # ---- 名单：组合口径已按预算算好股数；缺省退回单策略满额
        legs = lt.get("legs") or []
        sized_by = "combo" if legs else "standalone"
        if not legs:
            legs = [{
                "code": h.get("code"), "name": h.get("name"), "sector": h.get("industry"),
                "price": h.get("price"), "shares": h.get("shares"),
                "target_value": h.get("target_value"), "weight": h.get("weight"),
                "turn20_bp": h.get("turn20_bp"), "mcap_yi": h.get("mcap_yi"),
                "adv20_yi": h.get("adv20_yi"),
            } for h in (sig.get("holdings") or [])]

        target_month = lt.get("target_month") or sig.get("target_month") or ""
        asof = lt.get("asof") or sig.get("asof") or ""
        status = lt.get("status") or sig.get("status") or ""
        status_cn = {
            "month_end": "月末已算出下月名单，等次月首个交易日开盘建仓",
            "in_month": "本月名单生效中，下月首个交易日再调仓",
        }.get(status, "按月末信号执行")

        # ---- 上期名单：picks.json 只记已完成月份，最新一期要拿它做「新进/移出」
        prev = None
        try:
            picks = _json.load(open(C.BUILD / "runs" / "lowturn" / "picks.json",
                                   encoding="utf-8"))
            if picks:
                idx = next((i for i, e in enumerate(picks)
                            if str(e.get("month")) == str(target_month)), None)
                prev = picks[idx - 1] if idx else picks[-1]
        except Exception:
            prev = None
        prev_codes = [str(c) for c in ((prev or {}).get("codes") or [])]
        name_of = {}
        for c, n in zip((prev or {}).get("codes") or [], (prev or {}).get("names") or []):
            name_of[str(c)] = n
        for h in (sig.get("holdings") or []):
            name_of[str(h.get("code"))] = h.get("name")
        for l in legs:
            name_of.setdefault(str(l.get("code")), l.get("name"))

        cur_codes = [str(l.get("code")) for l in legs]
        entered = [c for c in cur_codes if c not in prev_codes]
        exited = [c for c in prev_codes if c not in cur_codes]
        kept = [c for c in cur_codes if c in prev_codes]

        # 组合口径的 legs 不带 adv20/市值，用单策略信号里的同名持仓补齐
        extra = {str(h.get("code")): h for h in (sig.get("holdings") or [])}

        rows = []
        for l in legs:
            t = _f(l.get("turn20_bp"))
            _c = str(l.get("code") or "")
            _x = extra.get(_c) or {}
            rows.append({
                "code": _c,
                "name": l.get("name") or name_of.get(_c) or "",
                "sector": l.get("sector") or _x.get("industry") or "—",
                "price": _f(l.get("price")),
                "shares": int(l.get("shares") or 0),
                "amount": _f(l.get("target_value")),
                "weight": _f(l.get("weight")),
                # turn20_bp 是基点数：15.38bp = 0.1538% = 0.001538（前端 fmtPctU 再乘 100）
                "turn20_pct": (t / 10000.0) if t is not None else None,
                "mcap_yi": _f(l.get("mcap_yi"), _f(_x.get("mcap_yi"))),
                "adv20_yi": _f(l.get("adv20_yi"), _f(_x.get("adv20_yi"))),
                "is_new": _c in entered,
            })
        total = sum(r["amount"] or 0 for r in rows)

        # ---- 行业分布（这策略本质是红利/低波组合，行业集中度必须让人看见）
        dist = {}
        for r in rows:
            d0 = dist.setdefault(r["sector"], {"sector": r["sector"], "n": 0, "amount": 0.0})
            d0["n"] += 1
            d0["amount"] += r["amount"] or 0
        sector_dist = sorted(dist.values(), key=lambda x: -x["amount"])
        for x in sector_dist:
            x["pct"] = (x["amount"] / total) if total else 0.0

        # ---- 回测绩效（净口径 = 只换变动；gross = 零成本；strict = 每月全清全建）
        run = {}
        run_p = C.BUILD / "runs" / "lowturn" / "result.json"
        if run_p.exists():
            try:
                run = _json.load(open(run_p, encoding="utf-8"))
            except Exception:
                run = {}
        pr, pg = (run.get("perf_real") or {}, run.get("perf_gross") or {})
        # strict（每月全清全建）必须读 perf_strict；旧产物没有该字段时退回 perf_net
        pn = run.get("perf_strict") or run.get("perf_net") or {}
        bt = sig.get("backtest") or {}
        perf = {
            "cagr": _f(pr.get("cagr"), _f(bt.get("cagr_net"))),
            "sharpe": _f(pr.get("sharpe"), _f(bt.get("sharpe"))),
            "max_dd": _f(pr.get("max_dd"), _f(bt.get("max_dd"))),
            "vol": _f(pr.get("ann_vol")),
            "gross_cagr": _f(pg.get("cagr"), _f(bt.get("cagr_gross"))),
            "strict_cagr": _f(pn.get("cagr"), _f(bt.get("cagr_strict"))),
            "win_rate": _f(run.get("monthly_win_rate")),
            "n_months": run.get("n_months") or bt.get("n_months"),
            "avg_overlap": _f(run.get("avg_overlap"), _f(bt.get("keep_ratio"))),
            "avg_mcap_yi": _f(run.get("avg_mcap_yi")),
            "avg_turn20_pct": (_f(run.get("avg_turn20_bp")) or 0) / 10000.0
                              if run.get("avg_turn20_bp") is not None else None,
            "turnover_ann": _f(run.get("turnover_x_year_actual"), _f(bt.get("turnover_ann"))),
            "turnover_full_switch": _f(run.get("turnover_x_year_full_switch")),
            "corr_main": _f((run.get("main_strategy") or {}).get("corr_with_lowturn")),
            "cost_bp": _f(self.cost_target_bp("lowturn")),
        }
        if perf["cagr"] is not None and perf["gross_cagr"] is not None:
            perf["cost_drag_pp"] = (perf["gross_cagr"] - perf["cagr"]) * 100.0

        yearly = run.get("yearly_net") or {}
        bench = []
        for k, v in (run.get("bench") or {}).items():
            bench.append({"key": k, "name": {"csi_all": "中证全指", "all_eq": "全A等权"}.get(k, k),
                          "cagr": _f(v.get("cagr")), "sharpe": _f(v.get("sharpe")),
                          "max_dd": _f(v.get("max_dd"))})

        # ---- 近 12 个月收益 + 净值曲线（散点用，最近 500 个交易日归一）
        hist, hist_nav = [], []
        nav_p = C.BUILD / "runs" / "lowturn" / "nav.parquet"
        if nav_p.exists():
            try:
                nv = pd.read_parquet(nav_p)
                col = "nav_real" if "nav_real" in nv.columns else nv.columns[-1]
                s = nv[col].astype(float).dropna()
                # 注意：不能 groupby(pd.Series(index))——索引不对齐会得到 0 组
                me = s.groupby(s.index.to_period("M")).last()
                ret = me.pct_change().dropna()
                hist = [{"month": str(p), "ret": float(v)} for p, v in ret.tail(12).items()]
                tail = s.tail(500)
                if len(tail) > 2 and float(tail.iloc[0]) > 0:
                    base = float(tail.iloc[0])
                    hist_nav = [{"date": str(i.date()), "nav": float(v) / base}
                                for i, v in tail.items()]
            except Exception:
                hist, hist_nav = [], []

        # ---- 补一条红利基准（2026-11-02）：低换手持仓常年是四大行+中石油+人寿，
        # 本质是红利组合。只给中证全指会虚增超额（全指不含股息、也不做红利筛选）。
        # 用 515080 中证红利ETF 的后复权价（含分红再投资），与策略同区间。
        try:
            dp = C.BUILD / "etf" / "div_panel.parquet"
            if dp.exists() and "s" in locals() and len(s) > 60:
                dd = pd.read_parquet(dp)
                dv = dd[dd["code"] == "515080"].sort_values("date")
                if not dv.empty:
                    sv = dv.set_index("date")["close_adj"].astype(float)
                    sv = sv[~sv.index.duplicated()]
                    sv = sv[(sv.index >= s.index[0]) & (sv.index <= s.index[-1])]
                    if len(sv) > 60 and float(sv.iloc[0]) > 0:
                        q = qfmetrics.performance(sv / float(sv.iloc[0]), rf=C.CASH_YIELD)
                        bench.append({"key": "div_etf", "name": "中证红利ETF",
                                      "cagr": _f(q.get("cagr")),
                                      "sharpe": _f(q.get("sharpe")),
                                      "max_dd": _f(q.get("max_dd")),
                                      "note": "对照基准（含股息再投资）"})
        except Exception:
            pass

        return {
            "strategy": "lowturn_monthly",
            "label": "低换手月度",
            "asof": asof,
            "target_month": target_month,
            "status": status,
            "status_cn": status_cn,
            "next_action": sig.get("next_action") or "",
            "rule": (lt.get("rule") or sig.get("rule")
                     or "月末算过去 20 日平均换手率 → 剔 ST/退市/停牌/上市不满一年 "
                        "→ 取换手率最低 10 只等权，月频调仓，只换变动"),
            "capital": cap,
            "weight": wt,
            "budget": budget,
            "sized_by": sized_by,
            "invested": total,
            "invested_pct": (total / budget) if budget else None,
            "cash": max(0.0, budget - total),
            "n_pick": len(rows),
            "rows": rows,
            "rotation": {
                "prev_month": (prev or {}).get("month"),
                "entered": [{"code": c, "name": name_of.get(c, "")} for c in entered],
                "exited": [{"code": c, "name": name_of.get(c, "")} for c in exited],
                "kept": len(kept),
            },
            "sector_dist": sector_dist,
            "perf": perf,
            "yearly": yearly,
            "bench": bench,
            "monthly_returns": hist,
            "nav_series": hist_nav,
            "stale": self._stale_diag(asof or None),
            "params": sig.get("params") or {},
        }

    def combo_plan(self) -> dict:
        """组合方案：四腿（主策略 + ETF 趋势 + 512890 红利低波 + 可转债）为生产口径。

        2026-11-04 接第四条腿（`reports/第四条腿可转债_2026-11-04.md`）：
        转债腿 ρ=0.06~0.07、净 SR 1.30，接入后 Sharpe 0.863→1.018、波动 8.46%→7.44%
        （主策略腿已按缩小后的真实资金重跑，不是线性拼接）。
        但 DSR@71 = 63.4% 未过 95% → 按预登记规则「DSR 不过而其余三条全过 → 最高按
        15% 降权试行」，生产口径取转债 15%，剩余 85% 按原三腿 50/25/25 等比压缩。

        combo4_stats.json 缺失时退回 combo3（三腿）口径，旧的 three_strategy_combo.json
        （含板块腿）始终作对照，板块腿已否两次。
        """
        import json as _json

        out = {"recommended": None, "recommended_label": None, "legs": [],
               "legs_corr": None, "schemes": [], "controls": [], "notes": [],
               "weights": {}, "marginal": None, "erc": None, "corr": None,
               "basis": None, "prod_weights": None}

        p4 = C.BUILD / "eval" / "combo4_stats.json"
        p3 = C.BUILD / "eval" / "combo3_stats.json"

        if p4.exists():
            c4 = _json.load(open(p4, encoding="utf-8"))
            lc = c4.get("legs_corr") or {}
            legs = list(lc) or ["主策略", "ETF趋势", "红利低波512890", "可转债"]
            res = c4.get("results") or {}
            prod_w = float(c4.get("prod_w_cb") or 0.15)
            prod_key = f"cb{int(round(prod_w * 100)):02d}"
            note4 = {
                "cb00": ("control", "接入前基线（原三腿 50/25/25）"),
                "cb10": ("control", "同方向低配档"),
                prod_key: ("recommended",
                           "生产口径：DSR 未过 95%，按预登记规则降到 15% 试行，"
                           "剩余 85% 按原三腿 50/25/25 等比压缩"),
                "cb20": ("control", "3 个月后目标档（实盘跟踪偏差 <1.5pp 再提）"),
                "cb25": ("control", "上限档，回撤已开始反向"),
            }
            for k in ("cb00", "cb10", "cb15", "cb20", "cb25"):
                v = res.get(k)
                if not v:
                    continue
                ws = [float(x) for x in (v.get("weights") or [])]
                st, note = note4.get(k, ("control", ""))
                if k == prod_key:
                    st = "recommended"
                out["schemes"].append({
                    "name": v.get("label") or k, "key": k, "status": st, "note": note,
                    "legs": {legs[i]: ws[i] for i in range(min(len(ws), len(legs)))},
                    "full": v.get("full"), "val": v.get("val"), "oos": v.get("oos"),
                })
                out["weights"][k] = {"full": v.get("full"), "val": v.get("val"),
                                     "oos": v.get("oos"), "weights": ws}
            rec = next((s for s in out["schemes"] if s["status"] == "recommended"), None)
            if rec:
                lab = " + ".join(f"{k} {_wfmt(v)}" for k, v in rec["legs"].items())
                out["recommended"] = rec["name"]
                out["recommended_label"] = f"{rec['name']}（{lab}）"
                out["prod_weights"] = rec["legs"]
            out["legs"] = [{"name": n, "corr": lc.get(n)} for n in legs]
            out["legs_corr"] = lc
            out["basis"] = "combo4"
            out["notes"] = [
                "接入转债不是加杠杆：波动 8.46%→7.44%，因为转债与三腿的日收益相关只有 "
                "0.03~0.07（几乎各走各的）。",
                "DSR 未过：扣掉 71 次选腿折扣后只剩 63.4%，所以只给 15% 试行，不算"
                "「采纳」。转债腿本身单算 DSR@13 = 96.7%，过。",
                "最大风险：当前转债极贵（YTM 中位 −3.04%、价格中位 125.85 元、"
                "100 元以下仅 1.0%），历史同类环境后一年收益中位仅 6.46%，"
                "回测的 9.56% 在坏环境下大概率达不到。",
                "实盘下单已切四腿 40/30/20/10（主策略/低换手/宽基ETF/板块，"
                "signals/latest_combo.json mode=four）；本面板的转债 15% 属于另一组腿，"
                "与本信号不是同一套口径，切换需单独确认。",
                "区间 2020-01-08 ~ 2026-10-29（转债数据末端），比三腿少 3 个交易日。",
                "口径：主策略腿按「分到的钱」重跑（15% 档只有 12.75 万，单只约 1,900 元），"
                "不是拿全资金净值乘权重。线性拼接会把规模损失漏掉约 0.36pp。",
            ]
        elif p3.exists():
            c3 = _json.load(open(p3, encoding="utf-8"))
            legs_corr = c3.get("legs_corr") or {}
            legs = list(legs_corr) or ["主策略", "ETF趋势", "红利低波512890"]
            res = c3.get("results") or {}
            meta = {
                "50/25/25": ("recommended",
                             "主方案：主权重固定 50%，两个 ETF 腿均分剩余，权重事先声明不事后改"),
                "40/30/30": ("control", "数字更好但不采用：方案是事先声明的"),
                "50/30/20": ("control", "三腿内变体（主/ETF/红利）"),
                "50/50(现状)": ("live", "当前实盘权重（主策略 + ETF 趋势，无红利腿）"),
            }
            for k in ("50/25/25", "40/30/30", "50/30/20", "50/50(现状)"):
                v = res.get(k)
                if not v:
                    continue
                ws = [float(x) for x in (v.get("weights") or [])]
                st, note = meta.get(k, ("control", ""))
                out["schemes"].append({
                    "name": k, "status": st, "note": note,
                    "legs": {legs[i]: ws[i] for i in range(min(len(ws), len(legs)))},
                    "full": v.get("full"), "val": v.get("val"), "oos": v.get("oos"),
                })
                out["weights"][k] = {"full": v.get("full"), "val": v.get("val"),
                                     "oos": v.get("oos"), "weights": ws}
            rec = next((s for s in out["schemes"] if s["status"] == "recommended"), None)
            if rec:
                lab = " + ".join(f"{k} {v:.0%}" for k, v in rec["legs"].items())
                out["recommended"] = rec["name"]
                out["recommended_label"] = f"{rec['name']}（{lab}）"
            out["legs"] = [{"name": n, "corr": legs_corr.get(n)} for n in legs]
            out["legs_corr"] = legs_corr
            out["basis"] = "combo3"
            out["notes"] = [
                "三腿未过全部闸门：CAGR 与样本外改善，但 Sharpe 0.844 < 0.871、"
                "MaxDD −7.99% 深于 −6.98%，本质是「加收益加一点风险」，不是免费午餐。",
                "相关性：主-ETF 0.21 / 主-红利 0.49 / ETF-红利 0.23；主策略持仓 bp 分位 0.744 "
                "与红利低波同向，是组合第一处真风格叠加。",
                "实盘信号已切四腿 40/30/20/10（signals/latest_combo.json mode=four）；"
                "本面板三腿口径保留作对照。",
                "combo4_stats.json 未生成，回退三腿口径（跑 scripts/combo4.py 可切四腿）。",
            ]

        p = C.BUILD / "eval" / "three_strategy_combo.json"
        if p.exists():
            d = _json.load(open(p, encoding="utf-8"))
            if not out["schemes"]:
                for k, v in (d.get("weights") or {}).items():
                    bad = "板块" in k
                    out["controls"].append({
                        "name": k,
                        "status": "rejected" if bad else "control",
                        "note": ("板块腿已否（2026-11-03）：换 mom250 后 OOS 3.02% < 基线 3.58%；"
                                 "纯动量 + 仓位管理风险改善真、收益改善假，最多当 ≤10~15% 卫星"
                                 if bad else "旧口径对照（主 + 宽基）"),
                        "full": v.get("full"), "val": v.get("val"), "oos": v.get("oos"),
                    })
            out["marginal"] = d.get("marginal")
            out["erc"] = d.get("erc")
            out["corr"] = d.get("corr")
            if not out["recommended"]:
                out["recommended"] = "50/50 主+宽基（现行）"
                out["recommended_label"] = "50/50 主+宽基（现行）"
                out["notes"].append("组合统计缺失，回退旧口径推荐。")

        if not out["schemes"] and not out["controls"]:
            raise RuntimeError("组合测试未运行，请先执行 scripts/combo4.py 或 scripts/etf_combo_3leg.py")
        return out
    def cb_signal(self) -> dict:
        """可转债高 YTM 策略：最新信号、市场环境、净值、多资产组合对比。"""
        import json as _json

        sig_p = C.SIGNALS / "latest_cb_signal.json"
        if not sig_p.exists():
            raise RuntimeError("转债信号未生成，请先执行 scripts/cb_signal.py")
        sig = _json.load(open(sig_p, encoding="utf-8"))

        out = {
            "asof": sig.get("signal_date"),
            "capital": sig.get("capital"),
            "cb_weight": sig.get("cb_weight"),
            "invested": sig.get("invested"),
            "invested_pct": sig.get("invested_pct"),
            "rank": sig.get("rank"), "pool_mult": sig.get("pool_mult"),
            "environment": sig.get("environment") or {},
            "advice": sig.get("advice") or {},
            "holdings": sig.get("holdings") or [],
            "notes": sig.get("notes") or [],
        }

        # ---- 策略净值 + 环境曲线
        nav_p = C.BUILD / "runs" / "cb" / "nav_prod.parquet"
        if not nav_p.exists():
            nav_p = C.BUILD / "runs" / "cb" / "nav_high_ytm.parquet"
        if nav_p.exists():
            nav = pd.read_parquet(nav_p).iloc[:, 0]
            out["nav"] = [{"date": str(d.date()), "nav": _f(v)} for d, v in nav.tail(600).items()]
        yh_p = C.BUILD / "cb" / "ytm_history.parquet"
        if yh_p.exists():
            yh = pd.read_parquet(yh_p).sort_values("date").tail(600)
            out["ytm_env"] = [{"date": str(pd.Timestamp(r.date).date()),
                               "ytm": _f(r.ytm_median), "price": _f(r.price_median),
                               "n": int(r.n_live)} for r in yh.itertuples()]

        # ---- 回测指标（本地重算，避免依赖实验脚本输出）
        if nav_p.exists():
            try:
                nav_s = pd.read_parquet(nav_p).iloc[:, 0]
                out["backtest"] = self._bt_periods(nav_s, "2019-01-01", "2026-12-31")
                out["backtest"]["yearly"] = [
                    {"year": int(d.year), "ret": _f(v)}
                    for d, v in nav_s.resample("YE").last().pct_change().dropna().items()]
            except Exception:
                pass

        # ---- 多资产组合对比（build/exp_combo3.py 产出）
        cj = C.BUILD / "runs" / "combo3" / "summary.json"
        if cj.exists():
            try:
                c = _json.load(open(cj, encoding="utf-8"))
                out["single"] = c.get("single")
                out["corr"] = c.get("corr")
                out["period"] = c.get("period")
                out["schemes"] = [
                    {"name": k, "weights": v.get("weights"), "stats": v.get("stats")}
                    for k, v in (c.get("free_schemes") or {}).items()]
            except Exception:
                pass
        return out

    def sector_rotation(self) -> dict:
        """板块 ETF 轮动策略：回测指标、最新信号、板块池状态。"""
        import json as _json

        from qf import sector_rotation as qfsr

        stats_p = C.BUILD / "eval" / "sector_rotation_stats.json"
        if not stats_p.exists():
            raise RuntimeError("板块轮动尚未运行，请先执行 python qf/sector_rotation.py")
        st = _json.load(open(stats_p, encoding="utf-8"))

        # 板块池与可交易状态
        dates, codes, close, amount, sector_of = qfsr.load_pool()
        t = len(dates) - 1
        amt20 = pd.DataFrame(amount).rolling(20, min_periods=10).mean().to_numpy()
        cs = pd.DataFrame(close)
        hi = cs.rolling(qfsr.ENTRY_N).max().shift(1).to_numpy()
        mom = (cs / cs.shift(qfsr.MOM_N) - 1.0).to_numpy()
        univ = []
        for j, c in enumerate(codes):
            if not np.isfinite(close[t, j]):
                continue
            tradable = bool(np.isfinite(amt20[t, j]) and amt20[t, j] >= qfsr.MIN_AMOUNT)
            brk = bool(np.isfinite(hi[t, j]) and close[t, j] > hi[t, j])
            univ.append({
                "code": c, "sector": sector_of.get(c, c),
                "price": _f(close[t, j]),
                "mom60": _f(mom[t, j]),
                "breakout": brk, "tradable": tradable,
                "dist_to_high": _f(close[t, j] / hi[t, j] - 1.0) if np.isfinite(hi[t, j]) else None,
                "amount_20d": _f(amt20[t, j]),
            })
        univ.sort(key=lambda x: -(x["mom60"] if x["mom60"] is not None else -9))
        picks = set((st.get("latest_signal") or {}).get("picks") and
                    [p["code"] for p in st["latest_signal"]["picks"]] or [])
        for u in univ:
            u["in_target"] = u["code"] in picks

        nav_p = C.BUILD / "runs" / "sector_rotation" / "nav.parquet"
        nav_series = []
        if nav_p.exists():
            nav = pd.read_parquet(nav_p)["nav"]
            sub = nav.tail(500)
            base = sub.iloc[0]
            nav_series = [{"date": str(d.date()), "nav": _f(v / base)} for d, v in sub.items()]

        return {
            "asof": st.get("latest_signal", {}).get("asof") or str(dates[-1].date()),
            "backtest": st.get("full"),
            "val": st.get("val"), "oos": st.get("oos"),
            "yearly": st.get("yearly"), "params": st.get("params"),
            "avg_exposure": st.get("avg_exposure"), "n_trades": st.get("n_trades"),
            "notes": st.get("note"),
            "latest_signal": st.get("latest_signal"),
            "universe": univ, "nav_series": nav_series,
            "n_sectors": len(set(sector_of.values())),
        }

    def factors_summary(self) -> dict:
        cat = self.catalog
        n_quant = int(len(cat)) if cat is not None else 800
        n_mom = int(len(self.mom_cat)) if self.mom_cat is not None else 0
        n_fund = int(len(self.fund_cat)) if self.fund_cat is not None else 0
        n_trend = int(len(self.trend_cat)) if self.trend_cat is not None else 0
        m = self.metrics
        stage2 = int(len(m)) if m is not None else 0
        n_pool = int(len(self.pool_meta)) if self.pool_meta is not None else 0
        families = []
        try:
            fam = self.family
            if fam is not None and len(fam):
                if "family_type" in fam.columns and "family" in fam.columns:
                    show = fam[fam["family_type"] == "category_en"] if (fam["family_type"] == "category_en").any() else fam
                    icol = "dev_abs_icir_mean" if "dev_abs_icir_mean" in show.columns else "abs_icir_mean"
                    show = show.sort_values(icol, ascending=False)
                    for _, r in show.head(20).iterrows():
                        families.append({"family": str(r["family"]), "n": int(r.get("n_candidates", 0)),
                                         "dev_abs_icir": _f(r.get(icol)),
                                         "coverage": _f(r.get("coverage_mean")),
                                         "turnover": _f(r.get("turnover_mean"))})
            elif m is not None:
                g = m.groupby("category_en").agg(n=("factor_key", "size"),
                                                 dev=("dev_icir_5d", lambda x: float(np.nanmean(np.abs(x)))),
                                                 cov=("coverage", "mean"))
                for k, r in g.iterrows():
                    families.append({"family": str(k), "n": int(r["n"]), "dev_abs_icir": _f(r["dev"]),
                                     "coverage": _f(r["cov"]), "turnover": None})
        except Exception:
            families = []
        pool_detail = []
        try:
            pm = self.pool_meta.copy()
            if "factor_key" not in pm.columns:
                pm = pm.reset_index().rename(columns={"index": "factor_key"})
            names = {}
            for c in (cat, self.mom_cat, self.fund_cat, self.trend_cat):
                if c is not None and len(c) and "factor_key" in c.columns:
                    for _, r in c.iterrows():
                        names[str(r["factor_key"])] = str(r.get("name_cn") or "")
            latest_w = {}
            if self.weights is not None and len(self.weights):
                last = self.weights["refit_date"].max()
                for _, r in self.weights[self.weights["refit_date"] == last].iterrows():
                    latest_w[str(r["factor_key"])] = _f(r["weight"])
            for _, r in pm.iterrows():
                key = str(r.get("factor_key", ""))
                dev_icir = _f(r.get("icir", r.get("dev_icir", np.nan)))
                # 族名：优先 category_en；pool_meta 实际只有 category(人类可读) 与 lib，
                # 所以回退到 factor_key 前缀（risk-pattern / newf-extra / momentum-extra …），
                # 与实验报告里的族口径一致。
                fam_key = str(r.get("category_en") or "").strip() or key.split(":")[0]
                pool_detail.append({"factor_key": key, "name_cn": names.get(key, ""),
                                    "category_en": str(r.get("category_en", "")),
                                    "family": fam_key,
                                    "ic": _f(r.get("ic")), "icir": dev_icir,
                                    "weight": latest_w.get(key, 0.0)})
            pool_detail.sort(key=lambda x: abs(x["weight"] or 0), reverse=True)
        except Exception:
            pool_detail = []
        fund_stats = []
        try:
            fcat = self.fund_cat
            fname = {}
            if fcat is not None and len(fcat):
                fname = {str(r["factor_key"]): str(r.get("name_cn") or r.get("slug") or "")
                         for _, r in fcat.iterrows()}
            for k, s in self.fund_stats.items():
                fund_stats.append({"factor_key": k, "name_cn": fname.get(k, k),
                                   "dev_ic": _f(s.get("dev_ic_5d")), "dev_icir": _f(s.get("dev_icir_5d")),
                                   "coverage": _f(s.get("coverage"))})
        except Exception:
            fund_stats = []
        comp = {}
        try:
            if self.comp_ic is not None and len(self.comp_ic):
                ic = self.comp_ic.copy()
                ic["date"] = pd.to_datetime(ic["date"])
                vals = ic["ic"].to_numpy()
                comp = {"mean": _f(np.nanmean(vals)),
                        "icir": _f(np.nanmean(vals) / max(np.nanstd(vals), 1e-9)),
                        "n_weeks": int(len(ic)),
                        "by_year": [{"year": int(y), "ic": _f(g["ic"].mean()), "n": int(len(g))}
                                    for y, g in ic.groupby(ic["date"].dt.year)]}
        except Exception:
            comp = {}
        stats = {}
        try:
            if m is not None:
                stats = {"stage2_ic_median": _f(m["ic_5d"].median()),
                         "stage2_icir_median": _f(m["icir_5d"].median()),
                         "stage2_icir_p90": _f(m["icir_5d"].quantile(0.9))}
        except Exception:
            stats = {}
        # 族占比监控：按 |权重| 汇总到 category_en，超阈值标黄（只展示，不干预）
        fam_w = []
        try:
            agg: dict[str, float] = {}
            cnt: dict[str, int] = {}
            for r in pool_detail:
                k = r.get("family") or r.get("category_en") or ""
                if not k:
                    continue
                agg[k] = agg.get(k, 0.0) + abs(float(r.get("weight") or 0.0))
                cnt[k] = cnt.get(k, 0) + 1
            tot = float(sum(agg.values()))
            if tot > 0:
                share = {k: v / tot for k, v in agg.items()}
                for r in pool_detail:
                    r["family_share"] = _f(share.get(r.get("family") or r.get("category_en") or ""))
                fam_w = sorted(
                    ({"family": k, "weight_abs": _f(v), "share": _f(v / tot),
                      "n": int(cnt.get(k, 0)), "over": bool(v / tot > FAMILY_SHARE_WARN)}
                     for k, v in agg.items()),
                    key=lambda x: x["share"] or 0.0, reverse=True)
        except Exception:
            fam_w = []
        return {"counts": {"total": n_quant + n_mom + n_fund + n_trend, "quant": n_quant,
                           "momentum": n_mom, "fundamental": n_fund, "trend": n_trend,
                           "models": [m["name"] for m in self.registry]},
                "stage2": stage2, "pool": n_pool, "families": families,
                "pool_detail": pool_detail, "fundamental_stats": fund_stats,
                "composite_ic": comp, "statistics": stats,
                "family_weights": fam_w, "family_share_warn": FAMILY_SHARE_WARN}

    def reports(self) -> dict:
        files = []
        for p in sorted(C.REPORTS.glob("*"), key=lambda x: -x.stat().st_mtime if x.exists() else 0):
            if not p.is_file():
                continue
            try:
                from datetime import datetime
                files.append({"name": p.name, "path": "/reports/" + p.name,
                              "size": int(p.stat().st_size),
                              "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                              "kind": p.suffix.lstrip(".")})
            except Exception:
                pass
        return {"files": files}

    # ------------------------------------------------------------ 图表
    def _cached_chart(self, key, fn):
        with self.lock:
            if key in self._chart_cache:
                return self._chart_cache[key]
        data = fn()
        with self.lock:
            if len(self._chart_cache) > 80:
                self._chart_cache.clear()
            self._chart_cache[key] = data
        return data

    def chart_index_line(self, code: str = "csi_all", days: int = 250) -> bytes:
        self._require_loaded()
        code = code if code in IDX_NAMES else "csi_all"
        days = int(np.clip(days, 60, 1200))
        key = ("index_line", code, days, str(self.md.dates[-1].date()))

        def _gen():
            s = self.derived["bench"][code]
            return charts.index_line(s.index, s.to_numpy(), IDX_NAMES.get(code, code), days)
        return self._cached_chart(key, _gen)

    def chart_stock_kline(self, code: str, days: int = 250) -> bytes:
        self._require_loaded()
        md = self.md
        j = md.code_index(str(code))
        if j < 0:
            raise KeyError(f"找不到股票 {code}")
        days = int(np.clip(days, 60, 800))
        key = ("kline", str(md.codes[j]), days, str(md.dates[-1].date()))

        def _gen():
            return charts.stock_kline(md.dates, md.price_open[:, j], md.price_high[:, j],
                                      md.price_low[:, j], md.price[:, j], md.volume[:, j],
                                      str(md.names[j]), str(md.codes[j]), days)
        return self._cached_chart(key, _gen)

    def chart_industry_bar(self, days: int = 20) -> bytes:
        self._require_loaded()
        key = ("industry_bar", days, str(self.md.dates[-1].date()))
        data = self._cached_chart(key, lambda: charts.industry_bar(
            self.market_overview()["industries"], days=days))
        return data

    def chart_backtest_nav(self, jid: str) -> bytes:
        key = ("backtest_nav", jid)
        with self.lock:
            if key in self._chart_cache:
                return self._chart_cache[key]
        job = self.jobs.get(jid)
        if not job:
            raise KeyError("任务不存在")
        if job["status"] != "done" or not job.get("result"):
            raise RuntimeError("回测尚未完成")
        navd = job["result"].get("nav") or {}
        dates = pd.DatetimeIndex([pd.Timestamp(x) for x in navd.get("dates", [])])
        if len(dates) == 0:
            raise RuntimeError("回测图表数据为空")
        png = charts.nav_chart(dates, np.asarray(navd["nav"], dtype=float),
                               np.asarray(navd["dd"], dtype=float),
                               title="回测净值与回撤")
        with self.lock:
            self._chart_cache[key] = png
        return png


def create_service() -> DataService:
    return DataService()
