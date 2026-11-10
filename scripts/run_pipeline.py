# -*- coding: utf-8 -*-
"""一键更新管道：数据 → 因子 → 模型 → 回测 → 组合 → 信号。

用法::

    python scripts/run_pipeline.py                    # 全流程
    python scripts/run_pipeline.py --skip-data        # 跳过数据增量
    python scripts/run_pipeline.py --skip-data --skip-factors   # 只重跑下游（最快）
    python scripts/run_pipeline.py --only signal      # 只跑某个阶段
    python scripts/run_pipeline.py --from backtest    # 从某阶段开始
    python scripts/run_pipeline.py --force            # 已有实例在跑时强行再开一个（通常不要）

进度写到 build/pipeline/status.json，网页「数据」页可轮询显示。

两条硬约定（都是踩坑换来的）：

* **status.json 永远不能影响结果**。网页每 2~3 秒轮询这个文件，而 Windows 上
  Python 的 open() 不带 FILE_SHARE_DELETE，`os.replace` 会偶发 WinError 5。
  写入走「退避重试 → 原地写兜底 → 只警告不抛」，否则一次成功的更新会被判成失败
  （2026-11-07 事故，见下方 STATUS_WRITE_ATTEMPTS 注释）。
* **不允许重入**。已有存活实例时默认直接退出（退出码 0，不算失败），
  避免"以为失败了"→再点一次→factors 白跑 850 秒。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATUS = ROOT / "build" / "pipeline" / "status.json"
LOGDIR = ROOT / "build" / "pipeline" / "logs"
PY = sys.executable

# Windows 控制台默认 GBK，打印 ✓/✗/✅ 会抛 UnicodeEncodeError 并让整条管道中断
# （2026-10-27、10-28、10-29 连续三次在第一个阶段后崩掉，status.json 卡在 running=true）。
# 只放宽编码错误容忍度：不强制改编码，避免手动在 cmd 里跑时中文变乱码。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


# (key, 中文名, 命令, 预计秒数)
# 各阶段超时上限（秒）。超时判定失败，避免"看起来在跑实际卡死"
STAGE_TIMEOUT = {
    "data": 1800,        # 30 分钟
    "cb_data": 600,
    "cb_signal": 180,
    "factors": 2400,     # 40 分钟
    # extra：重建 --extra 引用的额外因子族面板。三个面板各 ~40~90s。
    # 这一步缺了会静默毁掉模型最新一周的打分（见 scripts/build_extra_panels.py 注释）
    "extra": 900,
    "model": 900,
    "backtest": 300,
    "phase_band": 900,
    "etf": 300,
    "sector": 300,
    "combo": 180,
    "three": 180,
    "erc": 180,
    "tracking": 300,
    "track_record": 120,
    "numbers": 120,
    "report_s1": 120,
    "signal": 300,
    "verify": 120,
}

# 第 5 个元素为可选项：{"optional": True} 表示失败不阻断整条管道
# （转债数据依赖外部网络，抓不到时不应让主策略信号也跑不出来）
STAGES = [
    ("data", "数据增量更新", [PY, "-u", "scripts/update_data.py"], 150),
    ("cb_data", "转债数据增量", [PY, "-u", "build/fetch_cb.py", "--incremental"], 25,
     {"optional": True}),
    ("factors", "因子面板重建", [PY, "-u", "scripts/build_factors.py"], 950),
    # 额外因子族：build_factors 只重建主面板，--extra 引用的面板是独立生成的。
    # 不在这里重建的话，它们每次都比 weekly_meta 少一周 → 模型最后一行打分全 NaN
    # → composite_score 停在上一周 → verify 报「模型滞后」。2026-11-04 22:56 的失败即此。
    # 只重建 "周数已落后" 的面板（对齐就跳过），正常情况下耗时接近 0。
    ("extra", "额外因子族重建", [PY, "-u", "scripts/build_extra_panels.py"], 200),
    ("model", "模型重训", [PY, "-u", "scripts/run_research.py",
                          "--out", "build/models/wf_h10", "--method", "blend",
                          # 34 → 20 / 每类 4 → 3：删掉 49 对 |ρ|>0.7 的冗余槽后，
                          # 8 相位均值 +0.50pp、最差相位 +0.83pp、离散度 1.08→0.90pp
                          # 20 → 21：叠加价值维度 bp（账面市值比），不顶替任何既有因子。
                          # 实测 8 相位均值 6.50→7.01%（+0.51pp）、最差相位 5.26→5.46%、
                          # Sharpe 0.530→0.579、MDD 均值 +0.40pp；ICIR 0.8124→0.8004 是唯一代价。
                          # 注意：删因子再加（17 池 / 20 换 3）两种做法都跑输，只能叠加。
                          "--max-factors", "21", "--max-per-category", "3",
                          "--horizon", "10",
                          "--no-fundamental",
                          # 换手族（turn5/turn20/turn60/turn_std20/turn_accel）：
                          # 8 相位配对检验中唯一通过双准则的新族（+1.04pp、胜率 88%、t=2.38）
                          "--extra", "build/factors/newf_turnover_extra.npz",
                          # 价值维度（bp 账面市值比，F002）：与复合分 ρ=0.299、
                          # 正交残差 IC +0.0284(t=5.2)、DEV 0.0461→OOS 0.0580 不衰减
                          "--extra", "build/factors/value_extra.npz",
                          "--pool-guard", "--pool-streak", "2"], 150),
    ("backtest", "主策略回测", [PY, "-u", "scripts/run_backtest.py"], 35),
    # 相位带（可选）：build/runs/main 的 nav 只是 8 个对齐相位里最好的一档，
    # 绩效披露必须同时给分布，否则「单点 CAGR」会被当成能力。
    ("phase_band", "主策略相位带", [PY, "-u", "scripts/phase_band.py"], 180,
     {"optional": True}),
    ("etf", "ETF 趋势回测", [PY, "-u", "scripts/etf_trend_strategy.py"], 6),
    ("sector", "板块轮动回测", [PY, "-u", "qf/sector_rotation.py"], 5),
    ("combo", "50/50 组合", [PY, "-u", "scripts/etf_combo_50_50.py"], 4),
    # 三腿：主策略 50 + ETF趋势 25 + 512890红利低波 25（季度再平衡、Σ|Δw|×5bp 双边）
    ("combo3", "三腿组合(+红利低波)", [PY, "-u", "scripts/etf_combo_3leg.py"], 4),
    ("three", "三策略组合", [PY, "-u", "scripts/three_strategy_combo.py"], 4),
    ("erc", "ERC 配置对比", [PY, "-u", "scripts/erc_deep.py"], 4),
    ("tracking", "候选跟踪", [PY, "-u", "scripts/candidate_tracking.py"], 18),
    ("track_record", "实盘台账", [PY, "-u", "scripts/track_record.py", "signal"], 2),
    ("numbers", "刷新基准数字", [PY, "-u", "scripts/refresh_final_numbers.py"], 2),
    ("report_s1", "同步报告摘要", [PY, "-u", "scripts/refresh_report_s1.py"], 2),
    # 2026-11-05 起默认四腿 40/30/20/10（主策略/低换手/宽基ETF/可转债）：
    # 实盘此前跑 --combo 两腿，与回测口径脱节；四腿同风险 +0.63pp、SR 0.895→0.958。
    # 同一版把第四腿从板块换成转债（板块 OOS 3.20%，转债与三腿相关 0.03~0.07）。
    # 转债信号必须在 signal 之前：signal 要读 signals/latest_cb_signal.json 建腿，
    # 顺序反了会一直用前一天的榜单。仍保留 optional：失败则该腿转现金，不阻断管道。
    ("cb_signal", "转债信号", [PY, "-u", "scripts/cb_signal.py"], 5, {"optional": True}),
    ("signal", "今日信号", [PY, "-u", "scripts/daily_signal.py", "--four"], 35),
    # 质量门禁：确认数据/因子/模型/信号/报告全链路一致（失败则管道失败）
    ("verify", "数据源一致性检测", [PY, "-u", "scripts/check_data_source.py"], 2),
]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_WRITE_LOCK = threading.Lock()

# 2026-11-07 事故：全 20 个阶段跑完（含 verify 一致），最后写 status.json 时
#   PermissionError: [WinError 5] 拒绝访问: status.tmp -> status.json
# 异常没人接 → 进程非零退出 + 网页的僵尸检测把这次运行判成 interrupted/ok=False
# → 界面上显示"更新失败"，而实际所有活都干完了。
#
# 根因是 Windows 语义：os.replace 需要目标文件的 DELETE 权限，而 Python 的 open()
# 打开文件时**不带 FILE_SHARE_DELETE**。网页每 2~3 秒轮询 /api/pipeline/status，
# 只要读的瞬间我们正好在替换，就必然撞上 WinError 5。这不是逻辑错误，是时序彩票：
# 一次运行写 ~150 次状态，撞上概率不低。
#
# 对策：退避重试 → 原地写兜底 → 永不抛出。状态文件写失败只该让"进度条滞后"，
# 绝不该让一次成功的更新被判成失败。
STATUS_WRITE_ATTEMPTS = 12
STATUS_WRITE_BACKOFF = 0.05      # 秒，指数退避；累计等待上限约 2.5 秒


def _replace_with_retry(tmp: pathlib.Path, target: pathlib.Path) -> None:
    """把 tmp 换到 target，被占用就退避重试；仍不行则原地写兜底。

    注意与 `scripts/update_data.py:_replace_with_retry` 的差别（不是重复实现）：
    那边写的是几百 MB 的面板 npz，**宁可报错也绝不能原地写**（半截文件会毁数据）；
    这边是几 KB 的状态 JSON，**宁可丢原子性也不能丢状态**（写不进去网页就永远卡
    running）。同一个坑，两种取舍。
    """
    delay = STATUS_WRITE_BACKOFF
    last: Exception | None = None
    for _ in range(STATUS_WRITE_ATTEMPTS):
        try:
            os.replace(tmp, target)
            return
        except OSError as exc:               # WinError 5 / 32 都归这里
            last = exc
            time.sleep(delay)
            delay = min(delay * 2, 0.5)
    # 实在换不动（对方长时间持有句柄）：退化成原地覆盖。
    # 非原子，读方理论上可能读到半截 JSON —— 但读方已有解析失败兜底，
    # 而"状态永远写不进去"会让网页永远卡在 running，两害相权取这个。
    try:
        target.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
        tmp.unlink(missing_ok=True)
    except Exception:
        if last is not None:
            raise last
        raise


def write_status(obj: dict) -> None:
    """原子写 status.json —— 绝不抛异常。

    心跳线程和主循环都会写这个文件（每 10 秒一次），共用同一个 .tmp 路径时必须
    串行化，否则一个线程 replace 掉的瞬间另一个线程会 FileNotFoundError。
    """
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    obj["updated_at"] = _now()
    with _WRITE_LOCK:
        tmp = STATUS.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
            _replace_with_retry(tmp, STATUS)
        except Exception as exc:
            # 只警告，不影响阶段结果与退出码。pid + heartbeat 仍然能让网页识别死进程。
            print(f"[warn] status.json 写入失败（{type(exc).__name__}: {exc}）"
                  f"，进度展示会滞后，阶段结果不受影响", file=sys.stderr, flush=True)



# ---------------------------------------------------------------- 存活检测
# 2026-11-03 21:44 那次更新：update_data.py 在「[4/6] 构建扩展面板」被系统杀掉，
# run_pipeline.py 也跟着消失，但 status.json 留在 running=true / finished_at=null，
# 网页一直显示「运行中·数据增量更新」，实际已经空转 24 小时。
# 根因是状态文件里没有任何"我是否还活着"的证据。这里补两样：
#   1) pid：读取方可以直接查进程是否还在
#   2) heartbeat：每 10 秒刷新 updated_at，长阶段也能证明还活着
HEARTBEAT_SEC = 10.0
STALE_AFTER_SEC = 180.0     # 超过 3 分钟没有心跳即判定为死


def pid_alive(pid) -> bool:
    """进程是否还活着。pid 缺失/非法一律返回 False。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            if k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def status_is_stale(st: dict, now: float | None = None) -> bool:
    """status.json 声称 running，但背后的进程已经不在（或心跳停了）。

    返回 True 表示这份状态不可信，应视为"上次运行已中断"。

    注意：只有**确实记录了 pid** 时才按 pid 判断。老格式（没有 pid 字段）不能因为
    `pid_alive(None)` 为 False 就判死，否则任何旧状态都会被误报成中断——宁可漏判，
    也不能把健康任务标成失败。
    """
    if not isinstance(st, dict) or not st.get("running"):
        return False
    now = time.time() if now is None else now
    pid = st.get("pid")
    if pid is not None:
        if not pid_alive(pid):
            return True
        hb = st.get("heartbeat_ts")
        if isinstance(hb, (int, float)) and now - float(hb) > STALE_AFTER_SEC:
            return True
        return False
    hb = st.get("heartbeat_ts")
    if isinstance(hb, (int, float)):
        return now - float(hb) > STALE_AFTER_SEC
    try:
        upd = datetime.fromisoformat(str(st.get("updated_at") or ""))
        return (datetime.now() - upd).total_seconds() > 600
    except Exception:
        return False


def another_instance_running() -> int | None:
    """已有实例在跑就返回它的 pid，否则 None。

    2026-11-07 事故的连带损失：那次崩在收尾，界面显示失败，于是又点了一次更新，
    结果 factors 阶段白重跑 850 秒。周期 15~35 分钟的管道不该允许误触发重入，
    也不该让"看起来失败了其实成功了"变成"再跑一遍"。
    """
    try:
        if not STATUS.exists():
            return None
        st = json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(st, dict) or not st.get("running"):
        return None
    if status_is_stale(st):
        return None
    pid = st.get("pid")
    if pid is None:
        return None
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid == os.getpid() or not pid_alive(pid):
        return None
    return pid


def _start_heartbeat(status: dict, lock: threading.Lock) -> threading.Event:
    """后台每 10 秒把心跳写进 status.json；返回停止信号。"""
    stop = threading.Event()

    def loop():
        while not stop.wait(HEARTBEAT_SEC):
            try:
                with lock:
                    status["heartbeat_ts"] = time.time()
                    write_status(status)
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True, name="pipeline-heartbeat").start()
    return stop


def _data_env() -> dict:
    """子进程统一使用 config 解析出的数据目录（避免混用新旧数据）。"""
    sys.path.insert(0, str(ROOT))
    try:
        from qf import config as _C
        return {**__import__("os").environ,
                "QF_DATA_RAW": str(_C.DATA_RAW),
                "PYTHONIOENCODING": "utf-8"}
    except Exception:
        return {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}


DURATIONS = ROOT / "build" / "pipeline" / "durations.json"


def record_duration(key: str, sec: float) -> None:
    """记录各阶段实际耗时（保留最近 6 次），供网页进度条校准预计时间。"""
    try:
        data = json.loads(DURATIONS.read_text(encoding="utf-8")) if DURATIONS.exists() else {}
    except Exception:
        data = {}
    arr = [x for x in data.get(key, []) if isinstance(x, (int, float))][-5:]
    arr.append(round(float(sec), 1))
    data[key] = arr
    try:
        DURATIONS.parent.mkdir(parents=True, exist_ok=True)
        DURATIONS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def run_stage(key: str, name: str, cmd: list[str]) -> dict:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGDIR / f"{datetime.now():%Y%m%d_%H%M%S}_{key}.log"
    t0 = time.time()
    tail: list[str] = []
    rc = -1
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(f"# {name}\n# cmd: {' '.join(cmd)}\n# start: {_now()}\n\n")
            f.flush()
            t_start = time.time()
            limit = STAGE_TIMEOUT.get(key, 1200)
            # 本脚本由 web 服务以 DETACHED_PROCESS 启动（无控制台），
            # 这里再 spawn console 程序时 Windows 会给它新建一个可见黑窗，
            # 必须显式 CREATE_NO_WINDOW（有控制台时该标志也无副作用）。
            _cf = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, env=_data_env(), **_cf)
            assert proc.stdout is not None
            import threading as _th
            _to = {"v": False}

            def _killer():
                while proc.poll() is None:
                    if time.time() - t_start > limit:
                        _to["v"] = True
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        return
                    time.sleep(2)

            _th.Thread(target=_killer, daemon=True).start()
            for line in proc.stdout:
                f.write(line)
                f.flush()
                tail.append(line.rstrip())
                if len(tail) > 40:
                    tail.pop(0)
            rc = proc.wait()
            if _to["v"]:
                tail.append(f"TIMEOUT: 超 {limit}s 已终止")
                rc = -2
    except Exception as e:
        tail.append(f"EXCEPTION: {type(e).__name__}: {e}")
        rc = -1
    return {
        "key": key, "name": name, "cmd": " ".join(cmd),
        "returncode": rc, "elapsed_sec": round(time.time() - t0, 1),
        "log": str(log_path.relative_to(ROOT)),
        "tail": tail[-12:],
        "ok": rc == 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-factors", action="store_true")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None, help="逗号分隔的阶段 key")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="已有实例在运行时仍然启动（默认拒绝，避免重复跑十几分钟）")
    args = ap.parse_args()

    # 重入保护：宁可什么都不做，也不要在同一条管线上并行跑两遍。
    # 退出码用 0 —— "已经有人在干了"不是失败，别让调度器再重试一次。
    other = another_instance_running()
    if other and not args.force:
        print(f"已有更新正在运行（pid {other}），本次不重复启动。", flush=True)
        print("  查看进度：build/pipeline/status.json 或网页「数据」页", flush=True)
        print("  确需并行（通常不需要）：加 --force", flush=True)
        raise SystemExit(0)

    only = {x.strip() for x in (args.only or "").split(",") if x.strip()}
    t_all = time.time()
    try:
        from qf import config as _C0
        print(f"[data] 数据目录: {_C0.DATA_RAW}（{_C0.DATA_SOURCE}）", flush=True)
    except Exception:
        pass

    # 本次真正要跑的阶段清单（写进 status，网页据此算准确的总进度）
    plan = []
    _started = args.start is None
    for key, _name, _cmd, _est, *_opt in STAGES:
        if not _started:
            if key != args.start:
                continue
            _started = True
        if only and key not in only:
            continue
        if key == "data" and args.skip_data:
            continue
        # extra 不随 --skip-factors 跳过：它在面板已对齐时只要 ~3s（只做检查），
        # 而漏检一次就会让下游模型静默丢掉末周打分。宁可多花 3 秒。
        if key == "factors" and args.skip_factors:
            continue
        plan.append(key)

    # 上一轮若"声称在跑但进程已不在"，先把它标记为中断，避免网页一直等一个死进程
    prev = {}
    try:
        if STATUS.exists():
            prev = json.loads(STATUS.read_text(encoding="utf-8"))
        if status_is_stale(prev):
            prev.update({"running": False, "interrupted": True, "ok": False,
                         "finished_at": prev.get("finished_at") or _now(),
                         "stale_reason": "上次运行的进程已退出，状态未收尾"})
    except Exception:
        prev = {}

    _lock = threading.Lock()
    status = {"running": True, "started_at": _now(), "finished_at": None,
              "current": None, "stages": [], "total_elapsed": None, "ok": None,
              "pid": os.getpid(), "heartbeat_ts": time.time(),
              "plan": plan, "plan_est": sum(e for k, _n, _c, e, *_o in STAGES if k in plan)}
    if prev:
        status["prev"] = {k: prev.get(k) for k in
                          ("started_at", "finished_at", "ok", "interrupted", "stale_reason")
                          if prev.get(k) is not None}
    write_status(status)
    _hb = _start_heartbeat(status, _lock)

    # 无论正常结束、抛异常还是 Ctrl+C，都要把 running 收尾，否则网页永久卡在"运行中"
    import atexit

    def _finalize():
        if not status.get("running"):
            return
        status["running"] = False
        status["finished_at"] = status.get("finished_at") or _now()
        status["interrupted"] = True
        status["ok"] = False
        try:
            write_status(status)
        except Exception:
            pass

    atexit.register(_finalize)

    reached = args.start is None
    for key, name, cmd, est, *_opt in STAGES:
        optional = bool(_opt and isinstance(_opt[0], dict) and _opt[0].get("optional"))
        if not reached:
            if key != args.start:
                status["stages"].append({"key": key, "name": name,
                                         "status": "skipped", "reason": "未到起点"})
                write_status(status)
                continue
            reached = True
        if only and key not in only:
            status["stages"].append({"key": key, "name": name,
                                     "status": "skipped", "reason": "不在 --only"})
            write_status(status)
            continue
        if key == "data" and args.skip_data:
            status["stages"].append({"key": key, "name": name,
                                     "status": "skipped", "reason": "--skip-data"})
            write_status(status)
            continue
        if key == "factors" and args.skip_factors:
            status["stages"].append({"key": key, "name": name,
                                     "status": "skipped", "reason": "--skip-factors"})
            write_status(status)
            continue

        status["current"] = {"key": key, "name": name,
                             "started_at": _now(), "est_sec": est}
        write_status(status)
        print(f"\n[{key}] {name} …", flush=True)
        res = run_stage(key, name, cmd)
        if res.get("ok") and res.get("elapsed_sec"):
            record_duration(key, res["elapsed_sec"])
        res["status"] = "ok" if res["ok"] else "failed"
        status["stages"].append(res)
        print(f"[{key}] {'✓' if res['ok'] else '✗'} {res['elapsed_sec']}s"
              f"{'' if res['ok'] else ' rc=' + str(res['returncode'])}", flush=True)
        write_status(status)
        if not res["ok"] and optional:
            print(f"  ⚠ {key} 失败但为可选阶段，继续（日志: {res['log']}）", flush=True)
            continue
        if not res["ok"] and not args.keep_going:
            if key == "verify":
                print("  ✗ 一致性检测未通过：数据/因子/模型/信号存在滞后。", flush=True)
                print(f"    详见: {res['log']}", flush=True)
            else:
                print(f"  ✗ {key} 失败，停止。日志: {res['log']}", flush=True)
            break

    status["running"] = False
    status["interrupted"] = False
    status["current"] = None
    status["finished_at"] = _now()
    status["total_elapsed"] = round(time.time() - t_all, 1)
    _opt_keys = {k for k, *_r in STAGES if _r and isinstance(_r[0], dict) and _r[0].get("optional")}
    status["ok"] = not any(s.get("status") == "failed" and s.get("key") not in _opt_keys
                           for s in status["stages"])
    write_status(status)
    _verify = next((x for x in status["stages"] if x.get("key") == "verify"), None)
    if _verify is not None and _verify.get("status") == "ok":
        print("\n✅ 一致性校验通过：数据 / 因子 / 模型 / 信号 / 报告 全部对齐", flush=True)
    elif _verify is not None and _verify.get("status") == "failed":
        print("\n⚠️ 一致性校验未通过（见上）", flush=True)
    print(f"\n完成：{'✅' if status['ok'] else '✗ 有失败'} "
          f"({status['total_elapsed']}s)", flush=True)
    if not status["ok"]:
        bad = [s["key"] for s in status["stages"]
               if s.get("status") == "failed" and s.get("key") not in _opt_keys]
        print("失败: " + ", ".join(bad), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
