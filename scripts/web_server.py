# -*- coding: utf-8 -*-
"""启动 Web UI 服务。

用法:
    python scripts/web_server.py                       # 默认 127.0.0.1:8765，数据用 was_stock_data_ext
    python scripts/web_server.py --port 8766
    python scripts/web_server.py --data orig           # 用原始数据包（回测数字与报告一致）
    python scripts/web_server.py --no-prewarm          # 不预热推荐（启动更快，首次点推荐会慢 ~20s）
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent          # ashare_quant
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

ORIG = REPO / "data_extract" / "was_stock_data"
EXT = REPO / "data_extract" / "was_stock_data_ext"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", choices=["auto", "ext", "orig"], default="auto",
                    help="auto=有增量数据用增量，否则原始；orig 与回测报告口径一致")
    ap.add_argument("--no-prewarm", action="store_true")
    args = ap.parse_args()

    if args.data == "orig" or (args.data == "auto" and not EXT.exists()):
        os.environ["QF_DATA_RAW"] = str(ORIG)
    else:
        os.environ["QF_DATA_RAW"] = str(EXT)
    print(f"[web] 数据目录: {os.environ['QF_DATA_RAW']}", flush=True)

    from web import server as webserver
    from web.services import DataService

    svc = DataService()
    try:
        srv = webserver.make_server(args.host, args.port, svc)
    except OSError as e:
        print(f"[web] 无法监听 {args.host}:{args.port}：{e}", flush=True)
        raise SystemExit(2)

    url = f"http://{args.host}:{args.port}/"
    print(f"[web] 服务已启动: {url}", flush=True)
    print("[web] 后台加载行情/模型（约 30~60s，接口就绪后 /api/meta 的 server.loaded=true）...", flush=True)
    threading.Thread(target=svc.load, daemon=True, name="web-load").start()

    def _prewarm():
        t0 = time.time()
        while not svc.loaded and time.time() - t0 < 300:
            time.sleep(1)
        if not svc.loaded:
            return
        try:
            print("[web] 预热推荐打分（约 20s）...", flush=True)
            # 1) 预热 live_scores（个股页/推荐页都依赖）
            svc._warm_live_scores()
            # 2) 预热 live features（个股页因子贡献，首次约 20s）
            svc._get_live_features()
            # 3) 预热 recommend 缓存（今日推荐/候选榜）
            svc.public_recommend(top=60)
            print("[web] 预热完成", flush=True)
        except Exception as e:
            print(f"[web] 预热失败（不影响使用）：{e}", flush=True)
    if not args.no_prewarm:
        threading.Thread(target=_prewarm, daemon=True, name="web-prewarm").start()

    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[web] 收到 Ctrl-C，关闭服务", flush=True)
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
