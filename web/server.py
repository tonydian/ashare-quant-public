# -*- coding: utf-8 -*-
"""Web UI HTTP 服务（Python 标准库 ThreadingHTTPServer，无第三方依赖）。

用法（见 scripts/web_server.py）：
    python scripts/web_server.py --port 8765 [--data ext|orig|auto] [--host 127.0.0.1]
"""
from __future__ import annotations

import json
import math
import mimetypes
import pathlib
import re
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent            # ashare_quant
REPO = ROOT.parent
STATIC = ROOT / "web" / "static"
REPORTS = ROOT / "reports"
SIGNALS = ROOT / "signals"
from qf import config as C  # noqa: E402

JSON_CT = "application/json; charset=utf-8"


def _sanitize(obj):
    """递归转 JSON 安全类型（NaN/inf → None，numpy → python，Timestamp → iso）。"""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        import numpy as np
        import pandas as pd
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return v if math.isfinite(v) else None
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
    except Exception:
        pass
    return obj


class Handler(BaseHTTPRequestHandler):
    server_version = "AStockQuantWeb/2.0"
    protocol_version = "HTTP/1.1"

    # ---------- 基础 IO
    def log_message(self, fmt, *args):  # 静默访问日志（错误仍会打印）
        pass

    def _send(self, status: int, body: bytes, ctype: str = JSON_CT,
              extra_headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, status: int = 200):
        body = json.dumps(_sanitize(obj), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, body)

    def _error(self, msg: str, status: int = 500):
        self._json({"error": str(msg)}, status=status)

    def _query(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        return parsed.path, {k: v[-1] for k, v in q.items()}

    @property
    def svc(self):
        return self.server.svc  # type: ignore[attr-defined]

    # ---------- 路由
    def do_GET(self):
        path, q = self._query()
        try:
            if path in ("/", "/index.html"):
                return self._serve_file(STATIC / "index.html")
            if path.startswith("/static/"):
                return self._serve_file(STATIC / path[len("/static/"):])
            if path.startswith("/reports/"):
                return self._serve_file(REPORTS / urllib.parse.unquote(path[len("/reports/"):]))
            if path.startswith("/signals/"):
                return self._serve_file(SIGNALS / urllib.parse.unquote(path[len("/signals/"):]))
            if path == "/favicon.ico":
                return self._send(204, b"")
            if path == "/api/health":
                m = self.svc.meta()
                return self._json({"status": "ok", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   "data_last_date": m["data"]["last_date"]})
            if path == "/api/meta":
                return self._json(self.svc.meta())
            if path == "/api/market/overview":
                return self._json(self.svc.market_overview(asof=q.get("asof"), bench=q.get("bench", "csi_all")))
            if path == "/api/realtime/overview":
                return self._json(self.svc.realtime_overview())
            if path == "/api/recommend":
                top = int(q.get("top", 60))
                return self._json(self.svc.public_recommend(
                    top=top, asof=q.get("asof"), n_holdings=int(q.get("n_holdings", 40)),
                    model=q.get("model")))
            if path == "/api/models":
                meta = self.svc.meta().get("model", {})
                return self._json({"default": meta.get("default"), "models": meta.get("models", [])})
            if path == "/api/recommend/export":
                return self._export_csv(q.get("type", "target"), model=q.get("model"))
            if path == "/api/stock/search":
                return self._json(self.svc.stock_search(q.get("q", ""), int(q.get("limit", 20))))
            m = re.match(r"^/api/stock/([A-Za-z0-9_.]+)$", path)
            if m:
                return self._json(self.svc.stock_detail(m.group(1), model=q.get("model")))
            m = re.match(r"^/api/jobs/([A-Za-z0-9_]+)$", path)
            if m:
                job = self.svc.jobs.get(m.group(1))
                if not job:
                    return self._error("任务不存在", 404)
                log = job.pop("log", None)
                if log is not None:
                    try:
                        job["log_tail"] = list(log)[-30:]
                    except Exception:
                        job["log_tail"] = []
                return self._json(job)
            if path == "/api/data/status":
                return self._json(self.svc.data_status())
            if path == "/api/pipeline/status":
                return self._json(self.svc.pipeline_status())
            if path == "/api/consistency":
                return self._json(self.svc.consistency_check())
            if path == "/api/research/overview":
                return self._json(self.svc.research_overview())
            if path == "/api/research/cost":
                return self._json(self.svc.cost_sensitivity())
            if path == "/api/research/rebalance":
                return self._json(self.svc.rebalance_study())
            if path == "/api/research/pool":
                return self._json(self.svc.etf_pool_health())
            if path == "/api/tracking":
                return self._json(self.svc.tracking_data())
            if path == "/api/etf_trend":
                return self._json(self.svc.etf_trend())
            if path == "/api/sector_rotation":
                return self._json(self.svc.sector_rotation())
            if path == "/api/cb_signal":
                return self._json(self.svc.cb_signal())
            if path == "/api/combo_plan":
                return self._json(self.svc.combo_plan())
            if path == "/api/combo_methods":
                return self._json(self.svc.combo_methods())
            if path == "/api/candidates":
                return self._json(self.svc.candidates())
            if path == "/api/today_recommend":
                return self._json(self.svc.today_recommend())
            if path == "/api/monthly_recommend":
                return self._json(self.svc.monthly_recommend())
            if path == "/api/today_export":
                name, body = self.svc.today_export_csv()
                from urllib.parse import quote as _q
                return self._send(200, body, "text/csv; charset=utf-8",
                                  {"Content-Disposition":
                                   f"attachment; filename*=UTF-8''{_q(name)}"})
            if path == "/api/strategy_kinds":
                return self._json({"kinds": self.svc.STRATEGY_KINDS})
            if path == "/api/factors/summary":
                return self._json(self.svc.factors_summary())
            if path == "/api/reports":
                return self._json(self.svc.reports())
            if path == "/api/chart/index_line":
                png = self.svc.chart_index_line(q.get("code", "csi_all"), int(q.get("days", 250)))
                return self._send(200, png, "image/png")
            if path == "/api/chart/stock_kline":
                png = self.svc.chart_stock_kline(q.get("code", ""), int(q.get("days", 250)))
                return self._send(200, png, "image/png")
            if path == "/api/chart/industry_bar":
                png = self.svc.chart_industry_bar(int(q.get("days", 20)))
                return self._send(200, png, "image/png")
            m = re.match(r"^/api/chart/backtest_nav/([A-Za-z0-9_]+)$", path)
            if m:
                png = self.svc.chart_backtest_nav(m.group(1))
                return self._send(200, png, "image/png")
            if path.startswith("/api/"):
                return self._error(f"未知接口 {path}", 404)
            # SPA fallback
            return self._serve_file(STATIC / "index.html")
        except KeyError as e:
            return self._error(str(e), 404)
        except ValueError as e:
            return self._error(str(e), 400)
        except RuntimeError as e:
            return self._error(str(e), 503 if "加载" in str(e) else 400)
        except Exception:
            traceback.print_exc()
            return self._error("服务器内部错误（详见服务端日志）", 500)

    def do_POST(self):
        path, q = self._query()
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                return self._error("请求体不是合法 JSON", 400)
            if path == "/api/backtest_strategy":
                return self._json(self.svc.backtest_strategy(body))
            if path == "/api/backtest":
                return self._json({"job_id": self.svc.start_backtest(body or {})})
            if path == "/api/data/update":
                return self._json({"job_id": self.svc.start_data_update()})
            if path == "/api/pipeline/run":
                return self._json(self.svc.pipeline_run(
                    skip_data=bool(body.get("skip_data")),
                    skip_factors=bool(body.get("skip_factors")),
                    only=body.get("only") or None))
            if path == "/api/tracking/fill":
                return self._json(self.svc.tracking_fill(body or {}))
            if path == "/api/tracking/nav":
                return self._json(self.svc.tracking_nav(body or {}))
            if path == "/api/tracking/signal":
                return self._json(self.svc.tracking_signal())
            if path == "/api/tracking/delete":
                return self._json(self.svc.tracking_delete_trade(body or {}))
            if path == "/api/tracking/reset":
                return self._json(self.svc.tracking_reset(body or {}))
            return self._error(f"未知接口 {path}", 404)
        except ValueError as e:
            return self._error(str(e), 400)
        except RuntimeError as e:
            return self._error(str(e), 409)
        except Exception:
            traceback.print_exc()
            return self._error("服务器内部错误（详见服务端日志）", 500)

    # ---------- 文件
    def _serve_file(self, p: pathlib.Path):
        try:
            p = p.resolve()
        except Exception:
            return self._error("非法路径", 400)
        allowed = [STATIC.resolve(), REPORTS.resolve(), SIGNALS.resolve()]
        if not any(str(p).startswith(str(a)) for a in allowed) or not p.is_file():
            return self._error("文件不存在", 404)
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if p.suffix.lower() in (".md", ".txt"):
            ctype = "text/plain; charset=utf-8"
        body = p.read_bytes()
        self._send(200, body, ctype)

    # ---------- CSV 导出
    def _export_csv(self, kind: str, model: str | None = None):
        kind = "orders" if kind == "orders" else "target"
        try:
            text = self.svc.export_recommend_csv(kind, model=model)
            model_name = model or (self.svc.meta().get("model", {}) or {}).get("default", "wf_h10")
            fname = ("latest_orders" if kind == "orders" else "latest_target") + f"_{model_name}.csv"
            self._send(200, text.encode("utf-8-sig"), "text/csv; charset=utf-8",
                       {"Content-Disposition": f'attachment; filename="{fname}"'})
        except ValueError as e:
            return self._error(str(e), 400)
        except RuntimeError as e:
            return self._error(str(e), 503)


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError, TimeoutError, OSError)):
            return  # 浏览器/客户端提前断开，属于正常噪音
        super().handle_error(request, client_address)


def make_server(host: str, port: int, svc) -> WebServer:
    srv = WebServer((host, port), Handler)
    srv.svc = svc  # type: ignore[attr-defined]
    return srv
