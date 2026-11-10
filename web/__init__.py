# -*- coding: utf-8 -*-
"""A 股量化系统 Web UI 后端包（标准库 HTTP 服务 + matplotlib 图表）。"""
from .services import DataService, create_service  # noqa: F401
from .server import make_server, Handler  # noqa: F401
