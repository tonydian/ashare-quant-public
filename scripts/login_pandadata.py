# -*- coding: utf-8 -*-
"""PandaData SDK 登录诊断。

用法（务必用绝对路径的解释器）：
    D:/Python/Python313/python.exe D:/wkf/getrich/ashare_quant/scripts/login_pandadata.py --plain

密码输入方式（按优先级）：
    1. 环境变量 PANDA_PASSWORD（配合 --user 使用，适合自动化）
    2. --plain：明文回显输入（Git Bash / mintty 下 getpass 读不到键盘时用这个）
    3. 默认 getpass：不回显（仅在 Windows 原生控制台 cmd/PowerShell 下可靠）

成功后打印 token 摘要与 user.json 落盘位置，失败时打印完整原因与登录 URL。
"""
from __future__ import annotations

import getpass
import json
import os
import sys
import traceback
from hashlib import md5

BASE_URL = "http://pandadata.pandaaiquant.com"
LOGIN_URL = BASE_URL + "/pandaData/dataUser/login"

PLAIN = "--plain" in sys.argv
CLI_USER = ""
for i, a in enumerate(sys.argv):
    if a in ("--user", "-u") and i + 1 < len(sys.argv):
        CLI_USER = sys.argv[i + 1]
    elif a.startswith("--user="):
        CLI_USER = a.split("=", 1)[1]


def read_password() -> str:
    """按可用性依次尝试：环境变量 → 明文回显 → getpass。"""
    env = os.environ.get("PANDA_PASSWORD")
    if env:
        print("  密码: 从环境变量 PANDA_PASSWORD 读取（不回显）")
        return env
    if PLAIN:
        print("  密码: 明文输入（会显示在屏幕上，回车结束）")
        return input("  Password: ").strip()
    try:
        print("  密码: 不回显输入（若卡住无反应，请改用 --plain）")
        return getpass.getpass("  Password: ")
    except Exception as e:
        print(f"  getpass 不可用（{type(e).__name__}），改用明文输入")
        return input("  Password: ").strip()


def main() -> int:
    print("=" * 78)
    print("PandaData SDK 登录诊断")
    print("=" * 78)
    print(f"解释器      : {sys.executable}")
    print(f"登录端点    : {LOGIN_URL}")

    try:
        import panda_data
        from panda_data.auth_manager import _get_user_json_path
    except Exception as e:
        print(f"\n[FAIL] 无法导入 panda_data: {type(e).__name__}: {e}")
        print("请确认用的是 D:/Python/Python313/python.exe")
        return 1

    print(f"SDK 版本    : {getattr(panda_data, '__version__', 'unknown')}")
    print(f"user.json   : {_get_user_json_path()}")
    print(f"当前认证    : {panda_data.is_authenticated()}")
    print()

    # ---- 1. 连通性 ----
    try:
        import requests
        r = requests.post(LOGIN_URL, json={"username": "__probe__", "password": "x"}, timeout=10)
        print(f"[1/3] 端点连通 OK：HTTP {r.status_code}  {r.text[:200]}")
    except Exception as e:
        print(f"[1/3] 端点连通 FAIL：{type(e).__name__}: {str(e)[:200]}")
        return 1

    # ---- 2. 登录 ----
    print()
    print("[2/3] 输入凭据（账号通常是注册手机号，如 188xxxxxxxx）")
    username = CLI_USER or input("  账号: ").strip()
    if CLI_USER:
        print(f"  账号: {CLI_USER}（来自命令行参数）")
    password = read_password()
    if not username or not password:
        print("[FAIL] 账号或密码为空")
        return 1

    print(f"\n  提交用户名: {username}  长度 {len(username)}")
    print(f"  密码 md5   : {md5(password.encode('utf-8')).hexdigest()[:12]}...")

    # 先打一次裸请求，把服务端原始回执亮出来
    try:
        import requests
        r = requests.post(
            LOGIN_URL,
            json={"username": username, "password": md5(password.encode("utf-8")).hexdigest()},
            timeout=15,
        )
        print(f"  原始回执   : HTTP {r.status_code}  {r.text[:300]}")
    except Exception as e:
        print(f"  原始回执   : FATAL {type(e).__name__}: {str(e)[:200]}")

    print()
    try:
        token = panda_data.init_token(username=username, password=password, base_url=BASE_URL)
    except Exception as e:
        print(f"[2/3] 登录 FAIL: {type(e).__name__}: {e}")
        print("\n--- traceback ---")
        traceback.print_exc()
        return 1

    print(f"[2/3] 登录 OK   token 前 32 位: {token[:32]}...  长度 {len(token)}")

    # ---- 3. 落盘校验 ----
    print()
    path = _get_user_json_path()
    ok = panda_data.is_authenticated()
    print(f"[3/3] is_authenticated: {ok}")
    print(f"      user.json 存在: {os.path.exists(path)}  路径: {path}")
    if os.path.exists(path):
        print(f"      文件大小: {os.path.getsize(path)} 字节")
    try:
        info = panda_data.auth_info()
        print(f"      auth_info: {json.dumps({k: (str(v)[:40] if v is not None else None) for k, v in info.items()}, ensure_ascii=False)}")
    except Exception as e:
        print(f"      auth_info ERR: {e}")

    if ok:
        print("\n[OK] 认证就绪，可以开始取数。")
        return 0
    print("\n[WARN] token 已拿到但 is_authenticated 仍为 False，请看上面的 user.json 路径。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
