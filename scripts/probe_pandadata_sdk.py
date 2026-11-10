# -*- coding: utf-8 -*-
"""探测 panda_data SDK 的 API 面：装了没、怎么认证、get_factor 签名是什么。

用法：
    D:/Python/Python313/python.exe scripts/probe_pandadata_sdk.py

背景（docs/PANDADATA.md §7.5）：MCP 通道单次只回约 3.5 万字节，建不了数据仓库。
SDK 直连是唯一能一次拿完整 parquet 的路。本脚本只做只读探测，不取数。
"""
import inspect
import os
import sys


def sec(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


sec("1. SDK 是否已安装")
try:
    import panda_data
except ModuleNotFoundError:
    print("未安装。执行：")
    print("    D:/Python/Python313/python.exe -m pip install panda_data")
    sys.exit(1)

print("已安装：", getattr(panda_data, "__file__", "?"))
print("版本：", getattr(panda_data, "__version__", "未声明"))

sec("2. 顶层可调用对象（找 login / init / set_token 之类）")
pub = [n for n in dir(panda_data) if not n.startswith("_")]
funcs = []
for n in pub:
    obj = getattr(panda_data, n)
    if callable(obj) and not inspect.isclass(obj):
        funcs.append(n)
print(f"公开属性 {len(pub)} 个，其中可调用 {len(funcs)} 个：")
print("  " + ", ".join(funcs))

sec("3. 认证相关（关键字匹配）")
kw = ("login", "auth", "token", "init", "set_", "config", "connect", "session", "account")
hits = [n for n in funcs if any(k in n.lower() for k in kw)]
print("候选：", hits if hits else "（没有明显的认证入口，可能靠环境变量）")
for n in hits:
    try:
        print(f"  {n}{inspect.signature(getattr(panda_data, n))}")
        d = inspect.getdoc(getattr(panda_data, n))
        if d:
            print("    " + d.strip().split("\n")[0][:200])
    except Exception as e:
        print(f"  {n}  签名读取失败: {e}")

sec("4. get_factor 签名")
gf = getattr(panda_data, "get_factor", None)
if gf is None:
    print("!! SDK 里没有 get_factor，检查版本或换个包名")
else:
    try:
        print("签名：", inspect.signature(gf))
    except Exception as e:
        print("签名读取失败：", e)
    d = inspect.getdoc(gf)
    print("\n文档字符串（前 1500 字）：")
    print(d[:1500] if d else "（无）")

sec("5. 环境变量里是否已有凭据")
cands = [
    "PANDA_DATA_TOKEN", "PANDADATA_TOKEN", "PANDA_DATA_USER", "PANDADATA_USER",
    "PANDA_DATA_PASSWORD", "PANDADATA_PASSWORD", "PANDA_DATA_API_KEY",
    "PANDADATA_USERNAME", "PANDADATA_PASSWORD",
]
found = {k: ("***" if os.environ.get(k) else None) for k in cands}
if any(found.values()):
    print("发现（值已脱敏）：")
    for k, v in found.items():
        if v:
            print(f"  {k} = {v}")
else:
    print("都没有。SDK 认证需要单独提供凭据——MCP 的 OAuth 不通用（sdk_status 说 data_mode=gateway）。")

sec("6. 包内是否有配置文件 / 示例")
import pathlib
root = pathlib.Path(panda_data.__file__).resolve().parent
print("包目录：", root)
try:
    items = sorted(p.name for p in root.iterdir())[:40]
    print("内容：", ", ".join(items))
except Exception as e:
    print("列目录失败：", e)

print("\n下一步：把 §3 / §4 的结果贴回来，据此写取数脚本（资产负债表截面）。")
