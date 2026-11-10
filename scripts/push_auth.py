# -*- coding: utf-8 -*-
"""把 SDK 的 user.json 推送到项目共享目录，供 agent 进程复用。

背景：agent 侧的文件系统视图对 D:/Python 是会话快照，读不到你刚写入的
D:/Python/Python313/Lib/user.json；而项目目录 D:/wkf/getrich 是实时双向共享的。
所以把凭据副本放到 <project>/.secrets/ 下，agent 就能解密并复用。

用法：
    D:/Python/Python313/python.exe D:/wkf/getrich/ashare_quant/scripts/push_auth.py
"""
from __future__ import annotations

import os
import shutil
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(PROJ, ".secrets")
DEST = os.path.join(DEST_DIR, "user.json")


def main() -> int:
    try:
        import panda_data
        from panda_data.auth_manager import _get_user_json_path, _get_machine_fingerprint
    except Exception as e:
        print(f"[FAIL] 无法导入 panda_data: {type(e).__name__}: {e}")
        return 1

    src = _get_user_json_path()
    print(f"解释器      : {sys.executable}")
    print(f"机器指纹    : {_get_machine_fingerprint()}")
    print(f"源 user.json: {src}")
    print(f"目标        : {DEST}")

    if not os.path.exists(src):
        print("\n[FAIL] 源文件不存在——请先运行 login_pandadata.py 完成登录。")
        return 1

    os.makedirs(DEST_DIR, exist_ok=True)
    shutil.copy2(src, DEST)
    print(f"\n[OK] 已复制，{os.path.getsize(DEST)} 字节")

    # 校验：本项目目录下的副本能否被解密
    import panda_data.auth_manager as am
    am._user_json_dir = DEST_DIR
    creds = am.get_decrypted_credentials()
    if creds:
        u = creds.get("username", "")
        print(f"[OK] 副本可解密：username={u}  base_url={creds.get('base_url')}  "
              f"password={'*' * 8}（{len(creds.get('password', ''))} 位）")
    else:
        print("[WARN] 副本无法解密（指纹不一致？），agent 侧仍需你自己跑取数。")
        return 1

    print("\n现在可以让 agent 侧直接调用了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
