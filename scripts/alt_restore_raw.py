# -*- coding: utf-8 -*-
"""另类数据原始 jsonl 的归档恢复工具。

背景：build/alt 下的 8 个原始抓取文件（合计 5.33 GB）已用 gzip 压缩为 .jsonl.gz
      （合计约 1.0 GB，压缩率 18.5%）。对应的 (T,N) 面板 *_panel.npz 已独立存在，
      日常建模不需要原始文件。只有在**重建面板**（改口径/加字段）时才需要解压。

用法：
    python scripts/alt_restore_raw.py --list                 # 查看归档与状态
    python scripts/alt_restore_raw.py --restore flow_raw     # 解压指定文件
    python scripts/alt_restore_raw.py --restore-all          # 全部解压
    python scripts/alt_restore_raw.py --pack                 # 反向：把 jsonl 重新压回 .gz

解压后的 jsonl 与 .gz 并存；确认面板重建完成可再删 jsonl 释放空间。
"""
from __future__ import annotations

import argparse
import gzip
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ALT = ROOT / "build" / "alt"


def _mb(n: int | float) -> float:
    return round(float(n) / 1048576.0, 1)


def cmd_list() -> None:
    gz = sorted(ALT.glob("*.jsonl.gz"))
    raw = sorted(ALT.glob("*.jsonl"))
    if not gz:
        print(f"{ALT} 下没有 .jsonl.gz 归档。")
        return
    print(f"{'归档文件':<28}{'压缩':>10}{'原始':>10}  状态")
    print("-" * 62)
    tot_gz = tot_raw = 0
    for p in gz:
        name = p.name[: -len(".gz")]
        g = p.stat().st_size
        r = (ALT / name).stat().st_size if (ALT / name).exists() else 0
        tot_gz += g
        tot_raw += r if r else 0
        state = "已解压" if r else "仅归档"
        print(f"{name:<28}{_mb(g):>9.1f}M{r and _mb(r) or 0:>9.1f}M  {state}")
    print("-" * 62)
    print(f"合计：压缩 {_mb(tot_gz)} MB，当前已解压 {_mb(tot_raw)} MB")
    print(f"归档总数 {len(gz)}，存在未解压的原始 jsonl {len(raw)} 个")


def _decompress(p: pathlib.Path) -> None:
    dst = ALT / p.name[: -len(".gz")]
    if dst.exists():
        print(f"  跳过（已存在）：{dst.name}")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    with gzip.open(p, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 20)
    tmp.replace(dst)
    print(f"  解压完成：{dst.name}  {_mb(dst.stat().st_size)} MB")


def cmd_restore(names: list[str]) -> None:
    for nm in names:
        cands = [ALT / f"{nm}.jsonl.gz", ALT / f"{nm}.gz"]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            print(f"  未找到归档：{nm}")
            continue
        _decompress(p)


def cmd_restore_all() -> None:
    for p in sorted(ALT.glob("*.jsonl.gz")):
        _decompress(p)


def cmd_pack() -> None:
    n = 0
    for p in sorted(ALT.glob("*.jsonl")):
        gz = p.with_suffix(p.suffix + ".gz")
        if gz.exists() and gz.stat().st_mtime >= p.stat().st_mtime:
            print(f"  跳过（归档较新）：{p.name}")
            continue
        with open(p, "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, length=1 << 20)
        print(f"  已压缩：{p.name}  {_mb(p.stat().st_size)} → {_mb(gz.stat().st_size)} MB")
        n += 1
    print(f"完成 {n} 个。")


def main() -> None:
    ap = argparse.ArgumentParser(description="另类数据原始 jsonl 归档恢复")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="列出归档状态（默认）")
    g.add_argument("--restore", nargs="+", metavar="NAME", help="解压指定文件，如 flow_raw")
    g.add_argument("--restore-all", action="store_true", help="解压全部归档")
    g.add_argument("--pack", action="store_true", help="把已解压的 jsonl 重新压缩")
    args = ap.parse_args()

    if args.restore:
        print("解压：")
        cmd_restore(args.restore)
    elif args.restore_all:
        print("解压全部：")
        cmd_restore_all()
    elif args.pack:
        print("压缩：")
        cmd_pack()
    else:
        cmd_list()


if __name__ == "__main__":
    main()
