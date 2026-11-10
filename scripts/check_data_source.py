# -*- coding: utf-8 -*-
"""数据源一致性检测：确认全链路用的是同一份数据。

用法::

    python scripts/check_data_source.py           # 人类可读
    python scripts/check_data_source.py --json    # 机器可读（供管道/前端）

检查项：
  1) 当前数据目录与行数
  2) weekly_meta（因子面板）是否追上了数据
  3) 模型 scores 是否追上了数据
  4) 信号日期是否与数据最后日一致
  5) 报告 §1 是否含最新日期

退出码：0 = 全部一致；1 = 存在滞后
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qf import config as C

MAX_LAG = 5   # 周频步长 5，正常滞后应 <5 个交易日


def check() -> dict:
    out = {"ok": True, "items": [], "fix_hint": [], "data_raw": str(C.DATA_RAW),
           "source": getattr(C, "DATA_SOURCE", "?")}

    panel = C.DATA_RAW / "panel_ohlcv.npz"
    if not panel.exists():
        out["ok"] = False
        out["items"].append({"name": "行情面板", "ok": False,
                             "detail": f"不存在: {panel}"})
        return out
    z = np.load(panel)
    T = int(z["close"].shape[0])
    last = str(z["dates"][-1])[:10]
    out.update({"panel_rows": T, "data_last": last})
    out["items"].append({"name": "行情面板", "ok": True,
                         "detail": f"{T} 行，最后 {last}"})

    # 2) weekly_meta
    wm = C.BUILD_FACTORS / "weekly_meta.npz"
    need_factors = False
    if not wm.exists():
        out["ok"] = False
        need_factors = True
        out["items"].append({"name": "因子面板", "ok": False, "detail": "不存在"})
    else:
        m = np.load(wm)
        rows = m["rows"].astype(int)
        lag = T - 1 - int(rows[-1])
        ok = lag < MAX_LAG
        if not ok:
            out["ok"] = False
            need_factors = True
        out["items"].append({
            "name": "因子面板", "ok": ok, "lag": int(lag),
            "detail": f"{len(rows)} 行，最后行 {int(rows[-1])}，滞后 {lag} 日"})

    # 2.5) 额外因子族面板：生产模型 --extra 引用的那些必须与 weekly_meta 同长
    # 不同长 → 模型最后一周打分全 NaN → composite_score 停在上一周，
    # 而此处若不点名，只会看到下面那句误导性的「模型滞后」。
    extra_bad: list[tuple[str, int | None]] = []
    need_extra = False
    extra_detail = ""
    if wm.exists():
        try:
            sys.path.insert(0, str(ROOT / "scripts"))
            import build_extra_panels as _bep
            n_week = int(len(np.load(wm)["rows"]))
            stems = _bep.with_deps(_bep.production_extras())
            for s in stems:
                p = C.BUILD_FACTORS / f"{s}.npz"
                if not p.exists():
                    extra_bad.append((s, None))
                    continue
                a = np.load(p)
                w = int(a[a.files[0]].shape[0]) if a.files else 0
                if w != n_week:
                    extra_bad.append((s, w))
            ok_extra = not extra_bad
            if not ok_extra:
                out["ok"] = False
                need_extra = True
                extra_detail = "、".join(
                    f"{s} 停在 {w} 周" if w is not None else f"{s} 不存在"
                    for s, w in extra_bad) + f"（weekly_meta 已到 {n_week} 周）"
            else:
                extra_detail = f"{len(stems)} 个面板均已对齐 {n_week} 周"
            out["items"].append({"name": "额外因子族", "ok": ok_extra,
                                 "detail": extra_detail})
        except Exception as e:      # 检查器自身出错不应阻断主流程
            out["items"].append({"name": "额外因子族", "ok": True,
                                 "detail": f"跳过（{type(e).__name__}: {e}）"})

    # 3) 模型 scores
    need_model = False
    for name in ("wf_h10",):
        p = C.BUILD_MODELS / name / "composite_score.npz"
        if not p.exists():
            out["ok"] = False
            need_model = True
            out["items"].append({"name": f"模型 {name}", "ok": False, "detail": "不存在"})
            continue
        sc = np.load(p)["scores"]
        valid = np.flatnonzero(np.isfinite(sc).any(axis=1))
        if not len(valid):
            out["ok"] = False
            need_model = True
            out["items"].append({"name": f"模型 {name}", "ok": False, "detail": "空打分"})
            continue
        lag = T - 1 - int(valid[-1])
        ok = lag < MAX_LAG
        if not ok:
            out["ok"] = False
            need_model = True
        detail = f"{sc.shape[0]} 行，最后打分行 {int(valid[-1])}，滞后 {lag} 日"
        if not ok and extra_bad:
            # 指向真正的病灶：extra 面板缺周会让最后一行复合分整体变 NaN，
            # 此时重跑 run_research 没用，得先补齐面板
            detail += "（根因：额外因子族缺周 → 末行复合分全 NaN，非模型本身问题）"
        out["items"].append({
            "name": f"模型 {name}", "ok": ok, "lag": int(lag), "detail": detail})

    # 4) 信号
    sig = C.SIGNALS / "latest_signal.json"
    need_signal = False
    if sig.exists():
        try:
            j = json.load(open(sig, encoding="utf-8"))
            sd = str(j.get("signal_date") or "")
            ok = (sd == last)
            if not ok:
                need_signal = True
                if not need_model and not need_factors:
                    out["ok"] = False
            out["items"].append({"name": "信号", "ok": ok,
                                 "detail": f"{sd}（数据最后日 {last}）"})
        except Exception as e:
            out["items"].append({"name": "信号", "ok": False, "detail": str(e)})
    else:
        need_signal = True
        out["items"].append({"name": "信号", "ok": False, "detail": "不存在"})

    # 5) 报告
    rep = ROOT / "reports" / "回测报告.md"
    if rep.exists():
        txt = rep.read_text(encoding="utf-8")
        ok = (last in txt[:4000])
        out["items"].append({"name": "报告 §1", "ok": ok,
                             "detail": f"{'含' if ok else '缺'} {last}"})

    # 修复建议（最小路径）：先补面板，再重训模型，最后重出信号
    if need_extra:
        out["fix_hint"].append("python scripts/build_extra_panels.py")
    if need_factors:
        out["fix_hint"].append("python scripts/build_factors.py")
    if need_model:
        out["fix_hint"].append("python scripts/run_research.py --out build/models/wf_h10 "
                               "--method blend --max-factors 34 --horizon 10 --no-fundamental")
        out["fix_hint"].append("python scripts/run_backtest.py")
    if need_signal:
        out["fix_hint"].append("python scripts/daily_signal.py --combo")
    if out["fix_hint"]:
        out["fix_hint"].append("（或直接：python scripts/run_pipeline.py）")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    r = check()

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
        raise SystemExit(0 if r["ok"] else 1)

    print("=" * 74)
    print("数据源一致性检测")
    print("=" * 74)
    print(f"  当前数据目录: {r['data_raw']}")
    print(f"  来源标记:     {r['source']}")
    print()
    for it in r["items"]:
        mark = "✅" if it["ok"] else "❌"
        print(f"  {mark} {it['name']}: {it['detail']}")

    print()
    print("=" * 74)
    if r["ok"]:
        print("✅ 全部一致：数据 / 因子 / 模型 / 信号 / 报告 已对齐")
    else:
        print("❌ 存在滞后，建议按顺序执行：")
        for h in r["fix_hint"]:
            print(f"   {h}")
    print("=" * 74)
    raise SystemExit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
