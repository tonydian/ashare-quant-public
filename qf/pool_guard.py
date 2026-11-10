# -*- coding: utf-8 -*-
"""因子池缓冲机制（防抖动）——挤占式设计。

问题（实测）：
  因子池尾部因子 |ICIR| 只差 0.05（0.384~0.430），
  数据微变（多几日）就让排名尾部换人 → CAGR 波动约 0.9pp。

设计（挤占式，修 v1 的"僵化"缺陷）：
  1) 应然池 = 排名前 k
  2) 对每个因子跟踪两个 streak：
       in_streak  想进（在应然池、不在当前池）连续次数
       out_streak 想踢（不该在池）连续次数
  3) 只有 streak >= need_streak 才执行变更
  4) **极强因子**（排名 <= k*0.5）立即纳入，不受 streak 限制
  5) 最终池大小严格 == k

注意：v1 只在"有空位"时加新因子，导致真正变强的因子进不来（僵化）。
      v2 改为"挤占"：新因子达标就挤掉池内排名最差的。
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

ENTRY_RANK_RATIO = 0.75
EXIT_RANK_RATIO = 1.25
NEED_STREAK = 2
STRONG_ENTER_RATIO = 0.5


def apply_guard(
    prev_pool: list[str] | None,
    ranked: list[str],
    k: int,
    state: dict[str, dict[str, int]] | None = None,
    entry_rank: int | None = None,
    exit_rank: int | None = None,
    need_streak: int = NEED_STREAK,
    strong_enter: int | None = None,
    verbose: bool = True,
) -> tuple[list[str], dict[str, dict[str, int]], dict[str, Any]]:
    """挤占式缓冲。返回 (新池, 新状态, 诊断)。"""
    entry_rank = entry_rank or max(1, int(round(k * ENTRY_RANK_RATIO)))
    exit_rank = exit_rank or max(k + 1, int(round(k * EXIT_RANK_RATIO)))
    strong_enter = strong_enter or max(1, int(round(k * STRONG_ENTER_RATIO)))

    state = {kk: dict(vv) for kk, vv in (state or {}).items()}
    ranks = {key: i + 1 for i, key in enumerate(ranked)}

    def st(key: str) -> dict[str, int]:
        return state.setdefault(key, {"in_streak": 0, "out_streak": 0})

    # ---------- 首次构建
    if not prev_pool:
        pool = list(ranked[:k])
        for key in ranked[:k]:
            s = st(key)
            s["in_streak"], s["out_streak"] = need_streak, 0
        info = {"mode": "initial", "kept": [], "added": pool, "removed": [],
                "turnover": 0, "pool": pool}
        if verbose:
            print(f"[pool-guard] 首次构建：取前 {k} 名")
        return pool, state, info

    prev = list(prev_pool)
    prev_set = set(prev)
    ideal = set(ranked[:k])

    # ---------- Step 1：更新 streak（遍历 prev + 前 exit_rank 名）
    watch = list(dict.fromkeys(prev + ranked[: max(exit_rank, k) + 5]))
    for key in watch:
        s = st(key)
        in_prev = key in prev_set
        in_ideal = key in ideal
        rank = ranks.get(key, 10 ** 6)
        if in_prev and in_ideal:
            s["in_streak"], s["out_streak"] = need_streak, 0
        elif in_prev:
            s["in_streak"] = 0
            s["out_streak"] = (int(s.get("out_streak", 0)) + 1
                               if rank > exit_rank else 0)
        elif in_ideal:
            s["out_streak"] = 0
            s["in_streak"] = int(s.get("in_streak", 0)) + 1
        else:
            s["in_streak"], s["out_streak"] = 0, 0

    # ---------- Step 2：候选池 = prev 中"仍值得留"的
    keep: list[str] = []
    kick_candidates: list[str] = []      # 想踢但还没到次数
    for key in prev:
        s = st(key)
        rank = ranks.get(key, 10 ** 6)
        if key in ideal:
            keep.append(key)             # 仍在应然池
        elif rank > exit_rank and s["out_streak"] >= need_streak:
            continue                     # 确认踢出
        else:
            keep.append(key)             # 缓冲期内继续留
            kick_candidates.append(key)

    # ---------- Step 3：挤占（新因子达标时挤掉池内排名最差的）
    # 关键：先处理"极强因子"和"已确认该进"的因子，允许挤掉池内弱者
    keep_set = set(keep)
    outsiders = [key for key in ranked if key not in keep_set]
    for key in outsiders:
        s = st(key)
        rank = ranks[key]
        # 达标判定
        qualified = (rank <= strong_enter) or \
                    (s["in_streak"] >= need_streak and rank <= max(entry_rank, k))
        if not qualified:
            continue
        if len(keep) < k:
            keep.append(key)
            keep_set.add(key)
            s["in_streak"] = need_streak
            continue
        # 池已满 → 找一个"最该被挤掉"的（排名最差且在缓冲期的）
        weakest, weakest_rank = None, -1
        for cand in keep:
            cr = ranks.get(cand, 10 ** 6)
            if cr > weakest_rank:
                weakest, weakest_rank = cand, cr
        # 只有当新因子明显更优（排名更前）时才挤占
        if weakest is not None and rank < weakest_rank:
            keep.remove(weakest)
            keep_set.discard(weakest)
            keep.append(key)
            keep_set.add(key)
            s["in_streak"] = need_streak
            st(weakest)["in_streak"] = 0

    # ---------- Step 4：仍不足 → 按排名强制补足
    if len(keep) < k:
        for key in ranked:
            if len(keep) >= k:
                break
            if key not in set(keep):
                keep.append(key)
                st(key)["in_streak"] = need_streak

    # ---------- Step 5：超出 → 按排名截断（保留排名更好的）
    if len(keep) > k:
        keep.sort(key=lambda x: ranks.get(x, 10 ** 6))
        keep = keep[:k]

    added = [x for x in keep if x not in prev_set]
    removed = [x for x in prev if x not in set(keep)]
    info = {
        "mode": "guard",
        "entry_rank": entry_rank, "exit_rank": exit_rank,
        "strong_enter": strong_enter, "need_streak": need_streak,
        "kept": [x for x in prev if x in set(keep)],
        "added": added, "removed": removed,
        "turnover": len(added) + len(removed),
        "pool": keep,
    }
    if verbose:
        print(f"[pool-guard] 保留 {len(info['kept'])} | 新增 {len(added)} "
              f"| 移除 {len(removed)} | 换手 {info['turnover']}")
        if added:
            print(f"  + {added}")
        if removed:
            print(f"  - {removed}")
    return keep, state, info


# ---------------------------------------------------------------- 持久化
STATE_FILE = "build/models/_pool_guard_state.json"


def load_state(root: pathlib.Path) -> dict[str, dict[str, int]]:
    p = pathlib.Path(root) / STATE_FILE
    if not p.exists():
        return {}
    try:
        return (json.load(open(p, encoding="utf-8")).get("state") or {})
    except Exception:
        return {}


def save_state(root: pathlib.Path, state: dict, meta: dict | None = None) -> None:
    p = pathlib.Path(root) / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if p.exists():
        try:
            old = json.load(open(p, encoding="utf-8"))
        except Exception:
            old = {}
    hist = list(old.get("history") or [])
    if meta:
        hist.append(meta)
        hist = hist[-20:]
    p.write_text(json.dumps({"state": state, "history": hist},
                            ensure_ascii=False, indent=1), encoding="utf-8")


def prev_pool_from_meta(model_dir: pathlib.Path) -> list[str] | None:
    mp = pathlib.Path(model_dir) / "model_meta.json"
    if not mp.exists():
        return None
    try:
        d = json.load(open(mp, encoding="utf-8"))
        return list(d["pool"]) if d.get("pool") else None
    except Exception:
        return None
