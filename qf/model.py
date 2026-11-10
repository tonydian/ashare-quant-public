# -*- coding: utf-8 -*-
"""因子选择 / 权重估计 / 复合打分（walk-forward）。

方法论（与 docs/ARCH.md 铁律一致）：
1. 候选池：只用开发期（2017-01-03 ~ 2019-12-31）的周频 ICIR + 覆盖率 + 相关性剪枝
   + 因子族配额选出 K 个因子（固定池，避免全样本挑选）；
2. 权重：从验证期起点（2020-01）开始，每 13 周用**截至当时的滚动 3 年**周频数据
   重新估计（ICIR 加权 + 岭回归 50/50 混合），标签只用到 refit 时点已知的周；
3. 复合打分：Σ_k w_k · 中性化横截面 z-score(factor_k)，再对复合分做行业+市值中性化；
4. 输出 (T,N) 打分矩阵（仅周频行有值），供 qf.backtest.run_backtest 使用。

防未来函数要点：
- 特征 = t 日收盘可得的因子值，横截面排名/中性化只用 t 日截面信息；
- 权重只用 refit 时点之前已实现标签的周（滞后 2 周）；
- 候选池只用 DEV 期（<2020）信息。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import ops


# ------------------------------------------------------------------ 配置
@dataclass
class ModelConfig:
    horizon: int = C.FWD_HORIZON
    train_weeks: int = 156          # 3 年
    refit_weeks: int = 12           # 季度（月份季度起点 + 最小间隔 12 周，避免季度被跳过）
    max_factors: int = C.MAX_FACTORS
    corr_max: float = 0.70
    max_per_category: int = 4
    min_coverage: float = 0.90
    min_ic_obs: int = 80
    neutralize: bool = True
    method: str = "blend"           # icir / ridge / blend
    ridge_alpha_rel: float = 1.0e-3
    # ---- 两个「死参数」：当前 21 池下从不绑定，改它们结果逐位相同（2026-11-03 实测）
    # icir_clip=3.0：只在 |ICIR| 超过 3 时才截断，而池内 |ICIR| 最大仅 1.01（中位 0.88），
    #   改成 1.5 / 5.0 回测结果完全一致。保留是为了极端行情下的保护，不要当成可调参数。
    # weight_floor=0.05：权重绝对值 < max|w|×0.05 才置 0，实测归零因子数均值 = 0，
    #   从不生效。真要控因子数请调 max_factors，不要动这个。
    icir_clip: float = 3.0
    weight_floor: float = 0.05      # 权重绝对值 < max*floor 置 0
    # ---- 族预算（Family Budget）：按「族」分配权重预算，族间等分、族内按原比例
    # 默认关闭。开启后每个族拿 1/F 的总 |权重|，再按族内原始权重比例分给因子。
    # 目的：防止低波/换手这类因子家族靠数量优势吃掉大半权重（21 池里 risk-pattern
    #       一个族曾占 35%）。代价是 ICIR/岭回归的横截面最优性被牺牲一部分。
    family_budget: bool = False
    family_strength: float = 1.0    # 0=不干预，1=完全族等权（线性插值）
    # ---- 族上限（Family Cap）：只在该族超过上限时踩刹车，未越界不干预。
    # 与族预算的区别：族预算把 6 个低波因子的信息压成 1/6 份额；族上限不改变族内结构，
    # 只在越界时把超出部分按比例让给其他族。cap<=0 表示关闭。cap 必须 >= 1/F 才可行。
    family_cap: float = 0.0
    score_smooth_alpha: float = 0.6  # 复合分时序平滑：α·当期 + (1-α)·上期（只用过去信息）
    universe: str | None = None     # 可选选股域过滤：above_ma60 / above_ma60_slope / above_ma200
    fdr_q: float | None = None      # BH-FDR 控制水平（None=不启用；0.05 推荐）：多因子筛选假阳性控制
    # ---- 因子池缓冲（防抖动）：见 qf/pool_guard.py
    pool_guard: bool = False        # 启用"连续确认 + 双阈值"缓冲
    pool_prev: list | None = None   # 上一次的因子池（None=首次构建）
    pool_state: dict = field(default_factory=dict)  # 跨次 streak 状态
    pool_need_streak: int = 2       # 连续几次重估才动作
    pool_entry_rank: int | None = None   # 新因子入场线（None=按 k*0.75）
    pool_exit_rank: int | None = None    # 旧因子退出线（None=按 k*1.25）
    # ---- 选池第三判据：DEV→OOS 不衰减（2026-11-03 加，默认关闭）
    # 只按 DEV |ICIR| 排序选池，会选中「越往后越弱」的因子：ep 0.0436→0.0221、
    # ROE 0.0467→0.0072；而 bp 是唯一 OOS 反超 DEV 的（0.0231→0.0368）。
    # 价值维度那轮已证明这条判据能救 bp，这里把它变成选池的正式约束。
    # 注意：这是**选池口径**改动，会同时改变池成员与权重，必须重跑 8 相位。
    require_oos_nondecay: bool = False
    oos_decay_tol: float = 0.0       # 允许衰减比例：|OOS| >= |DEV|×(1-tol)，0=严格不衰减
    # 衰减检验窗口。**必须落在 DEV 内（<2020）**，否则等于用回测窗口挑因子（前视）。
    # 默认 2019 全年：基线窗 = DEV 中早于 oos_window_start 的部分（2017-2018）。
    oos_window_start: str | None = "2019-01-01"
    oos_window_end: str | None = "2019-12-31"

# ModelConfig 的声明字段（供 load_result 过滤旧版的临时属性）
_MODEL_CFG_FIELDS = set(ModelConfig.__dataclass_fields__.keys())



# ------------------------------------------------------------------ 横截面工具
def cs_rank_rows(panel: np.ndarray, rows: np.ndarray, mask: np.ndarray | None = None,
                 clip: float = 5.0) -> np.ndarray:
    """对 panel (T,N) 的指定行做横截面 rank → [-0.5,0.5]；无效/未选 mask 为 NaN。

    mask 支持两种形状：与 panel 同形的 (T,N)（用 rows 索引），或与 rows 对齐的
    (len(rows), N)（用局部下标索引）。
    """
    n = len(rows)
    out = np.full((n, panel.shape[1]), np.nan, dtype=np.float32)
    mask_rowwise = mask is not None and mask.shape[0] == n
    for i, t in enumerate(rows):
        x = panel[t]
        valid = np.isfinite(x)
        if mask is not None:
            valid &= mask[i] if mask_rowwise else mask[t]
        cnt = int(valid.sum())
        if cnt < 10:
            continue
        idx = np.flatnonzero(valid)
        order = np.argsort(x[idx], kind="stable")
        r = np.empty(cnt, dtype=np.float64)
        r[order] = np.arange(cnt, dtype=np.float64)
        out[i, idx] = (r / (cnt - 1) - 0.5).astype(np.float32)
    return out


def neutralize_rows(X: np.ndarray, industry: np.ndarray, size: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    """逐行做 行业均值 + log 市值 线性中性化；返回残差（NaN 保持）。"""
    n, N = X.shape
    out = np.full_like(X, np.nan)
    for i in range(n):
        m = mask[i] & np.isfinite(X[i])
        if m.sum() < 20:
            continue
        f = X[i].astype(np.float64).copy()
        ind = industry[i]
        sz = size[i]
        fm = f[m]
        # 行业去均值
        codes = ind[m]
        order = np.argsort(codes, kind="stable")
        cs = codes[order]
        starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        ends = np.r_[starts[1:], len(cs)]
        fclean = f.copy()
        vals = fm[order]
        for s, e in zip(starts, ends):
            sel = order[s:e]
            fclean[np.flatnonzero(m)[sel]] -= vals[s:e].mean()
        # 市值回归（在选股域内）
        s_ok = m & np.isfinite(sz)
        if s_ok.sum() > 30:
            ss = sz[s_ok].astype(np.float64)
            ss = ss - ss.mean()
            ff = fclean[s_ok]
            denom = float(np.dot(ss, ss))
            if denom > 1e-12:
                beta = float(np.dot(ss, ff)) / denom
                fclean[s_ok] -= beta * ss
        out[i] = fclean.astype(np.float32)
    return out


def zscore_rows(X: np.ndarray, mask: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """逐行 z-score（选股域内），NaN/域外为 NaN。"""
    out = np.full_like(X, np.nan)
    valid = mask & np.isfinite(X)
    n = valid.sum(axis=1).astype(np.float64)
    x = np.where(valid, X, 0.0).astype(np.float64)
    mean = x.sum(axis=1) / np.maximum(n, 1)
    var = (np.where(valid, (x - mean[:, None]) ** 2, 0.0)).sum(axis=1) / np.maximum(n - 1, 1)
    sd = np.sqrt(np.maximum(var, 1e-12))
    z = np.where(valid, (x - mean[:, None]) / sd[:, None], np.nan)
    return np.clip(z, -clip, clip).astype(np.float32)


def build_feature_block(md, rows: np.ndarray, panels: dict, key: str,
                        mask: np.ndarray) -> np.ndarray:
    """单个因子 → (n_rows, N) 的中性化 z-score 特征。"""
    raw = panels[key]  # (n_rows, N)，与 rows 对齐
    r = cs_rank_rows(raw, np.arange(len(rows)), mask=mask)
    if np.isfinite(r).sum() == 0:
        return r
    if ModelConfig.neutralize:
        r = neutralize_rows(r, md.industry[rows], md.logmcap[rows], mask)
    r = zscore_rows(r, mask)
    return r


# ------------------------------------------------------------------ 候选池
def weekly_ic(feat: np.ndarray, label: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐周 Rank IC（feat/label 已是横截面 rank 或连续值）。"""
    valid = mask & np.isfinite(feat) & np.isfinite(label)
    n = valid.sum(axis=1).astype(np.float64)
    f = np.where(valid, feat, 0.0).astype(np.float64)
    y = np.where(valid, label, 0.0).astype(np.float64)
    mf = f.sum(axis=1) / np.maximum(n, 1)
    my = y.sum(axis=1) / np.maximum(n, 1)
    cov = (np.where(valid, (f - mf[:, None]) * (y - my[:, None]), 0.0)).sum(axis=1) / np.maximum(n, 1)
    vf = (np.where(valid, (f - mf[:, None]) ** 2, 0.0)).sum(axis=1) / np.maximum(n, 1)
    vy = (np.where(valid, (y - my[:, None]) ** 2, 0.0)).sum(axis=1) / np.maximum(n, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = cov / np.sqrt(vf * vy)
    ic = np.where((n >= 100) & np.isfinite(ic), ic, np.nan)
    return ic


def dev_metrics(feat: np.ndarray | None, label: np.ndarray, mask: np.ndarray) -> dict:
    ic = weekly_ic(feat, label, mask) if feat is not None else np.array([])
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"ic": 0.0, "icir": 0.0, "t": 0.0, "pos": 0.0, "n": 0}
    m, s = float(np.mean(ic)), float(np.std(ic))
    return {"ic": m, "icir": m / s if s > 1e-12 else 0.0,
            "t": m / s * np.sqrt(len(ic)) if s > 1e-12 else 0.0,
            "pos": float(np.mean(ic > 0)), "n": int(len(ic))}


def select_pool(md, rows_dev: np.ndarray, panels_dev: dict, label_dev: np.ndarray,
                mask_dev: np.ndarray, cfg: ModelConfig, catalog: pd.DataFrame,
                verbose: bool = True,
                oos: tuple | None = None) -> tuple[list[str], pd.DataFrame]:
    """开发期候选池选择：|ICIR| 排序 + 相关性剪枝 + 族配额。

    `oos=(rows_oos, panels_oos, label_oos, mask_oos)` 可选：当
    `cfg.require_oos_nondecay=True` 时，额外要求「|OOS ICIR| ≥ |DEV ICIR|×(1-tol)」，
    把「DEV 排名高但样本外衰减」的因子挡在池外。
    """
    keys = list(panels_dev.keys())
    feats = {}
    metrics = []
    for k in keys:
        f = build_feature_block(md, rows_dev, panels_dev, k, mask_dev)
        met = dev_metrics(f, label_dev, mask_dev)
        met["factor_key"] = k
        metrics.append(met)
        feats[k] = f
    metdf = pd.DataFrame(metrics).set_index("factor_key")
    metdf["abs_icir"] = metdf["icir"].abs()
    cat = catalog.set_index("factor_key")
    metdf["category"] = cat["category_en"].reindex(metdf.index)
    metdf["lib"] = cat["lib"].reindex(metdf.index)
    metdf["base"] = cat["base"].reindex(metdf.index)
    metdf = metdf.sort_values("abs_icir", ascending=False)

    # ---- BH-FDR：控制"800 因子筛 34 个"的多重检验假阳性（cfg.fdr_q=None 时不启用，保持向后兼容）
    if getattr(cfg, "fdr_q", None) is not None:
        from .stats_adjust import bh_fdr, p_value_two_sided
        n_obs = metdf["n"].to_numpy(dtype=np.float64)
        icir = metdf["icir"].to_numpy(dtype=np.float64)
        tstat = icir * np.sqrt(np.maximum(n_obs, 1.0))
        pvals = np.array([p_value_two_sided(t) for t in tstat])
        fdr = bh_fdr(pvals, q=float(cfg.fdr_q))
        n_before = int((metdf["n"] >= cfg.min_ic_obs).sum())
        metdf = metdf.loc[fdr["reject"]]
        if verbose:
            print(f"[pool] BH-FDR(q={cfg.fdr_q}) 剔除 {n_before - len(metdf)} 个不显著因子，"
                  f"保留 {len(metdf)} 个（原始 {n_before}）")
        if metdf.empty:
            raise ValueError("BH-FDR 后无因子通过，请放宽 fdr_q 或检查 DEV 期标签")

    # ---- 第三判据：DEV→OOS 不衰减（cfg.require_oos_nondecay，默认关闭）
    # 放在相关性剪枝之前：先做质量筛，再在通过筛的因子里做去冗余，避免
    # 「一个衰减因子挤掉一个不衰减因子」的顺序性偏误。
    if getattr(cfg, "require_oos_nondecay", False):
        if not oos:
            raise ValueError("require_oos_nondecay=True 但未传入衰减检验窗口")
        rows_o, panels_o, label_o, mask_o = oos["oos"]
        rows_b, panels_b, label_b, mask_b = oos["base"]
        oos_icir: dict[str, float] = {}
        base_icir: dict[str, float] = {}
        for k in keys:
            if k in panels_o:
                m = dev_metrics(build_feature_block(md, rows_o, panels_o, k, mask_o),
                                label_o, mask_o)
                oos_icir[k] = float(m.get("icir") or 0.0)
            if k in panels_b:
                m = dev_metrics(build_feature_block(md, rows_b, panels_b, k, mask_b),
                                label_b, mask_b)
                base_icir[k] = float(m.get("icir") or 0.0)
        metdf["oos_icir"] = metdf.index.map(lambda x: oos_icir.get(str(x), 0.0))
        metdf["oos_base_icir"] = metdf.index.map(lambda x: base_icir.get(str(x), 0.0))
        metdf["abs_oos_icir"] = metdf["oos_icir"].abs()
        metdf["abs_oos_base_icir"] = metdf["oos_base_icir"].abs()
        tol = 1.0 - float(getattr(cfg, "oos_decay_tol", 0.0) or 0.0)
        keep = metdf["abs_oos_icir"] >= metdf["abs_oos_base_icir"] * tol
        n_before = int(len(metdf))
        n_pass = int(keep.sum())
        metdf = metdf.loc[keep]
        if verbose:
            print(f"[pool] OOS 不衰减（{cfg.oos_window_start}~{cfg.oos_window_end} vs "
                  f"DEV 前半段）：{n_before} → {n_pass} 个，剔除 {n_before - n_pass}"
                  f"（容差 {getattr(cfg, 'oos_decay_tol', 0.0):.0%}）")
        if metdf.empty:
            raise ValueError("OOS 不衰减判据后无因子通过，请放宽 oos_decay_tol")

    # 逐周横截面相关矩阵（只算强候选，控制计算量）
    cand = metdf[(metdf["n"] >= cfg.min_ic_obs)].index.tolist()
    cand = cand[: min(120, len(cand))]
    corr = None
    if len(cand) >= 2:
        # 分块计算：原先一次性 stack (K, n_dev, N)，120 个候选在 16GB 机器上要
        # 同时持有 F/valid/Fz 三份 ~750MB 临时数组并触发 OOM。改为按候选分块，
        # 峰值降到 ~150MB。
        #
        # 归一化口径（2026-11-03 修正）：特征已是「逐周横截面 z-score」，所以
        # Σ_周 Σ_股 z_i·z_j / Σ_周 |有效交集| 才是相关系数。旧实现写的是
        # G / (mean_n_i · mean_n_j)，少除了周数 T、又多除了一个 n，把 |ρ|=1
        # 的镜像对算成 0.008 量级 —— corr_max=0.70 永远不可能触发，相关性剪枝
        # 实际上从未生效（实测 R737/R738 ρ=-1.000 被算成 -0.079，池内 60 对
        # |ρ|>0.7 全部漏过）。
        def _blk(keys: list[str]):
            A = np.stack([feats[k] for k in keys], axis=0)
            v = mask_dev[None, :, :] & np.isfinite(A)
            A[~v] = 0.0
            return A, v

        K = len(cand)
        blk = 24
        G = np.zeros((K, K), dtype=np.float64)   # Σ 周 Σ 股 z_i·z_j
        J = np.zeros((K, K), dtype=np.float64)   # Σ 周 |有效交集|
        for i0 in range(0, K, blk):
            i1 = min(i0 + blk, K)
            Fi, Vi = _blk(cand[i0:i1])
            Vf = Vi.astype(np.float32)
            for j0 in range(0, i0 + blk, blk):
                j1 = min(j0 + blk, K)
                if j0 == i0:
                    Fj, Vj = Fi, Vi
                    Vjf = Vf
                else:
                    Fj, Vj = _blk(cand[j0:j1])
                    Vjf = Vj.astype(np.float32)
                G[i0:i1, j0:j1] = np.einsum("kwn,lwn->kl", Fi, Fj, optimize=True)
                J[i0:i1, j0:j1] = np.einsum("kwn,lwn->kl", Vf, Vjf, optimize=True)
                if j0 != i0:
                    G[j0:j1, i0:i1] = G[i0:i1, j0:j1].T
                    J[j0:j1, i0:i1] = J[i0:i1, j0:j1].T
        corr = np.clip(G / np.maximum(J, 1.0), -1, 1)
        np.fill_diagonal(corr, 1.0)
        feats.clear()          # 后续不再需要，及早释放 ~680MB

    _pos = {k: i for i, k in enumerate(cand)} if corr is not None else {}

    def _corr_ok(k: str, chosen: list[str]) -> bool:
        """k 与已入选因子的相关性是否全部在 corr_max 内。

        不在相关矩阵里（排序跌出前 120，没参与计算）时视为「无法判定」→ 放行，
        保持与原行为一致；只有能算的才拦。
        """
        if corr is None or not chosen or k not in _pos:
            return True
        js = [_pos[c] for c in chosen if c in _pos]
        if not js:
            return True
        return float(np.max(np.abs(corr[_pos[k], js]))) <= float(cfg.corr_max)

    selected = []
    cat_count: dict[str, int] = {}
    for k in cand:
        category = str(metdf.loc[k, "category"])
        if cat_count.get(category, 0) >= cfg.max_per_category:
            continue
        if not _corr_ok(k, selected):
            continue
        selected.append(k)
        cat_count[category] = cat_count.get(category, 0) + 1
        if len(selected) >= cfg.max_factors:
            break
    if verbose:
        print(f"[pool] 候选 {len(cand)}，选中 {len(selected)}；类别分布: {cat_count}")

    # ---- 因子池缓冲（防抖动）：只在 cfg.pool_guard=True 时启用
    guard_info = None
    if getattr(cfg, "pool_guard", False) and selected:
        from . import pool_guard as _pg
        # ranked = 本次"无缓冲时"的选池结果 + 其余候选（按 |ICIR| 排序）
        # 关键：selected 是经过了相关性剪枝与类别配额的结果，代表本次的"应然池"
        #       其余候选按 |ICIR| 排在后面，供挤占时参考
        _rest = [x for x in metdf.index if x not in set(selected)]
        ranked = list(selected) + _rest
        # 目标池大小：优先沿用上次池的大小（保持稳定），否则用 max_factors
        _prev = list(getattr(cfg, "pool_prev", None) or [])
        _k = len(_prev) if _prev else int(cfg.max_factors)
        _k = min(_k, int(cfg.max_factors))
        new_pool, _state_out, guard_info = _pg.apply_guard(
            prev_pool=_prev or None,
            ranked=ranked,
            k=_k,
            state=getattr(cfg, "pool_state", None) or {},
            entry_rank=getattr(cfg, "pool_entry_rank", None),
            exit_rank=getattr(cfg, "pool_exit_rank", None),
            need_streak=int(getattr(cfg, "pool_need_streak", 2)),
            verbose=verbose,
        )
        # 只保留能在 metdf 里取到指标的因子
        new_pool = [x for x in new_pool if x in metdf.index]
        if new_pool:
            # ---- 重新施加类别配额与相关性约束
            # 缓冲的「保留上次池」与「补足规模」两条路径原先都不过相关剪枝，导致
            # ρ≈±1 的镜像对可以长期占两个槽（R473/R474、R737/R738 都是这么进来的）。
            # 这里统一重施一次；若候选全撞相关，先放宽相关约束保规模（有兜底）。
            _cat_of = {x: str(metdf.loc[x, "category"]) for x in metdf.index}
            _rank_idx = {x: i for i, x in enumerate(ranked)}
            _max_per = int(cfg.max_per_category)
            _cnt: dict[str, int] = {}
            _final: list[str] = []
            _dropped: list[str] = []
            _corr_dropped: list[str] = []
            for x in sorted(new_pool, key=lambda z: _rank_idx.get(z, 10 ** 6)):
                c = _cat_of.get(x, "?")
                if _cnt.get(c, 0) >= _max_per:
                    _dropped.append(x)
                    continue
                if not _corr_ok(x, _final):
                    _corr_dropped.append(x)
                    continue
                _final.append(x)
                _cnt[c] = _cnt.get(c, 0) + 1
            # 补足到目标大小（从 ranked 里挑未超配额、且不撞相关的）
            if len(_final) < _k:
                for x in ranked:
                    if len(_final) >= _k:
                        break
                    if x in _final or x not in metdf.index:
                        continue
                    c = _cat_of.get(x, "?")
                    if _cnt.get(c, 0) >= _max_per:
                        continue
                    if not _corr_ok(x, _final):
                        continue
                    _final.append(x)
                    _cnt[c] = _cnt.get(c, 0) + 1
            # 兜底：候选全撞相关时，宁可放宽相关约束也不让池子缩水
            if len(_final) < _k and _corr_dropped:
                _n_fill = _k - len(_final)
                _recover: list[str] = []
                for x in _corr_dropped:
                    if len(_recover) >= _n_fill:
                        break
                    c = _cat_of.get(x, "?")
                    if _cnt.get(c, 0) >= _max_per:
                        continue
                    _recover.append(x)
                    _cnt[c] = _cnt.get(c, 0) + 1
                _final.extend(_recover)
                _corr_dropped = [x for x in _corr_dropped if x not in set(_recover)]
            if _dropped and verbose:
                print(f"[pool-guard] 类别配额回补：移除 {len(_dropped)} 个超配额因子")
                print(f"  超配额被移除: {_dropped}")
            if _corr_dropped and verbose:
                print(f"[pool-guard] 相关性剪枝：移除 {len(_corr_dropped)} 个与已入选 "
                      f"|ρ|>{cfg.corr_max} 的因子")
                print(f"  相关性被移除: {_corr_dropped}")
            if _final:
                new_pool = _final
                guard_info["category_fixed"] = _dropped
                guard_info["corr_fixed"] = _corr_dropped
                guard_info["cat_counts"] = _cnt
        if new_pool:
            selected = new_pool
        # 把状态挂到 cfg 上，供调用方持久化
        try:
            cfg.pool_state_out = _state_out
        except Exception:
            pass

    pool_meta = metdf.loc[selected].copy()
    if guard_info is not None:
        try:
            pool_meta.attrs["guard"] = guard_info
        except Exception:
            pass
    return selected, pool_meta


# ------------------------------------------------------------------ 权重
def fit_weights_icir(F: np.ndarray, y: np.ndarray, mask: np.ndarray,
                     cfg: ModelConfig, strict: np.ndarray | None = None) -> np.ndarray:
    """F: (K, n, N) 特征；y: (n,N)；返回 (K,) 权重。

    ``strict`` 为 True 的因子（原有量价/动量因子）保持 NaN 语义；基本面因子允许缺失，
    缺失按 0（中性）处理，避免因财报缺失把股票整行排除。
    """
    if strict is not None and (~strict).any():
        F = F.copy()
        F[~strict] = np.where(np.isfinite(F[~strict]), F[~strict], 0.0)
    K = F.shape[0]
    ic = np.stack([weekly_ic(F[k], y, mask) for k in range(K)], axis=0)  # (K, n)
    m = np.nanmean(ic, axis=1)
    s = np.nanstd(ic, axis=1)
    w = np.where(s > 1e-12, m / s, 0.0)
    w = np.clip(w, -cfg.icir_clip, cfg.icir_clip)
    return w


def fit_weights_ridge(F: np.ndarray, y: np.ndarray, mask: np.ndarray,
                      cfg: ModelConfig, strict: np.ndarray | None = None) -> np.ndarray:
    K, n, N = F.shape
    if strict is None:
        # 与 v1 完全一致：valid 为 (K,n,N)，要求所有因子在样本上都有限
        valid = mask & np.isfinite(y)[None, :, :] & np.isfinite(F)
    else:
        # 非严格因子（基本面）缺失填 0；严格因子保持 NaN 并参与 valid 掩码
        F = F.copy()
        if (~strict).any():
            F[~strict] = np.where(np.isfinite(F[~strict]), F[~strict], 0.0)
        base = mask & np.isfinite(y)[None, :, :]
        valid = np.broadcast_to(base, (K, n, N)).copy()
        for k in range(K):
            if strict[k]:
                valid[k] &= np.isfinite(F[k])
    # 逐周横截面 demean 后堆叠
    Xs, ys = [], []
    for w_i in range(n):
        v = valid[:, w_i, :].T  # (N, K)
        keep = v.all(axis=1)
        if keep.sum() < 100:
            continue
        Xi = np.where(v, F[:, w_i, :].T, 0.0)
        Xs.append(Xi[keep])
        ys.append(y[w_i][keep])
    if not Xs:
        return np.zeros(K)
    X = np.concatenate(Xs, axis=0).astype(np.float64)
    Y = np.concatenate(ys, axis=0).astype(np.float64)
    Y = Y - Y.mean()
    XtX = X.T @ X
    Xty = X.T @ Y
    lam = cfg.ridge_alpha_rel * float(np.mean(np.diag(XtX))) * K
    A = XtX + lam * np.eye(K)
    try:
        w = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ Xty
    return w


def normalize_weights(w: np.ndarray, floor: float = 0.05) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).copy()
    mx = np.max(np.abs(w))
    if mx < 1e-12:
        return w
    w[np.abs(w) < floor * mx] = 0.0
    l1 = np.sum(np.abs(w))
    if l1 > 1e-12:
        w /= l1
    return w


def apply_family_budget(w: np.ndarray, families: list[str], strength: float = 1.0) -> np.ndarray:
    """族等预算：族间等分 1/F，族内保留原权重的相对比例。

    为什么需要：因子数量决定了「因子级权重」隐含的族暴露。risk-pattern 在 21 池里占
    6 个槽，即使每个因子权重不高，整族也能吃到 35%。族预算把预算单位从「因子」换成
    「族」，让每个信息维度拿到固定份额。

    族内权重全被 weight_floor 清零时，该族预算留空（不强行摊给零信息因子），
    最后按实际占比重新归一到 sum|w|=1。
    """
    if not families or len(set(families)) <= 1:
        return w
    w = np.asarray(w, dtype=np.float64)
    uniq = sorted(set(families))
    n_fam = len(uniq)
    out = np.zeros_like(w)
    skipped: list[str] = []
    for f in uniq:
        idx = np.array([i for i, x in enumerate(families) if x == f])
        seg = w[idx]
        s = float(np.sum(np.abs(seg)))
        if s < 1e-12:
            skipped.append(f)
            continue
        out[idx] = seg / s * (1.0 / n_fam)
    tot = float(np.sum(np.abs(out)))
    if tot < 1e-12:
        return w
    out /= tot
    if skipped:
        print(f"[family-budget] {len(skipped)} 个族权重全零、预算留空：{skipped}")
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength >= 1.0:
        return out
    return (1.0 - strength) * w + strength * out


def apply_family_cap(w: np.ndarray, families: list[str], cap: float = 0.0,
                     verbose: bool = False) -> np.ndarray:
    """族上限：只在该族 |权重| 之和超过 cap 时踩刹车，未越界的族一动不动。

    与 `apply_family_budget` 的关键区别：
      - 族预算把每族**拉平**到 1/F —— 越界的削、没越界的抬，等于把 6 个低波因子的
        合计信息量当成 1 个（实测相位均值 −0.93pp）。
      - 族上限只**削越界族的超出部分**，让出的预算按"未越界族的原占比"分配，
        族内相对结构完整保留。没越界时不产生任何改动。

    削峰用迭代水位法（钉住一族后剩余预算重新分配，可能把另一族顶过线，需再迭代）。
    不可行条件：cap × 有效族数 < 1.0 时装不下 1.0，退化为族等分。

    Parameters
    ----------
    cap : float  单族权重上限（0 < cap < 1）。<=0 表示关闭。
    """
    cap = float(cap)
    if not families or cap <= 0.0 or cap >= 1.0:
        return w
    uniq = sorted(set(families))
    if len(uniq) <= 1:
        return w
    w = np.asarray(w, dtype=np.float64)
    idx_of = {f: np.array([i for i, x in enumerate(families) if x == f]) for f in uniq}
    s = {f: float(np.sum(np.abs(w[idx_of[f]]))) for f in uniq}
    active = [f for f in uniq if s[f] > 1e-12]
    if len(active) < 2 or max(s[f] for f in active) <= cap + 1e-12:
        return w          # 没有族越界 → 原样返回（这是与族预算最重要的行为差异）

    if cap * len(active) < 1.0 - 1e-9:
        # cap 太小、装不下 1.0 —— 任何分配都会违反 cap，此时"不动"才是唯一诚实的做法
        # （曾误用族等分兜底：2 族 + cap=0.4 会给出各 0.5，反而破了上限）
        print(f"[family-cap] cap={cap:.2f} × {len(active)} 个有效族 < 1.0，约束不可行 → 不干预")
        return w

    alloc: dict[str, float] = {}
    free = list(active)
    for _ in range(len(active) + 1):
        rem = 1.0 - cap * len(alloc)
        base = sum(s[f] for f in free)
        if base < 1e-12:
            break
        trial = {f: rem * s[f] / base for f in free}
        viol = [f for f in free if trial[f] > cap + 1e-12]
        if not viol:
            alloc.update(trial)
            break
        for f in viol:
            alloc[f] = cap
        free = [f for f in free if f not in alloc]
        if not free:
            break

    out = np.zeros_like(w)
    for f in active:
        if f not in alloc:
            continue
        seg = w[idx_of[f]]
        tot = float(np.sum(np.abs(seg)))
        if tot > 1e-12:
            out[idx_of[f]] = seg / tot * alloc[f]
    l1 = float(np.sum(np.abs(out)))
    if l1 < 1e-12:
        return w
    out /= l1
    if verbose:
        capped = [f for f in active if abs(alloc.get(f, 0.0) - cap) < 1e-9]
        print(f"[family-cap] cap={cap:.2f} 削峰族={capped} 最大族={max(alloc.values()):.3f}")
    return out


def fit_weights(F: np.ndarray, y: np.ndarray, mask: np.ndarray, cfg: ModelConfig,
                verbose: bool = False, strict: np.ndarray | None = None,
                families: list[str] | None = None) -> np.ndarray:
    if cfg.method == "icir":
        w = normalize_weights(fit_weights_icir(F, y, mask, cfg, strict), cfg.weight_floor)
    elif cfg.method == "ridge":
        w = normalize_weights(fit_weights_ridge(F, y, mask, cfg, strict), cfg.weight_floor)
    else:
        a = normalize_weights(fit_weights_icir(F, y, mask, cfg, strict), cfg.weight_floor)
        b = normalize_weights(fit_weights_ridge(F, y, mask, cfg, strict), cfg.weight_floor)
        w = 0.5 * a + 0.5 * b
        w = normalize_weights(w, cfg.weight_floor)
    if getattr(cfg, "family_budget", False) and families is not None:
        w = apply_family_budget(w, families, float(getattr(cfg, "family_strength", 1.0)))
    cap = float(getattr(cfg, "family_cap", 0.0) or 0.0)
    if cap > 0 and families is not None:
        w = apply_family_cap(w, families, cap)
    if verbose:
        print(f"[weights] nonzero={int((np.abs(w)>0).sum())} sum|w|={np.abs(w).sum():.3f}")
    return w


# ------------------------------------------------------------------ 标签
def weekly_label(md, rows: np.ndarray, mask_rows: np.ndarray, horizon: int,
                 rank: bool = True) -> np.ndarray:
    lab = md.fwd_open_ret(horizon)[rows]
    # 标签要求 t+1 开盘可买（停牌/涨停/退市等不可执行样本从训练集中剔除，见 verifier P2#2）
    buy = np.zeros_like(mask_rows, dtype=bool)
    valid = rows + 1 < md.T
    buy[valid] = md.buyable[rows[valid] + 1]
    mask_rows = mask_rows & buy
    if rank:
        lab = cs_rank_rows(lab, np.arange(len(rows)), mask=mask_rows)
    return np.where(mask_rows, lab, np.nan).astype(np.float32)


# ------------------------------------------------------------------ walk-forward
@dataclass
class WalkForwardResult:
    rows_weekly: np.ndarray
    scores: np.ndarray                 # (T,N) 仅周频行有值
    weights: pd.DataFrame              # refit_date, factor_key, weight
    pool: list[str]
    pool_meta: pd.DataFrame
    composite_ic: pd.DataFrame         # 逐周 OOS IC
    cfg: ModelConfig = field(default_factory=ModelConfig)


def walk_forward(md, rows_weekly: np.ndarray, panels: dict, catalog: pd.DataFrame,
                 cfg: ModelConfig | None = None, start_date: str | None = None,
                 verbose: bool = True,
                 pool_override: list[str] | None = None) -> WalkForwardResult:
    cfg = cfg or ModelConfig()
    start_date = start_date or C.VALID_START
    n_weeks = len(rows_weekly)
    mask_weekly = md.eligible[rows_weekly]
    label = weekly_label(md, rows_weekly, mask_weekly, cfg.horizon, rank=True)

    # ---- 1) DEV 期选池（可用已有池覆盖，做稳健性对比时复用）
    dev_mask = md.dates[rows_weekly] <= pd.Timestamp(C.DEV_END)
    rows_dev = rows_weekly[dev_mask]
    panels_dev = {k: v[dev_mask] for k, v in panels.items()}
    if pool_override:
        pool = [k for k in pool_override if k in panels]
        # 用 DEV 期 metric 填充 pool_meta
        met = []
        for k in pool:
            f = build_feature_block(md, rows_dev, panels_dev, k, mask_weekly[dev_mask])
            m = dev_metrics(f, label[dev_mask], mask_weekly[dev_mask])
            m["factor_key"] = k
            met.append(m)
        pool_meta = pd.DataFrame(met).set_index("factor_key")
    else:
        # 第三判据需要 DEV 内的「衰减检验窗 + 基线窗」；只在开启时构造，避免常态多算特征
        oos_arg = None
        if getattr(cfg, "require_oos_nondecay", False):
            dw = md.dates[rows_weekly]
            ws = pd.Timestamp(cfg.oos_window_start or C.DEV_END)
            we = pd.Timestamp(cfg.oos_window_end) if cfg.oos_window_end else dw[-1]
            if ws > pd.Timestamp(C.DEV_END):
                raise ValueError(
                    f"衰减检验窗 {ws.date()} 晚于 DEV_END {C.DEV_END}：等于用回测窗口选因子，"
                    "会引入前视。请把 oos_window_start 设在 DEV 内。")
            oos_mask = (dw >= ws) & (dw <= we)
            base_mask = dev_mask & (dw < ws)
            if not base_mask.any():
                base_mask = dev_mask
            oos_arg = {
                "oos": (rows_weekly[oos_mask], {k: v[oos_mask] for k, v in panels.items()},
                        label[oos_mask], mask_weekly[oos_mask]),
                "base": (rows_weekly[base_mask], {k: v[base_mask] for k, v in panels.items()},
                         label[base_mask], mask_weekly[base_mask]),
            }
        pool, pool_meta = select_pool(md, rows_dev, panels_dev,
                                      label[dev_mask], mask_weekly[dev_mask],
                                      cfg, catalog, verbose=verbose, oos=oos_arg)
    K = len(pool)
    if K == 0:
        raise RuntimeError("候选池为空")

    # ---- 2) 全期特征
    feats = np.empty((K, n_weeks, md.N), dtype=np.float32)
    for i, k in enumerate(pool):
        feats[i] = build_feature_block(md, rows_weekly, panels, k, mask_weekly)
        if verbose and (i + 1) % 10 == 0:
            print(f"[features] {i+1}/{K}")

    # ---- 3) 再估计时点：从 start_date 起每个自然季度初的第一个周频行
    dates_w = md.dates[rows_weekly]
    start_ts = pd.Timestamp(start_date)
    quarters = pd.date_range(start=start_ts, end=dates_w[-1], freq="QS")
    refit_idx: list[int] = []
    for q in quarters:
        pos = int(np.searchsorted(dates_w, q))
        if pos >= n_weeks:
            continue
        i = pos
        if refit_idx and i - refit_idx[-1] < cfg.refit_weeks:
            continue
        refit_idx.append(i)
    if not refit_idx:
        refit_idx = [int(np.searchsorted(dates_w, start_ts))]
    if verbose:
        print(f"[wf] refit 次数 {len(refit_idx)}，{dates_w[refit_idx[0]].date()} ~ {dates_w[refit_idx[-1]].date()}")

    # ---- 4) 走查打分
    # 基本面因子允许缺失（财报未覆盖的股票/退市股），缺失按中性 0 处理；
    # 原有量价/动量因子仍要求数值完整，保持与 v1 模型一致的行为。
    lenient_idx = np.array([i for i, k in enumerate(pool) if k.startswith(_LENIENT_PREFIXES)],
                           dtype=np.int64)
    strict = np.ones(len(pool), dtype=bool)
    strict[lenient_idx] = False
    strict_arg = strict if lenient_idx.size else None
    scores = np.full((md.T, md.N), np.nan, dtype=np.float32)
    weight_rows = []
    ic_rows = []
    blocks = refit_idx + [n_weeks]
    prev_smooth: np.ndarray | None = None
    alpha = float(getattr(cfg, "score_smooth_alpha", 1.0))
    for bi in range(len(refit_idx)):
        r = refit_idx[bi]
        r_end = blocks[bi + 1]
        # 训练窗口：r-train_weeks ~ r-2（标签已实现）
        tr_lo = max(0, r - cfg.train_weeks)
        tr_hi = max(0, r - 2)
        Ftr = feats[:, tr_lo:tr_hi, :]
        ytr = label[tr_lo:tr_hi]
        mtr = mask_weekly[tr_lo:tr_hi]
        if tr_hi - tr_lo < 52:
            if bi == 0:
                if verbose:
                    print(f"[warn] 训练样本不足 ({tr_hi-tr_lo} 周)，跳过 {dates_w[r].date()}")
                continue
        w = fit_weights(Ftr, ytr, mtr, cfg, verbose=False, strict=strict_arg,
                        families=[str(k).split(":")[0] for k in pool])
        for k, wk in zip(pool, w):
            weight_rows.append({"refit_date": dates_w[r], "factor_key": k, "weight": float(wk)})
        # 应用区间 [r, r_end)
        Fb = feats[:, r:r_end, :].copy()
        if lenient_idx.size:
            Fb[lenient_idx] = np.where(np.isfinite(Fb[lenient_idx]), Fb[lenient_idx], 0.0)
        for j in range(r_end - r):
            wi = r + j
            row = np.tensordot(w, Fb[:, j, :], axes=(0, 0)).astype(np.float32)  # (N,)
            # 复合分再中性化 + z 分数
            bb = row[None, :]
            mm = mask_weekly[wi][None, :]
            if np.isfinite(bb[mm]).sum() > 20:
                rr = neutralize_rows(bb, md.industry[rows_weekly[wi]][None, :],
                                     md.logmcap[rows_weekly[wi]][None, :], mm)
                zz = zscore_rows(rr, mm)[0]
            else:
                zz = np.full(md.N, np.nan, dtype=np.float32)
            # 时序平滑（只使用过去信息，降低周度换手）
            if alpha < 1.0 and prev_smooth is not None:
                both = np.isfinite(zz) & np.isfinite(prev_smooth)
                zz = np.where(both, alpha * zz + (1.0 - alpha) * prev_smooth,
                              np.where(np.isfinite(zz), zz, np.nan)).astype(np.float32)
            prev_smooth = zz.copy()
            scores[rows_weekly[wi]] = zz
        # 区间 OOS IC
        sl = slice(r, r_end)
        ic = weekly_ic(np.nan_to_num(scores[rows_weekly[sl]]), label[sl], mask_weekly[sl])
        for ww, v in zip(dates_w[sl], ic):
            if np.isfinite(v):
                ic_rows.append({"date": ww, "ic": float(v), "refit": dates_w[r]})
        if verbose:
            print(f"[wf] refit {dates_w[r].date()} → 持有 {r_end - r} 周，非零权重 {int((np.abs(w)>0).sum())}")

    weights_df = pd.DataFrame(weight_rows)
    ic_df = pd.DataFrame(ic_rows)
    return WalkForwardResult(rows_weekly=rows_weekly, scores=scores, weights=weights_df,
                             pool=pool, pool_meta=pool_meta, composite_ic=ic_df, cfg=cfg)


# ------------------------------------------------------------------ 实盘打分
_FUND_CACHE: dict[int, object] = {}
_TREND_CACHE: dict[int, dict] = {}
_ALPHA_RS_CACHE: dict[int, dict] = {}
_NEWF_CACHE: dict[int, dict] = {}

# 允许缺失（填 0）的因子族前缀：低频/长窗口因子对新股天然缺失，不应因此排除股票
_LENIENT_PREFIXES = ("fundamental-extra:", "trend-extra:", "alpha-rs:")


def build_live_features(md, inp, pool: list[str], cfg: ModelConfig) -> np.ndarray:
    """用最新交易日计算池内因子的中性化 z-score 特征 (K, N)。"""
    from . import factors as qff
    from . import momentum as qfmom
    from . import fundamentals as qffd
    from . import newf as qffnewf
    t = md.T - 1
    mask = md.eligible[[t]]
    feats = np.full((len(pool), md.N), np.nan, dtype=np.float32)
    cat = qff.load_catalog().set_index("factor_key")
    for i, k in enumerate(pool):
        if k.startswith(f"{qfmom.LIB}:"):
            panel = qfmom.build_live(md, k)
        elif k.startswith(f"{qffd.LIB}:"):
            key = id(md)
            fd = _FUND_CACHE.get(key)
            if fd is None:
                fd = qffd.FundamentalData(md)
                _FUND_CACHE.clear()
                _FUND_CACHE[key] = fd
            panel = fd.build_live(md, k)
        elif k.startswith("trend-extra:"):
            key = id(md)
            lr = _TREND_CACHE.get(key)
            if lr is None:
                from . import trend as qftrend
                lr = qftrend.live_row(md)
                _TREND_CACHE.clear()
                _TREND_CACHE[key] = lr
            panel = np.asarray(lr[k], dtype=np.float32)
        elif k.startswith("alpha-rs:"):
            key = id(md)
            lr = _ALPHA_RS_CACHE.get(key)
            if lr is None:
                from . import alpha_rs as qfalpha
                lr = qfalpha.live_row(md)
                _ALPHA_RS_CACHE.clear()
                _ALPHA_RS_CACHE[key] = lr
            panel = np.asarray(lr[k], dtype=np.float32)
        elif k.startswith(f"{qffnewf.PREFIX}"):
            # 换手/流动性/量价/隔夜族（2026-11-03 起可进生产池）。
            # 算式在 qf.newf 单点维护，与 build_newf_extra.py 落盘的面板同源。
            key = id(md)
            lr = _NEWF_CACHE.get(key)
            if lr is None:
                lr = qffnewf.live_row(md)
                _NEWF_CACHE.clear()
                _NEWF_CACHE[key] = lr
            panel = np.asarray(lr[k], dtype=np.float32)
        else:
            entry = cat.loc[k]
            panel = qff.compute_factor(inp, entry.to_dict())
        if panel.ndim == 1:
            panel = panel[None, :]
            rows = np.array([0])
        else:
            rows = np.array([t])
        r = cs_rank_rows(panel, rows, mask=mask)
        if cfg.neutralize:
            r = neutralize_rows(r, md.industry[[t]], md.logmcap[[t]], mask)
        feats[i] = zscore_rows(r, mask)[0]
    return feats


def latest_weights(res: WalkForwardResult) -> tuple[np.ndarray, pd.Timestamp]:
    """取最近一次 refit 的权重向量。"""
    last = res.weights["refit_date"].max()
    sub = res.weights[res.weights["refit_date"] == last]
    w = sub.set_index("factor_key")["weight"].reindex(res.pool).fillna(0.0).to_numpy()
    return w.astype(np.float64), pd.Timestamp(last)


def live_scores(md, inp, res: WalkForwardResult,
                feats: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """返回 (scores_row(N,), weights(K,), refit_date)。

    feats: 可选的已算好的池内特征 (K,N)。Web 服务先算一次特征、缓存后重复使用，
    避免同一天把 build_live_features 跑两遍（各约 20s）。传 None 时行为与历史一致。
    """
    w, refit = latest_weights(res)
    F = build_live_features(md, inp, res.pool, res.cfg) if feats is None else np.array(feats, copy=True)
    lenient_idx = [i for i, k in enumerate(res.pool) if k.startswith(_LENIENT_PREFIXES)]
    if lenient_idx:
        # 与 walk-forward 打分口径一致：低频/长窗口因子允许缺失并按中性 0 处理，
        # 避免因 52 周新高/财报缺失把次新股整行排除。
        F[lenient_idx] = np.where(np.isfinite(F[lenient_idx]), F[lenient_idx], 0.0)
    row = np.tensordot(w, F, axes=(0, 0)).astype(np.float32)
    t = md.T - 1
    mask = md.eligible[[t]]
    rr = neutralize_rows(row[None, :], md.industry[[t]], md.logmcap[[t]], mask)
    zz = zscore_rows(rr, mask)[0]
    zz = apply_universe_filter(md, zz[None, :], [t], res.cfg.universe)[0]
    return zz, w, refit


def apply_universe_filter(md, scores_rows: np.ndarray, rows, universe: str | None) -> np.ndarray:
    """按 models 的 universe 设置过滤选股域（rows 为日频行号列表，scores_rows 与 rows 对齐）。

    过滤只用 t 及以前数据；不满足的股票分数置 NaN（不参与排名）。
    above_ma60 / above_ma60_slope / above_ma200：收盘 > 对应均线；_slope 另要求均线较 20 日前上行。
    """
    if not universe:
        return scores_rows
    close = np.asarray(md.close, dtype=np.float64)
    T, N = close.shape
    out = np.array(scores_rows, dtype=np.float32, copy=True)

    def _ma(t: int, w: int) -> np.ndarray:
        start = max(0, t - w + 1)
        seg = close[start:t + 1]
        cnt = np.isfinite(seg).sum(axis=0)
        s = np.nansum(seg, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt >= max(2, int(w * 0.8)), s / cnt, np.nan)

    u = str(universe)
    if u.startswith("above_ma"):
        parts = u.split("_")
        win = int(parts[1][2:]) if parts[1].startswith("ma") else 60
        slope = len(parts) > 2 and parts[2] == "slope"
        for i, t in enumerate(rows):
            ma = _ma(int(t), win)
            ok = np.isfinite(close[t]) & np.isfinite(ma) & (close[t] > ma)
            if slope:
                ma_prev = _ma(int(t) - 20, win)
                ok &= np.isfinite(ma_prev) & (ma > ma_prev)
            out[i] = np.where(ok, out[i], np.nan).astype(np.float32)
    return out


# ------------------------------------------------------------------ 持久化
def save_result(res: WalkForwardResult, out_dir) -> None:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "composite_score.npz", scores=res.scores,
                        rows=res.rows_weekly)
    res.weights.to_parquet(out / "model_weights.parquet", index=False)
    res.pool_meta.to_parquet(out / "pool_meta.parquet")
    res.composite_ic.to_parquet(out / "composite_ic.parquet", index=False)
    # 只序列化 ModelConfig 的声明字段（排除运行时临时属性如 pool_state_out）
    _cfg_d = {k: v for k, v in res.cfg.__dict__.items()
              if not k.startswith("_") and k not in ("pool_state_out", "pool_state")}
    json.dump({"pool": res.pool, "cfg": _cfg_d,
               "refits": [str(d.date()) for d in sorted(res.weights["refit_date"].unique())]},
              open(out / "model_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def load_result(out_dir) -> WalkForwardResult:
    from pathlib import Path
    out = Path(out_dir)
    zp = out / "composite_score.npz"
    if zp.exists():
        z = np.load(zp)
        rows_weekly, scores = z["rows"], z["scores"]
    else:
        # 2026-11-10：分发版模型目录允许只带权重/池（约 40KB），history 分数矩阵
        # composite_score.npz（4.7MB）只服务回测复盘，出实盘信号不需要它。
        rows_weekly, scores = np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    import pandas as pd
    return WalkForwardResult(
        rows_weekly=rows_weekly, scores=scores,
        weights=pd.read_parquet(out / "model_weights.parquet"),
        pool=json.load(open(out / "model_meta.json", encoding="utf-8"))["pool"],
        pool_meta=pd.read_parquet(out / "pool_meta.parquet"),
        composite_ic=pd.read_parquet(out / "composite_ic.parquet"),
        cfg=ModelConfig(**{k: v for k, v in
                           json.load(open(out / "model_meta.json", encoding="utf-8"))["cfg"].items()
                           if k in _MODEL_CFG_FIELDS}),
    )
