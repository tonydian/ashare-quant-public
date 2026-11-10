# -*- coding: utf-8 -*-
"""Alpha158 / Alpha101 右侧因子族（alpha-rs）。

背景
----
用户要求用两个因子库重建**右侧交易模型**：(1) 现有 directional-alpha 296 个方向因子
（Lead 负责 directional-only 右侧模型）；(2) 本模块：从 Qlib Alpha158（29 统计类型 ×
[5,10,20,30,60] 窗口）与 WorldQuant Alpha101（101 公式）知识库中挑选的经典因子，
语义统一为「值越高 = 趋势越强 / 越右侧」，供 wf_rs2 使用。因子定义来自
https://github.com/seuzxh/quant-factor-skill 的 ``references/alpha158-factors.md`` /
``references/alpha101-factors.md``；算子语义与 microsoft/qlib 的
``qlib/data/ops.py`` / ``qlib/data/_libs/rolling.pyx`` 对齐（已核对源码）。

因子清单（16 个，每个因子一个窗口）
----------------------------------
A001 roc20     20 日动量        close/Ref(close,20)-1                         Momentum
A002 roc60     60 日动量        close/Ref(close,60)-1                         Momentum
A003 ma5_dev   收盘距 MA5 偏离   close/Mean(close,5)-1                         MA-Trend
A004 ma20_dev  收盘距 MA20 偏离  close/Mean(close,20)-1                        MA-Trend
A005 ma60_dev  收盘距 MA60 偏离  close/Mean(close,60)-1                        MA-Trend
A006 rsv20     20 日 RSV        (close-Min(low,20))/(Max(high,20)-Min(low,20)) Mean-Reversion
A007 imax20    20 日最高点位置   (IdxMax(high,20)+1)/20（1=最高点在最近）        Pattern
A008 imin20    20 日最低点位置   1-(IdxMin(low,20)+1)/20（1=最低点最久远）      Pattern
A009 qtlu20    距 20 日 80% 分位  close/Quantile(close,20,0.8)-1              Pattern
A010 qtld20    距 20 日 20% 分位  close/Quantile(close,20,0.2)-1              Pattern
A011 beta20    20 日趋势斜率     Slope(close,20)/close                        Momentum
A012 rsqr20    20 日趋势拟合 R²  Corr(close, t, 20)²                          Pattern
A013 resi20    20 日回归残差     (close - 拟合值)/close（q_lib Resi 口径）     Volatility
A014 corr20    20 日价量相关     Corr(close, Log(volume+1), 20)                Volume-Price
A015 sump20    20 日涨幅占比     Sum(pos(Δclose))/(Sum(|Δclose|)+1e-12)       Mean-Reversion
A016 vsum20    20 日放量占比     Sum(pos(Δvolume))/(Sum(|Δvolume|)+1e-12)     Volume-Price

方向约定（重要）
----------------
目标库用于**右侧长多**模型，故统一为「越大越右侧」。与 Alpha158 原式相比，以下因子
取了**反向单调变换**（原式方向为左侧/反转，括号内为原式）：
- roc20/roc60：实现 ``close/Ref-1``（原式 ``Ref/close``，与动量反向）
- ma5/20/60_dev：实现 ``close/Mean-1``（原式 ``Mean/close``，与"站上均线"反向）
- imin20：实现 ``1-(IdxMin+1)/w``（原式 ``IdxMin/w``，近期低点=弱）
- qtlu20/qtld20：实现 ``close/Quantile-1``（原式 ``Quantile/close``）
其余因子（rsv/imax/beta/rsqr/resi/corr/sump/vsum）方向与 Alpha158 原式一致。
每个因子的 ``catalog().params`` 中同时给出 ``expr``（实现表达式）与
``expr_alpha158``（原式）便于审计；如需原始方向，取负号即可（RankIC 会随之变号）。

未实现（及原因，详见 scripts/build_alpha_rs.py 的 stats["skipped"]）
-------------------------------------------------------------------
- ROC5：A 股 5 日偏反转，且 directional-alpha 已有 return/skip_return w=5 短动量；
- CNTP/CNTN/CNTD：与 SUMP 高度共线（仅缺幅度信息）；
- CORD：与 CORR 高度相关，且量比在停牌日为 0 会放大噪声；
- Alpha101 趋势类（alpha32/33/34/41/53 等）：依赖 vwap（本数据包无 vwap）、
  长窗口或截面 rank，且多为均值回复/微观结构类，与"右侧"目标不符。

约定
----
- ``factor_key = f"alpha-rs:{factor_id}"``，全系统唯一主键；
- 因子只用 t 日及以前数据（收盘/最高/最低/成交量），无未来函数；
- 滚动窗口要求满窗口有效（min_periods=window，与 Alpha158 的 min_periods=1
  只在上市初期有差异；本库评估的选股域已要求上市满 120 交易日，故实际无差异）；
- 输出 float32；NaN 表示数据不足/未上市/回撤不可算；
- 横截面 z-score / rank / 方向由 qf.model 统一处理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ops

LIB = "alpha-rs"

#: (factor_id, key/slug, name_cn, category_en) —— 与 Lead 约定的接口形态
DEFS = [
    ("A001", "roc20", "20日动量", "Momentum"),
    ("A002", "roc60", "60日动量", "Momentum"),
    ("A003", "ma5_dev", "收盘距MA5偏离", "MA-Trend"),
    ("A004", "ma20_dev", "收盘距MA20偏离", "MA-Trend"),
    ("A005", "ma60_dev", "收盘距MA60偏离", "MA-Trend"),
    ("A006", "rsv20", "20日RSV", "Mean-Reversion"),
    ("A007", "imax20", "20日最高点位置", "Pattern"),
    ("A008", "imin20", "20日最低点位置", "Pattern"),
    ("A009", "qtlu20", "距20日80%分位", "Pattern"),
    ("A010", "qtld20", "距20日20%分位", "Pattern"),
    ("A011", "beta20", "20日趋势斜率", "Momentum"),
    ("A012", "rsqr20", "20日趋势拟合R²", "Pattern"),
    ("A013", "resi20", "20日回归残差", "Volatility"),
    ("A014", "corr20", "20日价量相关", "Volume-Price"),
    ("A015", "sump20", "20日涨幅占比", "Mean-Reversion"),
    ("A016", "vsum20", "20日放量占比", "Volume-Price"),
]

_CATEGORY_CN = {
    "Momentum": "动量",
    "MA-Trend": "均线趋势",
    "Pattern": "形态",
    "Volatility": "波动率",
    "Mean-Reversion": "均值回复",
    "Volume-Price": "量价关系",
}

#: 每个因子的元信息（窗口 / 英文名 / 实现表达式 / Alpha158 原式）
_META = {
    "A001": dict(window=20, name_en="20-day momentum", expr="close/Ref($close,20)-1",
                 expr_alpha158="Ref($close,20)/$close"),
    "A002": dict(window=60, name_en="60-day momentum", expr="close/Ref($close,60)-1",
                 expr_alpha158="Ref($close,60)/$close"),
    "A003": dict(window=5, name_en="close vs MA5", expr="close/Mean($close,5)-1",
                 expr_alpha158="Mean($close,5)/$close"),
    "A004": dict(window=20, name_en="close vs MA20", expr="close/Mean($close,20)-1",
                 expr_alpha158="Mean($close,20)/$close"),
    "A005": dict(window=60, name_en="close vs MA60", expr="close/Mean($close,60)-1",
                 expr_alpha158="Mean($close,60)/$close"),
    "A006": dict(window=20, name_en="20-day RSV",
                 expr="($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-12)",
                 expr_alpha158="($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-12)"),
    "A007": dict(window=20, name_en="20-day position of highest high",
                 expr="(IdxMax($high,20)+1)/20", expr_alpha158="IdxMax($high,20)/20"),
    "A008": dict(window=20, name_en="20-day position of lowest low (right-side)",
                 expr="1-(IdxMin($low,20)+1)/20", expr_alpha158="IdxMin($low,20)/20"),
    "A009": dict(window=20, name_en="close vs 80% quantile",
                 expr="close/Quantile($close,20,0.8)-1",
                 expr_alpha158="Quantile($close,20,0.8)/$close"),
    "A010": dict(window=20, name_en="close vs 20% quantile",
                 expr="close/Quantile($close,20,0.2)-1",
                 expr_alpha158="Quantile($close,20,0.2)/$close"),
    "A011": dict(window=20, name_en="20-day trend slope", expr="Slope($close,20)/$close",
                 expr_alpha158="Slope($close,20)/$close"),
    "A012": dict(window=20, name_en="20-day trend R-square",
                 expr="Corr($close,t,20)^2", expr_alpha158="Rsquare($close,20)"),
    "A013": dict(window=20, name_en="20-day regression residual",
                 expr="(close-fitted)/close", expr_alpha158="Resi($close,20)/$close"),
    "A014": dict(window=20, name_en="20-day price-volume correlation",
                 expr="Corr($close,Log($volume+1),20)",
                 expr_alpha158="Corr($close,Log($volume+1),20)"),
    "A015": dict(window=20, name_en="20-day up-move ratio",
                 expr="Sum(pos(Δclose),20)/(Sum(|Δclose|,20)+1e-12)",
                 expr_alpha158="Sum(Greater($close-Ref($close,1),0),20)/(Sum(Abs($close-Ref($close,1)),20)+1e-12)"),
    "A016": dict(window=20, name_en="20-day volume-up ratio",
                 expr="Sum(pos(Δvolume),20)/(Sum(|Δvolume|,20)+1e-12)",
                 expr_alpha158="Sum(Greater($volume-Ref($volume,1),0),20)/(Sum(Abs($volume-Ref($volume,1)),20)+1e-12)"),
}

#: 未实现候选（供复盘/审计；写进 stats json）
SKIPPED = {
    "ROC5": "A 股 5 日偏反转，且 directional-alpha 已有 return/skip_return w=5 短动量",
    "CNTP20/CNTN20/CNTD20": "与 SUMP20 高度共线（仅缺涨跌幅度信息）",
    "CORD20": "与 CORR20 高度相关；停牌日 volume=0 会使量比出现伪跳变",
    "IMXD20": "与 IMAX20/IMIN20 线性组合高度相关，信息重复",
    "Alpha101 alpha32/33/34/41/53": "依赖 vwap（数据包无 vwap）/长窗口/截面 rank，且多为均值回复/微观结构类，不符合右侧目标",
}


# ================================================================ 小工具
def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _safe_ratio_dev(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den - 1（NaN / 0 安全）。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(np.isfinite(num) & np.isfinite(den) & (den != 0),
                     num / den - 1.0, np.nan)
    return v.astype(np.float32)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den（NaN / 0 安全）。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(np.isfinite(num) & np.isfinite(den) & (den != 0),
                     num / den, np.nan)
    return v.astype(np.float32)


def _chunk_rows(w: int, n_cols: int, elems: int = 6_000_000) -> int:
    """滑窗分块的行数（控制单块内存）。"""
    return max(1, int(elems // max(1, w * n_cols)))


def _roll_quantile_pair(a: np.ndarray, w: int, q_hi: float = 0.8,
                        q_lo: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """逐列滚动分位数（pandas rolling().quantile 线性插值口径，满窗口）。

    返回 (q_hi 面板, q_lo 面板)，均 float32；窗口内存在 NaN 时输出 NaN。
    """
    a = _f32(a)
    T, N = a.shape
    out_hi = np.full((T, N), np.nan, dtype=np.float32)
    out_lo = np.full((T, N), np.nan, dtype=np.float32)
    if T < w:
        return out_hi, out_lo

    def _pos(q: float):
        p = q * (w - 1)
        k0 = int(np.floor(p))
        k1 = min(k0 + 1, w - 1)
        return k0, k1, float(p - k0)

    h0, h1, fh = _pos(q_hi)
    l0, l1, fl = _pos(q_lo)
    kth = sorted({h0, h1, l0, l1})
    chunk = _chunk_rows(w, N)
    n_win = T - w + 1
    for s in range(0, n_win, chunk):
        e = min(n_win, s + chunk)
        # sliding_window_view 把窗口维放在最后 → (r, N, w)
        v = np.lib.stride_tricks.sliding_window_view(a[s:e + w - 1], w, axis=0)
        valid = np.isfinite(v).all(axis=-1)                       # (r, N)
        vv = np.where(np.isfinite(v), v, np.inf)
        pv = np.partition(vv, kth, axis=-1)                       # 指定位置已就位
        with np.errstate(invalid="ignore"):
            hi = pv[..., h0] if fh == 0 else pv[..., h0] + fh * (pv[..., h1] - pv[..., h0])
            lo = pv[..., l0] if fl == 0 else pv[..., l0] + fl * (pv[..., l1] - pv[..., l0])
        out_hi[s + w - 1:e + w - 1] = np.where(valid, hi, np.nan)
        out_lo[s + w - 1:e + w - 1] = np.where(valid, lo, np.nan)
    return out_hi, out_lo


def _roll_argidx(a: np.ndarray, w: int, mode: str = "max") -> np.ndarray:
    """逐列滚动极值位置：Qlib IdxMax/IdxMin 口径（窗口内从最早到最新 0..w-1）。

    返回 float32 面板（0..w-1），窗口内有 NaN 时输出 NaN。
    """
    a = _f32(a)
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    if T < w:
        return out
    fill = -np.inf if mode == "max" else np.inf
    chunk = _chunk_rows(w, N, elems=8_000_000)
    n_win = T - w + 1
    for s in range(0, n_win, chunk):
        e = min(n_win, s + chunk)
        # sliding_window_view 把窗口维放在最后 → (r, N, w)
        v = np.lib.stride_tricks.sliding_window_view(a[s:e + w - 1], w, axis=0)
        valid = np.isfinite(v).all(axis=-1)
        vv = np.where(np.isfinite(v), v, fill)
        idx = np.argmax(vv, axis=-1) if mode == "max" else np.argmin(vv, axis=-1)
        out[s + w - 1:e + w - 1] = np.where(valid, idx.astype(np.float32), np.nan)
    return out


# ================================================================ 全量面板
def _compute_full(md) -> dict[str, np.ndarray]:
    """计算 16 个因子的全量 (T,N) float32 面板（内部用，构建/live 共用）。"""

    def c_slice(arr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr, dtype=np.float32)

    close = _f32(md.close)
    high = _f32(md.high)
    low = _f32(md.low)
    volume = _f32(md.volume)
    T, N = close.shape
    out: dict[str, np.ndarray] = {}
    eps = 1e-12

    # ---- 动量 / 均线偏离（右侧方向：close/Ref-1, close/Mean-1）
    for d in (20, 60):
        out[f"alpha-rs:A{'001' if d == 20 else '002'}"] = c_slice(
            _safe_ratio_dev(close, ops.shift(close, d)))
    for d, fid in ((5, "A003"), (20, "A004"), (60, "A005")):
        ma = ops.roll_mean(close, d, minp=d)
        out[f"alpha-rs:{fid}"] = c_slice(_safe_ratio_dev(close, ma))
        del ma

    # ---- RSV20（原式即右侧：接近 1 = 接近 20 日高点）
    lo20 = ops.roll_min(low, 20, minp=20)
    hi20 = ops.roll_max(high, 20, minp=20)
    with np.errstate(invalid="ignore", divide="ignore"):
        rsv = (close - lo20) / (hi20 - lo20 + eps)
    out["alpha-rs:A006"] = c_slice(np.where(np.isfinite(lo20) & np.isfinite(hi20), rsv, np.nan))
    del lo20, hi20, rsv

    # ---- IMAX / IMIN（Qlib: argmax/argmin + 1；IMIN 取右向 = 1 - IdxMin/w）
    amax = _roll_argidx(high, 20, "max")
    out["alpha-rs:A007"] = c_slice((amax + 1.0) / 20.0)
    del amax
    amin = _roll_argidx(low, 20, "min")
    out["alpha-rs:A008"] = c_slice(1.0 - (amin + 1.0) / 20.0)
    del amin

    # ---- QTLU / QTLD（右侧方向：close/Quantile - 1）
    q_hi, q_lo = _roll_quantile_pair(close, 20, 0.8, 0.2)
    out["alpha-rs:A009"] = c_slice(_safe_ratio_dev(close, q_hi))
    out["alpha-rs:A010"] = c_slice(_safe_ratio_dev(close, q_lo))
    del q_hi, q_lo

    # ---- 趋势回归三件套（斜率 / R² / 当前点残差），共用滚动统计
    w = 20
    slope = ops.roll_slope(close, w, minp=w)
    std_close = ops.roll_std(close, w, minp=w)
    mean_close = ops.roll_mean(close, w, minp=w)
    out["alpha-rs:A011"] = c_slice(_safe_div(slope, close))          # BETA = Slope/close
    tmat = np.ascontiguousarray(
        np.broadcast_to(np.arange(T, dtype=np.float32)[:, None], (T, N)))
    corr_t = ops.roll_corr(close, tmat, w, minp=w)
    rsqr = (corr_t * corr_t).astype(np.float32)
    valid_std = np.isfinite(std_close) & (std_close > 2e-05)         # 与 Qlib Rsquare 一致
    out["alpha-rs:A012"] = c_slice(np.where(valid_std, rsqr, np.nan))
    # RESI：当前点相对拟合线的残差（Qlib Resi = y_cur - yhat_cur），再 /close
    yhat = mean_close + slope * ((w - 1) / 2.0)
    out["alpha-rs:A013"] = c_slice(_safe_div(close - yhat, close))
    del slope, std_close, mean_close, tmat, corr_t, rsqr, valid_std, yhat

    # ---- CORR20 = Corr(close, log(volume+1), 20)
    logv = np.log1p(np.where(np.isfinite(volume) & (volume >= 0), volume, np.nan))
    logv = np.ascontiguousarray(logv, dtype=np.float32)
    std_close = ops.roll_std(close, w, minp=w)
    std_logv = ops.roll_std(logv, w, minp=w)
    corr = ops.roll_corr(close, logv, w, minp=w)
    ok_corr = (np.isfinite(std_close) & (std_close > 2e-05)
               & np.isfinite(std_logv) & (std_logv > 2e-05))
    out["alpha-rs:A014"] = c_slice(np.where(ok_corr, corr, np.nan))
    del logv, std_close, std_logv, corr, ok_corr

    # ---- SUMP20 / VSUMP20（上涨/放量占比）
    d_close = close - ops.shift(close, 1)
    pos = np.where(d_close > 0, d_close, 0.0)
    pos = np.where(np.isfinite(d_close), pos, np.nan).astype(np.float32)
    absd = np.where(np.isfinite(d_close), np.abs(d_close), np.nan).astype(np.float32)
    sp = ops.roll_sum(pos, w, minp=w)
    sa = ops.roll_sum(absd, w, minp=w)
    out["alpha-rs:A015"] = np.where(np.isfinite(sp) & np.isfinite(sa),
                                    np.float32(sp / (sa + eps)), np.float32(np.nan))
    del d_close, pos, absd, sp, sa

    d_vol = volume - ops.shift(volume, 1)
    posv = np.where(d_vol > 0, d_vol, 0.0)
    posv = np.where(np.isfinite(d_vol), posv, np.nan).astype(np.float32)
    absv = np.where(np.isfinite(d_vol), np.abs(d_vol), np.nan).astype(np.float32)
    spv = ops.roll_sum(posv, w, minp=w)
    sav = ops.roll_sum(absv, w, minp=w)
    out["alpha-rs:A016"] = np.where(np.isfinite(spv) & np.isfinite(sav),
                                    np.float32(spv / (sav + eps)), np.float32(np.nan))
    del d_vol, posv, absv, spv, sav

    return {k: c_slice(v) for k, v in out.items()}


# ================================================================ 公开接口
def factor_keys() -> list[str]:
    return [f"{LIB}:{fid}" for fid, _, _, _ in DEFS]


def catalog() -> pd.DataFrame:
    """因子目录（列与全系统 catalog 对齐，额外含 key / category 字段）。"""
    rows = []
    for fid, key, name_cn, category_en in DEFS:
        meta = _META[fid]
        rows.append({
            "lib": LIB,
            "factor_id": fid,
            "slug": key,
            "key": key,
            "name_cn": name_cn,
            "name_en": meta["name_en"],
            "category_cn": _CATEGORY_CN[category_en],
            "category_en": category_en,
            "base": key,
            "transform": "raw",
            "window": int(meta["window"]),
            "norm": 20,
            "params": {
                "window": int(meta["window"]),
                "source": "qlib Alpha158",
                "expr": meta["expr"],
                "expr_alpha158": meta["expr_alpha158"],
            },
            "path": "",
            "factor_key": f"{LIB}:{fid}",
        })
    return pd.DataFrame(rows)


def build_panels(md, rows: np.ndarray) -> dict[str, np.ndarray]:
    """计算指定周频行的因子面板：{factor_key: (len(rows), N) float32}。

    内部逐因子计算全量 (T,N) 后立即切片，控制内存峰值；因子只用 t 及以前数据。
    """
    rows = np.asarray(rows, dtype=np.int64)
    full = _compute_full(md)
    out: dict[str, np.ndarray] = {}
    for k, arr in full.items():
        out[k] = np.ascontiguousarray(arr[rows], dtype=np.float32)
    return out


#: 实盘最新截面缓存：{id(md): {factor_key: (N,) float32}}
_LIVE_CACHE: dict[int, dict[str, np.ndarray]] = {}


def live_row(md) -> dict[str, np.ndarray]:
    """最新交易日截面 {factor_key: (N,) float32}；按 id(md) 缓存，供实盘打分一次取全。"""
    key_id = id(md)
    cached = _LIVE_CACHE.get(key_id)
    if cached is not None:
        return cached
    last = np.array([int(md.T) - 1], dtype=np.int64)
    panels = build_panels(md, last)
    out = {k: np.ascontiguousarray(v[0], dtype=np.float32) for k, v in panels.items()}
    _LIVE_CACHE.clear()
    _LIVE_CACHE[key_id] = out
    return out


def build_live(md, key: str) -> np.ndarray:
    """单因子最新截面（兼容 qf.model.build_live_features 的逐因子调用风格）。"""
    lr = live_row(md)
    if key not in lr:
        raise KeyError(key)
    return lr[key]
