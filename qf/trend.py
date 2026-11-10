# -*- coding: utf-8 -*-
"""额外因子族：右侧趋势 / 动量（trend-extra）。

背景
----
主线 wf_h10 的 34 因子来自 QuantSkills 三库（窗口 5/7/10/14 日，处在 A 股短周期
反转区）+ momentum-extra，最终权重集中在"反转 / 低波"（左侧）。本族用来构建
对照的**右侧模型**：所有因子语义统一为「值越高 = 趋势越强 / 越右侧」，
只使用 t 日收盘及以前的数据（无未来函数），便于长多权重合成与独立解释。

因子清单（16 个）
-----------------
T001 d_52w_high         距 52 周新高            close / max(close, 250) - 1
T002 d_60d_high         距 60 日新高            close / max(close, 60) - 1
T003 ma200_dev          MA200 偏离              close / MA200 - 1
T004 ma50_ma200         MA50 / MA200 - 1
T005 ma200_slope        MA200 20 日斜率         MA200 / MA200[t-20] - 1
T006 mom_120_5          6 个月动量(120-5)       close[t-5] / close[t-125] - 1
T007 mom_250_20         12-1 动量               close[t-20] / close[t-250] - 1
T008 rs_index_60        相对强度 vs 中证全指     个股 60 日收益 - 指数 60 日收益
T009 rs_industry_120    相对强度 vs 行业        个股 120 日收益 - 行业 120 日均值
T010 industry_mom_60    行业动量               行业 60 日收益均值（广播到个股）
T011 donchian_55        55 日通道突破           close / max(high,55)[t-1] - 1
T012 breakout_20_vol    20 日放量突破           (close / max(high,20)[t-1] - 1) * 量能比
T013 adx_di             ADX / DI 趋势强度       DI+ - DI-（14 日 Wilder）
T014 trend_eff_20       趋势效率                方向效率 ER(20) × 方向
T015 near_high_freq_60  近高天数占比            近 20 日中 close ≥ 95%·52周高 的占比
T016 trend_r2_60        趋势 R² × 方向          roll_corr(log close, t, 60) × sign(slope)

约定
----
- `factor_key = f"trend-extra:{factor_id}"`，全系统唯一主键；
- 输出 float32 (rows, N) / (T, N) 面板，NaN 表示数据不足或未上市；
- 横截面方向统一为「越大越右侧」，截面 z-score / rank 由 qf.model 统一处理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ops

LIB = "trend-extra"

#: (factor_id, key/slug, name_cn) —— 与 Lead 约定的接口形态
DEFS = [
    ("T001", "d_52w_high", "距52周新高"),
    ("T002", "d_60d_high", "距60日新高"),
    ("T003", "ma200_dev", "MA200偏离"),
    ("T004", "ma50_ma200", "MA50/MA200多头"),
    ("T005", "ma200_slope", "MA200斜率"),
    ("T006", "mom_120_5", "6个月动量(120-5)"),
    ("T007", "mom_250_20", "12-1动量"),
    ("T008", "rs_index_60", "相对强度vs中证全指"),
    ("T009", "rs_industry_120", "相对强度vs行业"),
    ("T010", "industry_mom_60", "行业动量"),
    ("T011", "donchian_55", "55日通道突破"),
    ("T012", "breakout_20_vol", "20日放量突破"),
    ("T013", "adx_di", "ADX/DI趋势强度"),
    ("T014", "trend_eff_20", "趋势效率"),
    ("T015", "near_high_freq_60", "近高天数占比"),
    ("T016", "trend_r2_60", "趋势R²×方向"),
]

#: 额外元信息（窗口 / 中英文名 / 类别），供 catalog() 使用
_META = {
    "T001": dict(window=250, name_en="52-week high proximity", category_cn="趋势突破", category_en="Trend-Breakout"),
    "T002": dict(window=60, name_en="60-day high proximity", category_cn="趋势突破", category_en="Trend-Breakout"),
    "T003": dict(window=200, name_en="deviation from MA200", category_cn="均线趋势", category_en="MA-Trend"),
    "T004": dict(window=200, name_en="MA50 vs MA200", category_cn="均线趋势", category_en="MA-Trend"),
    "T005": dict(window=200, name_en="MA200 20-day slope", category_cn="均线趋势", category_en="MA-Trend"),
    "T006": dict(window=120, name_en="6M momentum 120-5", category_cn="中期动量", category_en="Momentum"),
    "T007": dict(window=250, name_en="12-1 momentum", category_cn="中期动量", category_en="Momentum"),
    "T008": dict(window=60, name_en="relative strength vs CSI All", category_cn="相对强度", category_en="Relative-Strength"),
    "T009": dict(window=120, name_en="relative strength vs industry", category_cn="相对强度", category_en="Relative-Strength"),
    "T010": dict(window=60, name_en="industry momentum", category_cn="行业动量", category_en="Industry-Momentum"),
    "T011": dict(window=55, name_en="55-day Donchian breakout", category_cn="趋势突破", category_en="Trend-Breakout"),
    "T012": dict(window=20, name_en="20-day breakout with volume", category_cn="趋势突破", category_en="Trend-Breakout"),
    "T013": dict(window=14, name_en="ADX / DI trend strength", category_cn="趋势强度", category_en="Trend-Strength"),
    "T014": dict(window=20, name_en="trend efficiency (Kaufman ER)", category_cn="趋势强度", category_en="Trend-Strength"),
    "T015": dict(window=250, name_en="near-high frequency", category_cn="趋势突破", category_en="Trend-Breakout"),
    "T016": dict(window=60, name_en="trend R2 x direction", category_cn="趋势强度", category_en="Trend-Strength"),
}


# ------------------------------------------------------------------ 小工具
def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _safe_div_ret(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den - 1（NaN 安全）。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(np.isfinite(num) & np.isfinite(den) & (den != 0), num / den - 1.0, np.nan)
    return v.astype(np.float32)


def _industry_mean(md, ret: np.ndarray) -> np.ndarray:
    """逐日按行业求均值并广播到个股 (T,N)；未知行业用当日全市场均值。"""
    T, N = ret.shape
    ind = md.industry
    valid = np.isfinite(ret)
    out = np.full((T, N), np.nan, dtype=np.float32)
    # 全市场均值（兜底未知行业）
    cnt_all = valid.sum(axis=1)
    sum_all = np.where(valid, ret, 0.0).sum(axis=1)
    overall = np.where(cnt_all > 0, sum_all / np.maximum(cnt_all, 1), np.nan).astype(np.float32)
    for k in range(len(md.industry_cats)):
        sel = (ind == k)
        m = sel & valid
        cnt = m.sum(axis=1)
        s = np.where(m, ret, 0.0).sum(axis=1)
        mean_k = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).astype(np.float32)
        out[sel] = np.broadcast_to(mean_k[:, None], (T, N))[sel]
    unknown = ind < 0
    if unknown.any():
        out[unknown] = np.broadcast_to(overall[:, None], (T, N))[unknown]
    return out


def _bench_ret(md, n: int) -> np.ndarray:
    """中证全指 n 日收益 (T,)。"""
    bp = np.asarray(md.bench_prices["csi_all"].to_numpy(dtype=np.float64), dtype=np.float64)
    out = np.full(len(bp), np.nan, dtype=np.float64)
    if len(bp) > n:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[n:] = bp[n:] / bp[:-n] - 1.0
    return out.astype(np.float32)


def _roll_max(a: np.ndarray, w: int) -> np.ndarray:
    """尾部滚动最大值，等价 pandas ``rolling(w, min_periods=w).max()``。

    说明：``qf.ops.roll_max`` 的 chunk 实现在窗口 w 较大（约 w > 40）时会退化
    （chunk 行数小于窗口，返回全 NaN），本因子族涉及 20/55/250 日窗口，故在这里
    自实现 O(T·N) 的分块前缀/后缀最大值算法，并保持 min_periods=w 的 NaN 语义。
    """
    a = np.asarray(a, dtype=np.float32)
    T, N = a.shape
    valid = np.isfinite(a)
    if w <= 1:
        return np.where(valid, a, np.nan).astype(np.float32)
    x = np.where(valid, a, -np.inf).astype(np.float32)
    nb = (T + w - 1) // w
    pad = nb * w - T
    xp = np.vstack([x, np.full((pad, N), -np.inf, dtype=np.float32)])
    xp3 = xp.reshape(nb, w, N)
    pref = np.maximum.accumulate(xp3, axis=1).reshape(nb * w, N)[:T]
    suf = np.maximum.accumulate(xp3[:, ::-1, :], axis=1)[:, ::-1, :].reshape(nb * w, N)[:T]
    t = np.arange(T)
    start = np.maximum(t - w + 1, 0)
    same_block = (start // w) == (t // w)
    res = np.where(same_block[:, None], pref, np.fmax(suf[start], pref))
    cs = np.zeros((T + 1, N), dtype=np.int32)
    cs[1:] = np.cumsum(valid, axis=0)
    cnt = cs[t + 1] - cs[start]
    return np.where(cnt >= w, res, np.nan).astype(np.float32)


# ------------------------------------------------------------------ 单因子（返回全量 (T,N) 面板）
def _t001_52w_high(md):
    close = md.close
    mx = _roll_max(close, 250)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(np.isfinite(mx) & (mx != 0), close / mx - 1.0, np.nan)
    return _f32(v)


def _t002_60d_high(md):
    close = md.close
    mx = _roll_max(close, 60)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(np.isfinite(mx) & (mx != 0), close / mx - 1.0, np.nan)
    return _f32(v)


def _t003_ma200_dev(md):
    close = md.close
    ma = ops.roll_mean(close, 200, minp=200)
    return _safe_div_ret(close, ma)


def _t004_ma50_ma200(md):
    close = md.close
    ma50 = ops.roll_mean(close, 50, minp=50)
    ma200 = ops.roll_mean(close, 200, minp=200)
    return _safe_div_ret(ma50, ma200)


def _t005_ma200_slope(md):
    ma200 = ops.roll_mean(md.close, 200, minp=200)
    prev = ops.shift(ma200, 20)
    return _safe_div_ret(ma200, prev)


def _t006_mom_120_5(md):
    close = md.close
    cur = ops.shift(close, 5)
    prev = ops.shift(close, 125)
    return _safe_div_ret(cur, prev)


def _t007_mom_250_20(md):
    close = md.close
    cur = ops.shift(close, 20)
    prev = ops.shift(close, 250)
    return _safe_div_ret(cur, prev)


def _t008_rs_index_60(md):
    ret = ops.pct_change(md.close, 60)
    bench = _bench_ret(md, 60)[:, None]
    return _f32(ret - bench)


def _t009_rs_industry_120(md):
    ret = ops.pct_change(md.close, 120)
    ind = _industry_mean(md, ret)
    return _f32(ret - ind)


def _t010_industry_mom_60(md):
    ret = ops.pct_change(md.close, 60)
    return _f32(_industry_mean(md, ret))


def _t011_donchian_55(md):
    prior_high = ops.shift(_roll_max(md.high, 55), 1)
    return _safe_div_ret(md.close, prior_high)


def _t012_breakout_20_vol(md):
    prior_high = ops.shift(_roll_max(md.high, 20), 1)
    brk = _safe_div_ret(md.close, prior_high)
    vol_ma = ops.roll_mean(md.volume, 20, minp=20)
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = np.where(np.isfinite(vol_ma) & (vol_ma > 0), md.volume / vol_ma, np.nan)
    vr = np.clip(vr, 0.5, 3.0)
    return _f32(brk * vr)


def _t013_adx_di(md):
    high, low, close = md.high, md.low, md.close
    prev_close = ops.shift(close, 1)
    tr = high - low
    tr = np.fmax(tr, np.abs(high - prev_close))
    tr = np.fmax(tr, np.abs(low - prev_close))
    up = high - ops.shift(high, 1)
    dn = ops.shift(low, 1) - low
    with np.errstate(invalid="ignore"):
        plus_dm = np.where(np.isfinite(up) & np.isfinite(dn) & (up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where(np.isfinite(up) & np.isfinite(dn) & (dn > up) & (dn > 0), dn, 0.0)
    alpha = 1.0 / 14.0
    atr = ops.ema_alpha(tr.astype(np.float32), alpha, minp=14)
    p_dm = ops.ema_alpha(plus_dm.astype(np.float32), alpha, minp=14)
    m_dm = ops.ema_alpha(minus_dm.astype(np.float32), alpha, minp=14)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = np.where(atr > 0, 100.0 * p_dm / atr, np.nan)
        mdi = np.where(atr > 0, 100.0 * m_dm / atr, np.nan)
    return _f32(pdi - mdi)


def _t014_trend_eff_20(md):
    close = md.close
    chg = close - ops.shift(close, 1)
    abs_sum = ops.roll_sum(np.abs(chg), 20, minp=20)
    net = close - ops.shift(close, 20)
    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.where(abs_sum > 0, np.abs(net) / abs_sum, np.nan)
    return _f32(er * np.sign(net))


def _t015_near_high_freq_60(md):
    close = md.close
    mx = _roll_max(close, 250)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.isfinite(mx) & (mx != 0), close / mx, np.nan)
    near = np.where(np.isfinite(ratio), (ratio >= 0.95).astype(np.float32), np.nan)
    return _f32(ops.roll_mean(near, 20, minp=20))


def _t016_trend_r2_60(md):
    close = md.close
    with np.errstate(invalid="ignore", divide="ignore"):
        logp = np.where(close > 0, np.log(close.astype(np.float64)), np.nan).astype(np.float32)
    T, N = logp.shape
    tmat = np.tile(np.arange(T, dtype=np.float32)[:, None], (1, N))
    corr = ops.roll_corr(logp, tmat, 60, minp=60)
    slope = ops.roll_slope(logp, 60, minp=60)
    return _f32(corr * np.sign(slope))


_BUILDERS = {
    "T001": _t001_52w_high,
    "T002": _t002_60d_high,
    "T003": _t003_ma200_dev,
    "T004": _t004_ma50_ma200,
    "T005": _t005_ma200_slope,
    "T006": _t006_mom_120_5,
    "T007": _t007_mom_250_20,
    "T008": _t008_rs_index_60,
    "T009": _t009_rs_industry_120,
    "T010": _t010_industry_mom_60,
    "T011": _t011_donchian_55,
    "T012": _t012_breakout_20_vol,
    "T013": _t013_adx_di,
    "T014": _t014_trend_eff_20,
    "T015": _t015_near_high_freq_60,
    "T016": _t016_trend_r2_60,
}


# ------------------------------------------------------------------ 公开接口
def factor_keys() -> list[str]:
    return [f"{LIB}:{fid}" for fid, _, _ in DEFS]


def catalog() -> pd.DataFrame:
    """因子目录（列与全系统 catalog 对齐，额外含 key 列）。"""
    rows = []
    for fid, key, name_cn in DEFS:
        meta = _META[fid]
        rows.append({
            "lib": LIB,
            "factor_id": fid,
            "slug": key,
            "key": key,
            "name_cn": name_cn,
            "name_en": meta["name_en"],
            "category_cn": meta["category_cn"],
            "category_en": meta["category_en"],
            "base": key,
            "transform": "raw",
            "window": int(meta["window"]),
            "norm": 20,
            "params": {"window": int(meta["window"])},
            "path": "",
            "factor_key": f"{LIB}:{fid}",
        })
    return pd.DataFrame(rows)


def build_panels(md, rows: np.ndarray) -> dict[str, np.ndarray]:
    """计算指定周频行的因子面板：{factor_key: (len(rows), N) float32}。

    内部逐因子计算全量 (T,N) 后立即切片，控制内存峰值。
    """
    rows = np.asarray(rows, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for fid, _, _ in DEFS:
        full = _BUILDERS[fid](md)
        out[f"{LIB}:{fid}"] = np.ascontiguousarray(full[rows], dtype=np.float32)
        del full
    return out


#: 实盘最新截面缓存：{id(md): {factor_key: (N,) float32}}
_LIVE_CACHE: dict[int, dict[str, np.ndarray]] = {}


def live_row(md) -> dict[str, np.ndarray]:
    """最新交易日截面 {factor_key: (N,) float32}；按 id(md) 缓存，供实盘打分一次取全。"""
    key_id = id(md)
    cached = _LIVE_CACHE.get(key_id)
    if cached is not None:
        return cached
    out: dict[str, np.ndarray] = {}
    for fid, _, _ in DEFS:
        full = _BUILDERS[fid](md)
        out[f"{LIB}:{fid}"] = np.ascontiguousarray(full[-1], dtype=np.float32)
        del full
    _LIVE_CACHE.clear()
    _LIVE_CACHE[key_id] = out
    return out


def build_live(md, key: str) -> np.ndarray:
    """单因子最新截面（兼容 model.build_live_features 的逐因子调用风格）。"""
    rows = live_row(md)
    if key not in rows:
        raise KeyError(key)
    return rows[key]
