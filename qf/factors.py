# -*- coding: utf-8 -*-
"""QuantSkills 三大因子库（volume-stat / directional / risk-pattern，共 800 个因子）
的向量化实现。

每个因子由 ``factor_index.json`` 中的 ``base`` + ``transform`` + ``params`` 唯一确定；
参考实现位于 ``vendor/skill-*/factors/<id>/scripts/factor.py``（pandas 逐股票版本），
本模块在 (T, N) 面板上用 :mod:`qf.ops` 重写同一套语义，供全 A 面板批量计算。

因子族（base）:
    Volume: volume_ratio, volume_z, dollar_volume
    PriceVolume: price_volume_corr, obv_slope
    TimeSeriesRank: ts_rank_close, ts_rank_volume
    Distribution: ret_skew, ret_kurt
    Momentum/Reversal/Trend/Channel/Breakout: return, skip_return, reversal,
        sma_gap, ema_gap, dual_ema_gap, sma_slope, range_position, breakout,
        breakdown, trend_strength, efficiency
    Oscillator: rsi, rsi_reversal, stoch
    Volatility: atr_ratio, range_ratio, realized_vol, downside_vol
    Pattern: gap_sum, intraday, upper_wick, lower_wick
    Drawdown: drawdown

变换（transform）: raw, neg, z, delta, smooth, rank, vol_scaled, stability,
    confirm_volume, compress
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import ops

# ------------------------------------------------------------------ 因子目录
LIB_DIRS = {
    "volume-stat-alpha": C.VENDOR / "skill-volume-stat-alpha" / "skill-quant-factor-volume-stat-alpha-main",
    "directional-alpha": C.VENDOR / "skill-directional-alpha" / "skill-quant-factor-directional-alpha-main",
    "risk-pattern-alpha": C.VENDOR / "skill-risk-pattern-alpha" / "skill-quant-factor-risk-pattern-alpha-main",
}


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    rows = []
    for lib, d in LIB_DIRS.items():
        j = json.loads((d / "factor_index.json").read_text(encoding="utf-8"))
        for e in j:
            rows.append({
                "lib": lib,
                "factor_id": e["factor_id"],
                "slug": e["slug"],
                "name_cn": e.get("name_cn", ""),
                "name_en": e.get("name_en", ""),
                "category_cn": e.get("category_cn", ""),
                "category_en": e.get("category_en", ""),
                "base": e["base"],
                "transform": e["transform"],
                "window": int(e["params"].get("window", 10)),
                "norm": int(e["params"].get("norm", 20)),
                "params": e["params"],
                "path": str(d / e["path"]),
            })
    df = pd.DataFrame(rows)
    df["factor_key"] = df["lib"].str.replace("-alpha", "", regex=False) + ":" + df["factor_id"]
    return df


def align_panel_rows(a: np.ndarray, n_rows: int, name: str = "panel") -> np.ndarray | None:
    """把因子面板对齐到周频行数 ``n_rows``。

    早期构建的扩展因子族（momentum_extra / trend_extra）只有 475 周，而
    ``weekly_meta.npz`` 的 rows 是 477 周。``run_research.py`` 原先用
    ``shape[0] == len(rows)`` 过滤，导致这 24 个因子**被静默丢弃、从未参与选池**
    （2026-11-03 定位）。实测 475 行面板与 rows 是前对齐（offset=0 时与
    重算的 mom20_skip0 横截面相关最高），因此尾部补 NaN 即正确对齐。

    返回 None 表示该面板不可用（列数不符或比 rows 还长），由调用方跳过并告警。
    """
    if a.ndim != 2:
        return None
    if a.shape[0] == n_rows:
        return a
    if a.shape[0] > n_rows:
        return None
    pad = np.full((n_rows - a.shape[0], a.shape[1]), np.nan, dtype=a.dtype)
    # 补 NaN 只是让矩阵形状对齐；被补的那几周该因子是缺失的，一旦它进池，
    # 尾部几周的复合分就会整行为 NaN（模型"最后打分行"滞后、一致性检测报错）。
    # 这里必须吵：正常做法是重建该族面板，而不是靠补 NaN 糊过去。
    print(f"    [warn] {name} 比周频行少 {n_rows - a.shape[0]} 周，尾部补 NaN —— "
          f"该族面板已过期，请跑 scripts/build_extra_panels.py 重建全部 --extra 面板")
    return np.concatenate([a, pad], axis=0)


# ------------------------------------------------------------------ 输入面板
class FactorInputs:
    """因子计算所需的行情面板（一次构造，反复使用）。

    volume/amount 使用原始缺失口径（停牌日为 NaN），以贴近参考实现；
    价格使用停牌日前向填充（停牌期间价格不变）。
    """

    def __init__(self, md):
        self.md = md
        self.close = md.close
        self.open = md.open
        self.high = md.high
        self.low = md.low
        self.volume = np.where(md.alive, np.where(np.isfinite(md.volume_orig), md.volume_orig, np.nan), np.nan).astype(np.float32)
        self.amount = np.where(md.alive, np.where(np.isfinite(md.amount_orig), md.amount_orig, np.nan), np.nan).astype(np.float32)
        self.ret = ops.pct_change(self.close, 1)
        self.ret = np.where(np.isfinite(md.close_orig) | ~md.is_suspended, self.ret, self.ret).astype(np.float32)
        self.valid_close = np.isfinite(self.close)

    def __repr__(self) -> str:
        return f"FactorInputs(T={self.close.shape[0]}, N={self.close.shape[1]})"


# ------------------------------------------------------------------ base 计算
def _base_signal(inp: FactorInputs, base: str, p: dict) -> np.ndarray:
    close, open_, high, low = inp.close, inp.open, inp.high, inp.low
    volume = inp.volume
    ret = inp.ret
    w = int(p.get("window", 10))
    norm = int(p.get("norm", 20))
    fast = int(p.get("fast", max(3, w // 3)))
    slow = int(p.get("slow", w))
    skip = int(p.get("skip", 5))

    if base == "return":
        return ops.pct_change(close, w)
    if base == "skip_return":
        return ops.pct_change(ops.shift(close, skip), w)
    if base == "reversal":
        return -ops.pct_change(close, w)
    if base == "sma_gap":
        ma = ops.roll_mean(close, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (close / ma - 1.0).astype(np.float32)
    if base == "ema_gap":
        e = ops.ema(close, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (close / e - 1.0).astype(np.float32)
    if base == "dual_ema_gap":
        ef, es = ops.ema(close, fast), ops.ema(close, slow)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((ef - es) / close).astype(np.float32)
    if base == "sma_slope":
        ma = ops.roll_mean(close, w)
        sw = min(30, max(5, w // 2))
        slope = ops.roll_slope(ma, sw, minp=sw)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (slope / close).astype(np.float32)
    if base == "range_position":
        lo = ops.roll_min(low, w)
        hi = ops.roll_max(high, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((close - lo) / (hi - lo) - 0.5).astype(np.float32)
    if base == "breakout":
        hi = ops.roll_max(ops.shift(high, 1), w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (close / hi - 1.0).astype(np.float32)
    if base == "breakdown":
        lo = ops.roll_min(ops.shift(low, 1), w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (-(close / lo - 1.0)).astype(np.float32)
    if base in ("rsi", "rsi_reversal"):
        r = _rsi(close, w)
        if base == "rsi":
            return (r / 100.0 - 0.5).astype(np.float32)
        return (0.5 - r / 100.0).astype(np.float32)
    if base == "stoch":
        lo = ops.roll_min(low, w)
        hi = ops.roll_max(high, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((close - lo) / (hi - lo) - 0.5).astype(np.float32)
    if base == "atr_ratio":
        atr = _atr(inp, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (atr / close).astype(np.float32)
    if base == "range_ratio":
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = (high - low) / close
        return ops.roll_mean(rr, w)
    if base == "realized_vol":
        return ops.roll_std(ret, w)
    if base == "downside_vol":
        dn = np.where(np.isfinite(ret) & (ret < 0), ret, np.where(np.isfinite(ret), 0.0, np.nan)).astype(np.float32)
        return ops.roll_std(dn, w)
    if base == "trend_strength":
        num = ops.pct_change(close, w)
        den = ops.roll_std(ret, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (num / den).astype(np.float32)
    if base == "efficiency":
        ww = int(w)
        num = np.abs(close - ops.shift(close, ww))
        diff = close - ops.shift(close, 1)
        den = ops.roll_sum(np.abs(diff), ww)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (num / den).astype(np.float32)
    if base == "volume_ratio":
        vm = ops.roll_mean(volume, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (volume / vm - 1.0).astype(np.float32)
    if base == "volume_z":
        vm, vs = ops.roll_mean_std(volume, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(vs > 0, (volume - vm) / np.where(vs > 0, vs, np.nan), np.nan).astype(np.float32)
    if base == "dollar_volume":
        dollar = close * volume
        dm = ops.roll_mean(dollar, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (dollar / dm - 1.0).astype(np.float32)
    if base == "price_volume_corr":
        return ops.roll_corr(ret, ops.pct_change(volume, 1), w)
    if base == "obv_slope":
        sgn = np.where(np.isfinite(ret), np.sign(ret), np.nan)
        obv = np.nancumsum(np.where(np.isfinite(sgn) & np.isfinite(volume), sgn * volume, 0.0), axis=0).astype(np.float32)
        obv = np.where(np.isfinite(close), obv, np.nan)
        sw = min(ww := min(w, 60), 60)
        slope = ops.roll_slope(obv, ww, minp=ww)
        vm = ops.roll_mean(volume, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (slope / vm).astype(np.float32)
    if base == "gap_sum":
        gap = ops.shift(close, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = (open_ / gap - 1.0).astype(np.float32)
        return ops.roll_sum(g, w)
    if base == "intraday":
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (close / open_ - 1.0).astype(np.float32)
        return ops.roll_mean(v, w)
    if base == "upper_wick":
        top = np.maximum(open_, close)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = ((high - top) / (high - low)).astype(np.float32)
        return ops.roll_mean(v, w)
    if base == "lower_wick":
        bot = np.minimum(open_, close)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = ((bot - low) / (high - low)).astype(np.float32)
        return ops.roll_mean(v, w)
    if base == "drawdown":
        hi = ops.roll_max(close, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (close / hi - 1.0).astype(np.float32)
    if base == "ts_rank_close":
        return (ops.roll_rank_last(close, w) - 0.5).astype(np.float32)
    if base == "ts_rank_volume":
        return (ops.roll_rank_last(volume, w) - 0.5).astype(np.float32)
    if base == "ret_skew":
        return ops.roll_skew(ret, w, minp=ops._minp(w, 10))
    if base == "ret_kurt":
        return ops.roll_kurt(ret, w, minp=ops._minp(w, 10))
    raise ValueError(f"unknown base: {base}")


def _rsi(close: np.ndarray, w: int) -> np.ndarray:
    delta = close - ops.shift(close, 1)  # 参考实现用 close.diff()（价差，非收益率）
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0), np.nan).astype(np.float32)
    loss = np.where(np.isfinite(delta), np.maximum(-delta, 0), np.nan).astype(np.float32)
    minp = ops._minp(w)  # 参考实现: ewm(alpha=1/w, min_periods=_minp(w))
    ag = ops.ema_alpha(gain, 1.0 / w, minp=minp)
    al = ops.ema_alpha(loss, 1.0 / w, minp=minp)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = ag / al
        rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.astype(np.float32)


def _atr(inp: FactorInputs, w: int) -> np.ndarray:
    high, low, close = inp.high, inp.low, inp.close
    prev = ops.shift(close, 1)
    tr = np.maximum(np.maximum(high - low, np.abs(high - prev)), np.abs(low - prev)).astype(np.float32)
    return ops.roll_mean(tr, w)


# ------------------------------------------------------------------ transform
def _transform_signal(sig: np.ndarray, inp: FactorInputs, name: str, p: dict) -> np.ndarray:
    norm = int(p.get("norm", 20))
    if name == "raw":
        return sig.astype(np.float32)
    if name == "neg":
        return (-sig).astype(np.float32)
    if name == "z":
        m, s = ops.roll_mean_std(sig, norm)
        with np.errstate(invalid="ignore", divide="ignore"):
            # 与参考实现 _zscore 一致：std=0 处输出 NaN
            return np.where(s > 0, (sig - m) / np.where(s > 0, s, np.nan), np.nan).astype(np.float32)
    if name == "delta":
        return (sig - ops.shift(sig, int(p.get("lag", 1)))).astype(np.float32)
    if name == "smooth":
        return ops.ema(sig, int(p.get("smooth", 3)))
    if name == "rank":
        return (ops.roll_rank_last(sig, norm) - 0.5).astype(np.float32)
    if name == "vol_scaled":
        vol = ops.roll_std(inp.ret, norm)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(vol > 0, sig / np.where(vol > 0, vol, np.nan), np.nan).astype(np.float32)
    if name == "stability":
        vol = ops.roll_std(sig, norm)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(vol > 0, sig / np.where(vol > 0, vol, np.nan), np.nan).astype(np.float32)
    if name == "confirm_volume":
        vm = ops.roll_mean(inp.volume, norm)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(vm > 0, sig * (inp.volume / np.where(vm > 0, vm, np.nan)), np.nan).astype(np.float32)
    if name == "compress":
        return ops.tanh_clip(sig)
    raise ValueError(f"unknown transform: {name}")


# ------------------------------------------------------------------ 因子引擎
def compute_group(inp: FactorInputs, base: str, w: int, transforms: list[str],
                  params_list: list[dict]) -> dict[str, np.ndarray]:
    """计算同一 (base, window) 下的若干 transform，返回 {transform: panel}。"""
    p0 = dict(params_list[0])
    p0["window"] = w
    sig = _base_signal(inp, base, p0)
    out = {}
    for tr, p in zip(transforms, params_list):
        out[tr] = _transform_signal(sig, inp, tr, p)
    return out


def compute_factor(inp: FactorInputs, entry: dict) -> np.ndarray:
    """按 catalog 一行计算单因子面板。"""
    p = dict(entry["params"]) if not isinstance(entry.get("params"), dict) else dict(entry["params"])
    p["window"] = int(entry.get("window", p.get("window", 10)))
    p["norm"] = int(entry.get("norm", p.get("norm", 20)))
    base = entry["base"]
    with np.errstate(all="ignore"):
        sig = _base_signal(inp, base, p)
        return _transform_signal(sig, inp, entry["transform"], p)


def iter_groups(catalog: pd.DataFrame | None = None):
    """按 (base, window) 分组迭代，组内可复用 base 信号。"""
    cat = load_catalog() if catalog is None else catalog
    for (base, w), g in cat.groupby(["base", "window"], sort=False):
        yield base, w, g
