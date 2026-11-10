# -*- coding: utf-8 -*-
"""可转债「转股价值」反推口径诊断（负溢价率自洽检验）。

背景：
    cb_meta.parquet 的当前快照字段全部不可用——
        premium_now        = 100.0（东财 TRANSFER_PREMIUM_RATIO 占位值，100% 假）
        conv_value_raw     = init_conv_price 的复制（东财 TRANSFER_VALUE 填的是初始转股价）
        conv_price_raw     = 全空（TRANSFER_PRICE = None）
        stock_price_raw    = 全空（CONVERT_STOCK_PRICE = None）
    因此 fetch_cb.py 用 premium_now 锚定 C 常数的做法从一开始就错了，
    「偏差 p10/p90 = -70/+98pp」的根因是锚点取错，不是下修。

本脚本改用「初始转股价 + 复权因子」自洽反推，不依赖任何快照：
    转股价 P_t 随除权同比例调整 → P_t = P_0 * k_0 / k_t   （k = 后复权价/不复权价）
    转股价值_t = 100 / P_t * raw_t = 100 / (P_0 * k_0) * adj_t = C * adj_t
    其中 C = 100 / (init_conv_price * k_0)，k_0 为转债上市首日的正股复权因子。

自洽检验：
    正确口径下，转股期内「转债价格 / 转股价值」的负溢价应极少（存在无风险套利）。
    若反推的转股价值系统性偏低（如下修未反映），溢价率会整体偏高；
    若系统性偏高，会出现大量负溢价。两者都能从分布上看出来。

用法：python scripts/_diag_cb_convvalue.py
输出：build/cb/convvalue_diag.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qf import config as C

OUT = Path(C.BUILD) / "cb"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    z = np.load(C.PANEL_FILE, allow_pickle=False)
    dates = pd.DatetimeIndex(z["dates"])
    codes = [str(c) for c in z["codes"]]
    adj = z["close"].astype(np.float64)
    raw = np.load(C.RAWCLOSE_FILE)["rawclose"].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where((raw > 0) & (adj > 0), adj / raw, np.nan)

    meta = pd.read_parquet(Path(C.BUILD) / "altdata" / "cb_meta.parquet")
    for col in ("list_date", "delist_date", "maturity", "conv_start",
                "redeem_notice", "bond_start", "value_date"):
        meta[col] = pd.to_datetime(meta[col], errors="coerce")
    daily = pd.read_parquet(Path(C.BUILD) / "altdata" / "cb_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    pos = {c: i for i, c in enumerate(codes)}
    dpos = {t: i for i, t in enumerate(dates)}
    daily["ti"] = daily["date"].map(dpos)
    daily = daily[daily["ti"].notna()].copy()
    daily["ti"] = daily["ti"].astype(int)

    by_code = {c: g for c, g in daily.groupby("code")}
    cal = dates.values

    recs = []
    skipped = {"no_stock": 0, "no_init_cp": 0, "no_k0": 0, "no_daily": 0}
    for _, r in meta.iterrows():
        code = r["code"]
        sc = str(r["stock_code"])
        j = pos.get(sc)
        if j is None:
            skipped["no_stock"] += 1
            continue
        if pd.isna(r["init_conv_price"]) or r["init_conv_price"] <= 0:
            skipped["no_init_cp"] += 1
            continue
        g = by_code.get(code)
        if g is None or not len(g):
            skipped["no_daily"] += 1
            continue

        # 上市首日的正股复权因子 k0
        ld = r["list_date"]
        t0 = int(np.searchsorted(cal, np.datetime64(ld))) if pd.notna(ld) else g["ti"].min()
        t0 = min(max(t0, 0), len(cal) - 1)
        win = k[max(t0 - 3, 0): min(t0 + 6, len(cal)), j]
        win = win[np.isfinite(win)]
        if not win.size:
            skipped["no_k0"] += 1
            continue
        k0 = float(win[0])
        cc = 100.0 / (float(r["init_conv_price"]) * k0)

        ti = g["ti"].to_numpy()
        close = g["close"].to_numpy(dtype=np.float64)
        cv = cc * adj[ti, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            prem = np.where(cv > 0, close / cv - 1.0, np.nan)

        # 只保留：转股期开始后、上市 10 日后、且价格有效
        cs = r["conv_start"]
        ok = np.isfinite(cv) & (cv > 0) & np.isfinite(close) & (close > 0)
        if pd.notna(cs):
            ok &= ti >= int(np.searchsorted(cal, np.datetime64(cs)))
        if pd.notna(ld):
            ok &= ti >= int(np.searchsorted(cal, np.datetime64(ld))) + 10
        if pd.notna(r["maturity"]):
            ok &= np.datetime64(r["maturity"]) > cal[ti]
        for t_i, p_i, c_i, cv_i in zip(ti[ok], prem[ok], close[ok], cv[ok]):
            recs.append((code, dates[t_i], p_i, c_i, cv_i, float(adj[t_i, j])))

    df = pd.DataFrame(recs, columns=["code", "date", "premium", "cb_close", "conv_value", "adj"])
    df = df[np.isfinite(df["premium"])]

    print(f"可评估样本: {len(df):,} 行 / {df['code'].nunique()} 只转债")
    print(f"跳过: {skipped}")
    print()
    q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("反推溢价率分位(%)：")
    pv = df["premium"].to_numpy() * 100
    print("  " + "  ".join(f"p{x}={np.percentile(pv, x):.1f}" for x in q))
    print(f"  负溢价 < -1%: {(df['premium'] < -0.01).mean():.2%}"
          f"   |  < -5%: {(df['premium'] < -0.05).mean():.2%}")
    print(f"  溢价 > 100%: {(df['premium'] > 1.0).mean():.2%}")
    print()

    df["year"] = df["date"].dt.year
    print("逐年：负溢价占比 / 溢价中位数")
    rows = []
    for y, gg in df.groupby("year"):
        if len(gg) < 200:
            continue
        neg = float((gg["premium"] < -0.01).mean())
        med = float(gg["premium"].median())
        rows.append({"year": int(y), "n": int(len(gg)), "neg_share": round(neg, 4),
                     "median_premium": round(med, 4)})
        print(f"  {y}: n={len(gg):6d}  负溢价 {neg:6.2%}  中位 {med:7.1%}")

    # 溢价异常高的债 = 疑似下修未反映（转股价偏高→转股价值偏低→溢价虚高）
    hi = (df.groupby("code")["premium"].median().sort_values(ascending=False).head(15))
    print()
    print("溢价中位数最高的 15 只（疑似下修或转股价失真）：")
    nm = meta.set_index("code")["name"]
    for c, v in hi.items():
        print(f"  {c} {str(nm.get(c,''))[:10]:10s} 中位溢价 {v:8.1%}  n={int((df['code']==c).sum())}")

    # ---- 强赎公告日锚点检验 ----------------------------------------
    # 强赎触发条件：正股连续 30 个交易日中 >=15 日收盘价 >= 转股价的 130%。
    # 故公告日的转股价值理论上应 >=130。若反推值显著低于 130，
    # 说明 P_t 因为漏掉下修而偏大（转股价值被低估）。
    by_code = {c: g for c, g in daily.groupby("code")}
    anch = []
    for _, r in meta.iterrows():
        nd = r["redeem_notice"]
        if pd.isna(nd) or pd.isna(r["init_conv_price"]) or r["init_conv_price"] <= 0:
            continue
        j = pos.get(str(r["stock_code"]))
        if j is None or pd.isna(r["list_date"]):
            continue
        g = by_code.get(r["code"])
        if g is None:
            continue
        t0 = int(np.searchsorted(cal, np.datetime64(r["list_date"])))
        w = k[max(t0 - 3, 0): t0 + 6, j]
        w = w[np.isfinite(w)]
        if not w.size:
            continue
        ccs = 100.0 / (float(r["init_conv_price"]) * float(w[0]))
        gd = g[(g["date"] >= nd - pd.Timedelta(days=5)) & (g["date"] <= nd + pd.Timedelta(days=5))]
        if not len(gd):
            continue
        gi = gd.iloc[len(gd) // 2]
        ti, cl = int(gi["ti"]), float(gi["close"])
        cvv = ccs * adj[ti, j]
        if not (np.isfinite(cvv) and cvv > 0 and cl > 0):
            continue
        anch.append({"code": r["code"], "name": r["name"], "date": str(dates[ti].date()),
                     "conv_value": float(cvv), "cb_close": cl, "premium": float(cl / cvv - 1)})
    ad = pd.DataFrame(anch)
    print()
    if len(ad):
        cvp = ad["conv_value"].to_numpy()
        print(f"=== 强赎公告日锚点检验（n={len(ad)}）===")
        print("  反推转股价值分位:", np.round(np.percentile(cvp, [5, 25, 50, 75, 95]), 1))
        print(f"  <100 占比 {np.mean(cvp < 100):.1%}   <130 占比 {np.mean(cvp < 130):.1%}")
        print("  溢价率分位(%):", np.round(np.percentile(ad['premium'] * 100, [5, 25, 50, 75, 95]), 1))
    else:
        print("=== 强赎公告日锚点检验：无样本 ===")

    out = {
        "redeem_anchor": {
            "n": int(len(ad)),
            "conv_value_pct": ([float(x) for x in np.percentile(ad["conv_value"], [5, 25, 50, 75, 95])]
                               if len(ad) else []),
            "below_130_share": (float((ad["conv_value"] < 130).mean()) if len(ad) else None),
            "premium_pct": ([float(x) for x in np.percentile(ad["premium"] * 100, [5, 25, 50, 75, 95])]
                            if len(ad) else []),
        },

        "n_rows": int(len(df)), "n_codes": int(df["code"].nunique()),
        "skipped": skipped,
        "premium_pct": {f"p{x}": float(np.percentile(pv, x)) for x in q},
        "neg_share_1pct": float((df["premium"] < -0.01).mean()),
        "neg_share_5pct": float((df["premium"] < -0.05).mean()),
        "by_year": rows,
        "worst_codes": [{"code": c, "name": str(nm.get(c, "")), "median_premium": float(v)}
                        for c, v in hi.items()],
    }
    (OUT / "convvalue_diag.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {OUT / 'convvalue_diag.json'}")


if __name__ == "__main__":
    main()
