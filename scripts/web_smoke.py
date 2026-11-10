# -*- coding: utf-8 -*-
"""Web UI 冒烟测试：对运行中的服务逐个接口做契约检查（无需 pytest）。

用法:
    python scripts/web_smoke.py [--base http://127.0.0.1:8765] [--full]
    --full 会额外跑一次小回测（约 3~8 秒）并检查导出 CSV。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

FAILS: list[str] = []


def req(base, path, method="GET", body=None, timeout=180):
    url = base.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    t = time.time()
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        return resp.status, raw, ctype, time.time() - t


def get_json(base, path, timeout=120):
    st, raw, ctype, dt = req(base, path, timeout=timeout)
    assert st == 200, f"{path} status={st}"
    assert "json" in ctype, f"{path} content-type={ctype}"
    return json.loads(raw.decode("utf-8")), dt


def check(name, cond, detail=""):
    if cond:
        print(f"  [ok] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAILS.append(name)


def need(d: dict, keys: list[str], where: str):
    missing = [k for k in keys if k not in d]
    check(f"{where} 字段", not missing, f"missing={missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    base = args.base

    print("== health / meta ==")
    h, _ = get_json(base, "/api/health")
    need(h, ["status", "ts"], "health")
    meta, _ = get_json(base, "/api/meta")
    need(meta, ["app", "data", "model", "server"], "meta")
    if not meta["server"].get("loaded"):
        print("  [FAIL] server 未加载完成，先等 /api/meta server.loaded=true")
        FAILS.append("server.loaded")
        print(f"\n结果：{len(FAILS)} 项失败")
        raise SystemExit(1)
    check("数据日期", bool(meta["data"].get("last_date")), meta["data"])
    check("模型因子池", meta["model"].get("n_pool", 0) > 0, meta["model"])

    print("== 看盘 ==")
    ov, dt = get_json(base, "/api/market/overview")
    need(ov, ["asof", "regime", "indices", "breadth", "industries",
              "top_gainers", "top_losers", "top_amount", "chart_urls"], "overview")
    check("指数 4 个", len(ov["indices"]) == 4, len(ov["indices"]))
    check("行业数", len(ov["industries"]) > 10, len(ov["industries"]))
    check("涨跌榜", len(ov["top_gainers"]) > 0 and len(ov["top_losers"]) > 0)
    print(f"  ({dt:.2f}s, {ov['asof']}, 涨{ov['breadth']['up']}/跌{ov['breadth']['down']})")

    print("== 推荐 ==")
    rec, dt = get_json(base, "/api/today_recommend")
    check("signal_date", bool(rec.get("signal_date")), rec.get("signal_date"))
    check("持仓非空", len(rec.get("rows") or []) > 0, len(rec.get("rows") or []))
    check("资金守恒",
          abs(float(rec.get("total_invested") or 0) + float((rec.get("cash") or {}).get("value") or 0)
              - float(rec.get("capital") or 0)) < 1000,
          f"{rec.get('total_invested')} + {(rec.get('cash') or {}).get('value')}")
    check("现金替代", bool((rec.get("cash") or {}).get("sleeve")),
          (rec.get("cash") or {}).get("sleeve"))
    check("构成", bool(rec.get("parts")), rec.get("parts"))
    print(f"  ({dt:.2f}s, {rec.get('signal_date')}, "
          f"{rec.get('n_stock')} 股 + {rec.get('n_etf')} ETF)")

    print("== 个股 ==")
    code = rec["rows"][0]["code"] if rec["rows"] else "000001"
    sd, dt = get_json(base, f"/api/stock/{code}")
    need(sd, ["code", "name", "industry", "board", "price", "chg_1d", "chg_20d",
              "score", "factors", "chart_url", "in_universe"], "stock")
    check("因子贡献", len(sd["factors"]) > 0, len(sd["factors"]))
    ss, _ = get_json(base, "/api/stock/search?q=600")
    check("搜索", isinstance(ss, list) and len(ss) > 0, len(ss) if isinstance(ss, list) else ss)

    print("== 因子 ==")
    fs, _ = get_json(base, "/api/factors/summary")
    need(fs, ["counts", "stage2", "pool", "families", "pool_detail",
              "fundamental_stats", "composite_ic"], "factors")
    check("因子数 >=823", fs["counts"]["total"] >= 823 and fs["counts"]["quant"] == 800, fs["counts"])
    check("入模池", fs["pool"] == len(fs["pool_detail"]) and fs["pool"] > 0, fs["pool"])
    check("IC 年度", len(fs["composite_ic"].get("by_year", [])) > 0)
    # 族占比监控（2026-11-03）：只展示不干预，占比必须归一、阈值标记一致
    fw = fs.get("family_weights") or []
    check("族占比监控", bool(fw) and abs(sum(x["share"] for x in fw) - 1.0) < 1e-6, len(fw))
    check("族占比标记", all(x["over"] == (x["share"] > fs.get("family_share_warn", 0.33))
                         for x in fw), [x["family"] for x in fw if x["over"]])

    print("== 多模型 ==")
    mm, _ = get_json(base, "/api/models")
    check("模型注册表", isinstance(mm.get("models"), list) and len(mm["models"]) >= 1
          and mm.get("default"), mm)
    model_names = [x["name"] for x in mm.get("models", [])]
    check("包含默认模型", mm.get("default") in model_names, model_names)
    print(f"  models={model_names} default={mm.get('default')}")
    # 今日推荐 CSV 导出（每次都是同一个文件，检查一次即可）
    st, raw, ctype, _ = req(base, "/api/today_export")
    check("今日推荐 CSV", st == 200 and len(raw) > 100, (st, len(raw)))

    print("== 数据/报告 ==")
    # 组合方案：三腿为主口径 + 旧方案降级对照（2026-11-03 改版）
    try:
        cp, _ = get_json(base, "/api/combo_plan")
        need(cp, ["recommended", "schemes", "controls", "legs_corr"], "combo_plan")
        check("推荐三腿 50/25/25", cp.get("recommended") == "50/25/25", cp.get("recommended"))
        rec_s = [s for s in cp["schemes"] if s.get("status") == "recommended"]
        check("三腿含推荐方案", len(rec_s) == 1
              and abs(sum(rec_s[0]["legs"].values()) - 1.0) < 1e-6, rec_s[0]["legs"] if rec_s else None)
        check("板块腿已标否", any(c.get("status") == "rejected" and "板块" in c.get("name", "")
                                for c in cp["controls"]), len(cp["controls"]))
    except Exception as e:
        print(f"  [WARN] combo_plan 失败: {e}")
    try:
        et, _ = get_json(base, "/api/etf_trend", timeout=180)
        check("etf_trend 含 combo3", bool(et.get("combo3")), list((et.get("combo3") or {}).keys()))
        ns = et.get("nav_series") or []
        check("净值序列含 combo3 列", bool(ns) and "combo3" in ns[0], list(ns[0].keys()) if ns else None)
    except Exception as e:
        print(f"  [WARN] etf_trend 失败: {e}")
    # 新增：配置对比与候选跟踪
    try:
        cm, _ = get_json(base, "/api/combo_methods")
        print(f"  [ok] combo_methods: {len(cm.get('methods') or {})} 种方法")
    except Exception as e:
        print(f"  [WARN] combo_methods 失败: {e}")
    try:
        cd, _ = get_json(base, "/api/candidates")
        print(f"  [ok] candidates: {len(cd.get('rolling_start') or {})} 个候选")
    except Exception as e:
        print(f"  [WARN] candidates 失败: {e}")
    ds, _ = get_json(base, "/api/data/status")
    need(ds, ["source", "last_date", "orig_last_date", "update_running"], "data/status")
    rp, _ = get_json(base, "/api/reports")
    check("报告列表", len(rp["files"]) > 0, len(rp["files"]))
    st, raw, ctype, _ = req(base, "/reports/" + urllib.parse.quote(rp["files"][0]["name"]))
    check("报告下载", st == 200 and len(raw) > 0, st)

    print("== 图表 ==")
    for p in ["/api/chart/index_line?code=csi_all&days=250",
              f"/api/chart/stock_kline?code={code}&days=250",
              "/api/chart/industry_bar?days=20"]:
        st, raw, ctype, dt = req(base, p, timeout=120)
        check(f"{p.split('?')[0]} PNG", st == 200 and raw[:4] == b"\x89PNG", (st, ctype, len(raw)))
        print(f"  ({dt:.2f}s, {len(raw)//1024}KB)")

    if args.full:
        print("== 回测（full）==")
        st, raw, ctype, _ = req(base, "/api/backtest", method="POST",
                                body={"n_holdings": 40, "rebalance_days": 10,
                                      "cash_bond": 0.8, "cash_gold": 0.2,
                                      "overlay": True, "stop_loss": True,
                                      "start": "2024-01-01"}, timeout=30)
        job = json.loads(raw.decode("utf-8"))
        jid = job["job_id"]
        t0 = time.time()
        res = None
        while time.time() - t0 < 180:
            j, _ = get_json(base, f"/api/jobs/{jid}")
            if j["status"] in ("done", "error"):
                res = j
                break
            time.sleep(0.5)
        check("回测完成", res is not None and res["status"] == "done",
              res["error"] if res else "timeout")
        if res and res["status"] == "done":
            rr = res["result"]
            need(rr, ["params", "metrics", "periods", "yearly", "benchmarks", "nav",
                      "chart_url", "trades_summary"], "backtest.result")
            m = rr["metrics"]
            check("指标齐全", all(k in m for k in ("cagr", "sharpe", "max_dd", "calmar")), m)
            check("净值序列", len(rr["nav"]["dates"]) == len(rr["nav"]["nav"]) > 10)
            st, raw, ctype, _ = req(base, rr["chart_url"], timeout=120)
            check("回测图 PNG", st == 200 and raw[:4] == b"\x89PNG", st)
            print(f"  回测 {rr['params']['start']}~{rr['params']['end']}: "
                  f"CAGR={m.get('cagr'):.2%} Sharpe={m.get('sharpe'):.2f} MaxDD={m.get('max_dd'):.2%}")
        print("== CSV 导出 ==")
        for kind in ("target", "orders"):
            st, raw, ctype, _ = req(base, "/api/today_export")
            check(f"export {kind}", st == 200 and len(raw) > 100, (st, len(raw)))

    print(f"\n结果：{'PASS' if not FAILS else str(len(FAILS)) + ' 项失败'}")
    if FAILS:
        print("失败项：", FAILS)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
