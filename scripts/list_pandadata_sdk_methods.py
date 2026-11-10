# -*- coding: utf-8 -*-
"""枚举 panda_data SDK 的全部方法，按业务主题分类，输出 Markdown 清单。

不需要凭据（纯反射）。
用法：
    D:/Python/Python313/python.exe scripts/list_pandadata_sdk_methods.py
输出：
    docs/PANDADATA_SDK_METHODS.md
"""
import inspect
import pathlib

import panda_data


# 主题 -> (匹配规则, 说明)。按顺序匹配，先命中先归。
THEMES = [
    ("Barra 风格因子", lambda n: "style" in n or n in ("get_rf_rate",),
     "风格暴露 / 日度收益 / 残差 / 协方差 / 无风险利率。MCP 全部 Unknown，SDK 有"),
    ("因子与复权", lambda n: n in ("get_factor", "get_factor_hk", "get_adj_factor",
                                "get_adj_factor_hk", "get_adj_factor_us"),
     "get_factor 是 PIT 财报+量价宽表；get_adj_factor 是复权因子"),
    ("财务报表", lambda n: n.startswith("get_fina") or n in (
        "get_financial_statement_daily", "get_audit_opinion"),
     "fina_statement / fina_reports / fina_ex / fina_forecast / fina_performance"),
    ("可转债", lambda n: n.startswith("get_bond_conv") or n.startswith("get_bond_"),
     "MCP 整批 Unknown，SDK 有——这是当前唯一能改变「可转债不可回测」的东西"),
    ("股票行情", lambda n: n.startswith("get_stock_daily") or n.startswith("get_stock_min")
                  or n.startswith("get_stock_rt") or n in (
                      "get_stock_weekly", "get_stock_monthly", "get_stock_detail",
                      "get_stock_detail_all", "get_market_data", "get_market_min_data"),
     "日/周/月/分钟/实时行情"),
    ("股票指标", lambda n: n.startswith("get_stock_pv") or n.startswith("get_stock_mktfin")
                  or n.startswith("get_stock_operating")
                  or n in ("get_stock_industry_median", "get_stock_sector_median"),
     "pv / mktfin / operating 三套指标，以及行业中位数（可直接做横截面标准化）"),
    ("一致预期与分析师", lambda n: "consensus" in n or "recommendation" in n or "estimate" in n,
     "ncycl_consensus / recommendation_estimate / recommendation_consensus"),
    ("股东与股权", lambda n: n.startswith("get_stock_shareholder") or n in (
        "get_top_holders", "get_holder_count", "get_share_float", "get_stock_equity_nature",
        "get_stock_equity_illegal", "get_stock_equity_placard"),
     "股东持股 / 股东户数 / 股本结构"),
    ("质押与解禁", lambda n: "pledge" in n or "restricted" in n or "cumu_guarantee" in n,
     "质押、解禁、累计担保"),
    ("分红与股本变动", lambda n: "dividend" in n or "allotment" in n or "split" in n
                       or "repurchase" in n or "private_placement" in n or "over_allotment" in n,
     "分红、配股、拆股、回购、定增、超配"),
    ("公司事件", lambda n: any(k in n for k in (
        "litigation", "material_contract", "meeting", "ir_activity", "ir_event",
        "csrc_approval", "status_change", "market_event", "market_activity",
        "financial_event", "financial_activity", "status_over_allotment")),
     "诉讼、重大合同、股东大会、投资者关系、证监会审核、状态变更"),
    ("IPO", lambda n: "ipo" in n, "首发上市、发行、询价、网下配售"),
    ("内部人与高管", lambda n: "insider" in n, "内部人交易"),
    ("董监高与投资者", lambda n: n.startswith("get_investor") or "leaderboard" in n
                       or "concentration" in n or "centralization" in n,
     "投资者活动、集中度、榜单"),
    ("资金与龙虎榜", lambda n: n.startswith("get_lhb") or n in (
        "get_stock_flow", "get_margin", "get_hsgt_hold", "get_block_trade"),
     "龙虎榜、资金流、两融、北向、大宗。MCP 里北向是死的，SDK 待验"),
    ("经纪商", lambda n: n.startswith("get_broker_"), "席位盈亏、评级、净持仓"),
    ("指数", lambda n: n.startswith("get_index_"), "行情、成分、权重、估值"),
    ("行业与概念", lambda n: n.startswith("get_industry") or n.startswith("get_concept"),
     "行业成分、概念成分"),
    ("基金与 ETF", lambda n: n.startswith("get_fund_"), "净值、持仓、风格、ETF 申赎"),
    ("期货", lambda n: n.startswith("get_future_"), "行情、持仓、基差、库存、期限结构"),
    ("期权", lambda n: n.startswith("get_option_"), "行情、隐含波动率、风险指标"),
    ("宏观", lambda n: n.startswith("get_macro"), "约 70 个宏观序列 + 宏观日历"),
    ("港股/美股", lambda n: n.startswith("get_hk_") or n.startswith("get_us_")
                  or n.startswith("get_oversea"), "港股、美股、海外"),
    ("交易工具", lambda n: n.startswith("get_trade") or n.startswith("get_last_trade")
                  or n.startswith("get_prev_trade") or n == "get_suspended_list"
                  or n == "get_st_list", "交易日历、代码表、停牌、ST"),
    ("认证与客户端", lambda n: n in (
        "init", "init_token", "is_authenticated", "auth_info", "auth_expires_at",
        "auth_remaining_seconds", "get_token_ttl", "clear_auth", "get_client"), "会话管理"),
]


def main():
    names = sorted(n for n in dir(panda_data)
                   if n.startswith("get_") and callable(getattr(panda_data, n)))
    print(f"SDK 共 {len(names)} 个 get_* 方法")

    buckets = {t: [] for t, _, _ in THEMES}
    buckets["其他"] = []
    for n in names:
        for theme, rule, _ in THEMES:
            try:
                if rule(n):
                    buckets[theme].append(n)
                    break
            except Exception:
                pass
        else:
            buckets["其他"].append(n)

    lines = [
        "# PandaData SDK 方法清单",
        "",
        f"由 `scripts/list_pandadata_sdk_methods.py` 反射生成，共 **{len(names)}** 个 `get_*` 方法。",
        "包：`panda_data` 1.0.5（PyPI，纯 Python wheel）。",
        "",
        "> MCP 网关只暴露了 17 个可用方法，而 SDK 有这一整张表。",
        "> **MCP 报 `Unknown method` 不等于数据不存在，只是网关没部署那个接口。**",
        "",
        "## 分类汇总",
        "",
        "| 主题 | 方法数 | 说明 |",
        "|---|---:|---|",
    ]
    for theme, _, desc in THEMES:
        lines.append(f"| **{theme}** | {len(buckets[theme])} | {desc} |")
    if buckets["其他"]:
        lines.append(f"| 其他 | {len(buckets['其他'])} | — |")

    for theme, _, desc in THEMES:
        if not buckets[theme]:
            continue
        lines += ["", f"## {theme}（{len(buckets[theme])}）", "", f"{desc}", "", "```"]
        lines += buckets[theme]
        lines += ["```"]
    if buckets["其他"]:
        lines += ["", f"## 其他（{len(buckets['其他'])}）", "", "```"]
        lines += buckets["其他"]
        lines += ["```"]

    out = pathlib.Path(__file__).resolve().parent.parent / "docs" / "PANDADATA_SDK_METHODS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out}")
    for theme, _, _ in THEMES:
        if buckets[theme]:
            print(f"  {theme:16s} {len(buckets[theme]):3d}")


if __name__ == "__main__":
    main()
