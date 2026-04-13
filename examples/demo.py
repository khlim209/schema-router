"""
End-to-end demo of the GraphRAG Query Router.

Simulates an e-commerce data platform with three databases:
  - ecommerce   (orders, products, customers, order_items)
  - marketing   (campaigns, email_logs, ab_tests)
  - analytics   (daily_stats, funnel_events, cohorts)

Steps:
  1. Register schemas
  2. Ingest a synthetic access log (historical query → table accesses)
  3. Rebuild communities
  4. Route new queries and print results
  5. Show improvement over naive text similarity
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pprint import pprint
from router import QueryRouter
from graph_rag.indexer import SchemaDefinition
from trainer import Trainer


# ──────────────────────────────────────────────────────────────────────── #
#  1. Schema definitions                                                    #
# ──────────────────────────────────────────────────────────────────────── #

SCHEMAS = [
    SchemaDefinition(
        db_name="ecommerce",
        description="Core transactional database",
        tables={
            "orders": {
                "description": "Customer purchase orders",
                "columns": [
                    ("order_id", "bigint"), ("customer_id", "bigint"),
                    ("order_date", "date"), ("total_amount", "decimal"),
                    ("status", "varchar"),
                ],
                "joins": [("order_items", "order_id"), ("customers", "customer_id")],
            },
            "order_items": {
                "description": "Line items within each order",
                "columns": [
                    ("item_id", "bigint"), ("order_id", "bigint"),
                    ("product_id", "bigint"), ("quantity", "int"),
                    ("unit_price", "decimal"),
                ],
                "joins": [("orders", "order_id"), ("products", "product_id")],
            },
            "products": {
                "description": "Product catalogue",
                "columns": [
                    ("product_id", "bigint"), ("name", "varchar"),
                    ("category", "varchar"), ("price", "decimal"),
                    ("stock_qty", "int"),
                ],
                "joins": [],
            },
            "customers": {
                "description": "Registered customer accounts",
                "columns": [
                    ("customer_id", "bigint"), ("name", "varchar"),
                    ("email", "varchar"), ("signup_date", "date"),
                    ("country", "varchar"),
                ],
                "joins": [("orders", "customer_id")],
            },
        },
    ),
    SchemaDefinition(
        db_name="marketing",
        description="Marketing campaign and email data",
        tables={
            "campaigns": {
                "description": "Advertising campaigns",
                "columns": [
                    ("campaign_id", "bigint"), ("name", "varchar"),
                    ("channel", "varchar"), ("start_date", "date"),
                    ("budget", "decimal"),
                ],
                "joins": [("email_logs", "campaign_id")],
            },
            "email_logs": {
                "description": "Email send / open / click events",
                "columns": [
                    ("log_id", "bigint"), ("campaign_id", "bigint"),
                    ("customer_id", "bigint"), ("event_type", "varchar"),
                    ("event_time", "timestamp"),
                ],
                "joins": [("campaigns", "campaign_id")],
            },
            "ab_tests": {
                "description": "A/B experiment results",
                "columns": [
                    ("test_id", "bigint"), ("variant", "varchar"),
                    ("metric", "varchar"), ("value", "decimal"),
                    ("recorded_at", "date"),
                ],
                "joins": [],
            },
        },
    ),
    SchemaDefinition(
        db_name="analytics",
        description="Aggregated analytics and KPI data",
        tables={
            "daily_stats": {
                "description": "Daily aggregate metrics (revenue, orders, users)",
                "columns": [
                    ("stat_date", "date"), ("revenue", "decimal"),
                    ("order_count", "int"), ("new_users", "int"),
                    ("active_users", "int"),
                ],
                "joins": [],
            },
            "funnel_events": {
                "description": "User conversion funnel events",
                "columns": [
                    ("event_id", "bigint"), ("user_id", "bigint"),
                    ("step", "varchar"), ("event_time", "timestamp"),
                ],
                "joins": [],
            },
            "cohorts": {
                "description": "User cohort retention analysis",
                "columns": [
                    ("cohort_month", "date"), ("cohort_size", "int"),
                    ("retention_d7", "float"), ("retention_d30", "float"),
                ],
                "joins": [],
            },
        },
    ),
]


# ──────────────────────────────────────────────────────────────────────── #
#  2. Synthetic access log                                                  #
# ──────────────────────────────────────────────────────────────────────── #

ACCESS_LOG = [
    # ── Revenue / sales queries ─────────────────────────────────────────
    {"query_text": "지난달 총 매출액",                     "db_name": "ecommerce",  "table_name": "orders",       "count": 45},
    {"query_text": "monthly revenue last month",          "db_name": "ecommerce",  "table_name": "orders",       "count": 38},
    {"query_text": "이번 달 주문 건수",                    "db_name": "ecommerce",  "table_name": "orders",       "count": 32},
    {"query_text": "total sales this quarter",            "db_name": "analytics",  "table_name": "daily_stats",  "count": 28},
    {"query_text": "분기별 매출 합계",                     "db_name": "analytics",  "table_name": "daily_stats",  "count": 25},
    {"query_text": "revenue by day for the past week",    "db_name": "analytics",  "table_name": "daily_stats",  "count": 20},

    # ── Product / bestseller queries ─────────────────────────────────────
    {"query_text": "가장 많이 팔린 상품 TOP 10",           "db_name": "ecommerce",  "table_name": "order_items",  "count": 60},
    {"query_text": "best selling products last 30 days",  "db_name": "ecommerce",  "table_name": "order_items",  "count": 55},
    {"query_text": "인기 상품 목록",                       "db_name": "ecommerce",  "table_name": "order_items",  "count": 50},
    {"query_text": "상품별 판매 수량",                     "db_name": "ecommerce",  "table_name": "order_items",  "count": 42},
    {"query_text": "product sales ranking",               "db_name": "ecommerce",  "table_name": "products",     "count": 35},
    {"query_text": "카테고리별 판매 통계",                 "db_name": "ecommerce",  "table_name": "products",     "count": 30},

    # ── Customer queries ─────────────────────────────────────────────────
    {"query_text": "신규 가입 고객 수",                    "db_name": "ecommerce",  "table_name": "customers",    "count": 40},
    {"query_text": "new customer registrations this week","db_name": "ecommerce",  "table_name": "customers",    "count": 35},
    {"query_text": "고객별 구매 이력",                     "db_name": "ecommerce",  "table_name": "orders",       "count": 28},
    {"query_text": "customer lifetime value",             "db_name": "ecommerce",  "table_name": "orders",       "count": 22},
    {"query_text": "재구매율 분석",                        "db_name": "analytics",  "table_name": "cohorts",      "count": 18},

    # ── Marketing queries ─────────────────────────────────────────────────
    {"query_text": "이메일 오픈율",                        "db_name": "marketing",  "table_name": "email_logs",   "count": 35},
    {"query_text": "email open rate by campaign",         "db_name": "marketing",  "table_name": "email_logs",   "count": 30},
    {"query_text": "캠페인별 클릭률",                      "db_name": "marketing",  "table_name": "email_logs",   "count": 28},
    {"query_text": "campaign performance overview",       "db_name": "marketing",  "table_name": "campaigns",    "count": 25},
    {"query_text": "A/B 테스트 결과",                      "db_name": "marketing",  "table_name": "ab_tests",     "count": 20},

    # ── Funnel / conversion queries ───────────────────────────────────────
    {"query_text": "구매 전환율",                          "db_name": "analytics",  "table_name": "funnel_events","count": 32},
    {"query_text": "conversion funnel drop-off",          "db_name": "analytics",  "table_name": "funnel_events","count": 28},
    {"query_text": "사용자 퍼널 분석",                     "db_name": "analytics",  "table_name": "funnel_events","count": 25},
]


# ──────────────────────────────────────────────────────────────────────── #
#  Test queries (unseen — not in access log)                               #
# ──────────────────────────────────────────────────────────────────────── #

TEST_QUERIES = [
    # Should route to: ecommerce.order_items (bestseller community)
    "지난 3개월간 베스트셀러 상품",

    # Should route to: analytics.daily_stats (revenue community)
    "이번 주 일별 매출 트렌드",

    # Should route to: ecommerce.customers (customer community)
    "이번 달 새로 가입한 유저",

    # Should route to: marketing.email_logs (marketing community)
    "최근 발송한 뉴스레터의 클릭 수",

    # Should route to: analytics.funnel_events (funnel community)
    "checkout 페이지 이탈률",

    # Paraphrased (DBCopilot §4: paraphrase robustness test)
    "What were the top-performing SKUs this month?",
    "How many users converted last week?",
]


# ──────────────────────────────────────────────────────────────────────── #
#  Main demo                                                                #
# ──────────────────────────────────────────────────────────────────────── #

def main():
    print("=" * 60)
    print("  GraphRAG Query Router — End-to-End Demo")
    print("=" * 60)

    # Build router  (total_tables=10: 전체 등록 테이블 수 → brute-force 기준)
    print("\n[1] Building QueryRouter…")
    router = QueryRouter.build(total_tables=10)
    trainer = Trainer(router, router._neo4j)

    # Register schemas
    print("\n[2] Registering schemas…")
    router.register_schemas(SCHEMAS)
    print(f"  ✓ {len(SCHEMAS)} databases registered.")

    # Print DFS schema paths (DBCopilot-style)
    print("\n[3] DFS schema paths for 'ecommerce' (DBCopilot §3.2):")
    for path in router.dfs_schema_paths("ecommerce")[:8]:
        print(f"  {path}")
    print("  …")

    # Ingest access log
    print("\n[4] Ingesting access log…")
    router.load_history(ACCESS_LOG)
    print(f"  ✓ {len(ACCESS_LOG)} access records ingested.")

    # Rebuild communities
    print("\n[5] Running community detection (Leiden / Louvain)…")
    communities = router.rebuild_communities()
    print(f"  ✓ {len(communities)} communities detected.")
    for cid, members in communities.items():
        print(f"     Community {cid}: {len(members)} queries")

    # Routing test
    print("\n[6] Routing test queries:")
    print("-" * 60)
    for query in TEST_QUERIES:
        results = router.route(query, top_n=3)
        print(f"\nQuery: '{query}'")
        if not results:
            print("  → No results found.")
        for i, p in enumerate(results, 1):
            print(
                f"  [{i}] {p.db}.{p.table}  "
                f"score={p.score:.4f}  "
                f"(sim={p.evidence['embedding_sim']:.3f}, "
                f"access={p.evidence['access_count']}, "
                f"community={p.evidence['community_id']})"
            )

    # Full explanation for one query
    print("\n[7] Full explanation:")
    print("-" * 60)
    info = router.explain("지난달 베스트셀러 상품", top_n=3)
    pprint(info)

    # Training report
    print("\n[8] Training report:")
    print("-" * 60)
    pprint(trainer.report())

    # ── Tiered routing: 쿼리 횟수 최소화 비교 ───────────────────────────
    print("\n[9] Tiered routing — 쿼리 횟수 비교 (brute-force vs GraphRAG):")
    print("-" * 60)
    print(f"  {'Query':<40} {'Tier':<14} {'Brute':>6} {'GraphRAG':>9} {'Saved':>6}")
    print(f"  {'-'*40} {'-'*14} {'-'*6} {'-'*9} {'-'*6}")

    all_queries = TEST_QUERIES + [
        # 동일 쿼리 재등장 → Tier 1 캐시 적중 확인
        "지난 3개월간 베스트셀러 상품",
        "이번 주 일별 매출 트렌드",
    ]
    for q in all_queries:
        result = router.route_efficient(q, top_n=5)
        brute  = router._tiered._stats.total_tables // 2
        top    = result.paths[0] if result.paths else None
        dest   = f"{top.db}.{top.table}" if top else "N/A"
        print(
            f"  {q[:40]:<40} {result.tier.value:<14} "
            f"{brute:>6} {result.lookup_count:>9} {result.saved_lookups:>6}"
            f"  → {dest}"
        )

    print("\n[10] 누적 라우팅 통계:")
    print("-" * 60)
    pprint(router.routing_stats())

    print("\n[11] 시뮬레이션 요약 (brute-force vs GraphRAG):")
    print("-" * 60)
    sim = router.simulate_savings(TEST_QUERIES)
    print(f"  Brute-force 총 쿼리 수 : {sim['brute_force_total']}")
    print(f"  GraphRAG    총 쿼리 수 : {sim['graphrag_total']}")
    print(f"  절감된 쿼리 수         : {sim['total_saved']}")
    print(f"  쿼리 횟수 절감률       : {sim['reduction_pct']}")

    # Cleanup
    router.close()
    print("\n✓ Demo complete.")


if __name__ == "__main__":
    main()
