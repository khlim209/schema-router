"""
GraphRAG Query Router — Benchmark

4가지 방법을 동일한 ground-truth 레이블로 비교한다.

비교 방법
----------
  BruteForce       : 모든 테이블을 무작위 순서로 탐색 (기저선)
  TextSimilarity   : 순수 코사인 유사도만 사용 (그래프 없음)
  GraphRAG_NoComm  : FAISS + 접근 횟수 가중치 (커뮤니티 없음)
  GraphRAG_Full    : FAISS + 접근 가중치 + 커뮤니티 탐지 (완전한 시스템)

측정 지표
----------
  hit@1            : 1위 결과가 정답 테이블인 비율
  hit@3            : 상위 3위 안에 정답이 있는 비율
  mrr              : Mean Reciprocal Rank (1/정답_순위 평균)
  avg_lookup       : 정답을 찾기까지 평균 탐색 횟수 (= 정답 순위)
  total_lookups    : 모든 쿼리의 탐색 횟수 합계
  lookup_reduction : BruteForce 대비 탐색 횟수 절감률 (%)
  avg_latency_ms   : 쿼리당 평균 응답 시간

GraphRAG만의 기여 분리
-----------------------
  TextSimilarity → GraphRAG_NoComm : 접근 그래프 효과
  GraphRAG_NoComm → GraphRAG_Full  : 커뮤니티 탐지 효과
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pprint import pprint
from typing import Callable

import numpy as np

from embedding.embedder import get_embedder
from embedding.faiss_index import FaissQueryIndex
from graph_db.neo4j_client import Neo4jClient
from graph_rag.community import CommunityDetector
from graph_rag.retriever import GraphRetriever, SchemaPath
from router import QueryRouter
from examples.demo import ACCESS_LOG, SCHEMAS


# ──────────────────────────────────────────────────────────────────────── #
#  Ground truth                                                             #
#  (query_text, correct_db, correct_table)                                 #
#  히스토리에 없는 새로운 표현으로 작성 → 일반화 성능 측정                  #
# ──────────────────────────────────────────────────────────────────────── #

GROUND_TRUTH: list[tuple[str, str, str]] = [
    # ── 판매/상품 ──────────────────────────────────────────────────────
    ("지난 3개월간 베스트셀러 상품",           "ecommerce",  "order_items"),
    ("이번 달 가장 많이 팔린 카테고리",        "ecommerce",  "order_items"),
    ("상품 판매 순위 TOP 5",                   "ecommerce",  "order_items"),
    ("What were the top-selling items?",       "ecommerce",  "order_items"),
    ("재고 부족 상품 목록",                    "ecommerce",  "products"),
    ("카테고리별 평균 판매가",                 "ecommerce",  "products"),

    # ── 매출/주문 ──────────────────────────────────────────────────────
    ("이번 주 일별 매출 트렌드",               "analytics",  "daily_stats"),
    ("전월 대비 주문 증감률",                  "analytics",  "daily_stats"),
    ("Weekly revenue summary",                 "analytics",  "daily_stats"),
    ("오늘 신규 주문 건수",                    "ecommerce",  "orders"),
    ("주문 취소율 분석",                       "ecommerce",  "orders"),

    # ── 고객 ────────────────────────────────────────────────────────────
    ("이번 달 새로 가입한 유저",               "ecommerce",  "customers"),
    ("해외 고객 비율",                         "ecommerce",  "customers"),
    ("New user signups this week",             "ecommerce",  "customers"),
    ("고객 재구매 패턴 분석",                  "analytics",  "cohorts"),
    ("월별 고객 유지율",                       "analytics",  "cohorts"),

    # ── 마케팅 ──────────────────────────────────────────────────────────
    ("최근 뉴스레터 오픈율",                   "marketing",  "email_logs"),
    ("캠페인 클릭률 비교",                     "marketing",  "email_logs"),
    ("Email campaign performance",             "marketing",  "campaigns"),
    ("A/B 테스트 승자 결정",                   "marketing",  "ab_tests"),

    # ── 퍼널/전환 ───────────────────────────────────────────────────────
    ("checkout 페이지 이탈률",                 "analytics",  "funnel_events"),
    ("구매 전환 단계별 드롭오프",              "analytics",  "funnel_events"),
    ("User conversion funnel analysis",        "analytics",  "funnel_events"),
]

ALL_TABLES = [
    ("ecommerce", "orders"), ("ecommerce", "order_items"),
    ("ecommerce", "products"), ("ecommerce", "customers"),
    ("marketing", "campaigns"), ("marketing", "email_logs"),
    ("marketing", "ab_tests"), ("analytics", "daily_stats"),
    ("analytics", "funnel_events"), ("analytics", "cohorts"),
]
N_TABLES = len(ALL_TABLES)


# ──────────────────────────────────────────────────────────────────────── #
#  Metric helpers                                                           #
# ──────────────────────────────────────────────────────────────────────── #

@dataclass
class MethodResult:
    name:          str
    hit1:          list[bool]  = field(default_factory=list)
    hit3:          list[bool]  = field(default_factory=list)
    reciprocal_ranks: list[float] = field(default_factory=list)
    lookup_counts: list[int]   = field(default_factory=list)
    latencies_ms:  list[float] = field(default_factory=list)

    def add(
        self,
        ranked_tables: list[tuple[str, str]],
        correct: tuple[str, str],
        latency_ms: float,
    ) -> None:
        try:
            rank = ranked_tables.index(correct) + 1  # 1-indexed
        except ValueError:
            rank = N_TABLES + 1  # 정답이 후보에 없음

        self.hit1.append(rank == 1)
        self.hit3.append(rank <= 3)
        self.reciprocal_ranks.append(1.0 / rank)
        self.lookup_counts.append(rank)
        self.latencies_ms.append(latency_ms)

    def summary(self, brute_avg: float) -> dict:
        n = len(self.hit1) or 1
        avg_lookup = float(np.mean(self.lookup_counts))
        reduction  = (1 - avg_lookup / brute_avg) * 100 if brute_avg else 0
        return {
            "method":             self.name,
            "hit@1":              f"{sum(self.hit1)/n*100:.1f}%",
            "hit@3":              f"{sum(self.hit3)/n*100:.1f}%",
            "mrr":                f"{np.mean(self.reciprocal_ranks):.4f}",
            "avg_lookup":         round(avg_lookup, 2),
            "total_lookups":      sum(self.lookup_counts),
            "lookup_reduction":   f"{reduction:.1f}%",
            "avg_latency_ms":     f"{np.mean(self.latencies_ms):.2f}",
        }


def _rank_tables(paths: list[SchemaPath]) -> list[tuple[str, str]]:
    """SchemaPath 리스트 → (db, table) 순서 리스트."""
    return [(p.db, p.table) for p in paths]


# ──────────────────────────────────────────────────────────────────────── #
#  Method implementations                                                   #
# ──────────────────────────────────────────────────────────────────────── #

def brute_force_rank(
    _query: str,
    _neo4j: Neo4jClient,
    _faiss: FaissQueryIndex,
    _retriever: GraphRetriever,
) -> list[tuple[str, str]]:
    """무작위 순서 → 최악의 기저선."""
    order = ALL_TABLES.copy()
    random.shuffle(order)
    return order


def text_similarity_rank(
    query: str,
    neo4j: Neo4jClient,
    faiss: FaissQueryIndex,
    retriever: GraphRetriever,
) -> list[tuple[str, str]]:
    """
    순수 코사인 유사도만 사용.
    GraphRAG 그래프 구조(access count, community) 없이
    FAISS 검색 결과에서 가장 유사한 쿼리가 접근한 테이블 순으로 반환.
    """
    embedder = get_embedder()
    vec = embedder.embed(query)
    similar = faiss.search(vec, k=25)

    if not similar:
        return ALL_TABLES.copy()

    # 유사도만으로 테이블 스코어링 (access count 무시)
    scores: dict[tuple[str, str], float] = {}
    for qid, sim in similar:
        for row in neo4j.get_accessed_paths(qid):
            key = (row["db"], row["table"])
            # 순수 유사도만 누적 (접근 횟수 곱하지 않음)
            scores[key] = scores.get(key, 0.0) + sim

    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    # 후보에 없는 테이블 뒤에 붙이기
    remaining = [t for t in ALL_TABLES if t not in ranked]
    return ranked + remaining


def graphrag_no_community_rank(
    query: str,
    neo4j: Neo4jClient,
    faiss: FaissQueryIndex,
    retriever: GraphRetriever,
) -> list[tuple[str, str]]:
    """
    FAISS + 접근 횟수 가중치 (커뮤니티 없음).
    community_detector.assign_community() 결과를 0으로 고정.
    """
    import math
    embedder = get_embedder()
    vec = embedder.embed(query)
    similar = faiss.search(vec, k=25)

    if not similar:
        return ALL_TABLES.copy()

    sim_dict = dict(similar)
    accesses = neo4j.get_schema_paths_for_queries(list(sim_dict.keys()))

    scores: dict[tuple[str, str], dict] = {}
    for row in accesses:
        key = (row["db"], row["table"])
        cs  = sim_dict.get(row["query_id"], 0.0)
        c   = scores.setdefault(key, {"embed_sim": 0.0, "access_sum": 0})
        c["embed_sim"]  = max(c["embed_sim"], cs)
        c["access_sum"] += row["count"]

    if not scores:
        return ALL_TABLES.copy()

    max_access = max(v["access_sum"] for v in scores.values()) or 1

    import config
    final: dict[tuple[str, str], float] = {}
    for key, ev in scores.items():
        access_norm = math.log1p(ev["access_sum"]) / math.log1p(max_access)
        # community 항 제거 → α+β 비율로 재정규화
        w_embed  = config.ALPHA / (config.ALPHA + config.BETA)
        w_access = config.BETA  / (config.ALPHA + config.BETA)
        final[key] = w_embed * ev["embed_sim"] + w_access * access_norm

    ranked = sorted(final, key=final.__getitem__, reverse=True)
    remaining = [t for t in ALL_TABLES if t not in ranked]
    return ranked + remaining


def graphrag_full_rank(
    query: str,
    neo4j: Neo4jClient,
    faiss: FaissQueryIndex,
    retriever: GraphRetriever,
) -> list[tuple[str, str]]:
    """완전한 GraphRAG (FAISS + 접근 그래프 + 커뮤니티)."""
    paths = retriever.route(query, top_n=N_TABLES)
    ranked = _rank_tables(paths)
    remaining = [t for t in ALL_TABLES if t not in ranked]
    return ranked + remaining


# ──────────────────────────────────────────────────────────────────────── #
#  Benchmark runner                                                         #
# ──────────────────────────────────────────────────────────────────────── #

METHODS: list[tuple[str, Callable]] = [
    ("BruteForce",       brute_force_rank),
    ("TextSimilarity",   text_similarity_rank),
    ("GraphRAG_NoComm",  graphrag_no_community_rank),
    ("GraphRAG_Full",    graphrag_full_rank),
]


def run_benchmark(router: QueryRouter, n_brute_samples: int = 100) -> None:
    neo4j     = router._neo4j
    faiss     = router._faiss
    retriever = router._retriever

    results = {name: MethodResult(name) for name, _ in METHODS}

    print(f"\n  총 {len(GROUND_TRUTH)}개 쿼리 × {len(METHODS)}개 방법 = "
          f"{len(GROUND_TRUTH)*len(METHODS)}회 측정\n")

    for query, correct_db, correct_table in GROUND_TRUTH:
        correct = (correct_db, correct_table)

        for name, method_fn in METHODS:
            t0 = time.perf_counter()

            # BruteForce는 n_brute_samples번 실행해 평균 냄 (확률적 기저선)
            if name == "BruteForce":
                ranks = []
                for _ in range(n_brute_samples):
                    ranked = method_fn(query, neo4j, faiss, retriever)
                    try:
                        ranks.append(ranked.index(correct) + 1)
                    except ValueError:
                        ranks.append(N_TABLES + 1)
                rank = int(round(float(np.mean(ranks))))
                ranked_tables = ALL_TABLES.copy()
                # 평균 rank 위치에 correct를 두도록 재구성 (시각화용)
                ranked_tables = [t for t in ALL_TABLES if t != correct]
                ranked_tables.insert(rank - 1, correct)
            else:
                ranked_tables = method_fn(query, neo4j, faiss, retriever)

            latency_ms = (time.perf_counter() - t0) * 1000
            results[name].add(ranked_tables, correct, latency_ms)

    # ── 결과 출력 ──────────────────────────────────────────────────────
    brute_avg = float(np.mean(results["BruteForce"].lookup_counts))
    summaries = [r.summary(brute_avg) for r in results.values()]

    print_comparison_table(summaries, brute_avg)
    print_contribution_analysis(summaries)


def print_comparison_table(summaries: list[dict], brute_avg: float) -> None:
    cols = ["method", "hit@1", "hit@3", "mrr",
            "avg_lookup", "total_lookups", "lookup_reduction", "avg_latency_ms"]
    col_w = [18, 7, 7, 8, 11, 14, 17, 15]

    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, col_w))
    sep    = "  ".join("-" * w for w in col_w)
    print(f"  {header}")
    print(f"  {sep}")
    for s in summaries:
        row = "  ".join(f"{str(s[c]):<{w}}" for c, w in zip(cols, col_w))
        print(f"  {row}")
    print()


def print_contribution_analysis(summaries: list[dict]) -> None:
    """GraphRAG 각 구성요소의 기여도를 분리해서 출력."""
    def _float(s: dict, key: str) -> float:
        return float(str(s[key]).replace("%", ""))

    by_name = {s["method"]: s for s in summaries}

    print("  GraphRAG 구성요소별 기여 분석:")
    print("  " + "-" * 60)

    # 접근 그래프 효과: TextSim → GraphRAG_NoComm
    ts  = by_name.get("TextSimilarity")
    gnc = by_name.get("GraphRAG_NoComm")
    gf  = by_name.get("GraphRAG_Full")

    if ts and gnc:
        d_hit1   = _float(gnc, "hit@1") - _float(ts, "hit@1")
        d_lookup = gnc["avg_lookup"] - ts["avg_lookup"]
        print(f"  [접근 그래프 효과] TextSim → GraphRAG_NoComm")
        print(f"    hit@1:      {_float(ts,'hit@1'):+.1f}% → {_float(gnc,'hit@1'):+.1f}%  (Δ{d_hit1:+.1f}%)")
        print(f"    avg_lookup: {ts['avg_lookup']} → {gnc['avg_lookup']}  (Δ{d_lookup:+.2f})")

    if gnc and gf:
        d_hit1   = _float(gf, "hit@1") - _float(gnc, "hit@1")
        d_lookup = gf["avg_lookup"] - gnc["avg_lookup"]
        print(f"\n  [커뮤니티 탐지 효과] GraphRAG_NoComm → GraphRAG_Full")
        print(f"    hit@1:      {_float(gnc,'hit@1'):.1f}% → {_float(gf,'hit@1'):.1f}%  (Δ{d_hit1:+.1f}%)")
        print(f"    avg_lookup: {gnc['avg_lookup']} → {gf['avg_lookup']}  (Δ{d_lookup:+.2f})")

    bf = by_name.get("BruteForce")
    if bf and gf:
        total_red = _float(gf, "lookup_reduction")
        print(f"\n  [총 효과] BruteForce → GraphRAG_Full")
        print(f"    총 쿼리 횟수: {bf['total_lookups']} → {gf['total_lookups']}")
        print(f"    쿼리 횟수 절감률: {total_red:.1f}%")
        print(f"    MRR: {bf['mrr']} → {gf['mrr']}")
    print()


# ──────────────────────────────────────────────────────────────────────── #
#  Main                                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def main():
    print("=" * 70)
    print("  GraphRAG Query Router — Benchmark")
    print("=" * 70)

    print("\n[1] 라우터 초기화 및 데이터 준비…")
    router = QueryRouter.build(total_tables=N_TABLES)
    router.register_schemas(SCHEMAS)
    router.load_history(ACCESS_LOG)
    router.rebuild_communities()
    print(f"  ✓ {N_TABLES}개 테이블, {len(ACCESS_LOG)}개 히스토리 로드 완료")

    print("\n[2] 벤치마크 실행…")
    print("  (BruteForce는 100회 평균으로 확률적 기저선 계산)")
    run_benchmark(router)

    router.close()
    print("✓ Benchmark complete.")


if __name__ == "__main__":
    main()
