"""
Tiered Retriever — 쿼리 횟수 최소화에 특화된 라우터

핵심 아이디어
--------------
기존 시스템은 올바른 (DB, Table)을 찾을 때까지 순서대로 탐색하므로
최악의 경우 O(N_tables) 번 쿼리가 발생한다.

GraphRAG 커뮤니티 캐시 + 신뢰도 티어링으로 이를 줄인다:

  Tier 1 EXACT    : 동일 쿼리 재등장 → 메모리 캐시 즉시 반환 (0 추가 쿼리)
  Tier 2 HIGH     : score ≥ HIGH_THRESH → 1위 경로만 반환 (직접 이동)
  Tier 3 MEDIUM   : score ≥ MED_THRESH  → 상위 k개 반환 (k번 시도)
  Tier 4 FALLBACK : 신뢰도 낮음         → BFS 확장 + 전체 후보 반환

측정 지표
----------
- lookup_count  : 이번 쿼리에서 실제로 시도해야 할 경로 수
- saved_lookups : 시스템 없이 brute-force 했을 때 대비 절감된 횟수
- cache_hit     : Tier 1 캐시 적중 여부
- tier          : 적용된 티어

누적 통계
----------
- total_queries        : 처리한 쿼리 수
- cache_hits           : Tier 1 적중 수
- avg_lookup_count     : 평균 쿼리 횟수
- avg_saved_lookups    : 평균 절감 횟수
- lookup_reduction_pct : brute-force 대비 쿼리 절감률 (%)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from graph_rag.retriever import GraphRetriever, SchemaPath
from graph_db.neo4j_client import Neo4jClient
import config


# ──────────────────────────────────────────────────────────────────────── #
#  Constants                                                                #
# ──────────────────────────────────────────────────────────────────────── #

HIGH_CONF_THRESH = 0.88   # 이 이상이면 1위만 시도
MED_CONF_THRESH  = 0.70   # 이 이상이면 상위 3개만 시도
CACHE_SIZE       = 1024   # LRU 캐시 최대 항목 수


class RoutingTier(str, Enum):
    EXACT    = "exact_cache"    # Tier 1: 완전히 같은 쿼리
    HIGH     = "high_conf"      # Tier 2: 고신뢰도
    MEDIUM   = "medium_conf"    # Tier 3: 중신뢰도
    FALLBACK = "fallback"       # Tier 4: 저신뢰도 BFS


# ──────────────────────────────────────────────────────────────────────── #
#  Result dataclass                                                         #
# ──────────────────────────────────────────────────────────────────────── #

@dataclass
class TieredResult:
    paths:         list[SchemaPath]   # 라우팅 결과 (우선순위 순)
    tier:          RoutingTier        # 적용된 티어
    lookup_count:  int                # 실제로 시도해야 할 경로 수
    saved_lookups: int                # brute-force 대비 절감 횟수
    cache_hit:     bool               # Tier 1 캐시 적중 여부
    latency_ms:    float              # 응답 시간 (ms)
    top_score:     float              # 1위 경로의 신뢰도 점수
    metadata:      dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        top = self.paths[0] if self.paths else None
        top_str = f"{top.db}.{top.table}" if top else "N/A"
        return (
            f"[{self.tier.value}] {top_str} "
            f"(score={self.top_score:.3f}, "
            f"lookups={self.lookup_count}, "
            f"saved={self.saved_lookups}, "
            f"{self.latency_ms:.1f}ms)"
        )


# ──────────────────────────────────────────────────────────────────────── #
#  Stats tracker                                                            #
# ──────────────────────────────────────────────────────────────────────── #

class RoutingStats:
    def __init__(self, total_tables: int):
        self.total_tables      = total_tables  # brute-force 기준 (전체 테이블 수)
        self.total_queries     = 0
        self.cache_hits        = 0
        self.tier_counts       = {t: 0 for t in RoutingTier}
        self._lookup_sum       = 0
        self._saved_sum        = 0
        self._latency_sum      = 0.0

    def record(self, result: TieredResult) -> None:
        self.total_queries  += 1
        self._lookup_sum    += result.lookup_count
        self._saved_sum     += result.saved_lookups
        self._latency_sum   += result.latency_ms
        self.tier_counts[result.tier] += 1
        if result.cache_hit:
            self.cache_hits += 1

    def report(self) -> dict:
        n = self.total_queries or 1
        avg_lookup   = self._lookup_sum  / n
        avg_brute    = self.total_tables / 2   # 평균 brute-force 비용
        reduction    = (1 - avg_lookup / avg_brute) * 100 if avg_brute else 0

        return {
            "total_queries":        self.total_queries,
            "total_tables":         self.total_tables,
            "cache_hit_rate":       f"{self.cache_hits / n * 100:.1f}%",
            "avg_lookup_count":     round(avg_lookup, 2),
            "avg_brute_force_cost": round(avg_brute, 1),
            "lookup_reduction_pct": f"{reduction:.1f}%",
            "avg_saved_lookups":    round(self._saved_sum / n, 2),
            "avg_latency_ms":       round(self._latency_sum / n, 2),
            "tier_distribution":    {t.value: c for t, c in self.tier_counts.items()},
        }


# ──────────────────────────────────────────────────────────────────────── #
#  Tiered Retriever                                                         #
# ──────────────────────────────────────────────────────────────────────── #

class TieredRetriever:
    """
    쿼리 횟수 최소화를 목표로 하는 계층형 라우터.
    """

    def __init__(
        self,
        retriever: GraphRetriever,
        neo4j: Neo4jClient,
        total_tables: int = 10,       # 전체 등록 테이블 수 (brute-force 기준)
        cache_size: int = CACHE_SIZE,
    ):
        self._retriever   = retriever
        self._neo4j       = neo4j
        self._stats       = RoutingStats(total_tables)

        # Tier 1: LRU 캐시 (query_id → TieredResult)
        self._cache: OrderedDict[str, TieredResult] = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ #
    #  Main routing method                                                 #
    # ------------------------------------------------------------------ #

    def route(
        self,
        query_text: str,
        query_id:   str | None = None,
        top_n:      int = 5,
    ) -> TieredResult:
        t0 = time.perf_counter()

        # ── Tier 1: Exact cache ────────────────────────────────────────
        from embedding.embedder import get_embedder
        qid = query_id or get_embedder().text_id(query_text)

        if qid in self._cache:
            cached = self._cache[qid]
            self._cache.move_to_end(qid)   # LRU 갱신
            result = TieredResult(
                paths         = cached.paths,
                tier          = RoutingTier.EXACT,
                lookup_count  = 0,
                saved_lookups = self._stats.total_tables,
                cache_hit     = True,
                latency_ms    = (time.perf_counter() - t0) * 1000,
                top_score     = cached.top_score,
                metadata      = {"source": "lru_cache"},
            )
            self._stats.record(result)
            logger.debug(f"[Tier1 EXACT] {result.summary()}")
            return result

        # ── Full retrieval ─────────────────────────────────────────────
        paths = self._retriever.route(query_text, top_n=top_n)

        if not paths:
            result = TieredResult(
                paths=[], tier=RoutingTier.FALLBACK,
                lookup_count=self._stats.total_tables,
                saved_lookups=0, cache_hit=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                top_score=0.0,
            )
            self._stats.record(result)
            return result

        top_score = paths[0].score

        # ── Tier 2: High confidence ────────────────────────────────────
        if top_score >= HIGH_CONF_THRESH:
            tier         = RoutingTier.HIGH
            lookup_count = 1                           # 1위만 시도
            returned     = paths[:1]

        # ── Tier 3: Medium confidence ──────────────────────────────────
        elif top_score >= MED_CONF_THRESH:
            tier         = RoutingTier.MEDIUM
            lookup_count = min(3, len(paths))          # 최대 3개 시도
            returned     = paths[:lookup_count]

        # ── Tier 4: Fallback ───────────────────────────────────────────
        else:
            tier         = RoutingTier.FALLBACK
            lookup_count = len(paths)
            returned     = paths

        brute_force_cost = self._stats.total_tables // 2  # 평균 탐색 비용
        saved = max(0, brute_force_cost - lookup_count)

        result = TieredResult(
            paths         = returned,
            tier          = tier,
            lookup_count  = lookup_count,
            saved_lookups = saved,
            cache_hit     = False,
            latency_ms    = (time.perf_counter() - t0) * 1000,
            top_score     = top_score,
            metadata      = {
                "total_candidates": len(paths),
                "score_gap": round(
                    paths[0].score - paths[1].score, 4
                ) if len(paths) > 1 else 1.0,
            },
        )

        # LRU 캐시에 저장 (Tier 2~3만 캐시 가치 있음)
        if tier in (RoutingTier.HIGH, RoutingTier.MEDIUM):
            self._lru_put(qid, result)

        self._stats.record(result)
        logger.info(f"[{tier.value}] {result.summary()}")
        return result

    # ------------------------------------------------------------------ #
    #  Statistics                                                          #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return self._stats.report()

    def reset_stats(self) -> None:
        self._stats = RoutingStats(self._stats.total_tables)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ #
    #  LRU helpers                                                         #
    # ------------------------------------------------------------------ #

    def _lru_put(self, key: str, value: TieredResult) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)   # 가장 오래된 항목 제거

    # ------------------------------------------------------------------ #
    #  Lookup budget simulation                                            #
    # ------------------------------------------------------------------ #

    def simulate_savings(self, queries: list[str]) -> dict:
        """
        주어진 쿼리 목록에 대해 brute-force vs GraphRAG 비용을 시뮬레이션.
        실제 Neo4j 접근은 하지 않고 점수만 계산.
        """
        brute_total  = 0
        graphrag_total = 0
        details = []

        for q in queries:
            result = self.route(q)
            brute = self._stats.total_tables // 2
            brute_total    += brute
            graphrag_total += result.lookup_count
            details.append({
                "query":        q[:40],
                "tier":         result.tier.value,
                "brute_force":  brute,
                "graphrag":     result.lookup_count,
                "saved":        result.saved_lookups,
            })

        reduction = (1 - graphrag_total / brute_total) * 100 if brute_total else 0
        return {
            "brute_force_total":  brute_total,
            "graphrag_total":     graphrag_total,
            "total_saved":        brute_total - graphrag_total,
            "reduction_pct":      f"{reduction:.1f}%",
            "per_query":          details,
        }
