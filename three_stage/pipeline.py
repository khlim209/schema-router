"""
Three-Stage Pipeline — 전체 파이프라인 오케스트레이터
=======================================================

다이어그램의 3단계 구조를 직접 구현:

  NL Question
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │ Stage 1: Question Embedding + Cache Lookup  │  (속도 우선)
  │   cosine_sim > threshold?                   │
  │   YES → 저장된 스키마 경로 즉시 반환         │
  │   NO  → Stage 2로 진행                      │
  └─────────────────────────────────────────────┘
      │ Cache MISS
      ▼
  ┌─────────────────────────────────────────────┐
  │ Stage 2: Graph-RAG — 후보 축소 (속도)        │
  │   질문 노드 생성 → Graph 탐색 (1~2 hop)     │
  │   top-k 스키마 후보 반환                     │
  └─────────────────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │ Stage 3: DSI Re-ranker — 정확도 보정 (정확도)│
  │   질문-스키마 임베딩 유사도                   │
  │   컬럼 키워드 오버랩                          │
  │   FK·JOIN 구조 풍부도                        │
  │   → top-1 스키마 결정 + 캐시 저장            │
  └─────────────────────────────────────────────┘
      │
      ▼
  LLM → SQL 생성

피드백 루프 (선택적):
  SQL 실행 성공 → Graph 엣지 가중치 ↑, 캐시 신뢰도 강화
  SQL 실행 실패 → 캐시 무효화, DSI 재학습 트리거
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger

from embedding.embedder import get_embedder
from graph_rag.retriever import GraphRetriever, SchemaPath
from three_stage.stage1_cache import SimilaritySchemaCache
from three_stage.stage3_reranker import DSIReranker


# ──────────────────────────────────────────────────────────────────────── #
#  결과 타입                                                                #
# ──────────────────────────────────────────────────────────────────────── #

class PipelineStage(str, Enum):
    CACHE    = "stage1_cache"      # Stage 1 HIT → Stage 2·3 스킵
    FULL     = "stage2+3"          # Stage 2 + Stage 3 실행
    FALLBACK = "fallback"          # Graph-RAG 결과 없음 → SchemaRAG fallback


@dataclass
class PipelineResult:
    ranked:        list[tuple[str, str]]   # (db, table) 우선순위 순
    stage_used:    PipelineStage
    cache_hit:     bool
    latency_ms:    float
    top_db:        str   = ""
    top_table:     str   = ""
    cache_sim:     float = 0.0            # Stage 1 최고 유사도
    n_candidates:  int   = 0              # Stage 2 반환 후보 수
    metadata:      dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.ranked:
            self.top_db, self.top_table = self.ranked[0]

    def summary(self) -> str:
        top = f"{self.top_db}.{self.top_table}" if self.top_db else "N/A"
        return (
            f"[{self.stage_used.value}] {top} "
            f"(cache_hit={self.cache_hit}, "
            f"candidates={self.n_candidates}, "
            f"{self.latency_ms:.1f}ms)"
        )


@dataclass
class PipelineStats:
    total:        int   = 0
    stage1_hits:  int   = 0
    full_runs:    int   = 0
    fallbacks:    int   = 0
    _latency_sum: float = 0.0

    def record(self, result: PipelineResult) -> None:
        self.total        += 1
        self._latency_sum += result.latency_ms
        if result.stage_used == PipelineStage.CACHE:
            self.stage1_hits += 1
        elif result.stage_used == PipelineStage.FULL:
            self.full_runs += 1
        else:
            self.fallbacks += 1

    def report(self) -> dict:
        n = self.total or 1
        return {
            "total_queries":    self.total,
            "stage1_hit_rate":  f"{self.stage1_hits / n * 100:.1f}%",
            "full_run_rate":    f"{self.full_runs   / n * 100:.1f}%",
            "fallback_rate":    f"{self.fallbacks   / n * 100:.1f}%",
            "avg_latency_ms":   round(self._latency_sum / n, 2),
        }


# ──────────────────────────────────────────────────────────────────────── #
#  파이프라인                                                               #
# ──────────────────────────────────────────────────────────────────────── #

class ThreeStagePipeline:
    """
    Stage 1 → (Stage 2 → Stage 3) 오케스트레이터.

    Parameters
    ----------
    cache        : SimilaritySchemaCache  — Stage 1
    graph_ret    : GraphRetriever         — Stage 2
    dsi          : DSIReranker            — Stage 3
    top_k_graph  : Stage 2에서 추출할 후보 수 (DSI 입력)
    cache_threshold : Stage 1 HIT 판정 임계값 (None → cache 기본값 사용)
    """

    def __init__(
        self,
        cache:       SimilaritySchemaCache,
        graph_ret:   GraphRetriever,
        dsi:         DSIReranker,
        top_k_graph: int = 20,
    ):
        self._cache     = cache
        self._graph     = graph_ret
        self._dsi       = dsi
        self._top_k     = top_k_graph
        self._embedder  = get_embedder()
        self._stats     = PipelineStats()

    # ------------------------------------------------------------------ #
    #  메인 라우팅                                                         #
    # ------------------------------------------------------------------ #

    def route(
        self,
        question:      str,
        all_tables:    list[tuple[str, str]],
        question_vec:  np.ndarray | None = None,
    ) -> PipelineResult:
        """
        3단계 파이프라인으로 (db, table) 순위 리스트를 반환한다.

        Parameters
        ----------
        question     : 자연어 질문
        all_tables   : 전체 (db_id, table_name) 목록 (fallback 순위 구성용)
        question_vec : 미리 계산된 임베딩 (없으면 내부에서 계산)
        """
        t0  = time.perf_counter()
        vec = question_vec if question_vec is not None \
              else self._embedder.embed(question)

        # ── Stage 1: Cache Lookup ──────────────────────────────────────
        cache_result = self._cache.lookup(vec)
        if cache_result.hit:
            # HIT: Stage 2·3 완전 스킵
            top_pair = (cache_result.db, cache_result.table)
            ranked   = [top_pair] + [t for t in all_tables if t != top_pair]
            result = PipelineResult(
                ranked       = ranked,
                stage_used   = PipelineStage.CACHE,
                cache_hit    = True,
                latency_ms   = (time.perf_counter() - t0) * 1000,
                cache_sim    = cache_result.sim_score,
                n_candidates = 0,
                metadata     = {"stage1_entry_idx": cache_result.entry_idx},
            )
            self._stats.record(result)
            logger.debug(f"[Pipeline] {result.summary()}")
            return result

        # ── Stage 2: Graph-RAG ─────────────────────────────────────────
        candidates: list[SchemaPath] = self._graph.route(
            question, top_n=self._top_k
        )

        if not candidates:
            # Graph-RAG 결과 없음 → SchemaRAG fallback (호출자가 처리)
            result = PipelineResult(
                ranked       = all_tables,
                stage_used   = PipelineStage.FALLBACK,
                cache_hit    = False,
                latency_ms   = (time.perf_counter() - t0) * 1000,
                n_candidates = 0,
            )
            self._stats.record(result)
            return result

        # ── Stage 3: DSI Re-rank ───────────────────────────────────────
        ranked = self._dsi.rerank(
            question      = question,
            question_vec  = vec,
            candidates    = candidates,
            fallback_tables = all_tables,
        )

        # Stage 1 캐시에 결과 저장 (다음 유사 질문이 Stage 1에서 히트)
        if ranked:
            top_db, top_table = ranked[0]
            self._cache.store(vec, top_db, top_table)

        result = PipelineResult(
            ranked       = ranked,
            stage_used   = PipelineStage.FULL,
            cache_hit    = False,
            latency_ms   = (time.perf_counter() - t0) * 1000,
            n_candidates = len(candidates),
        )
        self._stats.record(result)
        logger.debug(f"[Pipeline] {result.summary()}")
        return result

    # ------------------------------------------------------------------ #
    #  피드백 루프                                                         #
    # ------------------------------------------------------------------ #

    def feedback_success(self, question_vec: np.ndarray) -> None:
        """
        SQL 실행 성공 → 현재 캐시 항목의 신뢰도 유지.
        (실질적으로 캐시 무효화 안 함)
        """
        pass   # 캐시는 이미 저장되어 있음

    def feedback_failure(self) -> None:
        """
        SQL 실행 실패 → 마지막으로 저장한 캐시 항목 무효화.
        다음 유사 질문은 Stage 2·3을 다시 통과한다.
        """
        self._cache.invalidate_last()
        logger.debug("[Pipeline] feedback_failure → cache invalidated")

    # ------------------------------------------------------------------ #
    #  통계                                                                #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "pipeline": self._stats.report(),
            "cache":    self._cache.stats(),
        }

    # ------------------------------------------------------------------ #
    #  팩토리                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        graph_retriever: GraphRetriever,
        schemas,                         # list[SchemaEntry]
        cache_threshold: float = 0.92,
        top_k_graph:     int   = 20,
    ) -> "ThreeStagePipeline":
        """
        SchemaEntry 목록에서 파이프라인을 구성한다.

        Parameters
        ----------
        graph_retriever : 기존 GraphRetriever (QueryRouter._retriever)
        schemas         : list[SchemaEntry]
        cache_threshold : Stage 1 HIT 임계값
        top_k_graph     : Stage 2 → Stage 3에 넘길 후보 수
        """
        cache = SimilaritySchemaCache(threshold=cache_threshold)
        dsi   = DSIReranker.from_schema_entries(schemas)
        return cls(
            cache       = cache,
            graph_ret   = graph_retriever,
            dsi         = dsi,
            top_k_graph = top_k_graph,
        )
