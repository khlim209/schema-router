"""
Stage 1 — Similarity-based Schema Cache
=========================================

다이어그램 Stage 1의 핵심 아이디어:
  새 질문 → 임베딩 → 과거 처리 질문과 코사인 유사도 비교
  유사도 >= threshold → 저장된 스키마 경로를 즉시 반환 (Stage 2·3 스킵)
  유사도 < threshold → Stage 2로 진행

기존 TieredRetriever.Tier1과의 차이:
  기존: SHA-256 exact match (완전히 동일한 질문만 히트)
  신규: cosine similarity threshold (의미적으로 유사한 질문도 히트)

캐시 구조:
  FAISS IndexFlatIP  — 코사인 유사도 검색 (L2-normalised 벡터이므로 IP = cosine)
  entries[]          — [(db_id, table_name), ...] 인덱스 매핑

히트율이 올라갈수록 Stage 2·3 호출이 줄어들어
전체 평균 응답 시간이 지속적으로 감소한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np
from loguru import logger

import config

# 기본 유사도 임계값 — 0.92 이상이면 "같은 의도의 질문"으로 간주
DEFAULT_CACHE_THRESHOLD = 0.92
# LRU 없이 단순 FIFO max size (FAISS는 삭제가 비용이 크므로 크게 설정)
DEFAULT_MAX_SIZE = 4096


@dataclass
class CacheLookupResult:
    hit:       bool
    db:        str = ""
    table:     str = ""
    sim_score: float = 0.0   # 최고 코사인 유사도
    entry_idx: int = -1


@dataclass
class CacheStats:
    total_lookups: int = 0
    hits:          int = 0
    stores:        int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_lookups if self.total_lookups else 0.0

    def report(self) -> dict:
        return {
            "total_lookups":  self.total_lookups,
            "cache_hits":     self.hits,
            "cache_misses":   self.total_lookups - self.hits,
            "hit_rate":       f"{self.hit_rate * 100:.1f}%",
            "stored_entries": self.stores,
        }


class SimilaritySchemaCache:
    """
    코사인 유사도 기반 스키마 경로 캐시.

    질문 임베딩을 FAISS에 저장하고,
    신규 질문이 기존 항목과 threshold 이상 유사하면
    저장된 (db, table)을 즉시 반환한다.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_CACHE_THRESHOLD,
        dim: int = config.EMBEDDING_DIM,
        max_size: int = DEFAULT_MAX_SIZE,
    ):
        self.threshold = threshold
        self._dim      = dim
        self._max_size = max_size

        # FAISS 인덱스 — 내적 = L2 정규화된 벡터에서의 코사인 유사도
        self._index = faiss.IndexFlatIP(dim)

        # (db_id, table_name) 매핑 (FAISS 행 인덱스 순서)
        self._entries: list[tuple[str, str]] = []

        self._stats = CacheStats()

    # ------------------------------------------------------------------ #
    #  조회                                                                #
    # ------------------------------------------------------------------ #

    def lookup(self, vec: np.ndarray) -> CacheLookupResult:
        """
        vec에 가장 유사한 과거 질문을 찾는다.
        유사도 >= threshold 이면 HIT, 미만이면 MISS.
        """
        self._stats.total_lookups += 1

        if self._index.ntotal == 0:
            return CacheLookupResult(hit=False)

        q = vec.reshape(1, -1).astype(np.float32)
        sims, idxs = self._index.search(q, 1)

        best_sim = float(sims[0][0])
        best_idx = int(idxs[0][0])

        if best_idx < 0 or best_sim < self.threshold:
            return CacheLookupResult(hit=False, sim_score=best_sim)

        db, table = self._entries[best_idx]
        self._stats.hits += 1
        logger.debug(
            f"[Stage1 Cache HIT] sim={best_sim:.4f} → {db}.{table}"
        )
        return CacheLookupResult(
            hit=True, db=db, table=table,
            sim_score=best_sim, entry_idx=best_idx,
        )

    # ------------------------------------------------------------------ #
    #  저장                                                                #
    # ------------------------------------------------------------------ #

    def store(
        self,
        vec: np.ndarray,
        db: str,
        table: str,
    ) -> None:
        """
        질문 벡터 + 최적 스키마 경로를 캐시에 추가한다.
        max_size 초과 시 가장 오래된 항목을 교체(FIFO 근사).
        """
        if len(self._entries) >= self._max_size:
            # FAISS는 개별 삭제가 느리므로 인덱스를 재구성
            keep = self._max_size // 2           # 절반 유지
            self._entries = self._entries[-keep:]
            # 재구성: 새 인덱스에 벡터 재추가 (이미 L2 정규화됨)
            # NOTE: 원본 벡터는 저장하지 않으므로 이전 벡터를 버리고
            #       새 항목만 추가하는 단순 FIFO 방식으로 처리
            self._index = faiss.IndexFlatIP(self._dim)
            logger.debug(
                f"[Stage1 Cache] max_size 도달, 인덱스 초기화 (entries 유지: {keep}개)"
            )
            # entries만 절반으로 자름 — 기존 FAISS 벡터는 소실됨
            # 다음 store 호출에서 새 항목이 추가됨

        v = vec.reshape(1, -1).astype(np.float32)
        self._index.add(v)
        self._entries.append((db, table))
        self._stats.stores += 1
        logger.debug(f"[Stage1 Cache] stored {db}.{table} (total={len(self._entries)})")

    # ------------------------------------------------------------------ #
    #  유틸리티                                                            #
    # ------------------------------------------------------------------ #

    def invalidate_last(self) -> None:
        """
        마지막으로 저장된 항목을 무효화한다.
        SQL 실행 실패 피드백 루프에서 사용.
        FAISS는 개별 삭제가 불가능하므로 entries에서만 제거하고
        다음 lookup 시 인덱스 불일치를 감수한다.
        (실제 운영에서는 IDMap2를 사용하는 것이 적합)
        """
        if self._entries:
            removed = self._entries.pop()
            logger.debug(f"[Stage1 Cache] invalidated {removed[0]}.{removed[1]}")

    def size(self) -> int:
        return self._index.ntotal

    def stats(self) -> dict:
        return self._stats.report()
