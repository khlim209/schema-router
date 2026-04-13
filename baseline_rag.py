"""
Schema RAG Baseline (그래프 없음, 히스토리 없음)

DBCopilot이 비교 기저선으로 사용한 "순수 임베딩 검색" 방식.

동작 원리
----------
1. 각 (db, table)의 스키마 텍스트를 임베딩해 FAISS에 저장
   스키마 텍스트 = "{db}.{table}: {col1} {col2} ... {description}"
2. 질의 임베딩 → FAISS 검색 → 코사인 유사도로 테이블 순위 반환
3. 접근 히스토리 없음, 커뮤니티 없음, 그래프 없음

이 방식과 GraphRAG의 차이가 순수 그래프/히스토리의 기여분이다.
"""

from __future__ import annotations

import numpy as np
import faiss

from dataclasses import dataclass
from loguru import logger

from bench_datasets.base import SchemaEntry
from embedding.embedder import get_embedder


@dataclass
class SchemaRAGResult:
    db:    str
    table: str
    score: float


class SchemaRAGBaseline:
    """
    스키마 임베딩 기반 순수 RAG.
    Neo4j / 접근 로그 불필요.
    """

    def __init__(self):
        self._embedder = get_embedder()
        self._index:  faiss.Index | None = None
        self._entries: list[tuple[str, str]] = []   # [(db_id, table_name), ...]

    # ------------------------------------------------------------------ #
    #  인덱스 구축                                                         #
    # ------------------------------------------------------------------ #

    def build_index(self, schemas: list[SchemaEntry]) -> None:
        """스키마 목록으로 FAISS 인덱스 구축."""
        texts: list[str] = []
        self._entries = []

        for schema in schemas:
            for table_name, table_info in schema.tables.items():
                cols = " ".join(c for c, _ in table_info.get("columns", []))
                desc = table_info.get("description", "")
                # DBCopilot §3.2 포맷: "db.table: col1 col2 ... description"
                text = f"{schema.db_id}.{table_name}: {cols}"
                if desc:
                    text += f" — {desc}"
                texts.append(text)
                self._entries.append((schema.db_id, table_name))

        if not texts:
            logger.warning("SchemaRAGBaseline: 스키마가 비어 있습니다.")
            return

        logger.info(f"SchemaRAG: {len(texts)}개 테이블 임베딩 중…")
        vecs = self._embedder.embed_batch(texts).astype(np.float32)

        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)
        logger.info(f"SchemaRAG 인덱스 구축 완료 ({len(texts)}개 테이블)")

    # ------------------------------------------------------------------ #
    #  검색                                                                #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        db_id: str | None = None,
        top_k: int | None = None,
    ) -> list[SchemaRAGResult]:
        """
        query 임베딩 → 유사 테이블 반환.
        db_id 지정 시 해당 DB 내에서만 검색.
        top_k=None 이면 전체 반환.
        """
        if self._index is None or not self._entries:
            return []

        vec = self._embedder.embed(query).reshape(1, -1).astype(np.float32)
        k   = top_k or len(self._entries)
        sims, idxs = self._index.search(vec, min(k, len(self._entries)))

        results: list[SchemaRAGResult] = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            db, table = self._entries[idx]
            if db_id and db != db_id:
                continue
            results.append(SchemaRAGResult(db=db, table=table, score=float(sim)))

        return results

    def ranked_tables(
        self,
        query: str,
        db_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """(db, table) 순위 리스트 반환."""
        return [(r.db, r.table) for r in self.search(query, db_id=db_id)]

    @property
    def total_tables(self) -> int:
        return len(self._entries)
