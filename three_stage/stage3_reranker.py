"""
Stage 3 — DSI Re-ranker
========================

다이어그램 Stage 3: "정확도 보정 (정확도)"

Graph-RAG(Stage 2)가 top-k 후보를 출력하면,
DSI Re-ranker가 각 후보를 질문과 더 정밀하게 비교해 재정렬한다.

Graph-RAG의 약점
-----------------
- 접근 히스토리 편향: 자주 쓰인 테이블이 과대평가될 수 있음
- 의미 유사도만으로는 FK·JOIN 관계 구조를 섬세하게 인식 못함
- Cold start 질문에서 graph 점수가 낮으면 SchemaRAG 점수에 의존

DSI Re-ranker가 보완하는 것
----------------------------
1. 질문-스키마 정밀 임베딩 유사도
   (테이블명 + 컬럼명 + 설명을 구조화된 텍스트로 임베딩)
2. 컬럼 키워드 오버랩
   (질문 토큰과 컬럼/테이블명의 직접 매칭)
3. FK·JOIN 구조 풍부도
   (관계가 많은 테이블 = 복잡한 질의 처리 가능성 ↑)
4. Reverse Generation 프록시
   (스키마 텍스트 임베딩 ≈ "이 스키마로 답할 수 있는 질문의 분포")

최종 DSI 점수 = α·embed_sim + β·keyword_overlap + γ·struct_richness

원논문(DBCopilot)의 학습된 DSI와 달리 학습 없이 작동하며,
Graph-RAG 점수와 블렌딩해 최종 순위를 결정한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from loguru import logger

from embedding.embedder import get_embedder
from graph_rag.retriever import SchemaPath


# ──────────────────────────────────────────────────────────────────────── #
#  스코어링 가중치                                                          #
# ──────────────────────────────────────────────────────────────────────── #

# DSI 내부 가중치
_W_EMBED   = 0.55   # 질문-스키마 임베딩 유사도 (가장 중요)
_W_KEYWORD = 0.30   # 컬럼 키워드 오버랩
_W_STRUCT  = 0.15   # FK·JOIN 구조 풍부도

# Graph-RAG vs DSI 블렌드
_W_GRAPHRAG = 0.40  # Graph-RAG 역사적 점수 기여
_W_DSI      = 0.60  # DSI 정밀 점수 기여 (새 방법의 핵심 차별점)


# ──────────────────────────────────────────────────────────────────────── #
#  키워드 추출 (경량, ML 불필요)                                            #
# ──────────────────────────────────────────────────────────────────────── #

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "in", "on", "at", "to", "for",
    "of", "by", "with", "from", "how", "many", "what", "which",
    "who", "when", "where", "why", "list", "show", "get", "find",
    "all", "count", "give", "tell", "이", "가", "은", "는", "을",
    "를", "의", "에", "에서", "로", "으로", "와", "과", "도", "만",
    "지난", "이번", "최근", "전체", "모든", "각", "및", "또는",
}


def _extract_tokens(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    return [t for t in text.split() if t not in _STOPWORDS and len(t) > 1]


# ──────────────────────────────────────────────────────────────────────── #
#  스키마 텍스트 직렬화                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def _schema_text(table_name: str, table_info: dict) -> str:
    """
    DBCopilot §3.2 포맷으로 스키마를 텍스트화.
    임베딩 유사도 계산에 사용.
    """
    cols = " ".join(
        col for col, _ in table_info.get("columns", [])
    )
    desc = table_info.get("description", "")
    joins = " ".join(
        other for other, _ in table_info.get("joins", [])
    )
    parts = [table_name, cols]
    if desc:
        parts.append(desc)
    if joins:
        parts.append(f"joins: {joins}")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────── #
#  DSI Reranker                                                             #
# ──────────────────────────────────────────────────────────────────────── #

@dataclass
class RankedCandidate:
    db:           str
    table:        str
    graphrag_score: float
    dsi_score:    float
    final_score:  float
    keyword_hits: int
    embed_sim:    float


class DSIReranker:
    """
    Graph-RAG top-k 후보를 질문-스키마 정밀 점수로 재정렬한다.

    Parameters
    ----------
    schema_registry : dict[str, dict[str, dict]]
        { db_id → { table_name → table_info } }
        table_info = { "columns": [(col, type), ...], "joins": [(other, via), ...] }
    w_graphrag : float
        Graph-RAG 기존 점수의 블렌딩 가중치 (0~1)
    w_dsi : float
        DSI 정밀 점수의 블렌딩 가중치 (= 1 - w_graphrag)
    """

    def __init__(
        self,
        schema_registry: dict[str, dict[str, dict]],
        w_graphrag: float = _W_GRAPHRAG,
        w_dsi: float = _W_DSI,
    ):
        self._registry  = schema_registry
        self._w_graph   = w_graphrag
        self._w_dsi     = w_dsi
        self._embedder  = get_embedder()

        # 스키마 텍스트 임베딩 캐시 (db.table → vec)
        self._schema_vec_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        question: str,
        question_vec: np.ndarray,
        candidates: list[SchemaPath],
        fallback_tables: list[tuple[str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        """
        candidates(Graph-RAG top-k)를 DSI 점수로 재정렬해
        (db, table) 순위 리스트를 반환한다.

        fallback_tables: 후보에 없는 테이블 (점수 0으로 뒤에 추가)
        """
        if not candidates:
            return fallback_tables or []

        q_tokens = _extract_tokens(question)
        scored: list[RankedCandidate] = []

        for path in candidates:
            dsi  = self._dsi_score(path.db, path.table, question_vec, q_tokens)
            final = self._w_graph * path.score + self._w_dsi * dsi.total

            scored.append(RankedCandidate(
                db=path.db, table=path.table,
                graphrag_score=path.score,
                dsi_score=dsi.total,
                final_score=final,
                keyword_hits=dsi.keyword_hits,
                embed_sim=dsi.embed_sim,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)

        ranked = [(c.db, c.table) for c in scored]

        # fallback: Graph-RAG에 없던 테이블 (점수 0, 뒤에 추가)
        if fallback_tables:
            ranked_set = set(ranked)
            ranked += [t for t in fallback_tables if t not in ranked_set]

        if scored:
            top = scored[0]
            logger.debug(
                f"[Stage3 DSI] top={top.db}.{top.table} "
                f"(final={top.final_score:.3f}, "
                f"dsi={top.dsi_score:.3f}, "
                f"kw_hits={top.keyword_hits})"
            )

        return ranked

    # ------------------------------------------------------------------ #
    #  DSI 점수 계산                                                       #
    # ------------------------------------------------------------------ #

    class _DSIScore:
        __slots__ = ("embed_sim", "keyword_score", "struct_score", "total", "keyword_hits")
        def __init__(self, embed_sim, keyword_score, struct_score, keyword_hits):
            self.embed_sim     = embed_sim
            self.keyword_score = keyword_score
            self.struct_score  = struct_score
            self.keyword_hits  = keyword_hits
            self.total = (
                _W_EMBED   * embed_sim
                + _W_KEYWORD * keyword_score
                + _W_STRUCT  * struct_score
            )

    def _dsi_score(
        self,
        db: str,
        table: str,
        q_vec: np.ndarray,
        q_tokens: list[str],
    ) -> "_DSIScore":
        # ── 1. 임베딩 유사도 ──────────────────────────────────────────
        embed_sim = self._schema_embed_sim(db, table, q_vec)

        # ── 2. 컬럼 키워드 오버랩 ────────────────────────────────────
        kw_score, kw_hits = self._keyword_overlap(db, table, q_tokens)

        # ── 3. 구조 풍부도 (FK·JOIN 수) ───────────────────────────────
        struct_score = self._struct_richness(db, table)

        return self._DSIScore(embed_sim, kw_score, struct_score, kw_hits)

    def _schema_embed_sim(self, db: str, table: str, q_vec: np.ndarray) -> float:
        """스키마 텍스트를 임베딩해 질문 벡터와 코사인 유사도를 계산."""
        cache_key = f"{db}.{table}"
        if cache_key not in self._schema_vec_cache:
            info = self._registry.get(db, {}).get(table, {})
            text = _schema_text(table, info)
            self._schema_vec_cache[cache_key] = self._embedder.embed(text)
        s_vec = self._schema_vec_cache[cache_key]
        # 둘 다 L2 정규화 → 내적 = 코사인 유사도
        sim = float(np.dot(q_vec, s_vec))
        return max(0.0, sim)

    def _keyword_overlap(
        self, db: str, table: str, q_tokens: list[str]
    ) -> tuple[float, int]:
        """
        질문 토큰과 (테이블명 + 컬럼명)의 직접 매칭 비율.
        Returns (score, n_hits)
        """
        info = self._registry.get(db, {}).get(table, {})
        columns = [col.lower() for col, _ in info.get("columns", [])]
        table_lower = table.lower()

        hits = 0
        for tok in q_tokens:
            # 테이블명 매칭 (부분 포함 허용)
            if tok in table_lower or table_lower in tok:
                hits += 2   # 테이블명 일치는 가중치 2배
                continue
            # 컬럼명 매칭
            if any(tok in col or col in tok for col in columns):
                hits += 1

        n_candidates = max(len(q_tokens), 1)
        score = min(hits / n_candidates, 1.0)
        return score, hits

    def _struct_richness(self, db: str, table: str) -> float:
        """
        테이블의 FK·JOIN 관계 수를 0~1로 정규화.
        관계가 많을수록 복합 질의에서 중요한 허브 테이블일 가능성이 높다.
        """
        info  = self._registry.get(db, {}).get(table, {})
        n_join = len(info.get("joins", []))
        # 3개 이상 JOIN → 1.0, 0개 → 0.0, 1~2개 → 선형 보간
        return min(n_join / 3.0, 1.0)

    # ------------------------------------------------------------------ #
    #  스키마 레지스트리 갱신                                               #
    # ------------------------------------------------------------------ #

    def update_registry(self, db: str, table: str, table_info: dict) -> None:
        """새 스키마 추가 또는 갱신 시 캐시를 지운다."""
        self._registry.setdefault(db, {})[table] = table_info
        cache_key = f"{db}.{table}"
        self._schema_vec_cache.pop(cache_key, None)

    # ------------------------------------------------------------------ #
    #  팩토리                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_schema_entries(cls, schemas) -> "DSIReranker":
        """
        bench_datasets.base.SchemaEntry 목록에서 레지스트리를 구성한다.
        """
        registry: dict[str, dict[str, dict]] = {}
        for s in schemas:
            registry[s.db_id] = dict(s.tables)
        return cls(registry)
