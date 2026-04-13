"""
Keyword-based Graph Router
==========================
FAISS / 벡터 인덱스 없이 순수 Cypher 그래프 탐색으로 테이블을 찾는다.

동작 흐름
---------
1. 질의에서 키워드 추출 (형태소 + 동의어 사전)
2. Cypher: 키워드와 컬럼명/테이블명이 매칭되는 테이블 탐색  ← 1번째 Cypher
3. Cypher: JOINS_WITH 엣지로 인접 테이블 BFS 확장            ← 2번째 Cypher (선택)
4. Cypher: 접근 히스토리 가중치 합산                          ← 3번째 Cypher (선택)
5. 최종 점수 = keyword_match * α + join_proximity * β + access_count * γ

기존 방식과의 차이
-----------------
기존: 질의 임베딩 → FAISS 유사 질의 검색 → 접근 기록 집계
신규: 질의 키워드 → Cypher 직접 탐색 → 히스토리 보정

쿼리 횟수 측정
--------------
CypherCounter 로 실행된 Cypher 수를 기록한다.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from graph_db.neo4j_client import Neo4jClient


# ──────────────────────────────────────────────────────────────────────── #
#  동의어 / 키워드 사전                                                     #
#  비즈니스 용어 → DB 컬럼/테이블명에서 자주 쓰이는 영어 패턴으로 확장     #
# ──────────────────────────────────────────────────────────────────────── #

_SYNONYM_MAP: dict[str, list[str]] = {
    # 매출/금액
    "매출": ["revenue", "sales", "amount", "total", "price", "income"],
    "금액": ["amount", "price", "total", "cost", "fee", "revenue"],
    "수익": ["revenue", "profit", "income", "earnings"],
    "판매": ["sales", "order", "sell", "transaction"],
    "가격": ["price", "cost", "fee", "amount"],

    # 주문
    "주문": ["order", "purchase", "transaction", "booking"],
    "구매": ["purchase", "order", "buy", "transaction"],
    "결제": ["payment", "transaction", "charge"],

    # 상품/제품
    "상품": ["product", "item", "goods", "commodity"],
    "제품": ["product", "item", "goods"],
    "재고": ["stock", "inventory", "quantity", "qty"],
    "카테고리": ["category", "type", "class", "group"],

    # 고객
    "고객": ["customer", "client", "user", "member", "buyer"],
    "사용자": ["user", "member", "account", "customer"],
    "회원": ["member", "user", "account"],
    "가입": ["signup", "register", "join", "created"],

    # 날짜/시간
    "날짜": ["date", "time", "at", "created", "updated"],
    "기간": ["date", "period", "range", "time", "duration"],
    "일별": ["daily", "date", "day"],
    "월별": ["monthly", "month"],
    "분기": ["quarter", "quarterly"],

    # 통계/분석
    "통계": ["stats", "statistics", "count", "total", "summary"],
    "분석": ["analysis", "stats", "metric", "report"],
    "현황": ["status", "stats", "summary", "current"],
    "추이": ["trend", "daily", "history", "log"],
    "순위": ["rank", "top", "order", "sort"],

    # 마케팅
    "캠페인": ["campaign", "marketing", "promotion"],
    "이메일": ["email", "mail", "message"],
    "클릭": ["click", "event", "action"],
    "오픈": ["open", "view", "read"],

    # 퍼널/전환
    "전환": ["conversion", "funnel", "convert"],
    "이탈": ["dropout", "abandon", "exit", "bounce"],
    "퍼널": ["funnel", "step", "stage"],

    # 공통 영어 → 영어 확장
    "revenue":    ["revenue", "sales", "amount", "total"],
    "order":      ["order", "purchase", "transaction"],
    "customer":   ["customer", "client", "user", "member"],
    "product":    ["product", "item", "goods"],
    "sales":      ["sales", "order", "amount", "revenue"],
    "user":       ["user", "member", "customer", "account"],
    "date":       ["date", "time", "at", "created"],
    "campaign":   ["campaign", "marketing"],
    "email":      ["email", "mail"],
    "conversion": ["conversion", "funnel"],
    "retention":  ["retention", "cohort", "churn"],
}

_STOPWORDS = {
    # 한국어 불용어
    "이", "가", "은", "는", "을", "를", "의", "에", "에서", "로", "으로",
    "와", "과", "도", "만", "이다", "하다", "있다", "없다", "되다",
    "지난", "이번", "최근", "전체", "모든", "각", "및", "또는", "그리고",
    "몇", "얼마", "어떤", "어느", "무슨",
    # 영어 불용어
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "shall",
    "in", "on", "at", "to", "for", "of", "by", "with", "from",
    "how", "many", "what", "which", "who", "when", "where", "why",
    "list", "show", "get", "find", "give", "tell", "count", "all",
}


# ──────────────────────────────────────────────────────────────────────── #
#  키워드 추출기                                                            #
# ──────────────────────────────────────────────────────────────────────── #

def extract_keywords(query: str) -> list[str]:
    """
    질의에서 의미 있는 키워드를 추출하고 동의어로 확장한다.
    ML 모델 없이 규칙 기반으로만 처리.
    """
    # 소문자화, 특수문자 제거
    text = query.lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text)

    # 토큰 분리 (공백 기준)
    tokens = [t.strip() for t in text.split() if t.strip()]

    # 불용어 제거
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    # 동의어 확장
    expanded: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if token not in seen:
            expanded.append(token)
            seen.add(token)
        # 동의어 추가
        for synonym in _SYNONYM_MAP.get(token, []):
            if synonym not in seen:
                expanded.append(synonym)
                seen.add(synonym)

    return expanded


# ──────────────────────────────────────────────────────────────────────── #
#  결과 타입                                                                #
# ──────────────────────────────────────────────────────────────────────── #

@dataclass
class KeywordRouteResult:
    db:              str
    table:           str
    score:           float
    matched_cols:    int     = 0
    access_count:    int     = 0
    hop_distance:    int     = 0    # 0 = 직접 매칭, 1~2 = JOIN 확장
    n_cypher_queries: int    = 0    # 실행된 Cypher 수


@dataclass
class KeywordRouteStats:
    """누적 통계."""
    total_queries:      int   = 0
    total_cypher_calls: int   = 0
    direct_hits:        int   = 0   # 그래프 탐색 1회로 찾은 경우
    fallback_hits:      int   = 0   # BFS 확장 후 찾은 경우

    def record(self, n_cypher: int, used_bfs: bool) -> None:
        self.total_queries      += 1
        self.total_cypher_calls += n_cypher
        if used_bfs:
            self.fallback_hits += 1
        else:
            self.direct_hits += 1

    def report(self) -> dict:
        n = self.total_queries or 1
        return {
            "total_queries":         self.total_queries,
            "avg_cypher_per_query":  round(self.total_cypher_calls / n, 2),
            "direct_hit_rate":       f"{self.direct_hits / n * 100:.1f}%",
            "bfs_expansion_rate":    f"{self.fallback_hits / n * 100:.1f}%",
        }


# ──────────────────────────────────────────────────────────────────────── #
#  메인 라우터                                                              #
# ──────────────────────────────────────────────────────────────────────── #

class KeywordGraphRouter:
    """
    FAISS 인덱스 없는 Cypher 기반 그래프 탐색 라우터.
    """

    # 점수 가중치
    W_KEYWORD  = 0.55   # 키워드 매칭 점수
    W_ACCESS   = 0.30   # 접근 히스토리 가중치
    W_JOIN     = 0.15   # JOIN 근접도 (직접 매칭 > 1hop > 2hop)

    def __init__(self, neo4j: Neo4jClient, max_hops: int = 2):
        self._neo4j    = neo4j
        self._max_hops = max_hops
        self._stats    = KeywordRouteStats()

    # ------------------------------------------------------------------ #
    #  메인 라우팅                                                         #
    # ------------------------------------------------------------------ #

    def route(
        self,
        query: str,
        top_n: int = 10,
    ) -> tuple[list[KeywordRouteResult], int]:
        """
        Returns (ranked_results, n_cypher_queries_used)
        """
        keywords   = extract_keywords(query)
        n_cypher   = 0
        used_bfs   = False

        if not keywords:
            self._stats.record(0, False)
            return [], 0

        # ── Cypher 1: 키워드 직접 매칭 ────────────────────────────────
        direct_results = self._keyword_match(keywords)
        n_cypher += 1

        scores: dict[tuple[str, str], dict] = {}
        for row in direct_results:
            key = (row["db"], row["table"])
            scores[key] = {
                "matched_cols": row["matched_cols"],
                "access_count": row.get("access_count", 0),
                "hop":          0,
            }

        # ── Cypher 2: 히스토리 가중치 병합 ────────────────────────────
        if scores:
            candidate_tables = [
                {"db": k[0], "table": k[1]}
                for k in scores
            ]
            access_data = self._get_access_counts(
                [k[1] for k in scores],
                [k[0] for k in scores],
            )
            n_cypher += 1
            for row in access_data:
                key = (row["db"], row["table"])
                if key in scores:
                    scores[key]["access_count"] = row.get("access_count", 0)

        # ── Cypher 3: JOIN BFS 확장 (직접 매칭 결과가 부족할 때) ───────
        if len(scores) < top_n:
            seed_tables = list(scores.keys())[:5]
            for db, table in seed_tables:
                bfs_results = self._bfs_expand(db, table)
                n_cypher   += 1
                used_bfs    = True
                for row in bfs_results:
                    key = (row["db"], row["table"])
                    if key not in scores:
                        scores[key] = {
                            "matched_cols": 0,
                            "access_count": row.get("access_count", 0),
                            "hop":          row.get("hops", 1),
                        }

        # ── 최종 점수 계산 ─────────────────────────────────────────────
        max_cols   = max((v["matched_cols"] for v in scores.values()), default=1) or 1
        max_access = max((v["access_count"] for v in scores.values()), default=1) or 1

        import math
        results: list[KeywordRouteResult] = []
        for (db, table), ev in scores.items():
            kw_score     = ev["matched_cols"] / max_cols
            access_score = math.log1p(ev["access_count"]) / math.log1p(max_access)
            hop_penalty  = 1.0 / (1 + ev["hop"])

            score = (
                self.W_KEYWORD * kw_score
                + self.W_ACCESS * access_score
                + self.W_JOIN   * hop_penalty
            )
            results.append(KeywordRouteResult(
                db=db, table=table, score=score,
                matched_cols=ev["matched_cols"],
                access_count=ev["access_count"],
                hop_distance=ev["hop"],
                n_cypher_queries=n_cypher,
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        self._stats.record(n_cypher, used_bfs)
        return results[:top_n], n_cypher

    def ranked_tables(
        self, query: str, all_tables: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], int]:
        """
        (db, table) 순위 리스트 + 실행된 Cypher 수 반환.
        all_tables에 있는 것만 포함, 나머지는 점수 0으로 뒤에 붙임.
        """
        results, n_cypher = self.route(query, top_n=len(all_tables))
        ranked = [(r.db, r.table) for r in results]
        remaining = [t for t in all_tables if t not in ranked]
        return ranked + remaining, n_cypher

    # ------------------------------------------------------------------ #
    #  Cypher 쿼리들                                                       #
    # ------------------------------------------------------------------ #

    def _keyword_match(self, keywords: list[str]) -> list[dict]:
        """
        키워드가 컬럼명/테이블명/설명에 포함된 테이블 반환.
        접근 히스토리도 함께 집계.
        """
        with self._neo4j.session() as s:
            result = s.run(
                """
                MATCH (db:Database)-[:HAS_TABLE]->(t:Table)
                WHERE any(kw IN $keywords
                      WHERE toLower(t.name) CONTAINS kw
                         OR kw CONTAINS toLower(t.name)
                         OR toLower(coalesce(t.description, '')) CONTAINS kw)
                WITH db, t,
                     size([kw IN $keywords
                           WHERE toLower(t.name) CONTAINS kw
                              OR kw CONTAINS toLower(t.name)]) AS name_matches
                OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                WHERE any(kw IN $keywords
                      WHERE toLower(c.name) CONTAINS kw
                         OR kw CONTAINS toLower(c.name))
                WITH db, t, name_matches, count(c) AS col_matches
                OPTIONAL MATCH (q:Query)-[r:ACCESSED]->(t)
                WITH db, t,
                     name_matches * 3 + col_matches AS matched_cols,
                     coalesce(sum(r.count), 0) AS access_count
                WHERE matched_cols > 0
                RETURN db.name AS db, t.name AS table,
                       matched_cols, access_count
                ORDER BY matched_cols DESC
                LIMIT 20
                """,
                keywords=keywords,
            )
            return [dict(row) for row in result]

    def _get_access_counts(
        self, table_names: list[str], db_names: list[str]
    ) -> list[dict]:
        """테이블 목록의 접근 횟수 조회."""
        with self._neo4j.session() as s:
            result = s.run(
                """
                UNWIND range(0, size($tables)-1) AS i
                MATCH (t:Table {name: $tables[i], db_name: $dbs[i]})
                OPTIONAL MATCH (q:Query)-[r:ACCESSED]->(t)
                RETURN t.db_name AS db, t.name AS table,
                       coalesce(sum(r.count), 0) AS access_count
                """,
                tables=table_names,
                dbs=db_names,
            )
            return [dict(row) for row in result]

    def _bfs_expand(
        self, db_name: str, table_name: str
    ) -> list[dict]:
        """JOINS_WITH 엣지로 인접 테이블 탐색."""
        with self._neo4j.session() as s:
            result = s.run(
                f"""
                MATCH path = (seed:Table {{name: $tname, db_name: $dname}})
                             -[:JOINS_WITH*1..{self._max_hops}]-(nb:Table)
                OPTIONAL MATCH (q:Query)-[r:ACCESSED]->(nb)
                RETURN nb.db_name AS db, nb.name AS table,
                       length(path) AS hops,
                       coalesce(sum(r.count), 0) AS access_count
                ORDER BY hops, access_count DESC
                """,
                tname=table_name, dname=db_name,
            )
            return [dict(row) for row in result]

    # ------------------------------------------------------------------ #
    #  통계                                                                #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return self._stats.report()
