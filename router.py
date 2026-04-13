"""
QueryRouter — the single entry point for the GraphRAG routing system.

Usage
-----
    router = QueryRouter.build()

    # Index a new access event (online learning)
    router.record(
        query_text="최근 30일간 가장 많이 팔린 상품",
        db_name="ecommerce",
        table_name="order_items",
    )

    # Route a new query → ranked (db, table) paths
    results = router.route("지난달 베스트셀러 상품 목록")
    for path in results:
        print(path.db, path.table, path.score)

    # Full explanation (for debugging / UI)
    info = router.explain("지난달 베스트셀러 상품 목록")
"""

from __future__ import annotations

from loguru import logger

from embedding.faiss_index import FaissQueryIndex
from graph_db.neo4j_client import Neo4jClient
from graph_rag.community import CommunityDetector
from graph_rag.indexer import AccessRecord, GraphIndexer, SchemaDefinition
from graph_rag.retriever import GraphRetriever, SchemaPath
from graph_rag.tiered_retriever import TieredResult, TieredRetriever


class QueryRouter:
    """
    Facade that wires together indexer, community detector, and retriever.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        faiss_idx: FaissQueryIndex,
        indexer: GraphIndexer,
        community: CommunityDetector,
        retriever: GraphRetriever,
        total_tables: int = 10,
    ):
        self._neo4j     = neo4j
        self._faiss     = faiss_idx
        self._indexer   = indexer
        self._community = community
        self._retriever = retriever
        self._tiered    = TieredRetriever(retriever, neo4j, total_tables)

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, total_tables: int = 10) -> "QueryRouter":
        """Instantiate all components and return a ready-to-use router."""
        neo4j     = Neo4jClient()
        faiss_idx = FaissQueryIndex()
        indexer   = GraphIndexer(neo4j, faiss_idx)
        community = CommunityDetector(neo4j, faiss_idx)
        retriever = GraphRetriever(neo4j, faiss_idx, community)

        neo4j.init_constraints()
        logger.info("QueryRouter ready.")
        return cls(neo4j, faiss_idx, indexer, community, retriever, total_tables)

    # ------------------------------------------------------------------ #
    #  Schema management                                                   #
    # ------------------------------------------------------------------ #

    def register_schema(self, schema: SchemaDefinition) -> None:
        """Register (or update) a database schema."""
        self._indexer.ingest_schema(schema)

    def register_schemas(self, schemas: list[SchemaDefinition]) -> None:
        self._indexer.ingest_schemas(schemas)

    # ------------------------------------------------------------------ #
    #  Online access recording                                             #
    # ------------------------------------------------------------------ #

    def record(
        self,
        query_text: str,
        db_name: str,
        table_name: str,
        count: int = 1,
    ) -> str:
        """
        Record one query→(db, table) access event.
        Returns the stable query_id (SHA-256 prefix).
        Updates graph + FAISS index incrementally.
        """
        return self._indexer.ingest_single_access(
            query_text, db_name, table_name, count
        )

    # ------------------------------------------------------------------ #
    #  Bulk historical ingestion                                           #
    # ------------------------------------------------------------------ #

    def load_history(self, records: list[dict]) -> None:
        """
        Bulk-ingest historical access log.
        Each record: { query_text, db_name, table_name, count (optional) }
        """
        access_records = [
            AccessRecord(
                query_text=r["query_text"],
                db_name=r["db_name"],
                table_name=r["table_name"],
                count=r.get("count", 1),
            )
            for r in records
        ]
        self._indexer.ingest_access_log(access_records)

    # ------------------------------------------------------------------ #
    #  Community (re)building                                              #
    # ------------------------------------------------------------------ #

    def rebuild_communities(self) -> dict[int, list[str]]:
        """
        Rerun community detection and write assignments to Neo4j.
        Call after bulk ingestion or periodically via a cron job.
        """
        return self._community.run()

    # ------------------------------------------------------------------ #
    #  Routing                                                             #
    # ------------------------------------------------------------------ #

    def route(
        self,
        query_text: str,
        top_n: int = 5,
    ) -> list[SchemaPath]:
        """
        Route a natural-language query to the top-N schema paths.
        Returns a list of SchemaPath (db, table, score, evidence).
        """
        return self._retriever.route(query_text, top_n=top_n)

    def route_efficient(
        self,
        query_text: str,
        top_n: int = 5,
    ) -> TieredResult:
        """
        쿼리 횟수 최소화 버전.
        신뢰도에 따라 Tier 1~4로 분기하며 lookup_count를 최소화한다.
        결과에 tier, lookup_count, saved_lookups 포함.
        """
        return self._tiered.route(query_text, top_n=top_n)

    def simulate_savings(self, queries: list[str]) -> dict:
        """brute-force vs GraphRAG 쿼리 횟수 비교 시뮬레이션."""
        return self._tiered.simulate_savings(queries)

    def routing_stats(self) -> dict:
        """누적 라우팅 통계 (캐시 적중률, 평균 절감량 등)."""
        return self._tiered.stats()

    def explain(self, query_text: str, top_n: int = 5) -> dict:
        """
        Return full routing explanation including evidence breakdown.
        """
        return self._retriever.explain(query_text, top_n=top_n)

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def dfs_schema_paths(self, db_name: str) -> list[str]:
        """DBCopilot-style DFS-serialised schema paths for a database."""
        return self._indexer.get_dfs_schema_paths(db_name)

    def rebuild_faiss(self) -> None:
        """Rebuild FAISS index from Neo4j (after restart or corruption)."""
        self._indexer.rebuild_faiss_from_neo4j()

    def close(self) -> None:
        self._faiss.persist()
        self._neo4j.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
