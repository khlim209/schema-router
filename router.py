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

from adaptive_retrieval import (
    ExecutionFeedback,
    ExecutionLogger,
    IndexGraphPruningPipeline,
    RetrievalBudget as IndexGraphBudget,
    RetrievalPlan as IndexGraphPlan,
    TableIndexRetriever,
    TableSchemaGraph,
)
from embedding.faiss_index import FaissQueryIndex
from graph_db.neo4j_client import Neo4jClient
from graph_rag.community import CommunityDetector
from graph_rag.indexer import AccessRecord, GraphIndexer, SchemaDefinition
from graph_rag.retriever import GraphRetriever, SchemaPath
from graph_rag.tiered_retriever import TieredResult, TieredRetriever
from planner import MultiHopSchemaPlanner, PathBudget, SchemaTraversalPlan


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
        self._schema_registry: dict[str, dict[str, dict]] = {}
        self._planner = MultiHopSchemaPlanner(
            schema_registry=self._schema_registry,
            graph_retriever=retriever,
        )
        self._table_index = TableIndexRetriever()
        self._schema_graph = TableSchemaGraph()
        self._execution_logger = ExecutionLogger()
        self._index_graph = IndexGraphPruningPipeline(
            self._table_index,
            self._schema_graph,
            execution_logger=self._execution_logger,
        )

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
        self._schema_registry[schema.db_name] = schema.tables
        self._sync_schema_views()

    def register_schemas(self, schemas: list[SchemaDefinition]) -> None:
        for schema in schemas:
            self._indexer.ingest_schema(schema)
            self._schema_registry[schema.db_name] = schema.tables
        self._sync_schema_views()

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
        qid = self._indexer.ingest_single_access(
            query_text, db_name, table_name, count
        )
        self._schema_graph.record_access(query_text, db_name, table_name, count)
        return qid

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
        self._schema_graph.ingest_access_log(access_records)

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

    def retrieve_subgraph(
        self,
        query_text: str,
        top_k: int = 8,
        max_hops: int = 2,
        max_seed_tables: int = 3,
        min_component_size: int = 2,
        max_subgraphs: int = 2,
        max_tables_per_subgraph: int = 8,
        query_type: str = "",
        gold_tables: list[str] | None = None,
    ) -> IndexGraphPlan:
        """
        Practical retrieval flow:
        query -> index top-k -> schema graph pruning -> small subgraph.
        """
        budget = IndexGraphBudget(
            top_k=top_k,
            max_hops=max_hops,
            max_seed_tables=max_seed_tables,
            min_component_size=min_component_size,
            max_subgraphs=max_subgraphs,
            max_tables_per_subgraph=max_tables_per_subgraph,
        )
        return self._index_graph.retrieve(
            query_text,
            budget=budget,
            query_type=query_type,
            gold_tables=gold_tables,
        )

    def retrieve_subgraph_dict(
        self,
        query_text: str,
        top_k: int = 8,
        max_hops: int = 2,
        max_seed_tables: int = 3,
        min_component_size: int = 2,
        max_subgraphs: int = 2,
        max_tables_per_subgraph: int = 8,
        query_type: str = "",
        gold_tables: list[str] | None = None,
    ) -> dict:
        return self.retrieve_subgraph(
            query_text=query_text,
            top_k=top_k,
            max_hops=max_hops,
            max_seed_tables=max_seed_tables,
            min_component_size=min_component_size,
            max_subgraphs=max_subgraphs,
            max_tables_per_subgraph=max_tables_per_subgraph,
            query_type=query_type,
            gold_tables=gold_tables,
        ).to_dict()

    def log_execution_feedback(
        self,
        run_id: str,
        query_text: str,
        executed_tables: list[str],
        contributing_tables: list[str],
        unnecessary_tables: list[str] | None = None,
        success: bool = True,
        latency_ms: float = 0.0,
        final_answer: str = "",
        notes: str = "",
        gold_tables: list[str] | None = None,
        query_type: str = "",
    ) -> None:
        """
        Persist runtime feedback for later DSI / GNN supervision.
        """
        self._execution_logger.log_feedback(
            ExecutionFeedback(
                run_id=run_id,
                query=query_text,
                query_type=query_type,
                executed_tables=executed_tables,
                contributing_tables=contributing_tables,
                unnecessary_tables=unnecessary_tables or [],
                success=success,
                latency_ms=latency_ms,
                final_answer=final_answer,
                notes=notes,
                gold_tables=gold_tables or [],
            )
        )

    def plan(
        self,
        query_text: str,
        max_hops: int = 3,
        max_tables: int = 5,
        max_entrypoints: int = 3,
        max_candidate_paths: int = 3,
        max_mcp_calls: int = 4,
    ) -> SchemaTraversalPlan:
        """
        Produce a budget-aware multi-hop schema traversal plan.
        """
        budget = PathBudget(
            max_hops=max_hops,
            max_tables=max_tables,
            max_entrypoints=max_entrypoints,
            max_candidate_paths=max_candidate_paths,
            max_mcp_calls=max_mcp_calls,
        )
        return self._planner.plan(query_text, budget=budget)

    def plan_dict(
        self,
        query_text: str,
        max_hops: int = 3,
        max_tables: int = 5,
        max_entrypoints: int = 3,
        max_candidate_paths: int = 3,
        max_mcp_calls: int = 4,
    ) -> dict:
        """
        Convenience wrapper for serializable API responses.
        """
        return self.plan(
            query_text=query_text,
            max_hops=max_hops,
            max_tables=max_tables,
            max_entrypoints=max_entrypoints,
            max_candidate_paths=max_candidate_paths,
            max_mcp_calls=max_mcp_calls,
        ).to_dict()

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def dfs_schema_paths(self, db_name: str) -> list[str]:
        """DBCopilot-style DFS-serialised schema paths for a database."""
        return self._indexer.get_dfs_schema_paths(db_name)

    def rebuild_faiss(self) -> None:
        """Rebuild FAISS index from Neo4j (after restart or corruption)."""
        self._indexer.rebuild_faiss_from_neo4j()

    @property
    def schema_registry(self) -> dict[str, dict[str, dict]]:
        return self._schema_registry

    @property
    def execution_log_path(self) -> str:
        return self._execution_logger.path

    def _sync_schema_views(self) -> None:
        self._planner.update_registry(self._schema_registry)
        self._table_index.build(self._schema_registry)
        self._schema_graph.rebuild(self._schema_registry)

    def close(self) -> None:
        self._faiss.persist()
        self._neo4j.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
