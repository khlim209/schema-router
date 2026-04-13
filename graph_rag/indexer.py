"""
G-Indexing  (GraphRAG Survey §3.1 + DBCopilot schema graph)

Responsibilities:
  1. Ingest schema definitions  → populate Neo4j schema nodes
  2. Ingest access-log records  → create Query nodes + ACCESSED edges
  3. Build / rebuild the FAISS vector index from all stored queries
  4. Expose DFS-serialised schema paths (DBCopilot §3.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from loguru import logger
from tqdm import tqdm

from embedding.embedder import Embedder, get_embedder
from embedding.faiss_index import FaissQueryIndex
from graph_db.neo4j_client import Neo4jClient


@dataclass
class SchemaDefinition:
    """Describes one database with its tables and columns."""
    db_name: str
    description: str = ""
    # { table_name: { "description": str, "columns": [(col, type), ...],
    #                  "joins": [(other_table, via_col), ...] } }
    tables: dict[str, dict] = field(default_factory=dict)


@dataclass
class AccessRecord:
    """Single (query_text, db, table) access event."""
    query_text: str
    db_name: str
    table_name: str
    count: int = 1


class GraphIndexer:
    """
    Builds and maintains the GraphRAG knowledge graph.
    """

    def __init__(self, neo4j: Neo4jClient, faiss_idx: FaissQueryIndex):
        self._neo4j   = neo4j
        self._faiss   = faiss_idx
        self._embedder: Embedder = get_embedder()

    # ------------------------------------------------------------------ #
    #  Schema ingestion                                                    #
    # ------------------------------------------------------------------ #

    def ingest_schema(self, schema: SchemaDefinition) -> None:
        """Register a full database schema into Neo4j."""
        self._neo4j.upsert_database(schema.db_name, schema.description)
        for table_name, table_info in schema.tables.items():
            self._neo4j.upsert_table(
                schema.db_name, table_name,
                table_info.get("description", "")
            )
            for col_name, col_type in table_info.get("columns", []):
                self._neo4j.upsert_column(
                    schema.db_name, table_name, col_name, col_type
                )
            for other_table, via_col in table_info.get("joins", []):
                self._neo4j.upsert_join(
                    schema.db_name, table_name, other_table, via_col
                )
        logger.info(
            f"Ingested schema for '{schema.db_name}' "
            f"({len(schema.tables)} tables)"
        )

    def ingest_schemas(self, schemas: Sequence[SchemaDefinition]) -> None:
        for s in schemas:
            self.ingest_schema(s)

    # ------------------------------------------------------------------ #
    #  Access-log ingestion                                                #
    # ------------------------------------------------------------------ #

    def ingest_access_log(self, records: Sequence[AccessRecord]) -> None:
        """
        Bulk-ingest historical access records.
        Creates Query nodes, ACCESSED edges, and updates FAISS index.
        """
        query_ids: list[str] = []
        texts:     list[str] = []

        for rec in tqdm(records, desc="Ingesting access log"):
            qid = self._embedder.text_id(rec.query_text)
            self._neo4j.upsert_query(qid, rec.query_text)
            self._neo4j.record_access(qid, rec.db_name, rec.table_name, rec.count)
            if qid not in {q for q in query_ids}:
                query_ids.append(qid)
                texts.append(rec.query_text)

        # Embed new queries and add to FAISS
        new_ids = [qid for qid in query_ids if not self._faiss.contains(qid)]
        if new_ids:
            new_texts = [texts[query_ids.index(qid)] for qid in new_ids]
            logger.info(f"Embedding {len(new_ids)} new queries…")
            vecs = self._embedder.embed_batch(new_texts)
            self._faiss.add_batch(new_ids, vecs)
            self._faiss.persist()

        logger.info(
            f"Ingested {len(records)} access records. "
            f"FAISS index size: {len(self._faiss)}"
        )

    def ingest_single_access(
        self, query_text: str, db_name: str, table_name: str, count: int = 1
    ) -> str:
        """
        Online: record one access and return the query_id.
        """
        qid = self._embedder.text_id(query_text)
        self._neo4j.upsert_query(qid, query_text)
        self._neo4j.record_access(qid, db_name, table_name, count)

        if not self._faiss.contains(qid):
            vec = self._embedder.embed(query_text)
            self._faiss.add(qid, vec)
            self._faiss.persist()

        return qid

    # ------------------------------------------------------------------ #
    #  Rebuild FAISS from Neo4j (recovery / sync)                         #
    # ------------------------------------------------------------------ #

    def rebuild_faiss_from_neo4j(self) -> None:
        """Rebuild the FAISS index from all Query nodes in Neo4j."""
        rows = self._neo4j.get_all_queries()
        if not rows:
            logger.warning("No queries found in Neo4j — nothing to rebuild.")
            return

        ids   = [r["id"]   for r in rows]
        texts = [r["text"] for r in rows]

        logger.info(f"Rebuilding FAISS index from {len(ids)} queries in Neo4j…")
        vecs = self._embedder.embed_batch(texts)
        self._faiss.add_batch(ids, vecs)
        self._faiss.persist()
        logger.info("FAISS rebuild complete.")

    # ------------------------------------------------------------------ #
    #  DFS schema path serialisation (DBCopilot §3.2)                     #
    # ------------------------------------------------------------------ #

    def get_dfs_schema_paths(self, db_name: str) -> list[str]:
        """
        Return DFS-ordered 'db.table.column' path strings.
        Used as structured context for an optional downstream LLM step.
        """
        return self._neo4j.get_dfs_schema_paths(db_name)
