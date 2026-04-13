"""
G-Retrieval  (GraphRAG Survey §3.2 + DBCopilot schema routing)

Routing score for a candidate (db, table) path:

  score = α × embedding_sim
        + β × access_count_weight     (normalised log-scale)
        + γ × community_coverage      (normalised)

Where:
  α + β + γ = 1   (see config.py)

Key advantage over plain text similarity
-----------------------------------------
- Even if a new query has low similarity to any individual historical query,
  it may still be assigned to a community whose COVERS edges point directly
  to the right tables.
- The schema-graph BFS expansion (JOINS_WITH) propagates evidence through
  related tables, capturing indirect access patterns.
- DBCopilot-style prefix constraint: only valid (db → table) paths are
  returned, so the result is always a coherent schema path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from embedding.embedder import Embedder, get_embedder
from embedding.faiss_index import FaissQueryIndex
from graph_db.neo4j_client import Neo4jClient
from graph_rag.community import CommunityDetector
import config


@dataclass
class SchemaPath:
    """A ranked candidate (db, table) result."""
    db: str
    table: str
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"SchemaPath({self.db}.{self.table}, score={self.score:.4f})"


class GraphRetriever:
    """
    Given a natural-language query, returns a ranked list of SchemaPath
    candidates using the GraphRAG knowledge graph.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        faiss_idx: FaissQueryIndex,
        community_detector: CommunityDetector,
    ):
        self._neo4j      = neo4j
        self._faiss      = faiss_idx
        self._community  = community_detector
        self._embedder: Embedder = get_embedder()

    # ------------------------------------------------------------------ #
    #  Main routing method                                                 #
    # ------------------------------------------------------------------ #

    def route(
        self,
        query_text: str,
        top_n: int = 5,
        expand_hops: int = config.MAX_GRAPH_HOPS,
    ) -> list[SchemaPath]:
        """
        Route a query to the top-N most likely (db, table) schema paths.
        """
        query_vec = self._embedder.embed(query_text)

        # ── Step 1: FAISS k-NN search ──────────────────────────────────
        similar = self._faiss.search(query_vec, k=config.TOP_K_SIMILAR)
        logger.debug(f"FAISS found {len(similar)} similar queries.")

        # ── Step 2: Aggregate access evidence from similar queries ─────
        candidate_scores: dict[tuple[str, str], dict] = {}

        if similar:
            sim_ids   = [qid for qid, _ in similar]
            sim_dict  = dict(similar)          # qid → cosine_sim
            accesses  = self._neo4j.get_schema_paths_for_queries(sim_ids)

            for row in accesses:
                key = (row["db"], row["table"])
                cs  = sim_dict.get(row["query_id"], 0.0)
                c   = candidate_scores.setdefault(key, {
                    "embed_sim": 0.0, "access_sum": 0, "n_queries": 0
                })
                c["embed_sim"]  = max(c["embed_sim"], cs)
                c["access_sum"] += row["count"]
                c["n_queries"]  += 1

        # ── Step 3: Community-coverage boost ─────────────────────────
        community_id = self._community.assign_community(query_vec)
        community_tables: dict[tuple[str, str], int] = {}
        if community_id >= 0:
            for row in self._neo4j.get_cluster_covered_tables(community_id):
                community_tables[(row["db"], row["table"])] = row["count"]

            # Add community evidence even for tables not yet in candidates
            for (db, table), cnt in community_tables.items():
                key = (db, table)
                if key not in candidate_scores:
                    candidate_scores[key] = {
                        "embed_sim": 0.0, "access_sum": 0, "n_queries": 0
                    }

        # ── Step 4: Schema-graph BFS expansion ────────────────────────
        if expand_hops > 0:
            seed_keys = list(candidate_scores.keys())
            for db, table in seed_keys:
                neighbours = self._neo4j.get_neighboring_tables(
                    db, table, max_hops=expand_hops
                )
                for nb in neighbours:
                    key = (nb["db"], nb["table"])
                    if key not in candidate_scores:
                        # Inherit a fraction of the seed's embed_sim
                        parent_sim = candidate_scores.get(
                            (db, table), {}
                        ).get("embed_sim", 0.0)
                        decay = 0.5 ** nb["hops"]
                        candidate_scores[key] = {
                            "embed_sim": parent_sim * decay,
                            "access_sum": 0,
                            "n_queries": 0,
                        }

        if not candidate_scores:
            logger.warning("No routing evidence found for query.")
            return []

        # ── Step 5: Compute final scores ──────────────────────────────
        max_access = max(
            c["access_sum"] for c in candidate_scores.values()
        ) or 1
        max_community = max(community_tables.values(), default=1)

        results: list[SchemaPath] = []
        for (db, table), ev in candidate_scores.items():
            embed_sim = ev["embed_sim"]

            # Log-normalised access count weight
            access_norm = math.log1p(ev["access_sum"]) / math.log1p(max_access)

            # Community coverage weight
            comm_cnt = community_tables.get((db, table), 0)
            comm_norm = math.log1p(comm_cnt) / math.log1p(max_community)

            score = (
                config.ALPHA * embed_sim
                + config.BETA  * access_norm
                + config.GAMMA * comm_norm
            )

            results.append(SchemaPath(
                db=db,
                table=table,
                score=score,
                evidence={
                    "embedding_sim":      round(embed_sim, 4),
                    "access_count":       ev["access_sum"],
                    "community_id":       community_id,
                    "community_coverage": comm_cnt,
                    "n_similar_queries":  ev["n_queries"],
                },
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        top = results[:top_n]

        logger.info(
            f"Routed '{query_text[:60]}' → "
            + ", ".join(f"{p.db}.{p.table}({p.score:.3f})" for p in top)
        )
        return top

    # ------------------------------------------------------------------ #
    #  Explain helper (for debugging / API)                               #
    # ------------------------------------------------------------------ #

    def explain(self, query_text: str, top_n: int = 5) -> dict:
        """
        Return routing results + full evidence breakdown.
        """
        query_vec = self._embedder.embed(query_text)
        similar   = self._faiss.search(query_vec, k=config.TOP_K_SIMILAR)
        cid       = self._community.assign_community(query_vec)

        paths = self.route(query_text, top_n=top_n)

        return {
            "query":           query_text,
            "community_id":    cid,
            "similar_queries": [
                {"id": qid, "similarity": round(sim, 4)}
                for qid, sim in similar
            ],
            "ranked_paths": [
                {
                    "db":       p.db,
                    "table":    p.table,
                    "score":    round(p.score, 4),
                    "evidence": p.evidence,
                }
                for p in paths
            ],
        }
