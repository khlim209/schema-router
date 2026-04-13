"""
Online / batch trainer for the GraphRAG Query Router.

Responsibilities
----------------
1. Apply exponential decay to old ACCESSED edge weights (prevents stale
   data from dominating routing scores indefinitely).
2. Periodic community re-detection trigger (after N new records or time).
3. Export / import training snapshots (JSON) for migration or debugging.
4. Provide a training summary report.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from graph_db.neo4j_client import Neo4jClient
from router import QueryRouter
import config


class Trainer:
    """
    Manages the training lifecycle for the QueryRouter.
    """

    def __init__(self, router: QueryRouter, neo4j: Neo4jClient):
        self._router = router
        self._neo4j  = neo4j
        self._new_records_since_community_rebuild = 0
        self._community_rebuild_threshold = 50  # rebuild after N new records

    # ------------------------------------------------------------------ #
    #  Online learning                                                     #
    # ------------------------------------------------------------------ #

    def learn(
        self,
        query_text: str,
        db_name: str,
        table_name: str,
        count: int = 1,
    ) -> str:
        """
        Record one access event and trigger community rebuild if needed.
        Returns the query_id.
        """
        qid = self._router.record(query_text, db_name, table_name, count)
        self._new_records_since_community_rebuild += 1

        if self._new_records_since_community_rebuild >= self._community_rebuild_threshold:
            logger.info(
                f"Threshold reached ({self._community_rebuild_threshold} new records). "
                "Rebuilding communities…"
            )
            self._router.rebuild_communities()
            self._new_records_since_community_rebuild = 0

        return qid

    # ------------------------------------------------------------------ #
    #  Exponential decay on stale edges                                   #
    # ------------------------------------------------------------------ #

    def apply_decay(self, decay: float = config.DECAY_FACTOR) -> int:
        """
        Multiply all ACCESSED edge weights by `decay`.
        Removes edges that fall below MIN_ACCESS_COUNT.
        Returns number of edges pruned.
        """
        pruned = 0
        with self._neo4j.session() as s:
            # Decay
            s.run(
                """
                MATCH ()-[r:ACCESSED]->()
                SET r.count = r.count * $decay
                """,
                decay=decay,
            )
            # Prune
            result = s.run(
                """
                MATCH ()-[r:ACCESSED]->()
                WHERE r.count < $min_count
                DELETE r
                RETURN count(r) AS pruned
                """,
                min_count=config.MIN_ACCESS_COUNT,
            )
            row = result.single()
            pruned = row["pruned"] if row else 0

        logger.info(f"Decay applied (factor={decay}). Pruned {pruned} stale edges.")
        return pruned

    # ------------------------------------------------------------------ #
    #  Bulk import / export                                               #
    # ------------------------------------------------------------------ #

    def export_access_log(self, path: str = "data/access_log_export.json") -> None:
        """Export all (query_text, db, table, count) records to JSON."""
        rows = self._neo4j.get_all_queries()
        records = []
        for row in rows:
            for access in self._neo4j.get_accessed_paths(row["id"]):
                records.append({
                    "query_text": row["text"],
                    "db_name":    access["db"],
                    "table_name": access["table"],
                    "count":      access["count"],
                })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(records)} access records to {path}")

    def import_access_log(self, path: str) -> None:
        """Import and ingest an access log JSON file."""
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        logger.info(f"Importing {len(records)} records from {path}…")
        self._router.load_history(records)
        self._router.rebuild_communities()
        logger.info("Import complete.")

    # ------------------------------------------------------------------ #
    #  Summary report                                                      #
    # ------------------------------------------------------------------ #

    def report(self) -> dict:
        """Return a summary of the current training state."""
        queries = self._neo4j.get_all_queries()
        total_accesses = sum(
            sum(a["count"] for a in self._neo4j.get_accessed_paths(q["id"]))
            for q in queries
        )
        communities = {}
        for q in queries:
            cid = q.get("community_id", -1)
            communities.setdefault(cid, 0)
            communities[cid] += 1

        return {
            "generated_at":     datetime.utcnow().isoformat(),
            "total_queries":    len(queries),
            "total_accesses":   total_accesses,
            "faiss_index_size": self._router._faiss.__len__(),
            "num_communities":  len([k for k in communities if k >= 0]),
            "community_sizes":  {
                str(k): v for k, v in communities.items() if k >= 0
            },
        }
