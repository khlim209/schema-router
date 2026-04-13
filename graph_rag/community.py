"""
Community detection over the Query similarity graph.  (GraphRAG Survey §3.1.3)

Algorithm:
  1. Pull all query embeddings from FAISS.
  2. Build a k-NN similarity graph using cosine similarity.
  3. Run Leiden (or Louvain fallback) community detection.
  4. Write community assignments back to Neo4j.
  5. Build QueryCluster summary + COVERS edges for fast routing.

Why this beats pure text similarity
------------------------------------
Two queries with different wording but the same *access pattern* will be
placed in the same community via shared ACCESSED neighbours, even if their
embedding similarity is below the threshold.  The community centroid then
acts as a richer routing signal.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import igraph as ig
    import leidenalg
    _LEIDEN_AVAILABLE = True
except ImportError:
    _LEIDEN_AVAILABLE = False
    import community as community_louvain  # python-louvain
    import networkx as nx

from embedding.embedder import get_embedder
from embedding.faiss_index import FaissQueryIndex
from graph_db.neo4j_client import Neo4jClient
import config


class CommunityDetector:
    """
    Clusters query nodes into communities and writes results to Neo4j.
    """

    def __init__(self, neo4j: Neo4jClient, faiss_idx: FaissQueryIndex):
        self._neo4j   = neo4j
        self._faiss   = faiss_idx
        self._embedder = get_embedder()

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(self) -> dict[int, list[str]]:
        """
        Full community-detection pipeline.
        Returns { community_id: [query_id, ...] }.
        """
        rows = self._neo4j.get_all_queries()
        if len(rows) < config.MIN_COMMUNITY_SIZE:
            logger.warning(
                f"Only {len(rows)} queries — skipping community detection "
                f"(need ≥ {config.MIN_COMMUNITY_SIZE})."
            )
            return {}

        ids   = [r["id"]   for r in rows]
        texts = [r["text"] for r in rows]

        logger.info(f"Embedding {len(ids)} queries for community detection…")
        vecs = self._embedder.embed_batch(texts)  # (N, dim)

        logger.info("Building k-NN similarity graph…")
        adj = self._build_knn_graph(ids, vecs)

        logger.info("Running community detection…")
        communities = (
            self._leiden(ids, adj)
            if _LEIDEN_AVAILABLE
            else self._louvain(ids, adj)
        )

        # Filter out tiny communities
        communities = {
            cid: members
            for cid, members in communities.items()
            if len(members) >= config.MIN_COMMUNITY_SIZE
        }

        logger.info(f"Found {len(communities)} communities.")
        self._write_to_neo4j(communities, vecs, ids, texts)
        return communities

    # ------------------------------------------------------------------ #
    #  k-NN graph construction                                            #
    # ------------------------------------------------------------------ #

    def _build_knn_graph(
        self, ids: list[str], vecs: np.ndarray, k: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Build an adjacency dict  id → [(neighbour_id, weight), ...]
        using FAISS inner-product search (= cosine on normalised vecs).
        """
        import faiss as _faiss

        n = len(ids)
        k = min(k + 1, n)  # +1 because the query itself is returned

        index = _faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs.astype(np.float32))
        sims, idxs = index.search(vecs.astype(np.float32), k)

        adj: dict[str, list[tuple[str, float]]] = {i: [] for i in ids}
        for row_i, (sim_row, idx_row) in enumerate(zip(sims, idxs)):
            for sim, col_i in zip(sim_row, idx_row):
                if col_i == row_i or col_i < 0:
                    continue
                if sim < config.SIMILARITY_THRESHOLD:
                    continue
                adj[ids[row_i]].append((ids[col_i], float(sim)))

        return adj

    # ------------------------------------------------------------------ #
    #  Leiden (preferred)                                                  #
    # ------------------------------------------------------------------ #

    def _leiden(
        self,
        ids: list[str],
        adj: dict[str, list[tuple[str, float]]],
    ) -> dict[int, list[str]]:
        id2int = {qid: i for i, qid in enumerate(ids)}

        edges, weights = [], []
        for src, neighbours in adj.items():
            for dst, w in neighbours:
                if id2int[src] < id2int[dst]:  # undirected, no duplicates
                    edges.append((id2int[src], id2int[dst]))
                    weights.append(w)

        g = ig.Graph(n=len(ids), edges=edges)
        g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            n_iterations=-1,
            seed=42,
        )
        communities: dict[int, list[str]] = {}
        for cid, members in enumerate(partition):
            communities[cid] = [ids[m] for m in members]
        return communities

    # ------------------------------------------------------------------ #
    #  Louvain fallback                                                    #
    # ------------------------------------------------------------------ #

    def _louvain(
        self,
        ids: list[str],
        adj: dict[str, list[tuple[str, float]]],
    ) -> dict[int, list[str]]:
        G = nx.Graph()
        G.add_nodes_from(ids)
        for src, neighbours in adj.items():
            for dst, w in neighbours:
                G.add_edge(src, dst, weight=w)

        partition: dict[str, int] = community_louvain.best_partition(
            G, weight="weight", resolution=config.COMMUNITY_RESOLUTION
        )
        communities: dict[int, list[str]] = {}
        for qid, cid in partition.items():
            communities.setdefault(cid, []).append(qid)
        return communities

    # ------------------------------------------------------------------ #
    #  Write results to Neo4j                                             #
    # ------------------------------------------------------------------ #

    def _write_to_neo4j(
        self,
        communities: dict[int, list[str]],
        vecs: np.ndarray,
        ids: list[str],
        texts: list[str],
    ) -> None:
        id2idx = {qid: i for i, qid in enumerate(ids)}
        id2text = dict(zip(ids, texts))

        for cid, members in communities.items():
            # Assign community to each member
            for qid in members:
                self._neo4j.set_query_community(qid, cid)

            # Compute centroid summary (use most central member's text)
            member_vecs = vecs[[id2idx[qid] for qid in members]]
            centroid = member_vecs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            sims = member_vecs @ centroid
            representative_qid = members[int(np.argmax(sims))]
            summary = f"[Community {cid}] Representative: {id2text[representative_qid]}"

            self._neo4j.upsert_cluster(cid, summary, len(members))

            # Aggregate COVERS edges: sum access counts from member queries
            all_accesses: dict[tuple[str, str], int] = {}
            for qid in members:
                for row in self._neo4j.get_accessed_paths(qid):
                    key = (row["db"], row["table"])
                    all_accesses[key] = all_accesses.get(key, 0) + row["count"]

            for (db, table), cnt in all_accesses.items():
                self._neo4j.update_cluster_coverage(cid, db, table, cnt)

        logger.info(
            f"Community assignments written to Neo4j "
            f"({sum(len(m) for m in communities.values())} queries)."
        )

    # ------------------------------------------------------------------ #
    #  Assign a new query to nearest community                            #
    # ------------------------------------------------------------------ #

    def assign_community(self, query_vec: np.ndarray) -> int:
        """
        Assign an unseen query to its nearest existing community
        by finding the closest query in FAISS and returning its community.
        Returns -1 if no community exists yet.
        """
        results = self._faiss.search(query_vec, k=5)
        if not results:
            return -1

        # Vote by community_id weighted by similarity
        rows = self._neo4j.get_all_queries()
        qid2cid = {r["id"]: r["community_id"] for r in rows}

        votes: dict[int, float] = {}
        for qid, sim in results:
            cid = qid2cid.get(qid, -1)
            if cid >= 0:
                votes[cid] = votes.get(cid, 0.0) + sim

        if not votes:
            return -1
        return max(votes, key=votes.__getitem__)
