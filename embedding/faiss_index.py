"""
Persistent FAISS index for fast approximate nearest-neighbour search over
historical query embeddings.

Layout (on disk):
  data/faiss_query.index   – raw FAISS flat inner-product index
  data/query_meta.json     – { faiss_row_id (int) -> query_id (str) } mapping
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

import config


class FaissQueryIndex:
    """
    Maps query texts → dense vectors, supports:
      - add(query_id, vector)
      - search(vector, k) → [(query_id, cosine_sim), ...]
      - persist / load
    """

    def __init__(
        self,
        index_path: str = config.FAISS_INDEX_PATH,
        meta_path: str  = config.QUERY_META_PATH,
        dim: int        = config.EMBEDDING_DIM,
    ):
        self._index_path = Path(index_path)
        self._meta_path  = Path(meta_path)
        self._dim        = dim

        # id2qid[faiss_row_idx] = query_id
        self._id2qid: list[str] = []
        # reverse map for dedup
        self._qid2id: dict[str, int] = {}

        self._index = self._load_or_create()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_or_create(self) -> faiss.Index:
        if self._index_path.exists() and self._meta_path.exists():
            idx = faiss.read_index(str(self._index_path))
            with open(self._meta_path) as f:
                raw = json.load(f)
            # json keys are strings; convert back to int-keyed list
            self._id2qid = [raw[str(i)] for i in range(len(raw))]
            self._qid2id = {qid: i for i, qid in enumerate(self._id2qid)}
            logger.info(f"Loaded FAISS index ({idx.ntotal} vectors) from {self._index_path}")
            return idx
        else:
            logger.info(f"Creating new FAISS IndexFlatIP (dim={self._dim})")
            return faiss.IndexFlatIP(self._dim)  # inner-product on L2-normalised = cosine

    def _ensure_dirs(self):
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._index.ntotal

    def contains(self, query_id: str) -> bool:
        return query_id in self._qid2id

    def add(self, query_id: str, vector: np.ndarray) -> None:
        """Add a single query vector. Skips duplicates."""
        if query_id in self._qid2id:
            return
        vec = vector.reshape(1, -1).astype(np.float32)
        self._index.add(vec)
        row_id = len(self._id2qid)
        self._id2qid.append(query_id)
        self._qid2id[query_id] = row_id

    def add_batch(self, query_ids: list[str], vectors: np.ndarray) -> None:
        """Bulk add, skipping already-indexed ids."""
        new_ids, new_vecs = [], []
        for qid, vec in zip(query_ids, vectors):
            if qid not in self._qid2id:
                new_ids.append(qid)
                new_vecs.append(vec)
        if not new_ids:
            return
        mat = np.vstack(new_vecs).astype(np.float32)
        self._index.add(mat)
        for qid in new_ids:
            row_id = len(self._id2qid)
            self._id2qid.append(qid)
            self._qid2id[qid] = row_id

    def search(
        self, vector: np.ndarray, k: int = config.TOP_K_SIMILAR
    ) -> list[tuple[str, float]]:
        """
        Return [(query_id, cosine_similarity), ...] sorted descending.
        Filters out pairs below SIMILARITY_THRESHOLD.
        """
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        vec = vector.reshape(1, -1).astype(np.float32)
        sims, idxs = self._index.search(vec, k)
        results = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            if sim < config.SIMILARITY_THRESHOLD:
                continue
            results.append((self._id2qid[idx], float(sim)))
        return results

    def persist(self) -> None:
        """Save index and metadata to disk."""
        self._ensure_dirs()
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w") as f:
            json.dump({str(i): qid for i, qid in enumerate(self._id2qid)}, f, indent=2)
        logger.debug(f"FAISS index persisted ({self._index.ntotal} vectors)")
