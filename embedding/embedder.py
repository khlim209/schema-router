"""
Query / schema text embedder using sentence-transformers.

Wraps the model with an LRU cache so repeated identical strings
don't re-run inference.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self._model = SentenceTransformer(model_name)
        self._dim = config.EMBEDDING_DIM

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Return a normalised float32 vector for a single string."""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Return (N, dim) normalised float32 matrix."""
        vecs = self._model.encode(list(texts), normalize_embeddings=True,
                                  batch_size=64, show_progress_bar=False)
        return vecs.astype(np.float32)

    @staticmethod
    def text_id(text: str) -> str:
        """Stable SHA-256 hex digest used as a deterministic query ID."""
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


# Module-level singleton so every component shares one loaded model
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
