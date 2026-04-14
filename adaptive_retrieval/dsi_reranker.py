from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from bench_datasets.base import BenchmarkSample

from .models import TableCandidate
from .schema_graph import TableSchemaGraph
from .table_index import TableIndexRetriever


FEATURE_NAMES = [
    "base_score",
    "embedding_score",
    "lexical_score",
    "join_score",
    "degree_score",
    "reciprocal_rank",
    "same_db_as_top1",
    "local_rank_recip",
    "graph_degree_norm",
    "same_db_candidate_ratio",
    "neighbor_candidate_ratio",
    "distance_to_seed_norm",
]


@dataclass
class DSITrainingStats:
    n_examples: int
    positive_rate: float
    average_precision: float
    roc_auc: float | None


class DSIRerankerModel:
    """
    Trainable DSI-style reranker for index candidates.

    The model is intentionally lightweight: it learns to re-rank top-k table
    candidates using structured features derived from the query, the table
    document, and the schema graph.
    """

    def __init__(
        self,
        model: LogisticRegression | None = None,
        blend_alpha: float = 0.7,
        top_k: int = 8,
        max_hops: int = 2,
    ):
        self._model = model or LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
        self._blend_alpha = blend_alpha
        self._top_k = top_k
        self._max_hops = max_hops

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def max_hops(self) -> int:
        return self._max_hops

    @staticmethod
    def _feature_dicts(
        query: str,
        candidates: list[TableCandidate],
        schema_graph: TableSchemaGraph,
        max_hops: int,
    ) -> list[dict[str, float]]:
        if not candidates:
            return []

        top_db = candidates[0].db
        top_table = candidates[0].table

        counts_by_db: dict[str, int] = {}
        for candidate in candidates:
            counts_by_db.setdefault(candidate.db, 0)
            counts_by_db[candidate.db] += 1
        max_db_count = max(counts_by_db.values()) or 1

        graph_degrees = [
            schema_graph.degree(candidate.db, candidate.table)
            for candidate in candidates
        ]
        max_graph_degree = max(graph_degrees) or 1.0

        same_db_groups: dict[str, list[TableCandidate]] = {}
        for candidate in candidates:
            same_db_groups.setdefault(candidate.db, []).append(candidate)

        feature_rows: list[dict[str, float]] = []
        for rank, candidate in enumerate(candidates, start=1):
            same_db = same_db_groups[candidate.db]
            local_rank = next(
                idx
                for idx, item in enumerate(same_db, start=1)
                if item.table == candidate.table
            )
            if candidate.db == top_db:
                if candidate.table == top_table:
                    distance = 0
                else:
                    path = schema_graph.shortest_path(
                        candidate.db,
                        top_table,
                        candidate.table,
                        max_hops=max_hops,
                    )
                    distance = max_hops + 1 if not path else len(path) - 1
            else:
                distance = max_hops + 1

            neighbors = set(schema_graph.neighbors(candidate.db, candidate.table))
            same_db_tables = {item.table for item in same_db if item.table != candidate.table}
            neighbor_hits = len(neighbors & same_db_tables)
            neighbor_ratio = (
                neighbor_hits / max(1, len(same_db_tables))
                if same_db_tables else 0.0
            )

            feature_rows.append(
                {
                    "base_score": candidate.score,
                    "embedding_score": candidate.embedding_score,
                    "lexical_score": candidate.lexical_score,
                    "join_score": candidate.join_score,
                    "degree_score": candidate.degree_score,
                    "reciprocal_rank": 1.0 / rank,
                    "same_db_as_top1": float(candidate.db == top_db),
                    "local_rank_recip": 1.0 / local_rank,
                    "graph_degree_norm": schema_graph.degree(candidate.db, candidate.table) / max_graph_degree,
                    "same_db_candidate_ratio": counts_by_db[candidate.db] / max_db_count,
                    "neighbor_candidate_ratio": neighbor_ratio,
                    "distance_to_seed_norm": distance / max(1, max_hops + 1),
                }
            )

        return feature_rows

    @classmethod
    def build_training_matrix(
        cls,
        samples: list[BenchmarkSample],
        table_index: TableIndexRetriever,
        schema_graph: TableSchemaGraph,
        top_k: int = 8,
        max_hops: int = 2,
        include_gold: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: list[list[float]] = []
        labels: list[int] = []

        for sample in samples:
            candidates = table_index.search(sample.question, top_k=top_k)
            if include_gold:
                existing = {(candidate.db, candidate.table) for candidate in candidates}
                for table_name in sample.used_tables:
                    key = (sample.db_id, table_name)
                    if key in existing:
                        continue
                    gold_candidate = table_index.score_table(
                        sample.question,
                        sample.db_id,
                        table_name,
                    )
                    if gold_candidate is not None:
                        candidates.append(gold_candidate)
                candidates.sort(key=lambda item: item.score, reverse=True)

            feature_dicts = cls._feature_dicts(
                sample.question,
                candidates,
                schema_graph=schema_graph,
                max_hops=max_hops,
            )
            gold = {(sample.db_id, table) for table in sample.used_tables}

            for candidate, feature_map in zip(candidates, feature_dicts):
                rows.append([feature_map[name] for name in FEATURE_NAMES])
                labels.append(int((candidate.db, candidate.table) in gold))

        if not rows:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def fit(
        self,
        samples: list[BenchmarkSample],
        table_index: TableIndexRetriever,
        schema_graph: TableSchemaGraph,
        top_k: int = 8,
        max_hops: int = 2,
        include_gold: bool = True,
    ) -> DSITrainingStats:
        self._top_k = top_k
        self._max_hops = max_hops

        x_train, y_train = self.build_training_matrix(
            samples,
            table_index=table_index,
            schema_graph=schema_graph,
            top_k=top_k,
            max_hops=max_hops,
            include_gold=include_gold,
        )
        if len(x_train) == 0:
            raise ValueError("No DSI training examples were generated.")
        if len(set(y_train.tolist())) < 2:
            raise ValueError(
                "DSI reranker training requires both positive and negative examples."
            )

        self._model.fit(x_train, y_train)
        probs = self._model.predict_proba(x_train)[:, 1]
        average_precision = float(average_precision_score(y_train, probs))
        roc_auc = None
        if len(set(y_train.tolist())) > 1:
            roc_auc = float(roc_auc_score(y_train, probs))

        stats = DSITrainingStats(
            n_examples=len(x_train),
            positive_rate=float(np.mean(y_train)),
            average_precision=average_precision,
            roc_auc=roc_auc,
        )
        logger.info(
            "DSI reranker fitted on {} examples (positive_rate={:.3f}, ap={:.4f}, auc={})",
            stats.n_examples,
            stats.positive_rate,
            stats.average_precision,
            f"{stats.roc_auc:.4f}" if stats.roc_auc is not None else "n/a",
        )
        return stats

    def rerank_candidates(
        self,
        query: str,
        candidates: list[TableCandidate],
        schema_graph: TableSchemaGraph,
    ) -> list[TableCandidate]:
        if not candidates:
            return []

        feature_dicts = self._feature_dicts(
            query,
            candidates,
            schema_graph=schema_graph,
            max_hops=self._max_hops,
        )
        x = np.asarray(
            [[row[name] for name in FEATURE_NAMES] for row in feature_dicts],
            dtype=np.float32,
        )
        probs = self._model.predict_proba(x)[:, 1]

        reranked: list[TableCandidate] = []
        for candidate, prob in zip(candidates, probs):
            final_score = self._blend_alpha * float(prob) + (1.0 - self._blend_alpha) * candidate.score
            reranked.append(
                TableCandidate(
                    db=candidate.db,
                    table=candidate.table,
                    score=final_score,
                    embedding_score=candidate.embedding_score,
                    lexical_score=candidate.lexical_score,
                    join_score=candidate.join_score,
                    degree_score=candidate.degree_score,
                    reasons=list(candidate.reasons) + ["dsi-reranked"],
                    metadata={**candidate.metadata, "dsi_probability": float(prob)},
                )
            )

        reranked.sort(
            key=lambda item: (
                item.score,
                float(item.metadata.get("dsi_probability", 0.0)),
            ),
            reverse=True,
        )
        return reranked

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "model": self._model,
                    "blend_alpha": self._blend_alpha,
                    "top_k": self._top_k,
                    "max_hops": self._max_hops,
                    "feature_names": FEATURE_NAMES,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "DSIRerankerModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)

        model = cls(
            model=payload["model"],
            blend_alpha=float(payload.get("blend_alpha", 0.7)),
            top_k=int(payload.get("top_k", 8)),
            max_hops=int(payload.get("max_hops", 2)),
        )
        return model
