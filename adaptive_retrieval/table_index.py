from __future__ import annotations

import re
from collections.abc import Iterable

import faiss
import numpy as np
from loguru import logger

from bench_datasets.base import SchemaEntry
from embedding.embedder import get_embedder

from .models import TableCandidate, TableDocument


_GENERIC_TAGS = {
    "table",
    "tables",
    "data",
    "info",
    "value",
    "values",
    "name",
    "text",
    "desc",
    "description",
    "type",
    "types",
    "status",
    "date",
    "time",
}


def _tokenise(text: str) -> list[str]:
    raw = re.findall(r"[\w]+", (text or "").lower(), flags=re.UNICODE)
    tokens: list[str] = []
    for token in raw:
        if not token:
            continue
        tokens.append(token)
        for part in token.split("_"):
            if part and part != token:
                tokens.append(part)
    return tokens


def _normalise_columns(columns: Iterable) -> list[str]:
    names: list[str] = []
    for column in columns or []:
        if isinstance(column, (list, tuple)) and column:
            names.append(str(column[0]).lower())
        elif isinstance(column, dict):
            names.append(str(column.get("name", "")).lower())
        else:
            names.append(str(column).lower())
    return [name for name in names if name]


def _normalise_joins(joins: Iterable) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for join in joins or []:
        if isinstance(join, (list, tuple)) and join:
            other = str(join[0]).lower()
            via = str(join[1]).lower() if len(join) > 1 else ""
        elif isinstance(join, dict):
            other = str(join.get("other_table", "")).lower()
            via = str(join.get("via_column", "")).lower()
        else:
            other = str(join).lower()
            via = ""
        if other:
            pairs.append((other, via))
    return pairs


def _key_columns(columns: list[str]) -> list[str]:
    return [
        column for column in columns
        if column == "id" or column.endswith("_id") or column.endswith("_key")
    ]


def _domain_tags(
    table_name: str,
    columns: list[str],
    description: str,
    join_targets: list[str],
) -> list[str]:
    tags = set(_tokenise(table_name))
    tags.update(_tokenise(description))
    for column in columns[:12]:
        tags.update(_tokenise(column))
    for target in join_targets[:6]:
        tags.update(_tokenise(target))
    return sorted(tag for tag in tags if tag and tag not in _GENERIC_TAGS)


def _schema_iter(
    schemas: list[SchemaEntry] | dict[str, dict[str, dict]],
):
    if isinstance(schemas, dict):
        for db_name, tables in schemas.items():
            for table_name, info in tables.items():
                yield db_name, table_name, info
        return

    for schema in schemas:
        for table_name, info in schema.tables.items():
            yield schema.db_id, table_name, info


class TableIndexRetriever:
    """
    Index-only baseline over table metadata.

    Each table becomes a structured document containing table name, columns,
    key columns, join targets, and light domain tags. Retrieval is embedding
    first, then re-ranked with lexical overlap and join richness.
    """

    def __init__(self):
        self._embedder = get_embedder()
        self._index: faiss.IndexFlatIP | None = None
        self._documents: list[TableDocument] = []
        self._documents_by_fqn: dict[str, TableDocument] = {}
        self._keyword_map: dict[str, set[str]] = {}
        self._max_degree: int = 1

    def build(
        self,
        schemas: list[SchemaEntry] | dict[str, dict[str, dict]],
    ) -> None:
        documents: list[TableDocument] = []

        for db_name, table_name, info in _schema_iter(schemas):
            columns = _normalise_columns(info.get("columns", []))
            joins = _normalise_joins(info.get("joins", []))
            join_targets = [other for other, _ in joins]
            description = str(info.get("description", "") or "")
            key_columns = _key_columns(columns)
            sample_queries = [
                str(item) for item in info.get("sample_queries", []) if item
            ]
            tags = list(info.get("domain_tags", []) or [])
            derived_tags = _domain_tags(table_name, columns, description, join_targets)
            domain_tags = sorted(set(tags + derived_tags))

            text_parts = [
                f"table_name: {table_name}",
                f"database: {db_name}",
                f"description: {description or 'n/a'}",
                f"columns: {', '.join(columns) if columns else 'n/a'}",
                f"key_columns: {', '.join(key_columns) if key_columns else 'n/a'}",
                f"joins: {', '.join(join_targets) if join_targets else 'n/a'}",
                f"domain_tags: {', '.join(domain_tags) if domain_tags else 'n/a'}",
            ]
            if sample_queries:
                text_parts.append(
                    f"sample_queries: {' | '.join(sample_queries[:3])}"
                )

            document = TableDocument(
                db=db_name,
                table=table_name,
                text="\n".join(text_parts),
                description=description,
                columns=columns,
                key_columns=key_columns,
                join_targets=join_targets,
                domain_tags=domain_tags,
                sample_queries=sample_queries,
                metadata={
                    "joins": joins,
                },
            )
            documents.append(document)

        self._documents = documents
        self._documents_by_fqn = {
            document.fqn: document for document in self._documents
        }
        self._keyword_map = {
            document.fqn: set(
                _tokenise(
                    " ".join(
                        [
                            document.table,
                            document.description,
                            " ".join(document.columns),
                            " ".join(document.key_columns),
                            " ".join(document.join_targets),
                            " ".join(document.domain_tags),
                        ]
                    )
                )
            )
            for document in self._documents
        }
        self._max_degree = max(
            (len(document.join_targets) for document in self._documents),
            default=1,
        )

        if not self._documents:
            self._index = None
            logger.warning("TableIndexRetriever: no documents to index.")
            return

        logger.info(f"Building table index over {len(self._documents)} tables...")
        matrix = self._embedder.embed_batch(
            [document.text for document in self._documents]
        ).astype(np.float32)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def search(
        self,
        query: str,
        top_k: int = 8,
        db_name: str | None = None,
    ) -> list[TableCandidate]:
        if self._index is None or not self._documents:
            return []

        query_tokens = set(_tokenise(query))
        query_vec = self._embedder.embed(query).reshape(1, -1).astype(np.float32)

        search_k = min(len(self._documents), max(top_k * 4, top_k))
        distances, indices = self._index.search(query_vec, search_k)

        max_degree = max(len(document.join_targets) for document in self._documents) or 1
        candidates: list[TableCandidate] = []

        for raw_score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            document = self._documents[idx]
            if db_name and document.db != db_name:
                continue

            embedding_score = max(0.0, float(raw_score))
            keyword_set = self._keyword_map.get(document.fqn, set())
            lexical_score = (
                len(query_tokens & keyword_set) / len(query_tokens)
                if query_tokens else 0.0
            )
            join_score = min(1.0, len(document.join_targets) / max_degree)
            degree_score = min(1.0, len(document.key_columns) / 4.0)
            final_score = (
                0.72 * embedding_score
                + 0.20 * lexical_score
                + 0.05 * join_score
                + 0.03 * degree_score
            )

            reasons: list[str] = []
            if lexical_score > 0:
                reasons.append("query-token overlap")
            if join_score > 0.3:
                reasons.append("rich join neighborhood")
            if document.key_columns:
                reasons.append("explicit key columns")

            candidates.append(
                TableCandidate(
                    db=document.db,
                    table=document.table,
                    score=final_score,
                    embedding_score=embedding_score,
                    lexical_score=lexical_score,
                    join_score=join_score,
                    degree_score=degree_score,
                    reasons=reasons,
                    metadata={"document": document.to_dict()},
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def score_table(
        self,
        query: str,
        db_name: str,
        table_name: str,
        rank_hint: int | None = None,
    ) -> TableCandidate | None:
        document = self._documents_by_fqn.get(f"{db_name}.{table_name}")
        if document is None:
            return None

        query_tokens = set(_tokenise(query))
        query_vec = self._embedder.embed(query)
        doc_vec = self._embedder.embed(document.text)
        embedding_score = max(0.0, float(np.dot(query_vec, doc_vec)))
        keyword_set = self._keyword_map.get(document.fqn, set())
        lexical_score = (
            len(query_tokens & keyword_set) / len(query_tokens)
            if query_tokens else 0.0
        )
        join_score = min(1.0, len(document.join_targets) / max(1, self._max_degree))
        degree_score = min(1.0, len(document.key_columns) / 4.0)
        final_score = (
            0.72 * embedding_score
            + 0.20 * lexical_score
            + 0.05 * join_score
            + 0.03 * degree_score
        )

        reasons: list[str] = []
        if lexical_score > 0:
            reasons.append("query-token overlap")
        if join_score > 0.3:
            reasons.append("rich join neighborhood")
        if document.key_columns:
            reasons.append("explicit key columns")
        if rank_hint is not None:
            reasons.append(f"rank_hint={rank_hint}")

        return TableCandidate(
            db=document.db,
            table=document.table,
            score=final_score,
            embedding_score=embedding_score,
            lexical_score=lexical_score,
            join_score=join_score,
            degree_score=degree_score,
            reasons=reasons,
            metadata={"document": document.to_dict(), "rank_hint": rank_hint},
        )

    def score_tables(
        self,
        query: str,
        tables: list[tuple[str, str]],
    ) -> list[TableCandidate]:
        scored: list[TableCandidate] = []
        for rank, (db_name, table_name) in enumerate(tables, start=1):
            candidate = self.score_table(query, db_name, table_name, rank_hint=rank)
            if candidate is not None:
                scored.append(candidate)
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    def get_document(self, db_name: str, table_name: str) -> TableDocument | None:
        return self._documents_by_fqn.get(f"{db_name}.{table_name}")

    def documents(self) -> list[TableDocument]:
        return list(self._documents)

    @property
    def total_tables(self) -> int:
        return len(self._documents)
