from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalBudget:
    """Controls how large the candidate set and pruned subgraph may become."""

    top_k: int = 8
    max_hops: int = 2
    max_seed_tables: int = 3
    min_component_size: int = 2
    max_subgraphs: int = 2
    max_tables_per_subgraph: int = 8

    def to_dict(self) -> dict[str, int]:
        return {
            "top_k": self.top_k,
            "max_hops": self.max_hops,
            "max_seed_tables": self.max_seed_tables,
            "min_component_size": self.min_component_size,
            "max_subgraphs": self.max_subgraphs,
            "max_tables_per_subgraph": self.max_tables_per_subgraph,
        }


@dataclass
class TableDocument:
    """Structured document used by the table index baseline."""

    db: str
    table: str
    text: str
    description: str = ""
    columns: list[str] = field(default_factory=list)
    key_columns: list[str] = field(default_factory=list)
    join_targets: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    sample_queries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fqn(self) -> str:
        return f"{self.db}.{self.table}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "db": self.db,
            "table": self.table,
            "text": self.text,
            "description": self.description,
            "columns": list(self.columns),
            "key_columns": list(self.key_columns),
            "join_targets": list(self.join_targets),
            "domain_tags": list(self.domain_tags),
            "sample_queries": list(self.sample_queries),
            "metadata": dict(self.metadata),
        }


@dataclass
class TableCandidate:
    """Ranked output from the index baseline."""

    db: str
    table: str
    score: float
    embedding_score: float
    lexical_score: float
    join_score: float
    degree_score: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fqn(self) -> str:
        return f"{self.db}.{self.table}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "db": self.db,
            "table": self.table,
            "score": round(self.score, 4),
            "embedding_score": round(self.embedding_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "join_score": round(self.join_score, 4),
            "degree_score": round(self.degree_score, 4),
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


@dataclass
class GraphEdge:
    """Table-level edge in the schema graph."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    via: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 4),
            "via": self.via,
        }


@dataclass
class CandidateSubgraph:
    """Pruned local schema graph that MCP should inspect."""

    db: str
    score: float
    seed_tables: list[str]
    retained_tables: list[str]
    dropped_tables: list[str]
    bridge_tables: list[str] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    retained_candidates: list[TableCandidate] = field(default_factory=list)
    connectivity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "db": self.db,
            "score": round(self.score, 4),
            "seed_tables": list(self.seed_tables),
            "retained_tables": list(self.retained_tables),
            "dropped_tables": list(self.dropped_tables),
            "bridge_tables": list(self.bridge_tables),
            "edges": [edge.to_dict() for edge in self.edges],
            "retained_candidates": [
                candidate.to_dict() for candidate in self.retained_candidates
            ],
            "connectivity": round(self.connectivity, 4),
        }


@dataclass
class RetrievalPlan:
    """Output of the index + graph pruning pipeline."""

    query: str
    budget: RetrievalBudget
    index_candidates: list[TableCandidate]
    candidate_subgraphs: list[CandidateSubgraph]
    selected_subgraph: CandidateSubgraph | None
    inspection_order: list[str]
    notes: list[str] = field(default_factory=list)
    run_id: str = ""
    query_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "query_type": self.query_type,
            "budget": self.budget.to_dict(),
            "index_candidates": [
                candidate.to_dict() for candidate in self.index_candidates
            ],
            "candidate_subgraphs": [
                subgraph.to_dict() for subgraph in self.candidate_subgraphs
            ],
            "selected_subgraph": (
                self.selected_subgraph.to_dict()
                if self.selected_subgraph is not None
                else None
            ),
            "inspection_order": list(self.inspection_order),
            "notes": list(self.notes),
        }


@dataclass
class ExecutionFeedback:
    """Optional runtime feedback after MCP inspects the suggested subgraph."""

    run_id: str
    query: str
    executed_tables: list[str]
    contributing_tables: list[str]
    unnecessary_tables: list[str] = field(default_factory=list)
    success: bool = True
    latency_ms: float = 0.0
    final_answer: str = ""
    notes: str = ""
    gold_tables: list[str] = field(default_factory=list)
    query_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "query_type": self.query_type,
            "executed_tables": list(self.executed_tables),
            "contributing_tables": list(self.contributing_tables),
            "unnecessary_tables": list(self.unnecessary_tables),
            "success": self.success,
            "latency_ms": round(self.latency_ms, 4),
            "final_answer": self.final_answer,
            "notes": self.notes,
            "gold_tables": list(self.gold_tables),
        }
