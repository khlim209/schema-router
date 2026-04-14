from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QueryConstraint:
    kind: str
    value: str
    matched_text: str = ""


@dataclass
class QueryUnderstanding:
    query: str
    intent: str
    entities: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    constraints: list[QueryConstraint] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    matched_schema_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PathBudget:
    max_hops: int = 3
    max_tables: int = 5
    max_entrypoints: int = 3
    max_candidate_paths: int = 3
    max_mcp_calls: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannedTable:
    db: str
    table: str
    score: float
    matched_columns: list[str] = field(default_factory=list)
    supporting_facts: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePath:
    path_id: str
    db: str
    tables: list[str]
    score: float
    coverage: float
    estimated_cost: int
    covered_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanningStep:
    step: int
    action: str
    targets: list[str]
    purpose: str
    stop_if: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchemaTraversalPlan:
    query: str
    understanding: QueryUnderstanding
    budget: PathBudget
    entrypoints: list[PlannedTable] = field(default_factory=list)
    candidate_paths: list[CandidatePath] = field(default_factory=list)
    execution_steps: list[PlanningStep] = field(default_factory=list)
    stop_condition: str = ""
    answer_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.understanding.intent,
            "entities": self.understanding.entities,
            "required_facts": self.understanding.facts,
            "constraints": [c.__dict__ for c in self.understanding.constraints],
            "tokens": self.understanding.tokens,
            "matched_schema_terms": self.understanding.matched_schema_terms,
            "budget": self.budget.to_dict(),
            "entrypoints": [table.to_dict() for table in self.entrypoints],
            "candidate_paths": [path.to_dict() for path in self.candidate_paths],
            "execution_steps": [step.to_dict() for step in self.execution_steps],
            "stop_condition": self.stop_condition,
            "answer_confidence": round(self.answer_confidence, 4),
            "metadata": self.metadata,
        }
