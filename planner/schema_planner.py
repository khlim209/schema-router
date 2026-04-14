from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Any

from loguru import logger

from planner.models import (
    CandidatePath,
    PathBudget,
    PlannedTable,
    PlanningStep,
    QueryUnderstanding,
    SchemaTraversalPlan,
)
from planner.query_understanding_v2 import QueryDecomposer


def _split_terms(text: str) -> set[str]:
    tokens = text.replace("_", " ").replace("-", " ").lower().split()
    return {token for token in tokens if token}


class MultiHopSchemaPlanner:
    """
    Produce a budget-aware schema traversal plan instead of a single table answer.
    """

    def __init__(
        self,
        schema_registry: dict[str, dict[str, dict]] | None = None,
        graph_retriever: Any | None = None,
        decomposer: QueryDecomposer | None = None,
    ):
        self._schema_registry = schema_registry or {}
        self._graph_retriever = graph_retriever
        self._decomposer = decomposer or QueryDecomposer()

    def update_registry(self, schema_registry: dict[str, dict[str, dict]]) -> None:
        self._schema_registry = schema_registry

    def plan(
        self,
        query: str,
        budget: PathBudget | None = None,
    ) -> SchemaTraversalPlan:
        budget = budget or PathBudget()
        understanding = self._decomposer.decompose(query, self._schema_registry)
        table_scores = self._score_tables(query, understanding)
        entrypoints = table_scores[:budget.max_entrypoints]
        candidate_paths = self._build_candidate_paths(entrypoints, understanding, budget)
        steps = self._build_execution_steps(candidate_paths, understanding, budget)
        confidence = candidate_paths[0].score if candidate_paths else 0.0

        return SchemaTraversalPlan(
            query=query,
            understanding=understanding,
            budget=budget,
            entrypoints=entrypoints,
            candidate_paths=candidate_paths,
            execution_steps=steps,
            stop_condition=(
                "Stop when fact coverage is high enough, or when the planner hits "
                f"{budget.max_tables} tables / {budget.max_hops} hops / {budget.max_mcp_calls} MCP calls."
            ),
            answer_confidence=confidence,
            metadata={
                "n_ranked_tables": len(table_scores),
                "candidate_dbs": sorted({table.db for table in entrypoints}),
            },
        )

    def _score_tables(
        self,
        query: str,
        understanding: QueryUnderstanding,
    ) -> list[PlannedTable]:
        prior_scores = self._historical_priors(query)
        ranked: list[PlannedTable] = []
        query_terms = set(understanding.tokens)
        entity_terms = set(understanding.entities)
        fact_terms = set(understanding.facts)

        for db_name, tables in self._schema_registry.items():
            for table_name, info in tables.items():
                table_terms = self._table_terms(db_name, table_name, info)
                matched_columns = self._matched_columns(info, query_terms)
                supporting_facts = self._supporting_facts(info, table_terms, fact_terms, query_terms)

                lexical_score = self._overlap_score(query_terms, table_terms)
                entity_score = self._overlap_score(entity_terms, table_terms)
                fact_score = self._overlap_score(fact_terms, table_terms.union(set(supporting_facts)))
                column_score = min(1.0, len(matched_columns) / 3.0)
                join_score = min(1.0, len(info.get("joins", [])) / 3.0)
                prior_score = prior_scores.get((db_name, table_name), 0.0)

                total = (
                    0.30 * lexical_score
                    + 0.20 * entity_score
                    + 0.20 * fact_score
                    + 0.10 * column_score
                    + 0.10 * join_score
                    + 0.10 * prior_score
                )
                if total <= 0:
                    continue

                reasons: list[str] = []
                if lexical_score > 0:
                    reasons.append("query terms overlap with table or column names")
                if matched_columns:
                    reasons.append(f"matched columns: {', '.join(matched_columns[:3])}")
                if supporting_facts:
                    reasons.append(f"supports facts: {', '.join(supporting_facts[:3])}")
                if prior_score > 0:
                    reasons.append("historical GraphRAG evidence boosts this table")

                ranked.append(
                    PlannedTable(
                        db=db_name,
                        table=table_name,
                        score=round(total, 4),
                        matched_columns=matched_columns[:5],
                        supporting_facts=supporting_facts[:5],
                        reasons=reasons[:4],
                    )
                )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _build_candidate_paths(
        self,
        entrypoints: list[PlannedTable],
        understanding: QueryUnderstanding,
        budget: PathBudget,
    ) -> list[CandidatePath]:
        facts = understanding.facts or understanding.matched_schema_terms or understanding.tokens[:4]
        paths: list[CandidatePath] = []

        for index, entrypoint in enumerate(entrypoints, start=1):
            tables = self._expand_path(entrypoint, facts, budget)
            coverage = self._fact_coverage(entrypoint.db, tables, facts)
            covered = self._covered_facts(entrypoint.db, tables, facts)
            missing = [fact for fact in facts if fact not in covered]
            avg_score = mean([entrypoint.score] + [self._table_score(entrypoint.db, table) for table in tables[1:]]) if tables else 0.0
            cost = len(tables)
            score = max(0.0, 0.55 * coverage + 0.35 * avg_score - 0.05 * max(0, cost - 1))

            rationale = [f"start from {entrypoint.db}.{entrypoint.table}"]
            if len(tables) > 1:
                rationale.append("expand only across join-connected neighbors with new fact coverage")
            if covered:
                rationale.append(f"covers facts: {', '.join(covered[:4])}")

            paths.append(
                CandidatePath(
                    path_id=f"path_{index}",
                    db=entrypoint.db,
                    tables=tables,
                    score=round(score, 4),
                    coverage=round(coverage, 4),
                    estimated_cost=cost,
                    covered_facts=covered,
                    missing_facts=missing,
                    rationale=rationale,
                )
            )

        deduped: list[CandidatePath] = []
        seen: set[tuple[str, ...]] = set()
        for path in sorted(paths, key=lambda item: item.score, reverse=True):
            key = tuple([path.db] + path.tables)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
            if len(deduped) >= budget.max_candidate_paths:
                break
        return deduped

    def _build_execution_steps(
        self,
        candidate_paths: list[CandidatePath],
        understanding: QueryUnderstanding,
        budget: PathBudget,
    ) -> list[PlanningStep]:
        steps: list[PlanningStep] = []
        if not candidate_paths:
            steps.append(
                PlanningStep(
                    step=1,
                    action="fallback",
                    targets=[],
                    purpose="No strong schema path was found. Fall back to the existing single-table router.",
                )
            )
            return steps

        primary = candidate_paths[0]
        steps.append(
            PlanningStep(
                step=1,
                action="probe_entrypoints",
                targets=[f"{primary.db}.{table}" for table in primary.tables[:2]],
                purpose="Fetch lightweight metadata, aggregates, or samples from the highest-priority entry tables first.",
                stop_if="Stop if the first probes already cover the key facts.",
            )
        )

        if len(primary.tables) > 2:
            steps.append(
                PlanningStep(
                    step=2,
                    action="expand_join_path",
                    targets=[f"{primary.db}.{table}" for table in primary.tables[2:]],
                    purpose="Expand only if uncovered facts remain after the first probes.",
                    stop_if="Skip this step if answer confidence is already sufficient.",
                )
            )

        if len(candidate_paths) > 1:
            alt = candidate_paths[1]
            steps.append(
                PlanningStep(
                    step=len(steps) + 1,
                    action="try_alternative_branch",
                    targets=[f"{alt.db}.{table}" for table in alt.tables],
                    purpose="Use the secondary path only when the primary branch leaves important facts uncovered.",
                    stop_if=f"Do not exceed {budget.max_mcp_calls} MCP calls overall.",
                )
            )

        steps.append(
            PlanningStep(
                step=len(steps) + 1,
                action="aggregate_evidence",
                targets=understanding.facts[:6],
                purpose="Merge the collected facts and generate the final answer or SQL.",
                stop_if="Stop once all required facts are covered or the evidence is stable enough to answer.",
            )
        )
        return steps

    def _expand_path(
        self,
        entrypoint: PlannedTable,
        facts: list[str],
        budget: PathBudget,
    ) -> list[str]:
        path = [entrypoint.table]
        visited = {entrypoint.table}
        db_tables = self._schema_registry.get(entrypoint.db, {})
        remaining_facts = set(facts) - set(entrypoint.supporting_facts)

        while len(path) < budget.max_tables and remaining_facts:
            candidate = self._best_neighbor(entrypoint.db, path, visited, remaining_facts)
            if candidate is None:
                break
            path.append(candidate)
            visited.add(candidate)
            covered = self._covered_facts(entrypoint.db, path, list(remaining_facts))
            remaining_facts -= set(covered)

            if len(path) - 1 >= budget.max_hops:
                break
            if candidate not in db_tables:
                break
        return path

    def _best_neighbor(
        self,
        db_name: str,
        current_path: list[str],
        visited: set[str],
        remaining_facts: set[str],
    ) -> str | None:
        best_table: str | None = None
        best_score = 0.0

        for table_name in current_path:
            for neighbor in self._neighbors(db_name, table_name):
                if neighbor in visited:
                    continue
                table_terms = self._table_terms(db_name, neighbor, self._schema_registry[db_name][neighbor])
                newly_covered = len([fact for fact in remaining_facts if fact in table_terms])
                score = 0.65 * min(1.0, newly_covered) + 0.35 * self._table_score(db_name, neighbor)
                if score > best_score:
                    best_score = score
                    best_table = neighbor
        return best_table

    def _table_terms(self, db_name: str, table_name: str, info: dict) -> set[str]:
        terms = set()
        terms.update(_split_terms(db_name))
        terms.update(_split_terms(table_name))
        terms.update(_split_terms(info.get("description", "")))
        for column_name, _ in info.get("columns", []):
            terms.update(_split_terms(column_name))
        for joined_table, _ in info.get("joins", []):
            terms.update(_split_terms(joined_table))
        return terms

    def _matched_columns(self, info: dict, query_terms: set[str]) -> list[str]:
        matched: list[str] = []
        for column_name, _ in info.get("columns", []):
            column_terms = _split_terms(column_name)
            if query_terms.intersection(column_terms):
                matched.append(column_name)
        return matched

    def _supporting_facts(
        self,
        info: dict,
        table_terms: set[str],
        fact_terms: set[str],
        query_terms: set[str],
    ) -> list[str]:
        supporting: list[str] = []
        for fact in fact_terms:
            fact_terms_set = _split_terms(fact)
            if fact in table_terms or fact_terms_set.intersection(table_terms):
                supporting.append(fact)
                continue
            for column_name, _ in info.get("columns", []):
                column_terms = _split_terms(column_name)
                if fact_terms_set.intersection(column_terms):
                    supporting.append(fact)
                    break

        if not supporting:
            fallback = sorted(query_terms.intersection(table_terms))
            supporting.extend(fallback[:3])
        return supporting

    def _overlap_score(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = left.intersection(right)
        return len(overlap) / max(1, len(left))

    def _neighbors(self, db_name: str, table_name: str) -> list[str]:
        tables = self._schema_registry.get(db_name, {})
        info = tables.get(table_name, {})
        neighbors = [other for other, _ in info.get("joins", []) if other in tables]
        reverse_neighbors = [
            other_name
            for other_name, other_info in tables.items()
            if table_name in {joined for joined, _ in other_info.get("joins", [])}
        ]
        ordered: list[str] = []
        for neighbor in neighbors + reverse_neighbors:
            if neighbor not in ordered:
                ordered.append(neighbor)
        return ordered

    def _shortest_path(self, db_name: str, start: str, goal: str, max_hops: int) -> list[str]:
        if start == goal:
            return [start]

        queue: deque[list[str]] = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            node = path[-1]
            if len(path) - 1 > max_hops:
                continue
            for neighbor in self._neighbors(db_name, node):
                if neighbor == goal:
                    return path + [neighbor]
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(path + [neighbor])
        return [start]

    def _covered_facts(self, db_name: str, tables: list[str], facts: list[str]) -> list[str]:
        covered: list[str] = []
        for fact in facts:
            fact_terms = _split_terms(fact)
            for table in tables:
                info = self._schema_registry.get(db_name, {}).get(table, {})
                table_terms = self._table_terms(db_name, table, info)
                if fact in table_terms or fact_terms.intersection(table_terms):
                    covered.append(fact)
                    break
        return covered

    def _fact_coverage(self, db_name: str, tables: list[str], facts: list[str]) -> float:
        if not facts:
            return 0.0
        covered = self._covered_facts(db_name, tables, facts)
        return len(covered) / len(facts)

    def _table_score(self, db_name: str, table_name: str) -> float:
        info = self._schema_registry.get(db_name, {}).get(table_name, {})
        base = 0.1
        base += min(0.3, len(info.get("columns", [])) / 20.0)
        base += min(0.3, len(info.get("joins", [])) / 4.0)
        return round(base, 4)

    def _historical_priors(self, query: str) -> dict[tuple[str, str], float]:
        if self._graph_retriever is None:
            return {}
        try:
            results = self._graph_retriever.route(query, top_n=10)
        except Exception as exc:  # pragma: no cover - defensive path for missing Neo4j/history
            logger.debug(f"Planner prior lookup skipped: {exc}")
            return {}

        if not results:
            return {}

        max_score = max(result.score for result in results) or 1.0
        return {
            (result.db, result.table): round(result.score / max_score, 4)
            for result in results
        }
