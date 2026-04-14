from __future__ import annotations

from collections import defaultdict

from .execution_log import ExecutionLogger
from .models import (
    CandidateSubgraph,
    RetrievalBudget,
    RetrievalPlan,
    TableCandidate,
)
from .schema_graph import TableSchemaGraph
from .table_index import TableIndexRetriever


class IndexGraphPruningPipeline:
    """
    Runtime pipeline:

    query -> index top-k -> table-level schema graph pruning -> small subgraph
    """

    def __init__(
        self,
        table_index: TableIndexRetriever,
        schema_graph: TableSchemaGraph,
        execution_logger: ExecutionLogger | None = None,
    ):
        self._table_index = table_index
        self._schema_graph = schema_graph
        self._logger = execution_logger

    def retrieve(
        self,
        query: str,
        budget: RetrievalBudget | None = None,
        query_type: str = "",
        gold_tables: list[str] | None = None,
    ) -> RetrievalPlan:
        budget = budget or RetrievalBudget()
        index_candidates = self._table_index.search(query, top_k=budget.top_k)
        return self.retrieve_with_candidates(
            query=query,
            index_candidates=index_candidates,
            budget=budget,
            query_type=query_type,
            gold_tables=gold_tables,
            note="Index candidates pruned with the schema graph.",
        )

    def retrieve_index_only(
        self,
        query: str,
        budget: RetrievalBudget | None = None,
        query_type: str = "",
        gold_tables: list[str] | None = None,
    ) -> RetrievalPlan:
        budget = budget or RetrievalBudget()
        index_candidates = self._table_index.search(query, top_k=budget.top_k)
        return self.plan_from_subgraphs(
            query=query,
            index_candidates=index_candidates,
            candidate_subgraphs=[],
            budget=budget,
            query_type=query_type,
            gold_tables=gold_tables,
            note="Index only baseline. No graph pruning applied.",
        )

    def retrieve_with_candidates(
        self,
        query: str,
        index_candidates: list[TableCandidate],
        budget: RetrievalBudget | None = None,
        query_type: str = "",
        gold_tables: list[str] | None = None,
        note: str = "",
    ) -> RetrievalPlan:
        budget = budget or RetrievalBudget()
        candidate_subgraphs = self.build_candidate_subgraphs(
            index_candidates=index_candidates,
            budget=budget,
        )
        return self.plan_from_subgraphs(
            query=query,
            index_candidates=index_candidates,
            candidate_subgraphs=candidate_subgraphs,
            budget=budget,
            query_type=query_type,
            gold_tables=gold_tables,
            note=note or "Index candidates pruned with the schema graph.",
        )

    def plan_from_subgraphs(
        self,
        query: str,
        index_candidates: list[TableCandidate],
        candidate_subgraphs: list[CandidateSubgraph],
        budget: RetrievalBudget | None = None,
        query_type: str = "",
        gold_tables: list[str] | None = None,
        note: str = "",
    ) -> RetrievalPlan:
        budget = budget or RetrievalBudget()
        selected = candidate_subgraphs[0] if candidate_subgraphs else None
        inspection_order = (
            [candidate.fqn for candidate in index_candidates]
            if not candidate_subgraphs
            else self._inspection_order(selected)
        )

        notes: list[str] = []
        if not index_candidates:
            notes.append("No table candidates returned from index.")
        elif selected is None and candidate_subgraphs:
            notes.append("Candidate subgraphs were produced, but none survived scoring.")
        elif selected is None:
            notes.append(note or "Index only baseline. No graph pruning applied.")
        else:
            notes.append(
                note
                or "Inspect seed tables first, then expand only within the retained connected subgraph."
            )

        plan = RetrievalPlan(
            query=query,
            query_type=query_type,
            budget=budget,
            index_candidates=index_candidates,
            candidate_subgraphs=candidate_subgraphs,
            selected_subgraph=selected,
            inspection_order=inspection_order,
            notes=notes,
        )

        if self._logger is not None:
            plan.run_id = self._logger.log_plan(plan, gold_tables=gold_tables)

        return plan

    def build_candidate_subgraphs(
        self,
        index_candidates: list[TableCandidate],
        budget: RetrievalBudget,
    ) -> list[CandidateSubgraph]:
        by_db: dict[str, list[TableCandidate]] = defaultdict(list)
        for candidate in index_candidates:
            if candidate.score <= 0:
                continue
            by_db[candidate.db].append(candidate)

        subgraphs: list[CandidateSubgraph] = []
        for db_name, candidates in by_db.items():
            candidate_map = {candidate.table: candidate for candidate in candidates}
            ranked = sorted(
                candidates,
                key=lambda item: (
                    item.score + 0.02 * self._schema_graph.degree(item.db, item.table)
                ),
                reverse=True,
            )
            seed_tables = [
                candidate.table for candidate in ranked[: budget.max_seed_tables]
            ]

            retained_tables = set(seed_tables)
            bridge_tables: set[str] = set()
            dropped_tables: list[str] = []

            for candidate in ranked:
                keep = candidate.table in retained_tables
                if not keep:
                    for seed in seed_tables:
                        path = self._schema_graph.shortest_path(
                            db_name,
                            seed,
                            candidate.table,
                            max_hops=budget.max_hops,
                        )
                        if not path:
                            continue
                        retained_tables.update(path)
                        bridge_tables.update(
                            table
                            for table in path[1:-1]
                            if table not in candidate_map
                        )
                        keep = True
                        break

                if not keep:
                    dropped_tables.append(candidate.table)

            retained_tables = self._trim_component(
                db_name=db_name,
                retained_tables=retained_tables,
                seed_tables=seed_tables,
                budget=budget,
                candidate_map=candidate_map,
            )
            bridge_tables.intersection_update(retained_tables)

            if not retained_tables:
                continue

            retained_candidates = [
                candidate_map[table]
                for table in retained_tables
                if table in candidate_map
            ]
            retained_candidates.sort(key=lambda item: item.score, reverse=True)

            edges = self._schema_graph.subgraph_edges(db_name, retained_tables)
            connectivity = (
                len(edges) / max(1, len(retained_tables) - 1)
                if len(retained_tables) > 1
                else 0.0
            )
            avg_candidate_score = (
                sum(candidate.score for candidate in retained_candidates)
                / len(retained_candidates)
                if retained_candidates
                else 0.0
            )
            score = avg_candidate_score + 0.08 * min(connectivity, 1.0)

            subgraphs.append(
                CandidateSubgraph(
                    db=db_name,
                    score=score,
                    seed_tables=[table for table in seed_tables if table in retained_tables],
                    retained_tables=sorted(retained_tables),
                    dropped_tables=sorted(set(dropped_tables) - retained_tables),
                    bridge_tables=sorted(bridge_tables),
                    edges=edges,
                    retained_candidates=retained_candidates,
                    connectivity=connectivity,
                )
            )

        subgraphs.sort(key=lambda item: item.score, reverse=True)
        return subgraphs[: budget.max_subgraphs]

    def _trim_component(
        self,
        db_name: str,
        retained_tables: set[str],
        seed_tables: list[str],
        budget: RetrievalBudget,
        candidate_map: dict[str, TableCandidate],
    ) -> set[str]:
        connected = self._component_from_seeds(db_name, retained_tables, seed_tables)
        if len(connected) < budget.min_component_size and seed_tables:
            connected.update(seed_tables[:1])

        if len(connected) <= budget.max_tables_per_subgraph:
            return connected

        ordered = sorted(
            connected,
            key=lambda table: (
                table not in seed_tables,
                -candidate_map.get(
                    table,
                    TableCandidate(
                        db=db_name,
                        table=table,
                        score=0.0,
                        embedding_score=0.0,
                        lexical_score=0.0,
                        join_score=0.0,
                        degree_score=0.0,
                    ),
                ).score,
            ),
        )
        return set(ordered[: budget.max_tables_per_subgraph])

    def _component_from_seeds(
        self,
        db_name: str,
        retained_tables: set[str],
        seed_tables: list[str],
    ) -> set[str]:
        if not retained_tables:
            return set()
        if not seed_tables:
            return set(retained_tables)

        queue = list(seed_tables)
        seen = {table for table in seed_tables if table in retained_tables}

        while queue:
            table = queue.pop(0)
            for neighbor in self._schema_graph.neighbors(db_name, table):
                if neighbor not in retained_tables or neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)

        return seen or {seed_tables[0]}

    @staticmethod
    def _inspection_order(selected: CandidateSubgraph | None) -> list[str]:
        if selected is None:
            return []

        ordered: list[str] = [
            f"{selected.db}.{table}" for table in selected.seed_tables
        ]
        for candidate in selected.retained_candidates:
            fqn = f"{candidate.db}.{candidate.table}"
            if fqn not in ordered:
                ordered.append(fqn)
        for table in selected.retained_tables:
            fqn = f"{selected.db}.{table}"
            if fqn not in ordered:
                ordered.append(fqn)
        return ordered
