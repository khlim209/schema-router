from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from typing import Iterable

from embedding.embedder import get_embedder
from graph_rag.indexer import AccessRecord

from .models import GraphEdge


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


class TableSchemaGraph:
    """
    Table-level schema graph with join edges and query-log co-access edges.
    """

    def __init__(self):
        self._adjacency: dict[str, dict[str, dict]] = defaultdict(dict)
        self._query_tables: dict[str, set[str]] = defaultdict(set)
        self._embedder = get_embedder()

    @staticmethod
    def _fqn(db_name: str, table_name: str) -> str:
        return f"{db_name.lower()}.{table_name.lower()}"

    def rebuild(self, schema_registry: dict[str, dict[str, dict]]) -> None:
        self._adjacency = defaultdict(dict)
        for db_name, tables in schema_registry.items():
            for table_name, info in tables.items():
                source = self._fqn(db_name, table_name)
                for other_table, via in _normalise_joins(info.get("joins", [])):
                    target = self._fqn(db_name, other_table)
                    self._add_edge(source, target, relation="join", weight=1.0, via=via)

    def _add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float,
        via: str = "",
    ) -> None:
        if source == target:
            return

        current = self._adjacency[source].get(target)
        if current is None:
            self._adjacency[source][target] = {
                "weight": weight,
                "relation": relation,
                "via": via,
            }
        else:
            current["weight"] += weight
            if relation == "join":
                current["relation"] = "join"
                current["via"] = via or current.get("via", "")

        reverse = self._adjacency[target].get(source)
        if reverse is None:
            self._adjacency[target][source] = {
                "weight": weight,
                "relation": relation,
                "via": via,
            }
        else:
            reverse["weight"] += weight
            if relation == "join":
                reverse["relation"] = "join"
                reverse["via"] = via or reverse.get("via", "")

    def ingest_access_log(self, records: list[AccessRecord]) -> None:
        grouped: dict[str, set[str]] = defaultdict(set)
        for record in records:
            query_id = self._embedder.text_id(record.query_text)
            grouped[query_id].add(self._fqn(record.db_name, record.table_name))

        for query_id, fqns in grouped.items():
            self._query_tables[query_id].update(fqns)

        for fqns in grouped.values():
            for source, target in combinations(sorted(fqns), 2):
                if source.split(".", 1)[0] != target.split(".", 1)[0]:
                    continue
                self._add_edge(
                    source,
                    target,
                    relation="co_access",
                    weight=0.35,
                )

    def record_access(
        self,
        query_text: str,
        db_name: str,
        table_name: str,
        count: int = 1,
    ) -> None:
        query_id = self._embedder.text_id(query_text)
        current = self._fqn(db_name, table_name)
        seen = self._query_tables.setdefault(query_id, set())

        for other in seen:
            if other == current or other.split(".", 1)[0] != db_name.lower():
                continue
            self._add_edge(
                current,
                other,
                relation="co_access",
                weight=max(0.15, 0.15 * count),
            )
        seen.add(current)

    def neighbors(self, db_name: str, table_name: str) -> list[str]:
        node = self._fqn(db_name, table_name)
        return [
            neighbor.split(".", 1)[1]
            for neighbor in self._adjacency.get(node, {})
            if neighbor.startswith(f"{db_name.lower()}.")
        ]

    def degree(self, db_name: str, table_name: str) -> float:
        node = self._fqn(db_name, table_name)
        return float(sum(item["weight"] for item in self._adjacency.get(node, {}).values()))

    def shortest_path(
        self,
        db_name: str,
        start_table: str,
        goal_table: str,
        max_hops: int = 2,
    ) -> list[str]:
        start = self._fqn(db_name, start_table)
        goal = self._fqn(db_name, goal_table)
        if start == goal:
            return [start_table]

        queue: deque[list[str]] = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            if len(path) - 1 > max_hops:
                continue

            for neighbor in self._adjacency.get(path[-1], {}):
                if not neighbor.startswith(f"{db_name.lower()}."):
                    continue
                if neighbor == goal:
                    return [node.split(".", 1)[1] for node in (path + [neighbor])]
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(path + [neighbor])

        return []

    def expand_from_seeds(
        self,
        db_name: str,
        seed_tables: list[str],
        max_hops: int = 2,
    ) -> set[str]:
        prefix = f"{db_name.lower()}."
        queue: deque[tuple[str, int]] = deque(
            (self._fqn(db_name, table), 0) for table in seed_tables
        )
        visited = {
            self._fqn(db_name, table)
            for table in seed_tables
        }

        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self._adjacency.get(node, {}):
                if not neighbor.startswith(prefix) or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

        return {node.split(".", 1)[1] for node in visited}

    def subgraph_edges(self, db_name: str, tables: set[str]) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        seen_pairs: set[tuple[str, str]] = set()
        prefix = f"{db_name.lower()}."

        for table in tables:
            source = f"{prefix}{table}"
            for target, payload in self._adjacency.get(source, {}).items():
                if not target.startswith(prefix):
                    continue
                target_table = target.split(".", 1)[1]
                if target_table not in tables:
                    continue
                pair = tuple(sorted((table, target_table)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edges.append(
                    GraphEdge(
                        source=f"{db_name}.{table}",
                        target=f"{db_name}.{target_table}",
                        relation=str(payload.get("relation", "join")),
                        weight=float(payload.get("weight", 1.0)),
                        via=str(payload.get("via", "")),
                    )
                )

        return sorted(edges, key=lambda item: (item.source, item.target))
