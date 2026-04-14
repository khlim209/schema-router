from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn

from bench_datasets.base import BenchmarkSample

from .models import CandidateSubgraph, RetrievalBudget, TableCandidate
from .pipeline import IndexGraphPruningPipeline
from .schema_graph import TableSchemaGraph
from .table_index import TableIndexRetriever


@dataclass
class GraphTrainingStats:
    n_graphs: int
    avg_loss: float
    avg_positive_rate: float


@dataclass
class GraphExample:
    db: str
    query: str
    node_names: list[str]
    seed_tables: list[str]
    x: np.ndarray
    adj: np.ndarray
    labels: np.ndarray


class _SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        a_hat = adj + identity
        degree = torch.sum(a_hat, dim=1)
        d_inv_sqrt = torch.diag(torch.pow(torch.clamp(degree, min=1.0), -0.5))
        a_norm = d_inv_sqrt @ a_hat @ d_inv_sqrt

        h = torch.relu(a_norm @ self.lin1(x))
        h = torch.relu(a_norm @ self.lin2(h))
        logits = self.out(h).squeeze(-1)
        return logits


class LocalGraphNodeScorer:
    """
    Lightweight trainable node scorer over local schema graphs.
    """

    def __init__(
        self,
        model: _SimpleGCN | None = None,
        threshold: float = 0.45,
        max_hops: int = 2,
    ):
        self._model = model or _SimpleGCN(in_dim=8, hidden_dim=32)
        self._threshold = threshold
        self._max_hops = max_hops

    @staticmethod
    def _distance_to_seeds(
        schema_graph: TableSchemaGraph,
        db_name: str,
        table_name: str,
        seed_tables: list[str],
        max_hops: int,
    ) -> int:
        if table_name in seed_tables:
            return 0
        best = max_hops + 1
        for seed in seed_tables:
            path = schema_graph.shortest_path(
                db_name,
                seed,
                table_name,
                max_hops=max_hops,
            )
            if path:
                best = min(best, len(path) - 1)
        return best

    def _node_feature(
        self,
        query: str,
        db_name: str,
        table_name: str,
        table_index: TableIndexRetriever,
        schema_graph: TableSchemaGraph,
        seed_tables: list[str],
        candidate_tables: set[str],
    ) -> list[float]:
        candidate = table_index.score_table(query, db_name, table_name)
        if candidate is None:
            candidate = TableCandidate(
                db=db_name,
                table=table_name,
                score=0.0,
                embedding_score=0.0,
                lexical_score=0.0,
                join_score=0.0,
                degree_score=0.0,
            )

        distance = self._distance_to_seeds(
            schema_graph,
            db_name,
            table_name,
            seed_tables=seed_tables,
            max_hops=self._max_hops,
        )
        return [
            candidate.score,
            candidate.embedding_score,
            candidate.lexical_score,
            candidate.join_score,
            candidate.degree_score,
            float(table_name in seed_tables),
            float(table_name in candidate_tables),
            distance / max(1, self._max_hops + 1),
        ]

    @staticmethod
    def _adjacency_matrix(
        node_names: list[str],
        subgraph: CandidateSubgraph,
    ) -> np.ndarray:
        size = len(node_names)
        matrix = np.zeros((size, size), dtype=np.float32)
        idx = {name: i for i, name in enumerate(node_names)}
        for edge in subgraph.edges:
            source = edge.source.split(".", 1)[1]
            target = edge.target.split(".", 1)[1]
            if source not in idx or target not in idx:
                continue
            i, j = idx[source], idx[target]
            matrix[i, j] = max(matrix[i, j], edge.weight)
            matrix[j, i] = max(matrix[j, i], edge.weight)
        return matrix

    def _graph_example_from_subgraph(
        self,
        query: str,
        subgraph: CandidateSubgraph,
        gold_tables: list[str],
        table_index: TableIndexRetriever,
        schema_graph: TableSchemaGraph,
    ) -> GraphExample | None:
        node_names = list(subgraph.retained_tables)
        if not node_names:
            return None

        candidate_tables = {candidate.table for candidate in subgraph.retained_candidates}
        x = np.asarray(
            [
                self._node_feature(
                    query,
                    subgraph.db,
                    table_name,
                    table_index=table_index,
                    schema_graph=schema_graph,
                    seed_tables=subgraph.seed_tables,
                    candidate_tables=candidate_tables,
                )
                for table_name in node_names
            ],
            dtype=np.float32,
        )
        labels = np.asarray(
            [1.0 if table_name in gold_tables else 0.0 for table_name in node_names],
            dtype=np.float32,
        )
        if float(labels.sum()) == 0.0:
            return None

        return GraphExample(
            db=subgraph.db,
            query=query,
            node_names=node_names,
            seed_tables=list(subgraph.seed_tables),
            x=x,
            adj=self._adjacency_matrix(node_names, subgraph),
            labels=labels,
        )

    def build_training_examples(
        self,
        samples: list[BenchmarkSample],
        pipeline: IndexGraphPruningPipeline,
        table_index: TableIndexRetriever,
        schema_graph: TableSchemaGraph,
        budget: RetrievalBudget,
        include_gold: bool = True,
    ) -> list[GraphExample]:
        examples: list[GraphExample] = []

        for sample in samples:
            candidates = table_index.search(sample.question, top_k=budget.top_k)
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

            subgraphs = pipeline.build_candidate_subgraphs(candidates, budget=budget)
            target = next(
                (subgraph for subgraph in subgraphs if subgraph.db == sample.db_id),
                None,
            )

            if target is None:
                expanded = schema_graph.expand_from_seeds(
                    sample.db_id,
                    seed_tables=list(sample.used_tables),
                    max_hops=budget.max_hops,
                )
                retained = set(sample.used_tables) | expanded
                retained = set(list(sorted(retained))[: budget.max_tables_per_subgraph])
                retained_candidates = [
                    table_index.score_table(sample.question, sample.db_id, table_name)
                    for table_name in retained
                ]
                retained_candidates = [
                    candidate for candidate in retained_candidates if candidate is not None
                ]
                retained_candidates.sort(key=lambda item: item.score, reverse=True)
                target = CandidateSubgraph(
                    db=sample.db_id,
                    score=0.0,
                    seed_tables=list(sample.used_tables[: budget.max_seed_tables]),
                    retained_tables=sorted(retained),
                    dropped_tables=[],
                    bridge_tables=[],
                    edges=schema_graph.subgraph_edges(sample.db_id, retained),
                    retained_candidates=retained_candidates,
                    connectivity=0.0,
                )

            example = self._graph_example_from_subgraph(
                sample.question,
                target,
                gold_tables=sample.used_tables,
                table_index=table_index,
                schema_graph=schema_graph,
            )
            if example is not None:
                examples.append(example)

        return examples

    def fit(
        self,
        samples: list[BenchmarkSample],
        pipeline: IndexGraphPruningPipeline,
        table_index: TableIndexRetriever,
        schema_graph: TableSchemaGraph,
        budget: RetrievalBudget,
        epochs: int = 10,
        learning_rate: float = 1e-2,
    ) -> GraphTrainingStats:
        graphs = self.build_training_examples(
            samples,
            pipeline=pipeline,
            table_index=table_index,
            schema_graph=schema_graph,
            budget=budget,
            include_gold=True,
        )
        if not graphs:
            raise ValueError("No graph training examples were generated.")

        device = torch.device("cpu")
        self._model.to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        total_pos = sum(float(graph.labels.sum()) for graph in graphs)
        total_nodes = sum(len(graph.labels) for graph in graphs)
        total_neg = max(1.0, total_nodes - total_pos)
        pos_weight = torch.tensor(total_neg / max(1.0, total_pos), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        avg_loss = 0.0
        for epoch in range(epochs):
            losses: list[float] = []
            self._model.train()
            for graph in graphs:
                x = torch.tensor(graph.x, dtype=torch.float32, device=device)
                adj = torch.tensor(graph.adj, dtype=torch.float32, device=device)
                labels = torch.tensor(graph.labels, dtype=torch.float32, device=device)

                optimizer.zero_grad()
                logits = self._model(x, adj)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            avg_loss = float(np.mean(losses))
            logger.info(
                "GNN node scorer epoch {}/{} - loss={:.4f}",
                epoch + 1,
                epochs,
                avg_loss,
            )

        stats = GraphTrainingStats(
            n_graphs=len(graphs),
            avg_loss=avg_loss,
            avg_positive_rate=total_pos / max(1.0, total_nodes),
        )
        return stats

    def score_subgraphs(
        self,
        query: str,
        candidate_subgraphs: list[CandidateSubgraph],
        table_index: TableIndexRetriever,
        schema_graph: TableSchemaGraph,
        budget: RetrievalBudget,
    ) -> list[CandidateSubgraph]:
        rescored: list[CandidateSubgraph] = []
        self._model.eval()

        for subgraph in candidate_subgraphs:
            node_names = list(subgraph.retained_tables)
            if not node_names:
                continue
            candidate_tables = {candidate.table for candidate in subgraph.retained_candidates}
            x = np.asarray(
                [
                    self._node_feature(
                        query,
                        subgraph.db,
                        table_name,
                        table_index=table_index,
                        schema_graph=schema_graph,
                        seed_tables=subgraph.seed_tables,
                        candidate_tables=candidate_tables,
                    )
                    for table_name in node_names
                ],
                dtype=np.float32,
            )
            adj = self._adjacency_matrix(node_names, subgraph)

            with torch.no_grad():
                logits = self._model(
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(adj, dtype=torch.float32),
                )
                probs = torch.sigmoid(logits).cpu().numpy()

            scored_nodes = sorted(
                zip(node_names, probs.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
            selected_tables = [
                table_name for table_name, prob in scored_nodes
                if prob >= self._threshold
            ]
            if not selected_tables:
                selected_tables = [scored_nodes[0][0]]

            selected_tables = selected_tables[: budget.max_tables_per_subgraph]
            selected_set = set(selected_tables)
            filtered_edges = schema_graph.subgraph_edges(subgraph.db, selected_set)

            retained_candidates: list[TableCandidate] = []
            original_map = {candidate.table: candidate for candidate in subgraph.retained_candidates}
            prob_map = dict(scored_nodes)
            for table_name in selected_tables:
                base = original_map.get(table_name) or table_index.score_table(
                    query,
                    subgraph.db,
                    table_name,
                )
                if base is None:
                    continue
                retained_candidates.append(
                    TableCandidate(
                        db=base.db,
                        table=base.table,
                        score=0.75 * float(prob_map.get(table_name, 0.0)) + 0.25 * base.score,
                        embedding_score=base.embedding_score,
                        lexical_score=base.lexical_score,
                        join_score=base.join_score,
                        degree_score=base.degree_score,
                        reasons=list(base.reasons) + ["gnn-node-scored"],
                        metadata={**base.metadata, "gnn_probability": float(prob_map.get(table_name, 0.0))},
                    )
                )
            retained_candidates.sort(key=lambda item: item.score, reverse=True)
            connectivity = (
                len(filtered_edges) / max(1, len(selected_set) - 1)
                if len(selected_set) > 1
                else 0.0
            )
            score = float(np.mean([prob_map[table] for table in selected_tables]))

            rescored.append(
                CandidateSubgraph(
                    db=subgraph.db,
                    score=score,
                    seed_tables=[table for table in subgraph.seed_tables if table in selected_set],
                    retained_tables=sorted(selected_set),
                    dropped_tables=sorted(set(subgraph.retained_tables) - selected_set),
                    bridge_tables=[table for table in subgraph.bridge_tables if table in selected_set],
                    edges=filtered_edges,
                    retained_candidates=retained_candidates,
                    connectivity=connectivity,
                )
            )

        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "threshold": self._threshold,
                "max_hops": self._max_hops,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LocalGraphNodeScorer":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = _SimpleGCN(in_dim=8, hidden_dim=32)
        model.load_state_dict(payload["state_dict"])
        scorer = cls(
            model=model,
            threshold=float(payload.get("threshold", 0.45)),
            max_hops=int(payload.get("max_hops", 2)),
        )
        return scorer
