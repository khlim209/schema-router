"""
Benchmark the practical retrieval ladder:

  A. IndexOnly
  B. IndexGraph
  C. IndexGraph + DSI reranker
  D. IndexGraph + GNN node scorer
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from adaptive_retrieval import (
    DSIRerankerModel,
    IndexGraphPruningPipeline,
    LocalGraphNodeScorer,
    RetrievalBudget,
    TableIndexRetriever,
    TableSchemaGraph,
)
from adaptive_retrieval.experiment_utils import (
    access_records_from_samples,
    build_registry,
    load_samples,
    load_schemas,
)
from bench_datasets.base import BenchmarkSample


@dataclass
class SetMetrics:
    name: str
    recalls: list[float] = field(default_factory=list)
    full_coverages: list[bool] = field(default_factory=list)
    inspected_tables: list[int] = field(default_factory=list)
    unnecessary_tables: list[int] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def record(
        self,
        predicted: list[tuple[str, str]],
        sample: BenchmarkSample,
        latency_ms: float,
    ) -> None:
        gold = {(sample.db_id, table) for table in sample.used_tables}
        predicted_set = set(predicted)
        hit = predicted_set & gold
        recall = len(hit) / len(gold) if gold else 0.0

        self.recalls.append(recall)
        self.full_coverages.append(gold.issubset(predicted_set))
        self.inspected_tables.append(len(predicted))
        self.unnecessary_tables.append(len(predicted_set - gold))
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict[str, str | float]:
        n = len(self.recalls) or 1
        return {
            "method": self.name,
            "table_recall": f"{np.mean(self.recalls) * 100:.1f}%",
            "full_coverage": f"{sum(self.full_coverages) / n * 100:.1f}%",
            "avg_tables": round(float(np.mean(self.inspected_tables)), 2),
            "avg_unnecessary": round(float(np.mean(self.unnecessary_tables)), 2),
            "avg_latency_ms": f"{np.mean(self.latencies_ms):.2f}",
        }


def _predicted_tables_from_plan(plan) -> list[tuple[str, str]]:
    if plan.selected_subgraph is None:
        return []
    return [
        (plan.selected_subgraph.db, table)
        for table in plan.selected_subgraph.retained_tables
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare index only, graph pruning, DSI reranking, and GNN node scoring."
    )
    parser.add_argument("--dataset", choices=["spider", "bird", "fiben"], default="spider")
    parser.add_argument("--data_root", default="bench_datasets")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--max_seed_tables", type=int, default=3)
    parser.add_argument("--min_component_size", type=int, default=2)
    parser.add_argument("--max_subgraphs", type=int, default=2)
    parser.add_argument("--max_tables_per_subgraph", type=int, default=8)
    parser.add_argument("--use_history_graph", action="store_true")
    parser.add_argument("--history_max_samples", type=int, default=None)
    parser.add_argument("--dsi_model", default="", help="Path to a trained DSI reranker.")
    parser.add_argument("--gnn_model", default="", help="Path to a trained GNN node scorer.")
    args = parser.parse_args()

    schemas = load_schemas(args.dataset, args.data_root)
    samples = load_samples(
        args.dataset,
        args.data_root,
        split="dev",
        max_samples=args.max_samples,
    )
    registry = build_registry(schemas)

    table_index = TableIndexRetriever()
    table_index.build(schemas)

    schema_graph = TableSchemaGraph()
    schema_graph.rebuild(registry)
    if args.use_history_graph:
        history_samples = load_samples(
            args.dataset,
            args.data_root,
            split="train",
            max_samples=args.history_max_samples,
        )
        schema_graph.ingest_access_log(access_records_from_samples(history_samples))

    pipeline = IndexGraphPruningPipeline(table_index, schema_graph)
    budget = RetrievalBudget(
        top_k=args.top_k,
        max_hops=args.max_hops,
        max_seed_tables=args.max_seed_tables,
        min_component_size=args.min_component_size,
        max_subgraphs=args.max_subgraphs,
        max_tables_per_subgraph=args.max_tables_per_subgraph,
    )

    dsi_model = DSIRerankerModel.load(args.dsi_model) if args.dsi_model else None
    gnn_model = LocalGraphNodeScorer.load(args.gnn_model) if args.gnn_model else None

    metrics: dict[str, SetMetrics] = {
        "IndexOnly": SetMetrics("IndexOnly"),
        "IndexGraph": SetMetrics("IndexGraph"),
    }
    if dsi_model is not None:
        metrics["IndexGraph+DSI"] = SetMetrics("IndexGraph+DSI")
    if gnn_model is not None:
        metrics["IndexGraph+GNN"] = SetMetrics("IndexGraph+GNN")

    for sample in samples:
        t0 = time.perf_counter()
        plan = pipeline.retrieve_index_only(sample.question, budget=budget)
        lat = (time.perf_counter() - t0) * 1000
        predicted = [(candidate.db, candidate.table) for candidate in plan.index_candidates]
        metrics["IndexOnly"].record(predicted, sample, lat)

        t0 = time.perf_counter()
        plan = pipeline.retrieve(sample.question, budget=budget)
        lat = (time.perf_counter() - t0) * 1000
        metrics["IndexGraph"].record(_predicted_tables_from_plan(plan), sample, lat)

        if dsi_model is not None:
            t0 = time.perf_counter()
            candidates = table_index.search(sample.question, top_k=budget.top_k)
            reranked = dsi_model.rerank_candidates(
                sample.question,
                candidates,
                schema_graph=schema_graph,
            )
            plan = pipeline.retrieve_with_candidates(
                query=sample.question,
                index_candidates=reranked,
                budget=budget,
                note="Index candidates reranked by DSI before schema-graph pruning.",
            )
            lat = (time.perf_counter() - t0) * 1000
            metrics["IndexGraph+DSI"].record(
                _predicted_tables_from_plan(plan),
                sample,
                lat,
            )

        if gnn_model is not None:
            t0 = time.perf_counter()
            candidates = table_index.search(sample.question, top_k=budget.top_k)
            base_subgraphs = pipeline.build_candidate_subgraphs(candidates, budget=budget)
            rescored = gnn_model.score_subgraphs(
                sample.question,
                base_subgraphs,
                table_index=table_index,
                schema_graph=schema_graph,
                budget=budget,
            )
            plan = pipeline.plan_from_subgraphs(
                query=sample.question,
                index_candidates=candidates,
                candidate_subgraphs=rescored,
                budget=budget,
                note="Graph-pruned subgraphs rescored by the GNN node scorer.",
            )
            lat = (time.perf_counter() - t0) * 1000
            metrics["IndexGraph+GNN"].record(
                _predicted_tables_from_plan(plan),
                sample,
                lat,
            )

    print("\n" + "=" * 104)
    print(
        f"  {args.dataset.upper()} adaptive retrieval benchmark"
        f"  (samples={len(samples)}, top_k={args.top_k}, hops={args.max_hops})"
    )
    print("=" * 104)
    print(
        f"  {'method':<18}  {'table_recall':<14}  {'full_coverage':<14}  "
        f"{'avg_tables':<12}  {'avg_unnecessary':<17}  {'avg_latency_ms':<14}"
    )
    print(f"  {'-' * 102}")
    for item in metrics.values():
        summary = item.summary()
        print(
            f"  {summary['method']:<18}  {summary['table_recall']:<14}  "
            f"{summary['full_coverage']:<14}  {str(summary['avg_tables']):<12}  "
            f"{str(summary['avg_unnecessary']):<17}  {summary['avg_latency_ms']:<14}"
        )


if __name__ == "__main__":
    main()
