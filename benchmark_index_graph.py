"""
Benchmark for the practical retrieval direction:

  A. index only
  B. index + schema graph pruning

Metrics focus on table-set quality instead of single-table hit@1.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adaptive_retrieval import (
    IndexGraphPruningPipeline,
    RetrievalBudget,
    TableIndexRetriever,
    TableSchemaGraph,
)
from bench_datasets.base import BenchmarkSample, SchemaEntry


def _load_dataset(
    name: str,
    data_root: str,
    max_samples: int | None,
) -> tuple[list[SchemaEntry], list[BenchmarkSample]]:
    root = Path(data_root) / name

    if name == "spider":
        from bench_datasets.spider_loader import load_schemas, load_samples

        return load_schemas(root), load_samples(root, split="dev", max_samples=max_samples)
    if name == "bird":
        from bench_datasets.bird_loader import load_schemas, load_samples

        return load_schemas(root, split="dev"), load_samples(root, split="dev", max_samples=max_samples)
    if name == "fiben":
        from bench_datasets.fiben_loader import load_schemas, load_samples

        return load_schemas(root), load_samples(root, max_samples=max_samples)

    raise ValueError(f"Unsupported dataset: {name}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Index vs Index+Graph retrieval benchmark")
    parser.add_argument("--dataset", choices=["spider", "bird", "fiben"], default="spider")
    parser.add_argument("--data_root", default="bench_datasets")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--max_seed_tables", type=int, default=3)
    parser.add_argument("--min_component_size", type=int, default=2)
    parser.add_argument("--max_tables_per_subgraph", type=int, default=8)
    args = parser.parse_args()

    schemas, samples = _load_dataset(args.dataset, args.data_root, args.max_samples)
    registry = {schema.db_id: schema.tables for schema in schemas}

    table_index = TableIndexRetriever()
    table_index.build(schemas)

    schema_graph = TableSchemaGraph()
    schema_graph.rebuild(registry)

    pipeline = IndexGraphPruningPipeline(table_index, schema_graph)
    budget = RetrievalBudget(
        top_k=args.top_k,
        max_hops=args.max_hops,
        max_seed_tables=args.max_seed_tables,
        min_component_size=args.min_component_size,
        max_tables_per_subgraph=args.max_tables_per_subgraph,
    )

    metrics = {
        "IndexOnly": SetMetrics("IndexOnly"),
        "IndexGraph": SetMetrics("IndexGraph"),
    }

    for sample in samples:
        t0 = time.perf_counter()
        candidates = table_index.search(sample.question, top_k=args.top_k)
        lat = (time.perf_counter() - t0) * 1000
        predicted = [(candidate.db, candidate.table) for candidate in candidates]
        metrics["IndexOnly"].record(predicted, sample, lat)

        t0 = time.perf_counter()
        plan = pipeline.retrieve(sample.question, budget=budget)
        lat = (time.perf_counter() - t0) * 1000
        if plan.selected_subgraph is None:
            predicted = []
        else:
            predicted = [
                (plan.selected_subgraph.db, table)
                for table in plan.selected_subgraph.retained_tables
            ]
        metrics["IndexGraph"].record(predicted, sample, lat)

    print("\n" + "=" * 86)
    print(
        f"  {args.dataset.upper()} retrieval benchmark"
        f"  (samples={len(samples)}, top_k={args.top_k}, hops={args.max_hops})"
    )
    print("=" * 86)
    print(
        f"  {'method':<14}  {'table_recall':<14}  {'full_coverage':<14}  "
        f"{'avg_tables':<12}  {'avg_unnecessary':<17}  {'avg_latency_ms':<14}"
    )
    print(f"  {'-'*84}")
    for item in metrics.values():
        summary = item.summary()
        print(
            f"  {summary['method']:<14}  {summary['table_recall']:<14}  "
            f"{summary['full_coverage']:<14}  {str(summary['avg_tables']):<12}  "
            f"{str(summary['avg_unnecessary']):<17}  {summary['avg_latency_ms']:<14}"
        )


if __name__ == "__main__":
    main()
