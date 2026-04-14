from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from adaptive_retrieval import (
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight GNN node scorer over local schema graphs."
    )
    parser.add_argument("--dataset", choices=["spider", "bird", "fiben"], default="spider")
    parser.add_argument("--data_root", default="bench_datasets")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--max_seed_tables", type=int, default=3)
    parser.add_argument("--min_component_size", type=int, default=2)
    parser.add_argument("--max_subgraphs", type=int, default=2)
    parser.add_argument("--max_tables_per_subgraph", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument(
        "--output",
        default="models/adaptive_retrieval/gnn_spider.pt",
        help="Path to save the trained node scorer.",
    )
    args = parser.parse_args()

    schemas = load_schemas(args.dataset, args.data_root)
    train_samples = load_samples(
        args.dataset,
        args.data_root,
        split="train",
        max_samples=args.max_samples,
    )
    registry = build_registry(schemas)

    table_index = TableIndexRetriever()
    table_index.build(schemas)

    schema_graph = TableSchemaGraph()
    schema_graph.rebuild(registry)
    schema_graph.ingest_access_log(access_records_from_samples(train_samples))

    pipeline = IndexGraphPruningPipeline(table_index, schema_graph)
    budget = RetrievalBudget(
        top_k=args.top_k,
        max_hops=args.max_hops,
        max_seed_tables=args.max_seed_tables,
        min_component_size=args.min_component_size,
        max_subgraphs=args.max_subgraphs,
        max_tables_per_subgraph=args.max_tables_per_subgraph,
    )

    scorer = LocalGraphNodeScorer(
        threshold=args.threshold,
        max_hops=args.max_hops,
    )
    stats = scorer.fit(
        train_samples,
        pipeline=pipeline,
        table_index=table_index,
        schema_graph=schema_graph,
        budget=budget,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    output_path = Path(args.output)
    scorer.save(output_path)

    metadata = {
        "dataset": args.dataset,
        "budget": budget.to_dict(),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "threshold": args.threshold,
        "max_samples": args.max_samples,
        "stats": asdict(stats),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("  GNN node scorer training complete")
    print("=" * 72)
    print(f"dataset        : {args.dataset}")
    print(f"train_samples  : {len(train_samples)}")
    print(f"output         : {output_path}")
    print(f"metadata       : {metadata_path}")
    print(f"n_graphs       : {stats.n_graphs}")
    print(f"avg_loss       : {stats.avg_loss:.4f}")
    print(f"positive_rate  : {stats.avg_positive_rate:.4f}")


if __name__ == "__main__":
    main()
