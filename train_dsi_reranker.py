from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from adaptive_retrieval import DSIRerankerModel, TableIndexRetriever, TableSchemaGraph
from adaptive_retrieval.experiment_utils import (
    access_records_from_samples,
    build_registry,
    load_samples,
    load_schemas,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DSI-style reranker over index candidates."
    )
    parser.add_argument("--dataset", choices=["spider", "bird", "fiben"], default="spider")
    parser.add_argument("--data_root", default="bench_datasets")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument(
        "--output",
        default="models/adaptive_retrieval/dsi_spider.pkl",
        help="Path to save the trained reranker.",
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

    model = DSIRerankerModel(top_k=args.top_k, max_hops=args.max_hops)
    stats = model.fit(
        train_samples,
        table_index=table_index,
        schema_graph=schema_graph,
        top_k=args.top_k,
        max_hops=args.max_hops,
        include_gold=True,
    )

    output_path = Path(args.output)
    model.save(output_path)

    metadata = {
        "dataset": args.dataset,
        "top_k": args.top_k,
        "max_hops": args.max_hops,
        "max_samples": args.max_samples,
        "stats": asdict(stats),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("  DSI reranker training complete")
    print("=" * 72)
    print(f"dataset        : {args.dataset}")
    print(f"train_samples  : {len(train_samples)}")
    print(f"output         : {output_path}")
    print(f"metadata       : {metadata_path}")
    print(f"n_examples     : {stats.n_examples}")
    print(f"positive_rate  : {stats.positive_rate:.4f}")
    print(f"average_prec.  : {stats.average_precision:.4f}")
    print(
        "roc_auc        : "
        + (f"{stats.roc_auc:.4f}" if stats.roc_auc is not None else "n/a")
    )


if __name__ == "__main__":
    main()
