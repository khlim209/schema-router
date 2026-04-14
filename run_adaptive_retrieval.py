from __future__ import annotations

import argparse
import json

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run adaptive retrieval inference with index, graph, DSI, or GNN strategies."
    )
    parser.add_argument("--dataset", choices=["spider", "bird", "fiben"], default="spider")
    parser.add_argument("--data_root", default="bench_datasets")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--strategy",
        choices=["index", "graph", "dsi", "gnn"],
        default="graph",
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument("--max_seed_tables", type=int, default=3)
    parser.add_argument("--min_component_size", type=int, default=2)
    parser.add_argument("--max_subgraphs", type=int, default=2)
    parser.add_argument("--max_tables_per_subgraph", type=int, default=8)
    parser.add_argument("--use_history_graph", action="store_true")
    parser.add_argument("--history_max_samples", type=int, default=None)
    parser.add_argument("--query_type", default="")
    parser.add_argument("--dsi_model", default="")
    parser.add_argument("--gnn_model", default="")
    args = parser.parse_args()

    schemas = load_schemas(args.dataset, args.data_root)
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

    if args.strategy == "index":
        plan = pipeline.retrieve_index_only(
            args.query,
            budget=budget,
            query_type=args.query_type,
        )
    elif args.strategy == "graph":
        plan = pipeline.retrieve(
            args.query,
            budget=budget,
            query_type=args.query_type,
        )
    elif args.strategy == "dsi":
        if not args.dsi_model:
            raise ValueError("--dsi_model is required when --strategy dsi")
        model = DSIRerankerModel.load(args.dsi_model)
        candidates = table_index.search(args.query, top_k=budget.top_k)
        reranked = model.rerank_candidates(
            args.query,
            candidates,
            schema_graph=schema_graph,
        )
        plan = pipeline.retrieve_with_candidates(
            query=args.query,
            index_candidates=reranked,
            budget=budget,
            query_type=args.query_type,
            note="Index candidates reranked by DSI before schema-graph pruning.",
        )
    else:
        if not args.gnn_model:
            raise ValueError("--gnn_model is required when --strategy gnn")
        model = LocalGraphNodeScorer.load(args.gnn_model)
        candidates = table_index.search(args.query, top_k=budget.top_k)
        base_subgraphs = pipeline.build_candidate_subgraphs(candidates, budget=budget)
        rescored = model.score_subgraphs(
            args.query,
            base_subgraphs,
            table_index=table_index,
            schema_graph=schema_graph,
            budget=budget,
        )
        plan = pipeline.plan_from_subgraphs(
            query=args.query,
            index_candidates=candidates,
            candidate_subgraphs=rescored,
            budget=budget,
            query_type=args.query_type,
            note="Graph-pruned subgraphs rescored by the GNN node scorer.",
        )

    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
