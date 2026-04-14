# Schema Router

Natural-language query routing and schema planning for large database collections, powered by GraphRAG, a three-stage router, and a multi-hop schema planner.

This project started as a `question -> correct table` router for NL-to-SQL. It now also supports `question -> entry tables + candidate traversal paths + stop conditions`, which is useful when the answer is spread across multiple related tables and the goal is to reduce unnecessary MCP / DB lookups.

---

## Overview

The codebase currently has two complementary outputs:

1. `Routing`
   Returns ranked `(db, table)` candidates for classic schema routing.
2. `Planning`
   Returns a budget-aware multi-hop schema traversal plan with:
   `intent`, `entities`, `required_facts`, `entrypoints`, `candidate_paths`, `execution_steps`, and `stop_condition`.

This project implements and benchmarks the following retrieval approaches:

| Method | Core Idea |
|--------|-----------|
| `Index + Graph Pruning` | Rich table index -> top-k candidates -> schema-graph pruning -> small local subgraph |
| `SchemaRAG` | Pure embedding similarity over schema text |
| `GraphRAG` | Neo4j access graph + FAISS query history + community coverage |
| `GraphRAG Tiered` | Confidence-based lookup reduction on top of GraphRAG |
| `ThreeStage` | Similarity cache -> GraphRAG -> DSI-style reranker |
| `Schema Planner` | Query understanding -> candidate entrypoints -> graph-constrained multi-hop path planning |

The embedding model can also be fine-tuned on `(question, schema)` pairs using `MultipleNegativesRankingLoss`.

---

## Architecture

### 1. Index + Graph Pruning

```text
NL Question
  -> Rich Table Index
     - table name
     - columns
     - key columns
     - join targets
     - domain tags
  -> Top-k Table Candidates
  -> Table-Level Schema Graph
     - FK / join edges
     - query-log co-access edges
  -> Pruning
     - keep connected candidates only
     - allow 1~2 hop bridges
     - seed by score + connectivity
  -> Small DB-local Subgraph
  -> MCP inspection order
```

This is the practical retrieval path for reducing unnecessary table exploration.

### 2. Single-Table Routing

```text
NL Question
  -> Stage 1: Similarity Cache
  -> Stage 2: GraphRAG Retrieval
  -> Stage 3: DSI-style Re-ranking
  -> Top ranked (db, table)
```

This is still the default path for classic routing benchmarks.

### 3. Multi-Hop Schema Planning

```text
NL Question
  -> Query Understanding
     - intent
     - entities
     - required facts
     - constraints
  -> Schema Scoring
     - lexical overlap
     - fact / entity overlap
     - join richness
     - historical GraphRAG priors
  -> Candidate Entrypoints
  -> Multi-Hop Path Expansion
     - join-connected neighbors only
     - budget-aware search
     - fact coverage scoring
  -> Execution Steps
     - probe entrypoints
     - expand path if needed
     - try alternative branch if needed
     - aggregate evidence
```

The planner is designed for questions where the answer is not stored in one table, but must be assembled from several related tables with minimal exploration cost.

---

## Why The Planner Exists

Classic schema routing solves:

`question -> which table is most likely correct?`

The planner targets a harder problem:

`question -> which tables and traversal path should we inspect, in what order, under a fixed lookup budget?`

This is useful when:

- The answer is distributed across multiple tables
- MCP explores schemas iteratively
- The main objective is not only correctness but also lookup efficiency
- You want explicit stop conditions instead of open-ended schema exploration

---

## Current Planner Output

`QueryRouter.plan()` and `POST /plan` return a structure like:

```json
{
  "query": "What common patterns do high repurchase customers show?",
  "intent": "customer_order_purchase_count_campaign_response",
  "entities": ["customer", "order", "campaign"],
  "required_facts": ["purchase_count", "campaign_response", "pattern"],
  "entrypoints": [
    {
      "db": "ecommerce",
      "table": "order_items",
      "score": 0.2667
    }
  ],
  "candidate_paths": [
    {
      "path_id": "path_1",
      "db": "ecommerce",
      "tables": ["order_items", "orders", "customers", "campaign_events"],
      "coverage": 0.75,
      "estimated_cost": 4
    }
  ],
  "execution_steps": [
    {
      "step": 1,
      "action": "probe_entrypoints",
      "targets": ["ecommerce.order_items", "ecommerce.orders"]
    }
  ],
  "stop_condition": "Stop when fact coverage is high enough..."
}
```

---

## Fine-Tuning

The sentence-transformer embedding model is fine-tuned on Spider training data using **MultipleNegativesRankingLoss**.

- No explicit negative samples are needed
- Training pairs follow the shape `{"anchor": question, "positive": "table_name: col1 col2 description"}`
- Reverse-generated queries are also supported for synthetic supervision

```bash
# Basic: 3000 samples, 3 epochs
python finetune.py --n_samples 3000 --epochs 3

# Full: entire train set + reverse-generated queries, 5 epochs
python finetune.py --epochs 5 --use_generated --output models/finetuned_spider_full
```

### Fine-Tuning Results

Spider dev, 1034 questions:

| Model | hit@1 | hit@3 | MRR | avg_lookup |
|-------|-------|-------|-----|------------|
| Base (`all-MiniLM-L6-v2`) | 47.9% | 72.8% | 0.6225 | 7.3 |
| Fine-tuned (3k samples, 3 ep) | 53.3% | 73.0% | 0.6525 | 7.8 |
| **Fine-tuned (full + generated, 5 ep)** | **56.0%** | **77.9%** | **0.6846** | **6.7** |

These numbers come from `compare_models.py`, which evaluates the embedding model only.

- `hit@1 +8.1%p` with full fine-tuning vs base model
- `avg_lookup` drops from `7.3` to `6.7`
- Training time: ~2 min on CPU (7000+ pairs, 5 epochs)

> As more `(question, table)` pairs accumulate, the embedding space learns domain-specific alignment. New questions require fewer lookups and are routed more accurately.

---

## Reverse Query Generation

To augment training data without manual annotation, queries can be generated from schemas:

```text
Schema -> LLM -> "What are the top 10 best-selling products this month?"
         + ground truth: db=ecommerce, table=order_items
```

Generation strategies:

1. `Single-table`: one table at a time, Korean + English
2. `Multi-table`: pairs of tables with FK relationships
3. `Template fallback`: rule-based generation without an API key

```bash
python query_generator.py --n_dbs 20
```

Generated data is saved to `data/generated_queries.json`.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

OPENAI_API_KEY=sk-...          # for reverse query generation (optional)
ANTHROPIC_API_KEY=sk-ant-...   # fallback (optional)
```

### Spider Dataset

```bash
python bench_datasets/download_spider.py
```

Or manually download from [Yale Spider](https://yale-lily.github.io/spider) and place it under `bench_datasets/spider/`.

### Neo4j

Start a local Neo4j instance, then initialize the schema graph:

```bash
python init_db.py
```

---

## Usage

### Schema Routing

```python
from router import QueryRouter

router = QueryRouter.build()
paths = router.route("How many customers signed up last month?", top_n=3)

for path in paths:
    print(path.db, path.table, path.score)
```

### Multi-Hop Schema Planning

```python
from router import QueryRouter

router = QueryRouter.build()

plan = router.plan(
    "Show common patterns among high repurchase customers by category overlap and campaign response",
    max_hops=3,
    max_tables=5,
    max_mcp_calls=4,
)

print(plan.to_dict())
```

### Index + Graph Pruning

```python
from router import QueryRouter

router = QueryRouter.build()

plan = router.retrieve_subgraph(
    "최근 3개월 재구매율 높은 고객군의 공통 패턴은?",
    top_k=8,
    max_hops=2,
    max_seed_tables=3,
    max_tables_per_subgraph=8,
    query_type="customer_pattern",
)

print(plan.to_dict())
print(router.execution_log_path)
```

This flow returns:

- raw index candidates
- DB-local candidate subgraphs after schema-graph pruning
- the selected subgraph
- MCP inspection order
- a JSONL execution log path for later DSI / GNN supervision

### API

Run the API:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:

- `POST /route`
- `POST /explain`
- `POST /plan`
- `POST /retrieve/subgraph`
- `POST /retrieve/feedback`
- `POST /record`
- `POST /schemas`
- `POST /rebuild`
- `POST /train/decay`
- `GET /train/report`
- `GET /health`

Example `POST /plan` payload:

```json
{
  "query": "재구매가 높은 고객군의 공통 패턴을 카테고리와 캠페인 반응 기준으로 보고 싶어",
  "max_hops": 3,
  "max_tables": 5,
  "max_entrypoints": 3,
  "max_candidate_paths": 3,
  "max_mcp_calls": 4
}
```

Example `POST /retrieve/subgraph` payload:

```json
{
  "query": "최근 3개월 재구매율 높은 고객군의 공통 패턴은?",
  "top_k": 8,
  "max_hops": 2,
  "max_seed_tables": 3,
  "min_component_size": 2,
  "max_subgraphs": 2,
  "max_tables_per_subgraph": 8,
  "query_type": "customer_pattern"
}
```

### Benchmark

```bash
python benchmark_v2.py --datasets spider
python compare_models.py --finetuned models/finetuned_spider_full/final
python benchmark_index_graph.py --dataset spider --max_samples 200 --top_k 8 --max_hops 2
```

Note:
`benchmark_v2.py` is still centered on single-table routing metrics such as `hit@1` and `MRR`. The planner is implemented and callable, but planner-specific evaluation metrics are not yet integrated into the benchmark suite.

`benchmark_index_graph.py` compares:

- `IndexOnly`
- `Index + Graph Pruning`

It reports table-set metrics such as average gold-table recall, full coverage rate, average inspected tables, average unnecessary tables, and average latency.

### Fine-Tuning

```bash
python finetune.py --epochs 5 --use_generated
```

---

## Project Structure

```text
graphrag/
  config.py
  router.py
  api.py
  baseline_rag.py
  benchmark.py
  benchmark_v2.py
  benchmark_index_graph.py
  compare_models.py
  finetune.py
  init_db.py
  query_generator.py
  trainer.py
  adaptive_retrieval/
    models.py
    table_index.py
    schema_graph.py
    pipeline.py
    execution_log.py
  embedding/
    embedder.py
    faiss_index.py
  graph_db/
    neo4j_client.py
  graph_rag/
    community.py
    indexer.py
    keyword_router.py
    retriever.py
    tiered_retriever.py
  planner/
    __init__.py
    models.py
    query_understanding_v2.py
    schema_planner.py
  three_stage/
    pipeline.py
    stage1_cache.py
    stage3_reranker.py
  bench_datasets/
    base.py
    spider_loader.py
    bird_loader.py
  data/
    generated_queries.json
```

---

## Metrics

### Current Benchmark Metrics

| Metric | Description |
|--------|-------------|
| `hit@1` | Correct table is ranked first |
| `hit@3` | Correct table appears in top 3 |
| `MRR` | Mean reciprocal rank |
| `avg_lookup` | Average rank position of the correct table |
| `lookup_reduction` | Relative lookup savings vs baseline |

### Recommended Planner Metrics

The planner introduces a different objective, so these are the next logical evaluation targets:

- `fact_coverage`
- `path_cost`
- `unnecessary_tables`
- `answer_quality`
- `stop_efficiency`

---

## References

- [Spider: Yale NL-to-SQL Benchmark](https://yale-lily.github.io/spider)
- [DBCopilot: Scaling Natural Language Queries to Massive Databases](https://arxiv.org/abs/2312.03463)
- [Sentence Transformers: MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html)
- [Neo4j Graph Data Science: Leiden Algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)
