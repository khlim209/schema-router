# Schema Router

Natural language question → correct database table, powered by **GraphRAG + Sentence-Transformer fine-tuning**.

Given a question like *"How many orders were placed last month?"*, Schema Router identifies the target table (`orders`) across 160+ databases without writing a single SQL query first.

---

## Overview

Schema routing is the first step in any NL-to-SQL pipeline. Before an LLM can write SQL, it needs to know *which table* to query. Naive approaches embed all schema descriptions and do cosine similarity — this works but degrades as the number of tables grows.

This project implements and benchmarks four increasingly sophisticated approaches:

| Method | Core Idea |
|--------|-----------|
| **SchemaRAG** | Pure embedding similarity (FAISS). Baseline. |
| **GraphRAG** | Neo4j access graph + community detection + FAISS history |
| **GraphRAG Tiered** | GraphRAG with confidence-based fallback to SchemaRAG |
| **ThreeStage** | Cache → Graph-RAG → DSI Re-ranker pipeline |

On top of these, the embedding model itself is fine-tuned on (question, schema) pairs using `MultipleNegativesRankingLoss`.

---

## Architecture

```
NL Question
    │
    ▼
┌─────────────────────────────────────────────┐
│ Stage 1: Similarity Cache                   │
│   cosine_sim > 0.92?                        │
│   YES → return cached (db, table) instantly │
│   NO  → Stage 2                             │
└─────────────────────────────────────────────┘
    │ Cache MISS
    ▼
┌─────────────────────────────────────────────┐
│ Stage 2: Graph-RAG                          │
│   Question node → Neo4j BFS (1~2 hop)      │
│   757 tables → top-20 candidates            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Stage 3: DSI Re-ranker                      │
│   Embedding similarity   (55%)              │
│   Column keyword overlap (30%)              │
│   FK / JOIN richness     (15%)              │
│   → top-1 schema + store in Stage 1 cache  │
└─────────────────────────────────────────────┘
    │
    ▼
LLM → SQL Generation
```

**Feedback loop**: SQL execution success/failure updates cache confidence and graph edge weights.

---

## Fine-Tuning

The sentence-transformer embedding model is fine-tuned on Spider training data using **MultipleNegativesRankingLoss** (bi-encoder):

- No explicit negative samples needed — other pairs in the same batch act as negatives
- Training pair format: `{"anchor": question, "positive": "table_name: col1 col2 — description"}`
- Also supports reverse-generated queries (DBCopilot §3.3 style): schema → LLM → synthetic question

```bash
# Basic: 3000 samples, 3 epochs
python finetune.py --n_samples 3000 --epochs 3

# Full: entire train set + reverse-generated queries, 5 epochs
python finetune.py --epochs 5 --use_generated --output models/finetuned_spider_full
```

### Fine-Tuning Results (Spider dev, 1034 questions)

| Model | hit@1 | hit@3 | MRR | avg_lookup |
|-------|-------|-------|-----|------------|
| Base (`all-MiniLM-L6-v2`) | 47.9% | 72.8% | 0.6225 | 7.3 |
| Fine-tuned (3k samples, 3 ep) | 53.3% | 73.0% | 0.6525 | 7.8 |
| **Fine-tuned (full + generated, 5 ep)** | **56.0%** | **77.9%** | **0.6846** | **6.7** |

- **hit@1 +8.1%p** with full fine-tuning vs base model
- **avg_lookup drops from 7.3 → 6.7**: fewer tables inspected before finding the correct one
- Training time: ~2 min on CPU (7000+ pairs, 5 epochs)

> **Key insight**: As more (question, table) pairs accumulate, the embedding space learns domain-specific alignment. New questions require fewer lookups and are routed more accurately — this is the core learning loop.

---

## Reverse Query Generation

To augment training data without manual annotation, queries are auto-generated from schemas (DBCopilot §3.3):

```
Schema → LLM → "What are the top 10 best-selling products this month?"
               + ground truth: db=ecommerce, table=order_items
```

Three generation strategies:
1. **Single-table**: one table at a time, Korean + English
2. **Multi-table (JOIN)**: pairs of tables with FK relationships
3. **Template fallback**: rule-based if no API key available

```bash
# Generate queries for first N databases
python query_generator.py --n_dbs 20
```

Generated data is saved to `data/generated_queries.json` (306 queries included).

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

Or manually download from [Yale Spider](https://yale-lily.github.io/spider) and place under `bench_datasets/spider/`.

### Neo4j

Start a local Neo4j instance (Desktop or Docker), then initialize the schema graph:

```bash
python init_db.py
```

---

## Usage

### Schema Routing (single question)

```python
from router import QueryRouter

router = QueryRouter()
result = router.route("How many customers signed up last month?")
print(result.top_db, result.top_table)
```

### Benchmark

```bash
# Full benchmark: SchemaRAG / GraphRAG / ThreeStage
python benchmark_v2.py --datasets spider

# Compare base vs fine-tuned (no Neo4j needed)
python compare_models.py --finetuned models/finetuned_spider_full/final
```

### Fine-Tuning

```bash
python finetune.py --epochs 5 --use_generated
```

---

## Project Structure

```
schema-router/
├── config.py                  # Model name, Neo4j URI, thresholds
├── router.py                  # Main QueryRouter entry point
├── baseline_rag.py            # SchemaRAG baseline (FAISS only)
├── finetune.py                # Sentence-transformer fine-tuning
├── compare_models.py          # Base vs fine-tuned evaluation
├── benchmark_v2.py            # Full benchmark suite
├── query_generator.py         # Reverse query generation (DBCopilot §3.3)
│
├── embedding/
│   ├── embedder.py            # SentenceTransformer wrapper + singleton
│   └── faiss_index.py         # FAISS index helpers
│
├── graph_rag/
│   ├── indexer.py             # Schema → Neo4j ingestion
│   ├── retriever.py           # Graph BFS retriever
│   ├── community.py           # Leiden community detection
│   └── tiered_retriever.py    # Confidence-tiered retriever
│
├── three_stage/
│   ├── stage1_cache.py        # Similarity cache (FAISS + threshold)
│   ├── stage3_reranker.py     # DSI re-ranker (embed + keyword + struct)
│   └── pipeline.py            # ThreeStagePipeline orchestrator
│
├── graph_db/
│   └── neo4j_client.py        # Neo4j driver wrapper
│
├── bench_datasets/
│   ├── spider_loader.py       # Spider train/dev loader
│   ├── bird_loader.py         # BIRD benchmark loader
│   └── base.py                # BenchmarkSample, SchemaEntry types
│
└── data/
    └── generated_queries.json # 306 reverse-generated queries
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **hit@1** | Correct table is ranked #1 |
| **hit@3** | Correct table is in top 3 |
| **MRR** | Mean Reciprocal Rank — average of 1/rank |
| **avg_lookup** | Average rank position of the correct table (lower = better) |

---

## References

- [Spider: Yale NL-to-SQL Benchmark](https://yale-lily.github.io/spider)
- [DBCopilot: Scaling Natural Language Queries to Massive Databases](https://arxiv.org/abs/2312.03463)
- [Sentence Transformers: MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html)
- [Neo4j Graph Data Science: Leiden Algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)
