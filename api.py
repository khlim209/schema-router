"""
FastAPI REST API for the GraphRAG Query Router.

Endpoints
---------
POST /route          – Route a query to (db, table) schema paths
POST /explain        – Full routing explanation with evidence
POST /record         – Record a query access event (online learning)
POST /schemas        – Register database schemas
POST /rebuild        – Trigger community rebuild
POST /train/decay    – Apply exponential decay to stale edges
GET  /train/report   – Training state summary
GET  /health         – Liveness check

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from graph_db.neo4j_client import Neo4jClient
from graph_rag.indexer import SchemaDefinition
from router import QueryRouter
from trainer import Trainer


# ──────────────────────────────────────────────────────────────────────── #
#  App state                                                                #
# ──────────────────────────────────────────────────────────────────────── #

_router:  QueryRouter | None = None
_trainer: Trainer     | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router, _trainer
    logger.info("Starting GraphRAG Query Router API…")
    _router  = QueryRouter.build()
    _trainer = Trainer(_router, _router._neo4j)
    yield
    logger.info("Shutting down…")
    _router.close()


app = FastAPI(
    title="GraphRAG Query Router",
    description=(
        "Routes natural-language queries to the correct DB/Table schema paths "
        "using a knowledge graph (Neo4j) + community detection + FAISS ANN search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_router() -> QueryRouter:
    if _router is None:
        raise HTTPException(503, "Router not initialised")
    return _router

def _get_trainer() -> Trainer:
    if _trainer is None:
        raise HTTPException(503, "Trainer not initialised")
    return _trainer


# ──────────────────────────────────────────────────────────────────────── #
#  Request / Response models                                                #
# ──────────────────────────────────────────────────────────────────────── #

class RouteRequest(BaseModel):
    query: str = Field(..., description="Natural-language query")
    top_n: int = Field(5,  description="Number of schema paths to return")

class RouteResult(BaseModel):
    db: str
    table: str
    score: float
    evidence: dict[str, Any]

class RouteResponse(BaseModel):
    query: str
    results: list[RouteResult]

class ExplainResponse(BaseModel):
    query: str
    community_id: int
    similar_queries: list[dict]
    ranked_paths: list[dict]

class PlanRequest(BaseModel):
    query: str = Field(..., description="Natural-language query")
    max_hops: int = Field(3, description="Maximum join hops to explore")
    max_tables: int = Field(5, description="Maximum number of tables in one candidate path")
    max_entrypoints: int = Field(3, description="How many entry tables to prioritize")
    max_candidate_paths: int = Field(3, description="How many alternative paths to keep")
    max_mcp_calls: int = Field(4, description="Planner budget for downstream MCP calls")

class PlanResponse(BaseModel):
    plan: dict[str, Any]

class RetrieveSubgraphRequest(BaseModel):
    query: str = Field(..., description="Natural-language query")
    top_k: int = Field(8, description="Number of index candidates to keep")
    max_hops: int = Field(2, description="Maximum graph hops when connecting candidates")
    max_seed_tables: int = Field(3, description="How many strong seed tables to probe first")
    min_component_size: int = Field(2, description="Minimum connected component size to keep")
    max_subgraphs: int = Field(2, description="How many DB-local candidate subgraphs to return")
    max_tables_per_subgraph: int = Field(8, description="Cap on retained tables per subgraph")
    query_type: str = Field("", description="Optional question type tag")

class RetrieveSubgraphResponse(BaseModel):
    plan: dict[str, Any]
    execution_log_path: str

class ExecutionFeedbackRequest(BaseModel):
    run_id: str = Field(..., description="Run ID returned from /retrieve/subgraph")
    query: str = Field(..., description="Original query text")
    executed_tables: list[str] = Field(default_factory=list, description="Actual tables inspected by MCP")
    contributing_tables: list[str] = Field(default_factory=list, description="Tables that materially contributed to the answer")
    unnecessary_tables: list[str] = Field(default_factory=list, description="Inspected but unused tables")
    success: bool = Field(True, description="Whether the run produced a satisfactory answer")
    latency_ms: float = Field(0.0, description="Total runtime for the MCP execution")
    final_answer: str = Field("", description="Optional final answer or answer summary")
    notes: str = Field("", description="Optional notes about failure or pruning behaviour")
    gold_tables: list[str] = Field(default_factory=list, description="Optional gold table set for offline evaluation")
    query_type: str = Field("", description="Optional question type tag")

class ExecutionFeedbackResponse(BaseModel):
    message: str
    execution_log_path: str

class RecordRequest(BaseModel):
    query_text: str = Field(..., description="Query text")
    db_name:    str = Field(..., description="Database name")
    table_name: str = Field(..., description="Table name")
    count:      int = Field(1,   description="Access count")

class RecordResponse(BaseModel):
    query_id: str
    message: str

class ColumnDef(BaseModel):
    name: str
    type: str = ""

class JoinDef(BaseModel):
    other_table: str
    via_column:  str = ""

class TableDef(BaseModel):
    description: str = ""
    columns: list[ColumnDef] = []
    joins:   list[JoinDef]   = []

class SchemaRequest(BaseModel):
    db_name:     str = Field(..., description="Database name")
    description: str = ""
    tables: dict[str, TableDef] = Field(
        ..., description="Map of table_name → table definition"
    )

class RebuildResponse(BaseModel):
    num_communities: int
    message: str

class DecayResponse(BaseModel):
    pruned_edges: int
    message: str


# ──────────────────────────────────────────────────────────────────────── #
#  Endpoints                                                                #
# ──────────────────────────────────────────────────────────────────────── #

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    """Route a query to the top-N most likely schema paths."""
    router = _get_router()
    paths = router.route(req.query, top_n=req.top_n)
    return RouteResponse(
        query=req.query,
        results=[
            RouteResult(
                db=p.db, table=p.table,
                score=round(p.score, 4),
                evidence=p.evidence,
            )
            for p in paths
        ],
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: RouteRequest):
    """Return full routing explanation with evidence breakdown."""
    router = _get_router()
    info = router.explain(req.query, top_n=req.top_n)
    return ExplainResponse(**info)


@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):
    """Return a budget-aware multi-hop schema traversal plan."""
    router = _get_router()
    plan_result = router.plan_dict(
        query_text=req.query,
        max_hops=req.max_hops,
        max_tables=req.max_tables,
        max_entrypoints=req.max_entrypoints,
        max_candidate_paths=req.max_candidate_paths,
        max_mcp_calls=req.max_mcp_calls,
    )
    return PlanResponse(plan=plan_result)


@app.post("/retrieve/subgraph", response_model=RetrieveSubgraphResponse)
async def retrieve_subgraph(req: RetrieveSubgraphRequest):
    """
    Return index candidates plus a schema-graph-pruned local subgraph.
    """
    router = _get_router()
    plan_result = router.retrieve_subgraph_dict(
        query_text=req.query,
        top_k=req.top_k,
        max_hops=req.max_hops,
        max_seed_tables=req.max_seed_tables,
        min_component_size=req.min_component_size,
        max_subgraphs=req.max_subgraphs,
        max_tables_per_subgraph=req.max_tables_per_subgraph,
        query_type=req.query_type,
    )
    return RetrieveSubgraphResponse(
        plan=plan_result,
        execution_log_path=router.execution_log_path,
    )


@app.post("/retrieve/feedback", response_model=ExecutionFeedbackResponse)
async def log_execution_feedback(req: ExecutionFeedbackRequest):
    """
    Persist MCP execution feedback for later DSI / GNN supervision.
    """
    router = _get_router()
    router.log_execution_feedback(
        run_id=req.run_id,
        query_text=req.query,
        executed_tables=req.executed_tables,
        contributing_tables=req.contributing_tables,
        unnecessary_tables=req.unnecessary_tables,
        success=req.success,
        latency_ms=req.latency_ms,
        final_answer=req.final_answer,
        notes=req.notes,
        gold_tables=req.gold_tables,
        query_type=req.query_type,
    )
    return ExecutionFeedbackResponse(
        message=f"Stored execution feedback for {req.run_id}",
        execution_log_path=router.execution_log_path,
    )


@app.post("/record", response_model=RecordResponse)
async def record(req: RecordRequest):
    """Record a query access event (online learning)."""
    trainer = _get_trainer()
    qid = trainer.learn(
        req.query_text, req.db_name, req.table_name, req.count
    )
    return RecordResponse(
        query_id=qid,
        message=f"Recorded access: {req.db_name}.{req.table_name}",
    )


@app.post("/schemas")
async def register_schemas(schemas: list[SchemaRequest]):
    """Register one or more database schemas."""
    router = _get_router()
    for s in schemas:
        sd = SchemaDefinition(
            db_name=s.db_name,
            description=s.description,
            tables={
                tname: {
                    "description": tdef.description,
                    "columns": [(c.name, c.type) for c in tdef.columns],
                    "joins":   [(j.other_table, j.via_column) for j in tdef.joins],
                }
                for tname, tdef in s.tables.items()
            },
        )
        router.register_schema(sd)
    return {"message": f"Registered {len(schemas)} schema(s)."}


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild_communities():
    """Trigger a full community detection rebuild."""
    router = _get_router()
    communities = router.rebuild_communities()
    return RebuildResponse(
        num_communities=len(communities),
        message="Community detection completed.",
    )


@app.post("/train/decay", response_model=DecayResponse)
async def apply_decay():
    """Apply exponential decay to stale ACCESSED edge weights."""
    trainer = _get_trainer()
    pruned = trainer.apply_decay()
    return DecayResponse(
        pruned_edges=pruned,
        message=f"Decay applied. {pruned} edges pruned.",
    )


@app.get("/train/report")
async def training_report():
    """Return a summary of the current training state."""
    trainer = _get_trainer()
    return trainer.report()
