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
