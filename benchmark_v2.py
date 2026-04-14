"""
GraphRAG Query Router — 논문 수준 벤치마크 v2

Spider / Bird / FIBEN 세 데이터셋을 각각 점수화하고,
자동 생성 질의(DBCopilot §3.3)와 원본 질의를 모두 테스트한다.

비교 방법
----------
  SchemaRAG_Baseline  : 순수 스키마 임베딩 RAG (그래프·히스토리 없음)
  GraphRAG_NoComm     : FAISS + 접근 가중치 (커뮤니티 없음)
  GraphRAG_Full       : FAISS + 접근 가중치 + 커뮤니티 탐지
  ThreeStage          : Stage1 유사질문캐시 + Stage2 GraphRAG + Stage3 DSI Re-ranker

측정 지표 (DBCopilot §4.2 기준)
-----------------------------------
  hit@1   : 1위 결과가 정답 테이블인 비율
  hit@3   : 상위 3위 내 정답 비율
  hit@5   : 상위 5위 내 정답 비율
  mrr     : Mean Reciprocal Rank
  avg_lookup     : 정답 찾을 때까지 평균 탐색 횟수
  total_lookups  : 전체 탐색 횟수 합산
  lookup_reduction : SchemaRAG 대비 탐색 횟수 절감률
  avg_latency_ms : 평균 응답 시간 (ms)

실행 방법
----------
  python benchmark_v2.py --datasets spider bird fiben
  python benchmark_v2.py --datasets spider --max_samples 200
  python benchmark_v2.py --datasets demo   # 내장 데모 데이터로 실행

데이터셋 경로 (기본값):
  datasets/spider/
  datasets/bird/
  datasets/fiben/
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Callable

import numpy as np
from loguru import logger

from baseline_rag import SchemaRAGBaseline
from bench_datasets.base import BenchmarkSample, SchemaEntry
from embedding.embedder import get_embedder
from graph_db.neo4j_client import Neo4jClient
from graph_rag.community import CommunityDetector
from graph_rag.keyword_router import KeywordGraphRouter
from graph_rag.retriever import GraphRetriever
from query_generator import QueryGenerator
from router import QueryRouter
from three_stage.pipeline import ThreeStagePipeline
import config


# ──────────────────────────────────────────────────────────────────────── #
#  데이터셋 로더 레지스트리                                                 #
# ──────────────────────────────────────────────────────────────────────── #

def _load_dataset(
    name: str,
    data_root: str,
    max_samples: int | None,
) -> tuple[list[SchemaEntry], list[BenchmarkSample]]:
    """name에 따라 적절한 로더를 호출한다."""
    root = Path(data_root) / name

    if name == "spider":
        from bench_datasets.spider_loader import load_schemas, load_samples
        schemas = load_schemas(root)
        samples = load_samples(root, split="dev", max_samples=max_samples)

    elif name == "bird":
        from bench_datasets.bird_loader import load_schemas, load_samples
        schemas = load_schemas(root, split="dev")
        samples = load_samples(root, split="dev", max_samples=max_samples)

    elif name == "fiben":
        from bench_datasets.fiben_loader import load_schemas, load_samples
        schemas = load_schemas(root)
        samples = load_samples(root, max_samples=max_samples)

    elif name == "demo":
        from examples.demo import SCHEMAS, ACCESS_LOG
        from graph_rag.indexer import SchemaDefinition
        schemas = [
            SchemaEntry(
                db_id=s.db_name,
                tables={
                    t: {
                        "columns": info.get("columns", []),
                        "joins":   info.get("joins", []),
                    }
                    for t, info in s.tables.items()
                },
            )
            for s in SCHEMAS
        ]
        # demo 모드: ACCESS_LOG를 ground truth로 변환
        samples = [
            BenchmarkSample(
                question    = r["query_text"],
                db_id       = r["db_name"],
                used_tables = [r["table_name"]],
                source      = "demo",
            )
            for r in ACCESS_LOG
        ]
    else:
        raise ValueError(f"알 수 없는 데이터셋: {name}")

    return schemas, samples


# ──────────────────────────────────────────────────────────────────────── #
#  측정 지표                                                                #
# ──────────────────────────────────────────────────────────────────────── #

@dataclass
class MethodMetrics:
    name:              str
    hit1:              list[bool]  = field(default_factory=list)
    hit3:              list[bool]  = field(default_factory=list)
    hit5:              list[bool]  = field(default_factory=list)
    rr:                list[float] = field(default_factory=list)
    lookup_counts:     list[int]   = field(default_factory=list)
    latencies_ms:      list[float] = field(default_factory=list)
    cypher_counts:     list[int]   = field(default_factory=list)   # Cypher 실행 횟수

    def record(
        self,
        ranked: list[tuple[str, str]],
        correct_db: str,
        correct_tables: list[str],
        latency_ms: float,
        n_cypher: int = 0,
    ) -> None:
        # 첫 번째 정답 테이블의 순위 기준
        best_rank = len(ranked) + 1
        for table in correct_tables:
            try:
                rank = next(
                    i + 1 for i, (db, t) in enumerate(ranked)
                    if t == table and db == correct_db
                )
                best_rank = min(best_rank, rank)
            except StopIteration:
                pass

        self.hit1.append(best_rank == 1)
        self.hit3.append(best_rank <= 3)
        self.hit5.append(best_rank <= 5)
        self.rr.append(1.0 / best_rank)
        self.lookup_counts.append(best_rank)
        self.latencies_ms.append(latency_ms)
        self.cypher_counts.append(n_cypher)

    def summary(self, baseline_avg: float | None = None) -> dict:
        n          = len(self.hit1) or 1
        avg_lookup = float(np.mean(self.lookup_counts))
        reduction  = (
            (1 - avg_lookup / baseline_avg) * 100
            if baseline_avg and baseline_avg > 0 else 0.0
        )
        avg_cypher = float(np.mean(self.cypher_counts)) if self.cypher_counts else 0.0
        return {
            "method":           self.name,
            "n_samples":        n,
            "hit@1":            f"{sum(self.hit1)/n*100:.1f}%",
            "hit@3":            f"{sum(self.hit3)/n*100:.1f}%",
            "hit@5":            f"{sum(self.hit5)/n*100:.1f}%",
            "mrr":              f"{np.mean(self.rr):.4f}",
            "avg_lookup":       round(avg_lookup, 2),
            "total_lookups":    sum(self.lookup_counts),
            "lookup_reduction": f"{reduction:.1f}%",
            "avg_cypher":       round(avg_cypher, 2),
            "avg_latency_ms":   f"{np.mean(self.latencies_ms):.2f}",
        }


@dataclass
class PlannerMetrics:
    name: str = "SchemaPlanner"
    entrypoint_hit: list[bool] = field(default_factory=list)
    path_hit: list[bool] = field(default_factory=list)
    table_coverage: list[float] = field(default_factory=list)
    path_cost: list[int] = field(default_factory=list)
    unnecessary_tables: list[int] = field(default_factory=list)
    stop_efficiency: list[float] = field(default_factory=list)
    answer_confidence: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, plan, correct_db: str, correct_tables: list[str], latency_ms: float) -> None:
        correct_set = set(correct_tables)
        top_entry = plan.entrypoints[0] if plan.entrypoints else None
        top_path = plan.candidate_paths[0] if plan.candidate_paths else None

        entrypoint_ok = bool(
            top_entry
            and top_entry.db == correct_db
            and top_entry.table in correct_set
        )

        path_tables = top_path.tables if top_path else []
        if top_path and top_path.db == correct_db:
            matched_tables = len(correct_set.intersection(path_tables))
        else:
            matched_tables = 0

        coverage = matched_tables / max(1, len(correct_set))
        path_hit = matched_tables == len(correct_set) and len(correct_set) > 0
        path_cost = len(path_tables)
        unnecessary = max(0, path_cost - matched_tables)
        efficiency = matched_tables / max(1, path_cost)

        self.entrypoint_hit.append(entrypoint_ok)
        self.path_hit.append(path_hit)
        self.table_coverage.append(coverage)
        self.path_cost.append(path_cost)
        self.unnecessary_tables.append(unnecessary)
        self.stop_efficiency.append(efficiency)
        self.answer_confidence.append(plan.answer_confidence)
        self.latencies_ms.append(latency_ms)

    def summary(self) -> dict:
        n = len(self.table_coverage) or 1
        return {
            "method": self.name,
            "entrypoint_hit": f"{sum(self.entrypoint_hit)/n*100:.1f}%",
            "path_hit": f"{sum(self.path_hit)/n*100:.1f}%",
            "table_coverage": f"{np.mean(self.table_coverage)*100:.1f}%",
            "avg_path_cost": round(float(np.mean(self.path_cost)), 2) if self.path_cost else 0.0,
            "avg_unnecessary": round(float(np.mean(self.unnecessary_tables)), 2) if self.unnecessary_tables else 0.0,
            "stop_efficiency": f"{np.mean(self.stop_efficiency)*100:.1f}%",
            "avg_confidence": f"{np.mean(self.answer_confidence):.3f}",
            "avg_latency_ms": f"{np.mean(self.latencies_ms):.2f}",
        }


# ──────────────────────────────────────────────────────────────────────── #
#  GraphRAG NoComm 라우팅 (커뮤니티 제외 버전)                             #
# ──────────────────────────────────────────────────────────────────────── #

def _graphrag_no_comm_ranked(
    query: str,
    neo4j: Neo4jClient,
    faiss_idx,
    all_tables: list[tuple[str, str]],
    schema_baseline: SchemaRAGBaseline,
) -> list[tuple[str, str]]:
    """
    GraphRAG 점수(접근 그래프) + SchemaRAG fallback 하이브리드.
    히스토리가 없는 테이블은 SchemaRAG 임베딩 점수로 보완한다.
    """
    embedder = get_embedder()
    vec      = embedder.embed(query)
    similar  = faiss_idx.search(vec, k=50)

    # SchemaRAG fallback 점수 (모든 테이블 커버)
    schema_scores: dict[tuple[str, str], float] = {
        (r.db, r.table): r.score
        for r in schema_baseline.search(query)
    }
    # SchemaRAG 점수를 [0, 1]로 정규화
    if schema_scores:
        s_max = max(schema_scores.values()) or 1.0
        schema_scores = {k: v / s_max for k, v in schema_scores.items()}

    if not similar:
        # 히스토리 없음 → 순수 SchemaRAG
        return sorted(all_tables, key=lambda t: schema_scores.get(t, 0.0), reverse=True)

    sim_dict = dict(similar)
    accesses = neo4j.get_schema_paths_for_queries(list(sim_dict.keys()))

    graph_scores: dict[tuple[str, str], dict] = {}
    for row in accesses:
        key = (row["db"], row["table"])
        cs  = sim_dict.get(row["query_id"], 0.0)
        c   = graph_scores.setdefault(key, {"embed_sim": 0.0, "access_sum": 0})
        c["embed_sim"]  = max(c["embed_sim"], cs)
        c["access_sum"] += row["count"]

    max_access = max((v["access_sum"] for v in graph_scores.values()), default=1) or 1
    w_e = config.ALPHA / (config.ALPHA + config.BETA)
    w_a = config.BETA  / (config.ALPHA + config.BETA)

    final: dict[tuple[str, str], float] = {}
    for t in all_tables:
        if t in graph_scores:
            ev = graph_scores[t]
            graph_score = (
                w_e * ev["embed_sim"]
                + w_a * math.log1p(ev["access_sum"]) / math.log1p(max_access)
            )
            # 그래프 점수 우선, SchemaRAG로 보완
            final[t] = 0.7 * graph_score + 0.3 * schema_scores.get(t, 0.0)
        else:
            # 히스토리 없음 → SchemaRAG 점수만 (낮은 가중치)
            final[t] = 0.3 * schema_scores.get(t, 0.0)

    return sorted(all_tables, key=lambda t: final.get(t, 0.0), reverse=True)


# ──────────────────────────────────────────────────────────────────────── #
#  단일 데이터셋 벤치마크                                                   #
# ──────────────────────────────────────────────────────────────────────── #

def run_dataset_benchmark(
    dataset_name:   str,
    schemas:        list[SchemaEntry],
    samples:        list[BenchmarkSample],
    router:         QueryRouter,
    baseline:       SchemaRAGBaseline,
    generated_samples: list[BenchmarkSample],
    sample_label:   str = "original",
    three_stage_pipeline: ThreeStagePipeline | None = None,
    planner_max_hops: int = 3,
    planner_max_tables: int = 5,
    planner_max_entrypoints: int = 3,
    planner_max_candidate_paths: int = 3,
    planner_max_mcp_calls: int = 4,
) -> dict:
    """
    단일 데이터셋에 대해 세 방법을 비교한다.
    generated_samples가 있으면 원본 + 생성 합산해서 평가.
    """
    all_samples = samples + generated_samples
    if not all_samples:
        logger.warning(f"{dataset_name}: 샘플 없음, 건너뜀")
        return {}

    # 이 데이터셋의 전체 (db, table) 목록
    all_tables = [
        (s.db_id, t)
        for s in schemas
        for t in s.tables
    ]
    n_tables = len(all_tables)

    metrics = {
        "SchemaRAG_Baseline":  MethodMetrics("SchemaRAG_Baseline"),
        "KeywordGraph":        MethodMetrics("KeywordGraph"),        # Cypher 탐색
        "GraphRAG_NoComm":     MethodMetrics("GraphRAG_NoComm"),
        "GraphRAG_Full":       MethodMetrics("GraphRAG_Full"),
        "GraphRAG_Tiered":     MethodMetrics("GraphRAG_Tiered"),
        "ThreeStage":          MethodMetrics("ThreeStage"),          # 3단계 파이프라인
    }
    planner_metrics = PlannerMetrics()

    neo4j     = router._neo4j
    faiss     = router._faiss
    retriever = router._retriever
    tiered    = router._tiered
    kw_router = KeywordGraphRouter(neo4j)

    logger.info(
        f"{dataset_name}: {len(all_samples)}개 샘플 평가 중 "
        f"({n_tables}개 테이블)…"
    )

    for sample in all_samples:
        q  = sample.question
        db = sample.db_id
        correct_tables = sample.used_tables

        # ── SchemaRAG Baseline ────────────────────────────────────────
        t0 = time.perf_counter()
        ranked_baseline = baseline.ranked_tables(q, db_id=db)
        if not ranked_baseline:
            ranked_baseline = baseline.ranked_tables(q)
        lat = (time.perf_counter() - t0) * 1000
        metrics["SchemaRAG_Baseline"].record(ranked_baseline, db, correct_tables, lat, n_cypher=0)

        # ── KeywordGraph (신규: Cypher 탐색, 인덱스 없음) ─────────────
        t0 = time.perf_counter()
        ranked_kw, n_cypher_kw = kw_router.ranked_tables(q, all_tables)
        lat = (time.perf_counter() - t0) * 1000
        metrics["KeywordGraph"].record(ranked_kw, db, correct_tables, lat, n_cypher=n_cypher_kw)

        # ── GraphRAG NoComm ───────────────────────────────────────────
        t0 = time.perf_counter()
        ranked_nc = _graphrag_no_comm_ranked(q, neo4j, faiss, all_tables, baseline)
        lat = (time.perf_counter() - t0) * 1000
        metrics["GraphRAG_NoComm"].record(ranked_nc, db, correct_tables, lat, n_cypher=2)

        # ── GraphRAG Full (커뮤니티 + SchemaRAG fallback) ─────────────
        t0 = time.perf_counter()
        paths      = retriever.route(q, top_n=n_tables)
        graph_set  = {(p.db, p.table): p.score for p in paths}
        schema_scs = {(r.db, r.table): r.score for r in baseline.search(q)}
        s_max      = max(schema_scs.values(), default=1.0) or 1.0
        schema_scs = {k: v / s_max for k, v in schema_scs.items()}

        ranked_gf = sorted(
            all_tables,
            key=lambda t: (
                0.7 * graph_set[t] + 0.3 * schema_scs.get(t, 0.0)
                if t in graph_set
                else 0.3 * schema_scs.get(t, 0.0)
            ),
            reverse=True,
        )
        lat = (time.perf_counter() - t0) * 1000
        metrics["GraphRAG_Full"].record(ranked_gf, db, correct_tables, lat, n_cypher=3)

        # ── GraphRAG Tiered (신뢰도 게이팅) ──────────────────────────
        # 점수가 HIGH_CONF 이상이면 GraphRAG만, 아니면 SchemaRAG로 fallback
        t0 = time.perf_counter()
        tiered_result = tiered.route(q, top_n=n_tables)
        if tiered_result.top_score >= 0.88:
            # 고신뢰: GraphRAG 결과 그대로 사용
            ranked_tiered = [(p.db, p.table) for p in tiered_result.paths]
            remaining = [t for t in all_tables if t not in ranked_tiered]
            ranked_tiered += remaining
        else:
            # 저신뢰: SchemaRAG 결과 사용
            ranked_tiered = ranked_baseline
        lat = (time.perf_counter() - t0) * 1000
        metrics["GraphRAG_Tiered"].record(ranked_tiered, db, correct_tables, lat, n_cypher=2)

        # ── ThreeStage (3단계 파이프라인) ─────────────────────────────
        # Stage1 유사질문캐시 → Stage2 Graph-RAG → Stage3 DSI Re-ranker
        if three_stage_pipeline is not None:
            t0 = time.perf_counter()
            ts_result = three_stage_pipeline.route(q, all_tables)
            lat = (time.perf_counter() - t0) * 1000

            # Stage 1 HIT면 Cypher 0회, FULL이면 Stage2(~3회) + Stage3(0 Cypher)
            n_cypher_ts = 0 if ts_result.cache_hit else 3
            metrics["ThreeStage"].record(
                ts_result.ranked, db, correct_tables, lat, n_cypher=n_cypher_ts
            )
        else:
            # 파이프라인 미초기화 시 SchemaRAG 결과로 대체 (비교 제외)
            metrics["ThreeStage"].record(
                ranked_baseline, db, correct_tables, 0.0, n_cypher=0
            )

        # ── Schema Planner (멀티홉 경로 계획) ──────────────────────────────
        t0 = time.perf_counter()
        plan = router.plan(
            q,
            max_hops=planner_max_hops,
            max_tables=planner_max_tables,
            max_entrypoints=planner_max_entrypoints,
            max_candidate_paths=planner_max_candidate_paths,
            max_mcp_calls=planner_max_mcp_calls,
        )
        lat = (time.perf_counter() - t0) * 1000
        planner_metrics.record(plan, db, correct_tables, lat)

    # ── 요약 계산 ─────────────────────────────────────────────────────
    baseline_avg = float(np.mean(metrics["SchemaRAG_Baseline"].lookup_counts))
    summaries    = [m.summary(baseline_avg) for m in metrics.values()]

    return {
        "dataset":   dataset_name,
        "n_tables":  n_tables,
        "n_samples": len(all_samples),
        "n_original": len(samples),
        "n_generated": len(generated_samples),
        "results":   summaries,
        "planner": planner_metrics.summary(),
    }


# ──────────────────────────────────────────────────────────────────────── #
#  출력 포맷                                                                #
# ──────────────────────────────────────────────────────────────────────── #

def print_dataset_results(result: dict) -> None:
    if not result:
        return
    ds   = result["dataset"]
    cols = ["method", "hit@1", "hit@3", "mrr",
            "avg_lookup", "total_lookups", "lookup_reduction",
            "avg_cypher", "avg_latency_ms"]
    widths = [22, 7, 7, 8, 11, 14, 17, 11, 15]

    print(f"\n{'='*90}")
    print(f"  데이터셋: {ds.upper()}  "
          f"(테이블 {result['n_tables']}개, "
          f"샘플 {result['n_samples']}개 "
          f"= 원본 {result['n_original']} + 생성 {result['n_generated']})")
    print(f"{'='*90}")

    hdr = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    print(f"  {hdr}")
    print(f"  {sep}")
    for s in result["results"]:
        row = "  ".join(f"{str(s.get(c,'')):<{w}}" for c, w in zip(cols, widths))
        print(f"  {row}")

    # 기여도 분석
    by_name = {s["method"]: s for s in result["results"]}
    base = by_name.get("SchemaRAG_Baseline")
    kw   = by_name.get("KeywordGraph")
    gnc  = by_name.get("GraphRAG_NoComm")
    gf   = by_name.get("GraphRAG_Full")
    gt   = by_name.get("GraphRAG_Tiered")

    def pct(s): return float(str(s).replace("%", ""))

    if base and kw and gnc and gf:
        print(f"\n  비교 분석 (기준: SchemaRAG_Baseline):")
        print(f"  {'비교 구간':<40} {'hit@1':>8} {'avg_lookup':>11} {'avg_cypher':>11} {'lookup절감':>12}")
        print(f"  {'-'*85}")
        ts   = by_name.get("ThreeStage")
        rows = [
            ("SchemaRAG → KeywordGraph (키워드 탐색)", kw),
            ("SchemaRAG → GraphRAG_NoComm (접근그래프)", gnc),
            ("SchemaRAG → GraphRAG_Full (커뮤니티)", gf),
        ]
        if gt:
            rows.append(("SchemaRAG → GraphRAG_Tiered (게이팅)", gt))
        if ts:
            rows.append(("SchemaRAG → ThreeStage (3단계 파이프라인)", ts))
        for label, m in rows:
            print(
                f"  {label:<40} "
                f"  {pct(m['hit@1'])-pct(base['hit@1']):>+7.1f}%"
                f"  {m['avg_lookup']-base['avg_lookup']:>+10.2f}"
                f"  {m['avg_cypher']:>11}"
                f"  {m['lookup_reduction']:>12}"
            )

    planner = result.get("planner")
    if planner:
        print(f"\n  Planner metrics (top candidate path 기준):")
        print(f"  {'entrypoint_hit':<18} {planner['entrypoint_hit']}")
        print(f"  {'path_hit':<18} {planner['path_hit']}")
        print(f"  {'table_coverage':<18} {planner['table_coverage']}")
        print(f"  {'avg_path_cost':<18} {planner['avg_path_cost']}")
        print(f"  {'avg_unnecessary':<18} {planner['avg_unnecessary']}")
        print(f"  {'stop_efficiency':<18} {planner['stop_efficiency']}")
        print(f"  {'avg_confidence':<18} {planner['avg_confidence']}")
        print(f"  {'avg_latency_ms':<18} {planner['avg_latency_ms']}")


def print_cross_dataset_summary(all_results: list[dict]) -> None:
    """데이터셋 간 hit@1 / lookup_reduction 비교표."""
    print(f"\n{'='*90}")
    print("  Cross-dataset Summary")
    print(f"{'='*90}")
    header = f"  {'Dataset':<12}  {'Method':<22}  {'hit@1':>7}  {'mrr':>7}  {'avg_lookup':>11}  {'lookup_reduction':>17}"
    print(header)
    print(f"  {'-'*85}")
    for r in all_results:
        for s in r.get("results", []):
            print(
                f"  {r['dataset']:<12}  {s['method']:<22}  "
                f"{s['hit@1']:>7}  {s['mrr']:>7}  "
                f"{str(s['avg_lookup']):>11}  {s['lookup_reduction']:>17}"
            )
        print()

    print(f"\n{'='*90}")
    print("  Cross-dataset Planner Summary")
    print(f"{'='*90}")
    planner_header = (
        f"  {'Dataset':<12}  {'entrypoint_hit':>14}  {'path_hit':>10}  "
        f"{'table_coverage':>16}  {'avg_path_cost':>14}  {'avg_unnecessary':>17}"
    )
    print(planner_header)
    print(f"  {'-'*95}")
    for r in all_results:
        planner = r.get("planner")
        if not planner:
            continue
        print(
            f"  {r['dataset']:<12}  "
            f"{planner['entrypoint_hit']:>14}  "
            f"{planner['path_hit']:>10}  "
            f"{planner['table_coverage']:>16}  "
            f"{str(planner['avg_path_cost']):>14}  "
            f"{str(planner['avg_unnecessary']):>17}"
        )


# ──────────────────────────────────────────────────────────────────────── #
#  Neo4j에 스키마 + 접근 로그 적재                                          #
# ──────────────────────────────────────────────────────────────────────── #

def _load_train_split(
    ds_name: str,
    data_root: str,
    max_train: int = 3000,
) -> list[BenchmarkSample]:
    """train set을 로드한다. 없으면 빈 리스트 반환."""
    root = Path(data_root) / ds_name
    try:
        if ds_name == "spider":
            from bench_datasets.spider_loader import load_samples
            return load_samples(root, split="train", max_samples=max_train)
        elif ds_name == "bird":
            from bench_datasets.bird_loader import load_samples
            return load_samples(root, split="train", max_samples=max_train)
        elif ds_name == "fiben":
            # FIBEN은 test만 있는 경우 dev를 train으로 사용
            from bench_datasets.fiben_loader import load_samples
            return load_samples(root, max_samples=max_train)
        elif ds_name == "demo":
            from examples.demo import ACCESS_LOG
            return [
                BenchmarkSample(
                    question=r["query_text"], db_id=r["db_name"],
                    used_tables=[r["table_name"]], source="demo",
                )
                for r in ACCESS_LOG
            ]
    except Exception as e:
        logger.warning(f"train split 로드 실패 ({ds_name}): {e}")
    return []


def _load_dataset_into_router(
    router: QueryRouter,
    schemas: list[SchemaEntry],
    train_samples: list[BenchmarkSample],
    max_history: int = 3000,
) -> None:
    """
    스키마를 Neo4j에 등록하고, train_samples를 접근 히스토리로 학습시킨다.
    train_samples는 테스트 샘플과 완전히 분리된 학습 전용 데이터여야 한다.
    """
    for entry in schemas:
        router.register_schema(entry.to_schema_definition())

    history = [
        {
            "query_text": s.question,
            "db_name":    s.db_id,
            "table_name": s.primary_table,
            "count":      1,
        }
        for s in train_samples
        if s.primary_table
    ]
    train_size = min(max_history, len(history))
    router.load_history(history[:train_size])
    router.rebuild_communities()

    logger.info(
        f"  → {len(schemas)}개 스키마 등록, "
        f"{train_size}개 히스토리 학습 (train set)"
    )


# ──────────────────────────────────────────────────────────────────────── #
#  메인                                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="GraphRAG Benchmark v2")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["demo"],
        choices=["spider", "bird", "fiben", "demo"],
        help="평가할 데이터셋 목록",
    )
    parser.add_argument(
        "--data_root", default="bench_datasets",
        help="데이터셋 루트 디렉터리",
    )
    parser.add_argument(
        "--max_samples", type=int, default=200,
        help="데이터셋당 최대 샘플 수 (None = 전체)",
    )
    parser.add_argument(
        "--gen_per_table", type=int, default=3,
        help="테이블당 자동 생성 질의 수 (DBCopilot §3.3)",
    )
    parser.add_argument(
        "--no_generate", action="store_true",
        help="자동 질의 생성 비활성화",
    )
    parser.add_argument("--planner_max_hops", type=int, default=3,
                        help="schema planner 최대 join hop 수")
    parser.add_argument("--planner_max_tables", type=int, default=5,
                        help="schema planner 후보 path 최대 테이블 수")
    parser.add_argument("--planner_max_entrypoints", type=int, default=3,
                        help="schema planner가 유지할 entrypoint 수")
    parser.add_argument("--planner_max_candidate_paths", type=int, default=3,
                        help="schema planner가 유지할 candidate path 수")
    parser.add_argument("--planner_max_mcp_calls", type=int, default=4,
                        help="schema planner MCP call budget")
    args = parser.parse_args()

    print("=" * 90)
    print("  GraphRAG Query Router — Benchmark v2")
    print(f"  데이터셋: {args.datasets}  |  샘플 수: {args.max_samples}  |"
          f"  생성 질의: {'OFF' if args.no_generate else args.gen_per_table}/table")
    print(f"  Planner budget: hops={args.planner_max_hops}, tables={args.planner_max_tables}, "
          f"entrypoints={args.planner_max_entrypoints}, paths={args.planner_max_candidate_paths}, "
          f"mcp_calls={args.planner_max_mcp_calls}")
    print("=" * 90)

    # ── 라우터 초기화 ─────────────────────────────────────────────────
    print("\n[Init] QueryRouter 초기화…")
    router   = QueryRouter.build()
    baseline = SchemaRAGBaseline()
    gen      = QueryGenerator(n_per_table=args.gen_per_table)
    three_stage: ThreeStagePipeline | None = None   # 스키마 로드 후 초기화

    all_results: list[dict] = []

    for ds_name in args.datasets:
        print(f"\n{'─'*60}")
        print(f"  데이터셋 로드: {ds_name.upper()}")

        # ── 데이터 로드 ───────────────────────────────────────────────
        try:
            schemas, samples = _load_dataset(ds_name, args.data_root, args.max_samples)
        except FileNotFoundError as e:
            print(f"  ✗ 데이터셋 없음: {e}")
            print(f"  → bench_datasets/{ds_name}/ 에 데이터를 준비해 주세요")
            continue

        # ── 기저선 인덱스 구축 ────────────────────────────────────────
        print(f"  SchemaRAG 인덱스 구축…")
        baseline.build_index(schemas)

        # ── 자동 질의 생성 (DBCopilot §3.3) ──────────────────────────
        generated: list[BenchmarkSample] = []
        if not args.no_generate:
            print(f"  역방향 질의 생성 중 (DBCopilot §3.3)…")
            from graph_rag.indexer import SchemaDefinition
            schema_defs = [
                SchemaDefinition(
                    db_name=s.db_id,
                    tables=s.tables,
                )
                for s in schemas[:20]   # 처음 20개 DB만 생성 (속도)
            ]
            gen_queries = gen.generate_from_schemas(schema_defs)
            generated = [
                BenchmarkSample(
                    question    = q.query_text,
                    db_id       = q.db_name,
                    used_tables = [q.table_name],
                    source      = f"generated_{q.source}",
                )
                for q in gen_queries
            ]
            print(f"  ✓ {len(generated)}개 질의 자동 생성")

        # ── GraphRAG 학습: train set 사용 (dev와 완전 분리) ─────────────
        print(f"  GraphRAG 학습 중 (train set)…")
        train_samples = _load_train_split(ds_name, args.data_root)
        _load_dataset_into_router(router, schemas, train_samples)

        # ── ThreeStage 파이프라인 초기화 (스키마 로드 후) ────────────
        print(f"  ThreeStage 파이프라인 초기화…")
        three_stage = ThreeStagePipeline.build(
            graph_retriever = router._retriever,
            schemas         = schemas,
            cache_threshold = 0.92,
            top_k_graph     = 20,
        )

        # ── 벤치마크 실행: dev set 전체를 테스트에 사용 ─────────────
        print(f"  벤치마크 실행…")
        test_samples = samples  # dev set 전체

        result = run_dataset_benchmark(
            dataset_name          = ds_name,
            schemas               = schemas,
            samples               = test_samples,
            router                = router,
            baseline              = baseline,
            generated_samples     = generated,
            three_stage_pipeline  = three_stage,
            planner_max_hops      = args.planner_max_hops,
            planner_max_tables    = args.planner_max_tables,
            planner_max_entrypoints = args.planner_max_entrypoints,
            planner_max_candidate_paths = args.planner_max_candidate_paths,
            planner_max_mcp_calls = args.planner_max_mcp_calls,
        )
        all_results.append(result)
        print_dataset_results(result)

        # ── ThreeStage 내부 통계 출력 ─────────────────────────────────
        if three_stage is not None:
            ts_stats = three_stage.stats()
            pl = ts_stats["pipeline"]
            ca = ts_stats["cache"]
            print(f"\n  [ThreeStage 내부 통계]")
            print(f"    Stage1 캐시 히트율:  {pl['stage1_hit_rate']}"
                  f"  (전체 {ca['total_lookups']}회 중 {ca['cache_hits']}회)")
            print(f"    Stage2+3 실행율:     {pl['full_run_rate']}")
            print(f"    Fallback율:          {pl['fallback_rate']}")
            print(f"    캐시 저장 항목:      {ca['stored_entries']}개")
            print(f"    평균 응답 시간:      {pl['avg_latency_ms']} ms")

        # Neo4j 초기화 (다음 데이터셋을 위해)
        if len(args.datasets) > 1:
            print(f"\n  다음 데이터셋을 위해 Neo4j 초기화…")
            with router._neo4j.session() as s:
                s.run("MATCH (n) DETACH DELETE n")
            router.rebuild_faiss()

    # ── 전체 요약 ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        print_cross_dataset_summary(all_results)

    router.close()
    print("\n✓ Benchmark v2 complete.")


if __name__ == "__main__":
    main()
