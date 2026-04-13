"""
FIBEN 데이터셋 로더
(Financial Instruction Benchmark for Complex SQL — DBCopilot §4.1)

FIBEN 입수 방법:
  DBCopilot 저자 GitHub: https://github.com/tshu-w/DBCopilot
  → data/fiben/ 디렉터리 확인

  또는 논문 저자에게 직접 요청 (비공개 데이터셋일 수 있음)

예상 디렉터리 구조:
  datasets/fiben/
    tables.json   ← Spider 포맷 호환 스키마
    test.json     ← 질의 (question, db_id, query)

논문 통계 (DBCopilot Table 1):
  DB 수: 1 / 질의 수: 1,000 / 테이블 수: 51 (금융 도메인 단일 DB)
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from bench_datasets.base import BenchmarkSample, SchemaEntry
from bench_datasets.sql_parser import extract_tables


def load_schemas(fiben_dir: str | Path) -> list[SchemaEntry]:
    path = Path(fiben_dir) / "tables.json"
    if not path.exists():
        raise FileNotFoundError(f"FIBEN tables.json not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = list(raw.values())

    schemas: list[SchemaEntry] = []
    for db in raw:
        db_id       = db["db_id"].lower()
        table_names = [t.lower() for t in db.get("table_names_original", db["table_names"])]
        col_names   = db.get("column_names_original", db["column_names"])
        col_types   = db.get("column_types", [])

        tables: dict[str, dict] = {t: {"columns": [], "joins": []} for t in table_names}
        for i, (tbl_idx, col_name) in enumerate(col_names):
            if tbl_idx < 0:
                continue
            tbl   = table_names[tbl_idx]
            ctype = col_types[i] if i < len(col_types) else "text"
            tables[tbl]["columns"].append((col_name.lower(), ctype))

        col_to_table = {
            i: table_names[tbl_idx]
            for i, (tbl_idx, _) in enumerate(col_names)
            if tbl_idx >= 0
        }
        for fk_a, fk_b in db.get("foreign_keys", []):
            ta  = col_to_table.get(fk_a)
            tb  = col_to_table.get(fk_b)
            via = col_names[fk_a][1].lower() if fk_a < len(col_names) else ""
            if ta and tb and ta != tb and (tb, via) not in tables[ta]["joins"]:
                tables[ta]["joins"].append((tb, via))

        schemas.append(SchemaEntry(db_id=db_id, tables=tables))

    logger.info(f"FIBEN: {len(schemas)}개 DB 스키마 로드")
    return schemas


def load_samples(
    fiben_dir: str | Path,
    max_samples: int | None = None,
) -> list[BenchmarkSample]:
    # FIBEN은 test.json 또는 dev.json 사용
    for fname in ("test.json", "dev.json", "fiben.json"):
        path = Path(fiben_dir) / fname
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"FIBEN sample file not found in: {fiben_dir}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    samples: list[BenchmarkSample] = []
    for item in raw:
        sql    = item.get("query", item.get("SQL", ""))
        tables = extract_tables(sql)
        if not tables:
            continue
        samples.append(BenchmarkSample(
            question    = item.get("question", ""),
            db_id       = item.get("db_id", "").lower(),
            used_tables = tables,
            sql         = sql,
            source      = "fiben",
        ))
        if max_samples and len(samples) >= max_samples:
            break

    logger.info(f"FIBEN: {len(samples)}개 샘플 로드")
    return samples
