"""
Bird 데이터셋 로더

Bird 다운로드:
  https://bird-bench.github.io/  → "Download BIRD mini-dev"
  → 압축 해제 후 datasets/bird/ 에 위치

예상 디렉터리 구조:
  datasets/bird/
    dev/
      dev_databases/         ← 각 DB SQLite 파일
      dev.json               ← 검증 질의
      dev_tables.json        ← DB 스키마 (Spider 포맷 호환)
    train/
      train_databases/
      train.json
      train_tables.json

논문 통계 (DBCopilot Table 1):
  DB 수: 95 / 질의 수: 11,445 / 테이블 수: 638
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from bench_datasets.base import BenchmarkSample, SchemaEntry
from bench_datasets.sql_parser import extract_tables


def _load_tables_json(path: Path) -> list[SchemaEntry]:
    """Spider 포맷 호환 tables.json 파싱."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Bird tables.json은 리스트일 수도 있고 dict일 수도 있음
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

    return schemas


def load_schemas(bird_dir: str | Path, split: str = "dev") -> list[SchemaEntry]:
    bird_path = Path(bird_dir)
    tables_path = bird_path / split / f"{split}_tables.json"

    # fallback: dev_tables.json 직접
    if not tables_path.exists():
        tables_path = bird_path / f"{split}_tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(f"Bird tables.json not found near: {bird_path}")

    schemas = _load_tables_json(tables_path)
    logger.info(f"Bird ({split}): {len(schemas)}개 DB 스키마 로드")
    return schemas


def load_samples(
    bird_dir: str | Path,
    split: str = "dev",
    max_samples: int | None = None,
) -> list[BenchmarkSample]:
    bird_path   = Path(bird_dir)
    sample_path = bird_path / split / f"{split}.json"
    if not sample_path.exists():
        sample_path = bird_path / f"{split}.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Bird {split}.json not found near: {bird_path}")

    with open(sample_path, encoding="utf-8") as f:
        raw = json.load(f)

    samples: list[BenchmarkSample] = []
    for item in raw:
        sql    = item.get("SQL", item.get("query", ""))
        tables = extract_tables(sql)
        if not tables:
            continue
        samples.append(BenchmarkSample(
            question    = item.get("question", ""),
            db_id       = item.get("db_id", "").lower(),
            used_tables = tables,
            sql         = sql,
            source      = "bird",
        ))
        if max_samples and len(samples) >= max_samples:
            break

    logger.info(f"Bird ({split}): {len(samples)}개 샘플 로드")
    return samples
