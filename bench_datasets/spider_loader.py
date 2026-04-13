"""
Spider 데이터셋 로더

Spider 다운로드:
  https://drive.google.com/uc?export=download&id=1403EGqzIDoHMdQF4c9Bkql7PC5Ro2kb6
  → 압축 해제 후 datasets/spider/ 에 위치

예상 디렉터리 구조:
  datasets/spider/
    tables.json          ← DB 스키마
    train_spider.json    ← 훈련 질의
    dev.json             ← 검증 질의 (벤치마크용)

논문 통계 (DBCopilot Table 1):
  DB 수: 200 / 질의 수: 10,181 / 테이블 수: 1,020
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from bench_datasets.base import BenchmarkSample, SchemaEntry
from bench_datasets.sql_parser import extract_tables


def load_schemas(spider_dir: str | Path) -> list[SchemaEntry]:
    """tables.json → SchemaEntry 리스트."""
    path = Path(spider_dir) / "tables.json"
    if not path.exists():
        raise FileNotFoundError(f"Spider tables.json not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    schemas: list[SchemaEntry] = []
    for db in raw:
        db_id = db["db_id"].lower()

        # 테이블 이름 목록
        table_names = [t.lower() for t in db.get("table_names_original", db["table_names"])]

        # 컬럼 정보: column_names = [(table_idx, col_name), ...]
        col_names  = db.get("column_names_original", db["column_names"])
        col_types  = db.get("column_types", [])
        # col_names[0] = [-1, "*"] (와일드카드) → 건너뜀

        tables: dict[str, dict] = {t: {"columns": [], "joins": []} for t in table_names}
        for i, (tbl_idx, col_name) in enumerate(col_names):
            if tbl_idx < 0:
                continue
            tbl = table_names[tbl_idx]
            ctype = col_types[i] if i < len(col_types) else "text"
            tables[tbl]["columns"].append((col_name.lower(), ctype))

        # 외래키 → joins
        col_to_table = {}
        for i, (tbl_idx, _) in enumerate(col_names):
            if tbl_idx >= 0:
                col_to_table[i] = table_names[tbl_idx]

        for fk_a, fk_b in db.get("foreign_keys", []):
            ta = col_to_table.get(fk_a)
            tb = col_to_table.get(fk_b)
            if ta and tb and ta != tb:
                via = col_names[fk_a][1].lower() if fk_a < len(col_names) else ""
                if (tb, via) not in tables[ta]["joins"]:
                    tables[ta]["joins"].append((tb, via))

        schemas.append(SchemaEntry(db_id=db_id, tables=tables))

    logger.info(f"Spider: {len(schemas)}개 DB 스키마 로드")
    return schemas


def load_samples(
    spider_dir: str | Path,
    split: str = "dev",
    max_samples: int | None = None,
) -> list[BenchmarkSample]:
    """
    train_spider.json 또는 dev.json → BenchmarkSample 리스트.
    SQL에서 테이블 이름을 파싱해 used_tables를 채운다.
    """
    fname = "dev.json" if split == "dev" else "train_spider.json"
    path  = Path(spider_dir) / fname
    if not path.exists():
        raise FileNotFoundError(f"Spider {fname} not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    samples: list[BenchmarkSample] = []
    for item in raw:
        tables = extract_tables(item.get("query", ""))
        if not tables:
            continue
        samples.append(BenchmarkSample(
            question    = item["question"],
            db_id       = item["db_id"].lower(),
            used_tables = tables,
            sql         = item.get("query", ""),
            source      = "spider",
        ))
        if max_samples and len(samples) >= max_samples:
            break

    logger.info(f"Spider ({split}): {len(samples)}개 샘플 로드")
    return samples
