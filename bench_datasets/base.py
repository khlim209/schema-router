"""
Dataset 로더 공통 인터페이스.

모든 로더가 반환하는 공통 타입:
  BenchmarkSample : (question, db_id, used_tables)
  SchemaEntry     : (db_id, tables_dict)  ← SchemaDefinition 변환용
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkSample:
    """하나의 (질의, 정답) 쌍."""
    question:    str
    db_id:       str
    used_tables: list[str]           # 정답 테이블 이름 목록 (lower case)
    sql:         str        = ""
    source:      str        = ""     # "spider" | "bird" | "fiben" | "generated"

    @property
    def primary_table(self) -> str:
        """라우팅 평가 기준 테이블 (첫 번째 정답 테이블)."""
        return self.used_tables[0] if self.used_tables else ""


@dataclass
class SchemaEntry:
    """하나의 DB 스키마."""
    db_id:   str
    # { table_name: { "columns": [(col, type), ...], "joins": [(other, via), ...] } }
    tables:  dict[str, dict] = field(default_factory=dict)

    def to_schema_definition(self):
        """GraphIndexer.ingest_schema()에 넘길 SchemaDefinition으로 변환."""
        from graph_rag.indexer import SchemaDefinition
        return SchemaDefinition(
            db_name=self.db_id,
            tables=self.tables,
        )
