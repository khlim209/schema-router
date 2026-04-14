from __future__ import annotations

from pathlib import Path

from loguru import logger

from bench_datasets.base import BenchmarkSample, SchemaEntry
from graph_rag.indexer import AccessRecord


def dataset_root(dataset: str, data_root: str) -> Path:
    return Path(data_root) / dataset


def load_schemas(dataset: str, data_root: str) -> list[SchemaEntry]:
    root = dataset_root(dataset, data_root)

    if dataset == "spider":
        from bench_datasets.spider_loader import load_schemas as spider_load_schemas

        return spider_load_schemas(root)
    if dataset == "bird":
        from bench_datasets.bird_loader import load_schemas as bird_load_schemas

        return bird_load_schemas(root, split="dev")
    if dataset == "fiben":
        from bench_datasets.fiben_loader import load_schemas as fiben_load_schemas

        return fiben_load_schemas(root)

    raise ValueError(f"Unsupported dataset: {dataset}")


def load_samples(
    dataset: str,
    data_root: str,
    split: str = "dev",
    max_samples: int | None = None,
) -> list[BenchmarkSample]:
    root = dataset_root(dataset, data_root)

    if dataset == "spider":
        from bench_datasets.spider_loader import load_samples as spider_load_samples

        spider_split = "train" if split == "train" else "dev"
        return spider_load_samples(root, split=spider_split, max_samples=max_samples)
    if dataset == "bird":
        from bench_datasets.bird_loader import load_samples as bird_load_samples

        bird_split = "train" if split == "train" else "dev"
        return bird_load_samples(root, split=bird_split, max_samples=max_samples)
    if dataset == "fiben":
        from bench_datasets.fiben_loader import load_samples as fiben_load_samples

        if split == "train":
            logger.warning(
                "FIBEN does not expose a dedicated train split here. Reusing the available samples."
            )
        return fiben_load_samples(root, max_samples=max_samples)

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_registry(schemas: list[SchemaEntry]) -> dict[str, dict[str, dict]]:
    return {schema.db_id: schema.tables for schema in schemas}


def access_records_from_samples(samples: list[BenchmarkSample]) -> list[AccessRecord]:
    records: list[AccessRecord] = []
    for sample in samples:
        for table_name in sample.used_tables:
            records.append(
                AccessRecord(
                    query_text=sample.question,
                    db_name=sample.db_id,
                    table_name=table_name,
                    count=1,
                )
            )
    return records
