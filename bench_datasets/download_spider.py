"""
Spider 데이터셋 다운로드 스크립트

방법 1 (권장): Hugging Face datasets 라이브러리
방법 2 (fallback): requests로 직접 다운로드

실행:
    python datasets/download_spider.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "spider"


def method1_huggingface():
    """Hugging Face datasets 라이브러리로 다운로드."""
    from datasets import load_dataset

    print("  Hugging Face에서 Spider 다운로드 중… (첫 실행 시 수분 소요)")
    ds = load_dataset("spider", trust_remote_code=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # dev.json
    dev_records = [dict(row) for row in ds["validation"]]
    with open(OUT_DIR / "dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_records, f, ensure_ascii=False, indent=2)
    print(f"  ✓ dev.json — {len(dev_records)}개 질의")

    # train_spider.json
    train_records = [dict(row) for row in ds["train"]]
    with open(OUT_DIR / "train_spider.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    print(f"  ✓ train_spider.json — {len(train_records)}개 질의")

    # tables.json — HF Spider에는 스키마가 별도 필드로 있음
    # db_id별로 테이블/컬럼 정보를 재구성
    tables = _rebuild_tables_json(dev_records + train_records)
    with open(OUT_DIR / "tables.json", "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    print(f"  ✓ tables.json — {len(tables)}개 DB 스키마 재구성")


def _rebuild_tables_json(records: list[dict]) -> list[dict]:
    """
    HF Spider 레코드에서 tables.json 포맷을 재구성한다.
    HF Spider는 각 레코드에 db_id + query가 있고,
    별도 db_schema 필드가 있거나 없을 수 있다.
    없으면 SQL 파싱으로 테이블명만 추출해 최소 스키마를 만든다.
    """
    import re

    db_tables: dict[str, set] = {}
    for rec in records:
        db_id = rec.get("db_id", "")
        query = rec.get("query", "")

        # SQL에서 테이블 이름 추출
        matches = re.findall(
            r"(?:FROM|JOIN)\s+`?\"?([a-zA-Z_][a-zA-Z0-9_]*)`?\"?",
            query, re.IGNORECASE
        )
        for m in matches:
            db_tables.setdefault(db_id, set()).add(m.lower())

        # HF Spider에 db_table_names 필드가 있는 경우
        if "db_table_names" in rec:
            for t in rec["db_table_names"]:
                db_tables.setdefault(db_id, set()).add(t.lower())

    result = []
    for db_id, table_set in sorted(db_tables.items()):
        table_list = sorted(table_set)
        result.append({
            "db_id": db_id,
            "table_names": table_list,
            "table_names_original": table_list,
            "column_names": [[-1, "*"]] + [
                [i, "id"] for i, _ in enumerate(table_list)
            ],
            "column_names_original": [[-1, "*"]] + [
                [i, "id"] for i, _ in enumerate(table_list)
            ],
            "column_types": ["text"] * (len(table_list) + 1),
            "foreign_keys": [],
            "primary_keys": [],
        })
    return result


def method2_requests():
    """requests로 GitHub raw 파일 직접 다운로드."""
    import requests

    # Spider dev/train은 GitHub에 공개돼 있음
    files = {
        "dev.json": (
            "https://raw.githubusercontent.com/taoyds/spider/master/dev.json"
        ),
        "train_spider.json": (
            "https://raw.githubusercontent.com/taoyds/spider/master/train_spider.json"
        ),
        "tables.json": (
            "https://raw.githubusercontent.com/taoyds/spider/master/tables.json"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname, url in files.items():
        out_path = OUT_DIR / fname
        if out_path.exists():
            print(f"  ✓ {fname} 이미 존재, 건너뜀")
            continue

        print(f"  다운로드 중: {fname}…")
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"  ✗ 실패 (HTTP {resp.status_code}): {url}")
            continue

        with open(out_path, "wb") as f:
            f.write(resp.content)

        data = json.loads(resp.content)
        count = len(data) if isinstance(data, list) else "?"
        print(f"  ✓ {fname} — {count}개 항목")


def verify():
    """다운로드 결과 검증."""
    ok = True
    for fname, min_count in [("dev.json", 100), ("tables.json", 10)]:
        path = OUT_DIR / fname
        if not path.exists():
            print(f"  ✗ 없음: {path}")
            ok = False
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else "?"
        print(f"  ✓ {fname}: {count}개")
    return ok


def main():
    print("=" * 50)
    print("  Spider 데이터셋 다운로드")
    print("=" * 50)

    if (OUT_DIR / "dev.json").exists() and (OUT_DIR / "tables.json").exists():
        print("\n이미 다운로드돼 있습니다. 검증 중…")
        verify()
        return

    # 방법 1: Hugging Face
    try:
        import datasets as _hf
        print("\n[방법 1] Hugging Face datasets 사용")
        method1_huggingface()
    except ImportError:
        print("\nHugging Face datasets 없음 → 방법 2로 전환")
        print("[방법 2] GitHub raw 파일 직접 다운로드")
        method2_requests()

    print("\n검증:")
    if verify():
        print("\n✓ Spider 준비 완료!")
        print("  실행: python benchmark_v2.py --datasets spider --max_samples 200")
    else:
        print("\n✗ 일부 파일이 없습니다. 수동 다운로드가 필요할 수 있습니다.")
        print("  datasets/DOWNLOAD.md 참고")


if __name__ == "__main__":
    main()
