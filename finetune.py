"""
Sentence-Transformer Fine-tuning for Schema Routing
=====================================================

Spider train split의 (질문, 정답 테이블 스키마) 쌍으로
sentence-transformer를 파인튜닝한다.

학습 방법: MultipleNegativesRankingLoss (bi-encoder)
  - 같은 배치 내 다른 샘플들을 자동으로 negative로 사용
  - (질문, 정답 스키마) 쌍만 있으면 학습 가능
  - 데이터가 적어도 효과 있음 (100개~)

데이터 흐름:
  Spider train split
    → (question, table_name, db_id)
    → schema_text = "table_name: col1 col2 ... (description)"
    → InputPair(question, schema_text)
    → MultipleNegativesRankingLoss 학습

실행:
  python finetune.py                        # 기본: 전체 train set, 3 epoch
  python finetune.py --n_samples 100        # 100개만
  python finetune.py --n_samples 1000 --epochs 5
  python finetune.py --output models/my_model
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

import config
from bench_datasets.spider_loader import load_schemas, load_samples


# ──────────────────────────────────────────────────────────────────────── #
#  스키마 텍스트 직렬화                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def _build_schema_registry(schemas) -> dict[str, dict[str, str]]:
    """
    { db_id → { table_name → schema_text } }
    schema_text = "table: col1 col2 col3 — description"
    """
    registry: dict[str, dict[str, str]] = {}
    for s in schemas:
        registry[s.db_id] = {}
        for table_name, info in s.tables.items():
            cols = " ".join(col for col, _ in info.get("columns", []))
            desc = info.get("description", "")
            text = f"{table_name}: {cols}"
            if desc:
                text += f" — {desc}"
            registry[s.db_id][table_name] = text
    return registry


# ──────────────────────────────────────────────────────────────────────── #
#  학습 데이터 구성                                                         #
# ──────────────────────────────────────────────────────────────────────── #

def build_training_pairs(
    samples,
    registry: dict[str, dict[str, str]],
    extra_json: str | None = None,
) -> list[dict]:
    """
    (question, schema_text) 쌍 리스트 반환.
    extra_json: 역생성 데이터 파일 경로 (선택)
    """
    pairs: list[dict] = []

    # Spider train split
    for s in samples:
        db_tables = registry.get(s.db_id, {})
        for table in s.used_tables:
            schema_text = db_tables.get(table)
            if schema_text:
                pairs.append({
                    "anchor":   s.question,
                    "positive": schema_text,
                })

    # 역생성 데이터 추가 (선택)
    if extra_json and Path(extra_json).exists():
        with open(extra_json, encoding="utf-8") as f:
            generated = json.load(f)
        for item in generated:
            db_tables = registry.get(item["db"], {})
            schema_text = db_tables.get(item["table"])
            if schema_text:
                pairs.append({
                    "anchor":   item["question"],
                    "positive": schema_text,
                })
        logger.info(f"역생성 데이터 {len(generated)}개 추가")

    logger.info(f"총 학습 쌍: {len(pairs)}개")
    return pairs


# ──────────────────────────────────────────────────────────────────────── #
#  파인튜닝                                                                 #
# ──────────────────────────────────────────────────────────────────────── #

def finetune(
    n_samples:   int | None,
    epochs:      int,
    batch_size:  int,
    output_path: str,
    base_model:  str,
    use_generated: bool,
    data_root:   str,
) -> str:
    """
    파인튜닝 실행 후 저장 경로 반환.
    """
    # ── 데이터 로드 ───────────────────────────────────────────────────
    spider_dir = Path(data_root) / "spider"
    logger.info("Spider 스키마 로드…")
    schemas = load_schemas(spider_dir)
    registry = _build_schema_registry(schemas)

    logger.info(f"Spider train split 로드 (max={n_samples})…")
    samples = load_samples(spider_dir, split="train", max_samples=n_samples)

    extra = "data/generated_queries.json" if use_generated else None
    pairs = build_training_pairs(samples, registry, extra_json=extra)

    if not pairs:
        raise ValueError("학습 쌍이 없습니다. 데이터 경로를 확인하세요.")

    # ── 모델 로드 ─────────────────────────────────────────────────────
    logger.info(f"베이스 모델 로드: {base_model}")
    model = SentenceTransformer(base_model)

    # ── Dataset 구성 ──────────────────────────────────────────────────
    dataset = Dataset.from_list(pairs)
    # train/eval 9:1 분리
    split    = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]

    logger.info(f"학습: {len(train_ds)}개 / 검증: {len(eval_ds)}개")

    # ── Loss ──────────────────────────────────────────────────────────
    # MultipleNegativesRankingLoss:
    #   배치 내 다른 (anchor, positive) 쌍을 자동으로 negative로 사용
    #   → 별도 negative 샘플 필요 없음
    loss = MultipleNegativesRankingLoss(model)

    # ── 학습 설정 ─────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    warmup_steps    = max(1, steps_per_epoch // 5)

    args = SentenceTransformerTrainingArguments(
        output_dir              = output_path,
        num_train_epochs        = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        warmup_steps            = warmup_steps,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        greater_is_better       = False,
        logging_steps           = max(1, steps_per_epoch // 4),
        fp16                    = torch.cuda.is_available(),
        dataloader_num_workers  = 0,
    )

    # ── 학습 실행 ─────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        loss            = loss,
    )

    logger.info(f"파인튜닝 시작 (epochs={epochs}, batch={batch_size})…")
    trainer.train()

    # ── 저장 ──────────────────────────────────────────────────────────
    final_path = str(Path(output_path) / "final")
    model.save(final_path)
    logger.info(f"모델 저장 완료: {final_path}")

    return final_path


# ──────────────────────────────────────────────────────────────────────── #
#  메인                                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Sentence-Transformer 파인튜닝")
    parser.add_argument("--n_samples",  type=int,   default=None,
                        help="train 샘플 수 (None=전체)")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--output",     type=str,   default="models/finetuned_spider")
    parser.add_argument("--base_model", type=str,   default=config.EMBEDDING_MODEL)
    parser.add_argument("--data_root",  type=str,   default="bench_datasets")
    parser.add_argument("--use_generated", action="store_true",
                        help="역생성 데이터(data/generated_queries.json) 추가")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Sentence-Transformer 파인튜닝")
    print(f"  베이스 모델: {args.base_model}")
    print(f"  train 샘플: {args.n_samples or '전체'}")
    print(f"  epochs: {args.epochs}  batch: {args.batch_size}")
    print(f"  역생성 데이터: {'포함' if args.use_generated else '미포함'}")
    print(f"  저장 경로: {args.output}")
    print("=" * 60)

    saved_path = finetune(
        n_samples    = args.n_samples,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        output_path  = args.output,
        base_model   = args.base_model,
        use_generated= args.use_generated,
        data_root    = args.data_root,
    )

    print(f"\n파인튜닝 완료. 모델 경로: {saved_path}")
    print(f"\n벤치마크에서 사용하려면:")
    print(f"  python benchmark_v2.py --datasets spider --model_path {saved_path}")


if __name__ == "__main__":
    main()
