"""
Base vs Fine-tuned 모델 빠른 비교
===================================

Spider dev set에서 SchemaRAG(순수 임베딩 유사도)로
base 모델과 파인튜닝된 모델을 비교한다.

Neo4j 불필요 — 임베딩 유사도만 사용.

실행:
  python compare_models.py
  python compare_models.py --n_samples 200 --finetuned models/finetuned_spider/final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

import config
from bench_datasets.spider_loader import load_schemas, load_samples


# ──────────────────────────────────────────────────────────────────────── #
#  스키마 텍스트 직렬화 (baseline_rag.py 동일 포맷)                         #
# ──────────────────────────────────────────────────────────────────────── #

def _build_schema_texts(schemas) -> tuple[list[tuple[str, str]], list[str]]:
    """(entries, texts) — entries[i] = (db_id, table_name)"""
    entries, texts = [], []
    for s in schemas:
        for table_name, info in s.tables.items():
            cols = " ".join(c for c, _ in info.get("columns", []))
            desc = info.get("description", "")
            text = f"{s.db_id}.{table_name}: {cols}"
            if desc:
                text += f" {desc}"
            entries.append((s.db_id, table_name))
            texts.append(text)
    return entries, texts


# ──────────────────────────────────────────────────────────────────────── #
#  평가                                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def evaluate(
    model: SentenceTransformer,
    entries: list[tuple[str, str]],
    schema_vecs: np.ndarray,
    samples,
    label: str,
) -> dict:
    """
    dev samples에 대해 hit@1, hit@3, MRR, avg_lookup 계산.
    """
    # FAISS 인덱스 구축
    dim = schema_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(schema_vecs)

    hit1 = hit3 = mrr_sum = lookup_sum = 0
    total = 0

    for s in samples:
        q_vec = model.encode(s.question, normalize_embeddings=True).reshape(1, -1).astype(np.float32)
        D, I = index.search(q_vec, len(entries))

        ranked = [entries[i] for i in I[0]]
        correct = set((s.db_id, t) for t in s.used_tables)

        # hit@1
        if ranked[0] in correct:
            hit1 += 1

        # hit@3 & MRR & avg_lookup
        for rank, pair in enumerate(ranked, 1):
            if pair in correct:
                if rank <= 3:
                    hit3 += 1
                mrr_sum += 1.0 / rank
                lookup_sum += rank
                break

        total += 1

    return {
        "label":       label,
        "n":           total,
        "hit@1":       round(hit1 / total * 100, 1),
        "hit@3":       round(hit3 / total * 100, 1),
        "MRR":         round(mrr_sum / total, 4),
        "avg_lookup":  round(lookup_sum / total, 1),
    }


# ──────────────────────────────────────────────────────────────────────── #
#  메인                                                                     #
# ──────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Base vs Fine-tuned 모델 비교")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="dev 샘플 수 (None=전체)")
    parser.add_argument("--finetuned",  type=str, default="models/finetuned_spider/final",
                        help="파인튜닝된 모델 경로")
    parser.add_argument("--base",       type=str, default=config.EMBEDDING_MODEL,
                        help="베이스 모델")
    parser.add_argument("--data_root",  type=str, default="bench_datasets")
    args = parser.parse_args()

    spider_dir = Path(args.data_root) / "spider"

    # ── 데이터 로드 ───────────────────────────────────────────────────
    logger.info("Spider 스키마 로드…")
    schemas = load_schemas(spider_dir)
    entries, texts = _build_schema_texts(schemas)
    logger.info(f"총 {len(entries)}개 테이블")

    logger.info(f"Spider dev split 로드 (max={args.n_samples})…")
    samples = load_samples(spider_dir, split="dev", max_samples=args.n_samples)
    logger.info(f"dev 샘플: {len(samples)}개")

    results = []

    # ── Base 모델 ─────────────────────────────────────────────────────
    logger.info(f"베이스 모델 평가: {args.base}")
    base_model = SentenceTransformer(args.base)
    base_vecs  = base_model.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=True, batch_size=128)
    base_vecs  = base_vecs.astype(np.float32)
    results.append(evaluate(base_model, entries, base_vecs, samples, "Base"))

    # ── Fine-tuned 모델 ───────────────────────────────────────────────
    ft_path = args.finetuned
    if Path(ft_path).exists():
        logger.info(f"파인튜닝 모델 평가: {ft_path}")
        ft_model = SentenceTransformer(ft_path)
        ft_vecs  = ft_model.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=True, batch_size=128)
        ft_vecs  = ft_vecs.astype(np.float32)
        results.append(evaluate(ft_model, entries, ft_vecs, samples, "Fine-tuned"))
    else:
        logger.warning(f"파인튜닝 모델 없음: {ft_path}")

    # ── 결과 출력 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Spider dev 비교 (n={len(samples)})")
    print("=" * 60)
    print(f"{'모델':<18} {'hit@1':>6} {'hit@3':>6} {'MRR':>7} {'avg_lookup':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<18} {r['hit@1']:>5}% {r['hit@3']:>5}% {r['MRR']:>7.4f} {r['avg_lookup']:>10.1f}")
    print("-" * 60)

    if len(results) == 2:
        b, f = results[0], results[1]
        print(f"\n파인튜닝 효과:")
        print(f"  hit@1  : {b['hit@1']}% → {f['hit@1']}%  ({f['hit@1']-b['hit@1']:+.1f}%p)")
        print(f"  hit@3  : {b['hit@3']}% → {f['hit@3']}%  ({f['hit@3']-b['hit@3']:+.1f}%p)")
        print(f"  MRR    : {b['MRR']:.4f} → {f['MRR']:.4f}  ({f['MRR']-b['MRR']:+.4f})")
        print(f"  avg_lookup: {b['avg_lookup']} → {f['avg_lookup']}  ({f['avg_lookup']-b['avg_lookup']:+.1f})")


if __name__ == "__main__":
    main()
