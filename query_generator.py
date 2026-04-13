"""
DBCopilot §3.3 스타일 역방향 질의 생성기
(Reverse Schema-to-Question Generation)

원리
----
수동으로 (질의, 정답 테이블) 쌍을 만드는 대신,
스키마 정보를 LLM에 넣고 "이 테이블을 조회해야 답할 수 있는 질문"을
역방향으로 자동 생성한다.

  스키마 입력 → LLM → 자연어 질문 + ground truth 레이블(db, table)

생성 전략 (DBCopilot §3.3)
---------------------------
1. Single-table  : 단일 테이블 질의 (테이블당 n개)
2. Multi-table   : JOIN 관계 있는 테이블 조합 질의 (join당 n개)
3. Paraphrase    : 동일 의도 다른 표현 (한국어/영어 교차)

백엔드
------
- Claude API  : ANTHROPIC_API_KEY 있으면 자동 사용
- Template    : API 없어도 동작하는 룰 기반 fallback
"""

from __future__ import annotations

import itertools
import os
import random
from dataclasses import dataclass

from loguru import logger

from graph_rag.indexer import SchemaDefinition


@dataclass
class GeneratedQuery:
    query_text: str
    db_name:    str
    table_name: str
    source:     str          # "llm" | "template"
    lang:       str          # "ko" | "en"


# ──────────────────────────────────────────────────────────────────────── #
#  스키마 직렬화  (DBCopilot §3.2 포맷)                                    #
# ──────────────────────────────────────────────────────────────────────── #

def _serialize_table(db: str, table: str, table_info: dict) -> str:
    """tablename(col:type, col:type, ...) 형식으로 직렬화."""
    cols = ", ".join(
        f"{c}:{t}" for c, t in table_info.get("columns", [])
    )
    desc = table_info.get("description", "")
    base = f"{table}({cols})"
    if desc:
        base += f"  -- {desc}"
    return base


def _serialize_join(
    db: str,
    table_a: str, info_a: dict,
    table_b: str, info_b: dict,
    via_col: str,
) -> str:
    s_a = _serialize_table(db, table_a, info_a)
    s_b = _serialize_table(db, table_b, info_b)
    return f"{s_a}\n{s_b}\n(JOIN via {via_col})"


# ──────────────────────────────────────────────────────────────────────── #
#  LLM 백엔드  (Claude API)                                                #
# ──────────────────────────────────────────────────────────────────────── #

def _llm_generate(schema_text: str, n: int, lang: str) -> list[str]:
    """
    Claude API로 역방향 질의 생성.
    OPENAI_API_KEY 또는 ANTHROPIC_API_KEY가 없으면 빈 리스트 반환 → 템플릿으로 fallback.
    우선순위: OpenAI → Anthropic → 템플릿
    """
    lang_instruction = "in English" if lang == "en" else "한국어로"

    prompt = f"""Look at the database schema below and generate {n} natural language questions \
that can be answered by querying this table(s). Write {lang_instruction}. \
Separate each question with a newline. Write questions only, no numbering or symbols.

Schema:
{schema_text}

Questions:"""

    # ── OpenAI 우선 ───────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            questions = [q.strip() for q in text.split("\n") if q.strip()]
            return questions[:n]
        except Exception as e:
            logger.warning(f"OpenAI API 오류: {e} — 다음 백엔드 시도")

    # ── Anthropic fallback ────────────────────────────────────────────
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            questions = [q.strip() for q in text.split("\n") if q.strip()]
            return questions[:n]
        except Exception as e:
            logger.warning(f"Anthropic API 오류: {e} — 템플릿으로 fallback")

    return []


# ──────────────────────────────────────────────────────────────────────── #
#  템플릿 기반 fallback                                                     #
# ──────────────────────────────────────────────────────────────────────── #

# {db}.{table} → (한국어 템플릿 목록, 영어 템플릿 목록)
_TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "orders": (
        [
            "지난달 {db} 총 주문 건수는?",
            "이번 주 신규 주문 현황 조회",
            "주문 상태별 건수 집계",
            "최근 {n}일간 일별 주문 추이",
            "취소된 주문 목록",
        ],
        [
            "How many orders were placed last month?",
            "Show recent order status summary",
            "Total order count by status",
            "Daily order trend for the past {n} days",
        ],
    ),
    "order_items": (
        [
            "가장 많이 팔린 상품 TOP {n}",
            "상품별 판매 수량 집계",
            "최근 {n}일 베스트셀러 목록",
            "카테고리별 판매량 순위",
            "이번 달 판매된 상품 종류 수",
        ],
        [
            "Top {n} best-selling products this month",
            "Product sales quantity ranking",
            "Best sellers in the last {n} days",
            "Sales volume by product category",
        ],
    ),
    "products": (
        [
            "재고가 {n}개 미만인 상품 목록",
            "카테고리별 평균 판매가",
            "가장 비싼 상품 TOP {n}",
            "상품 카탈로그 전체 목록",
        ],
        [
            "Products with stock below {n}",
            "Average price by product category",
            "Top {n} most expensive products",
        ],
    ),
    "customers": (
        [
            "이번 달 신규 가입 고객 수",
            "국가별 고객 분포",
            "최근 {n}일 내 가입한 고객 목록",
            "가입일 기준 고객 증감 추이",
        ],
        [
            "New customer registrations this month",
            "Customer distribution by country",
            "Customers who signed up in the last {n} days",
        ],
    ),
    "campaigns": (
        [
            "채널별 캠페인 예산 현황",
            "현재 진행 중인 마케팅 캠페인 목록",
            "최근 {n}개 캠페인 성과 요약",
        ],
        [
            "Marketing campaign budget by channel",
            "List of active campaigns",
            "Recent {n} campaign performance overview",
        ],
    ),
    "email_logs": (
        [
            "최근 이메일 오픈율 통계",
            "캠페인별 클릭 이벤트 수",
            "이메일 발송 대비 오픈율 비교",
            "지난 {n}일간 이메일 반응 현황",
        ],
        [
            "Email open rate statistics",
            "Click events by campaign",
            "Email open rate vs send rate",
            "Email engagement in the last {n} days",
        ],
    ),
    "ab_tests": (
        [
            "A/B 테스트 결과 요약",
            "실험별 전환율 비교",
            "최근 진행한 A/B 테스트 목록",
        ],
        [
            "A/B test results summary",
            "Conversion rate comparison by variant",
            "Recent A/B experiment outcomes",
        ],
    ),
    "daily_stats": (
        [
            "이번 주 일별 매출 트렌드",
            "최근 {n}일 활성 사용자 수 추이",
            "전월 대비 매출 증감률",
            "일별 신규 가입자 수 집계",
        ],
        [
            "Daily revenue trend this week",
            "Active user count for the past {n} days",
            "Revenue growth vs last month",
            "Daily new user registrations",
        ],
    ),
    "funnel_events": (
        [
            "구매 전환 퍼널 단계별 이탈률",
            "checkout 페이지 전환율",
            "사용자 퍼널 진행 현황 분석",
            "가장 이탈이 많은 퍼널 단계",
        ],
        [
            "Drop-off rate at each funnel stage",
            "Checkout page conversion rate",
            "User funnel progression analysis",
            "Highest drop-off step in conversion funnel",
        ],
    ),
    "cohorts": (
        [
            "월별 고객 유지율 분석",
            "{n}일 후 재방문율 코호트 분석",
            "신규 고객 리텐션 트렌드",
        ],
        [
            "Monthly customer retention rate",
            "Day-{n} cohort retention analysis",
            "New customer retention trend",
        ],
    ),
}


def _template_generate(
    table: str, n_ko: int, n_en: int
) -> tuple[list[str], list[str]]:
    ko_tmpl, en_tmpl = _TEMPLATES.get(table, ([], []))
    ns = [3, 5, 7, 10, 30]

    def fill(tmpl: str) -> str:
        return tmpl.format(db="", n=random.choice(ns)).strip()

    ko = [fill(t) for t in random.sample(ko_tmpl, min(n_ko, len(ko_tmpl)))]
    en = [fill(t) for t in random.sample(en_tmpl, min(n_en, len(en_tmpl)))]
    return ko, en


# ──────────────────────────────────────────────────────────────────────── #
#  Public API                                                               #
# ──────────────────────────────────────────────────────────────────────── #

class QueryGenerator:
    """
    DBCopilot §3.3 역방향 질의 생성기.
    LLM 사용 가능하면 Claude API, 불가하면 템플릿 기반으로 자동 전환.
    """

    def __init__(self, n_per_table: int = 5, n_per_join: int = 3):
        self.n_per_table = n_per_table
        self.n_per_join  = n_per_join
        self._use_llm    = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        if os.getenv("OPENAI_API_KEY"):
            mode = "OpenAI API"
        elif os.getenv("ANTHROPIC_API_KEY"):
            mode = "Claude API"
        else:
            mode = "Template fallback"
        logger.info(f"QueryGenerator 초기화 — 백엔드: {mode}")

    def generate_from_schemas(
        self, schemas: list[SchemaDefinition]
    ) -> list[GeneratedQuery]:
        """
        모든 스키마에서 (single-table + join) 질의를 자동 생성한다.
        """
        all_queries: list[GeneratedQuery] = []

        for schema in schemas:
            # ── Single-table 질의 ──────────────────────────────────────
            for table_name, table_info in schema.tables.items():
                qs = self._generate_single(schema.db_name, table_name, table_info)
                all_queries.extend(qs)

            # ── Multi-table (JOIN) 질의 ────────────────────────────────
            join_pairs = self._extract_join_pairs(schema)
            for (ta, ia), (tb, ib), via in join_pairs:
                qs = self._generate_join(schema.db_name, ta, ia, tb, ib, via)
                all_queries.extend(qs)

        random.shuffle(all_queries)
        logger.info(
            f"총 {len(all_queries)}개 질의 생성 완료 "
            f"({'LLM' if self._use_llm else 'Template'})"
        )
        return all_queries

    # ------------------------------------------------------------------ #
    #  Internal generators                                                 #
    # ------------------------------------------------------------------ #

    def _generate_single(
        self, db: str, table: str, info: dict
    ) -> list[GeneratedQuery]:
        schema_text = _serialize_table(db, table, info)
        results: list[GeneratedQuery] = []

        n_ko = max(2, self.n_per_table // 2)
        n_en = self.n_per_table - n_ko

        if self._use_llm:
            ko_qs = _llm_generate(schema_text, n_ko, "ko")
            en_qs = _llm_generate(schema_text, n_en, "en")
            src = "llm"
        else:
            ko_qs, en_qs = _template_generate(table, n_ko, n_en)
            src = "template"

        for q in ko_qs:
            results.append(GeneratedQuery(q, db, table, src, "ko"))
        for q in en_qs:
            results.append(GeneratedQuery(q, db, table, src, "en"))

        return results

    def _generate_join(
        self,
        db: str,
        ta: str, ia: dict,
        tb: str, ib: dict,
        via: str,
    ) -> list[GeneratedQuery]:
        """
        JOIN 질의는 첫 번째 테이블을 ground-truth 레이블로 사용한다.
        (두 테이블 모두 필요하지만 라우팅 목적상 진입점 테이블이 중요)
        """
        schema_text = _serialize_join(db, ta, ia, tb, ib, via)
        results: list[GeneratedQuery] = []

        if self._use_llm:
            qs = _llm_generate(schema_text, self.n_per_join, "ko")
            src = "llm"
        else:
            # JOIN 질의: 두 테이블 템플릿을 섞어서 사용
            ko_a, _ = _template_generate(ta, self.n_per_join, 0)
            src = "template"
            qs = ko_a[:self.n_per_join]

        for q in qs:
            results.append(GeneratedQuery(q, db, ta, src, "ko"))

        return results

    def _extract_join_pairs(
        self, schema: SchemaDefinition
    ) -> list[tuple[tuple, tuple, str]]:
        """스키마에서 JOIN 관계 쌍을 추출."""
        pairs = []
        tables = schema.tables
        for ta, ia in tables.items():
            for tb, via in ia.get("joins", []):
                if tb in tables and ta < tb:  # 중복 방지
                    pairs.append(((ta, ia), (tb, tables[tb]), via))
        return pairs
