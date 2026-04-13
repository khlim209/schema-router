"""
SQL 쿼리에서 사용된 테이블 이름을 추출하는 경량 파서.

정규식 기반으로 FROM / JOIN 절의 테이블을 추출한다.
Spider / Bird / FIBEN SQL 구문을 커버할 수 있도록 작성됨.
"""

from __future__ import annotations

import re


_TABLE_PATTERN = re.compile(
    r"""
    (?:FROM|JOIN)\s+          # FROM 또는 JOIN 키워드 (대소문자 무시)
    `?\"?                     # 선택적 백틱/따옴표
    ([a-zA-Z_][a-zA-Z0-9_]*) # 테이블 이름
    `?\"?                     # 선택적 백틱/따옴표
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_tables(sql: str) -> list[str]:
    """
    SQL 문자열에서 사용된 테이블 이름을 소문자로 반환한다.
    서브쿼리 포함, 중복 제거, 순서 유지.
    """
    matches = _TABLE_PATTERN.findall(sql)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        t = m.lower()
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result
