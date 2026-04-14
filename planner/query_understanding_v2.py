from __future__ import annotations

import re
from collections import Counter

from planner.models import QueryConstraint, QueryUnderstanding


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "last", "many", "of", "on", "or", "show",
    "that", "the", "this", "to", "what", "which", "with",
    "\uac00", "\uacfc", "\ub294", "\ub3c4", "\ub97c", "\uc5d0", "\uc640",
    "\uc740", "\uc758", "\uc774", "\uc790", "\uc880",
    "\ucd5c\uadfc", "\uc9c0\ub09c", "\uc774\ubc88", "\uc804\uccb4",
    "\ubaa8\ub4e0", "\uc870\ud68c", "\uc9c8\ubb38",
}

_ENTITY_ALIASES = {
    "customer": {
        "customer", "customers", "user", "users", "member", "buyer",
        "\uace0\uac1d", "\ud68c\uc6d0", "\uc0ac\uc6a9\uc790",
    },
    "order": {
        "order", "orders", "purchase", "purchases", "transaction", "transactions",
        "\uc8fc\ubb38", "\uad6c\ub9e4", "\uac70\ub798", "\uacb0\uc81c",
    },
    "product": {
        "product", "products", "item", "items", "sku", "goods",
        "\uc0c1\ud488", "\uc81c\ud488", "\uce74\ud14c\uace0\ub9ac",
    },
    "campaign": {
        "campaign", "campaigns", "marketing", "promotion",
        "\ucea0\ud398\uc778", "\ud504\ub85c\ubaa8\uc158", "\ub9c8\ucf00\ud305",
    },
    "email": {"email", "emails", "mail", "mails", "\uc774\uba54\uc77c", "\uba54\uc77c"},
    "payment": {
        "payment", "payments", "charge", "charges", "refund",
        "\uc815\uc0b0", "\uacb0\uc81c", "\ud658\ubd88",
    },
    "funnel": {
        "funnel", "conversion", "conversions", "checkout",
        "\uc804\ud658", "\ud37c\ub110", "\uc774\ud0c8",
    },
    "cohort": {
        "cohort", "cohorts", "retention", "retentions",
        "\ucf54\ud638\ud2b8", "\ub9ac\ud150\uc158",
    },
}

_FACT_ALIASES = {
    "revenue": {"revenue", "sales", "income", "gmv", "\ub9e4\ucd9c", "\uc218\uc775"},
    "order_count": {"order_count", "count", "orders", "\uc8fc\ubb38\uc218", "\uac74\uc218"},
    "purchase_count": {"purchase_count", "repurchase", "repeat", "\uc7ac\uad6c\ub9e4", "\ubc18\ubcf5\uad6c\ub9e4"},
    "last_purchase_date": {"last_purchase", "last_order", "\ucd5c\uadfc\uad6c\ub9e4", "\ub9c8\uc9c0\ub9c9\uad6c\ub9e4"},
    "days_since_last_purchase": {"recency", "days_since", "\ud734\uba74", "\ucd5c\uadfc\uc131"},
    "campaign_response": {"response", "open_rate", "click_rate", "ctr", "\uc751\ub2f5", "\ubc18\uc751", "\uc624\ud508", "\ud074\ub9ad"},
    "product_category_overlap": {"category_overlap", "mix", "affinity", "\uce74\ud14c\uace0\ub9ac", "\uad50\ucc28\uad6c\ub9e4"},
    "conversion_rate": {"conversion_rate", "conversion", "\uc804\ud658\uc728", "\uc804\ud658"},
    "retention": {"retention", "retained", "\uc720\uc9c0\uc728", "\ub9ac\ud150\uc158"},
    "signup_count": {"signup", "registration", "\uac00\uc785", "\ub4f1\ub85d"},
    "average_order_value": {"aov", "avg_order", "average_order_value", "\uac1d\ub2e8\uac00", "\ud3c9\uade0\uc8fc\ubb38\uae08\uc561"},
    "status_breakdown": {"status", "state", "\uc0c1\ud0dc"},
}

_AGGREGATION_ALIASES = {
    "count": {"count", "number", "\uba87", "\uac74\uc218", "\uc218", "\ud569\uacc4"},
    "sum": {"sum", "total", "\ucd1d", "\ub204\uc801"},
    "average": {"average", "avg", "mean", "\ud3c9\uade0"},
    "top_k": {"top", "best", "highest", "largest", "\uc0c1\uc704", "\ubca0\uc2a4\ud2b8", "\uac00\uc7a5"},
    "pattern": {"pattern", "common", "shared", "\uacf5\ud1b5", "\ud328\ud134", "\ud2b9\uc9d5"},
}

_TEMPORAL_PATTERNS = [
    (r"\blast\s+\d+\s+(day|days|week|weeks|month|months|quarter|quarters|year|years)\b", "relative_window"),
    (r"\bthis\s+(week|month|quarter|year)\b", "current_window"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "date_literal"),
    ("(\ucd5c\uadfc|\uc9c0\ub09c|\uc774\ubc88)\\s*\\d+\\s*(\uc77c|\uc8fc|\uac1c\uc6d4|\ub2ec|\ubd84\uae30|\ub144)", "relative_window"),
    ("(\ucd5c\uadfc|\uc9c0\ub09c|\uc774\ubc88)\\s*(\uc8fc|\ub2ec|\uac1c\uc6d4|\ubd84\uae30|\ub144)", "relative_window"),
]

_LIMIT_PATTERNS = [
    re.compile(r"\btop\s+(\d+)\b"),
    re.compile("(\\d+)\\s*(\uac1c|\uba85|\uac74)"),
]


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[_/.-]", " ", text.lower())
    normalized = re.sub("[^\\w\\s\\uac00-\\ud7a3]", " ", normalized)
    return [token for token in normalized.split() if token and token not in _STOPWORDS]


def _term_matches(token: str, alias: str) -> bool:
    return token == alias or token in alias or alias in token


def _has_alias_match(tokens: list[str], aliases: set[str]) -> bool:
    return any(_term_matches(token, alias) for token in tokens for alias in aliases)


def _schema_terms(schema_registry: dict[str, dict[str, dict]]) -> set[str]:
    terms: set[str] = set()
    for db_name, tables in schema_registry.items():
        terms.update(_tokenize(db_name))
        for table_name, info in tables.items():
            terms.update(_tokenize(table_name))
            terms.update(_tokenize(info.get("description", "")))
            for column_name, _ in info.get("columns", []):
                terms.update(_tokenize(column_name))
    return terms


class QueryDecomposer:
    """Heuristic semantic planner for schema traversal queries."""

    def decompose(
        self,
        query: str,
        schema_registry: dict[str, dict[str, dict]] | None = None,
    ) -> QueryUnderstanding:
        registry = schema_registry or {}
        tokens = _tokenize(query)
        schema_terms = _schema_terms(registry)

        entities = self._extract_entities(tokens)
        facts = self._extract_facts(tokens)
        constraints = self._extract_constraints(query)
        matched_terms = sorted(
            {
                term
                for term in schema_terms
                if any(_term_matches(token, term) for token in tokens)
            }
        )

        if not facts:
            facts = matched_terms[:4]
        if not entities:
            entities = self._fallback_entities(tokens, matched_terms)

        intent = self._build_intent(entities, facts, tokens)
        return QueryUnderstanding(
            query=query,
            intent=intent,
            entities=entities,
            facts=facts[:6],
            constraints=constraints,
            tokens=tokens,
            matched_schema_terms=matched_terms[:10],
        )

    def _extract_entities(self, tokens: list[str]) -> list[str]:
        entities: list[str] = []
        for canonical, aliases in _ENTITY_ALIASES.items():
            if _has_alias_match(tokens, aliases):
                entities.append(canonical)
        return entities

    def _extract_facts(self, tokens: list[str]) -> list[str]:
        facts: list[str] = []
        for canonical, aliases in _FACT_ALIASES.items():
            if _has_alias_match(tokens, aliases):
                facts.append(canonical)
        for canonical, aliases in _AGGREGATION_ALIASES.items():
            if _has_alias_match(tokens, aliases):
                facts.append(canonical)
        deduped: list[str] = []
        for fact in facts:
            if fact not in deduped:
                deduped.append(fact)
        return deduped

    def _extract_constraints(self, query: str) -> list[QueryConstraint]:
        constraints: list[QueryConstraint] = []
        lowered = query.lower()

        for pattern, kind in _TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, lowered):
                constraints.append(
                    QueryConstraint(kind=kind, value=match.group(0), matched_text=match.group(0))
                )

        for pattern in _LIMIT_PATTERNS:
            match = pattern.search(lowered)
            if match:
                constraints.append(
                    QueryConstraint(kind="limit", value=match.group(1), matched_text=match.group(0))
                )

        if any(
            token in lowered
            for token in (
                "group",
                "segment",
                "cohort",
                "\uadf8\ub8f9",
                "\uc138\uadf8\uba3c\ud2b8",
                "\ucf54\ud638\ud2b8",
            )
        ):
            constraints.append(QueryConstraint(kind="grouping", value="segment", matched_text="segment"))
        return constraints

    def _fallback_entities(self, tokens: list[str], matched_terms: list[str]) -> list[str]:
        counts = Counter(token for token in tokens if token in matched_terms)
        return [token for token, _ in counts.most_common(3)]

    def _build_intent(self, entities: list[str], facts: list[str], tokens: list[str]) -> str:
        intent_tokens = entities[:2] + facts[:2]
        if not intent_tokens:
            intent_tokens = tokens[:3]
        return "_".join(intent_tokens) if intent_tokens else "schema_exploration"
