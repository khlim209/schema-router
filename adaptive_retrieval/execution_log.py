from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .models import ExecutionFeedback, RetrievalPlan


class ExecutionLogger:
    """
    JSONL logger for retrieval planning and downstream MCP execution feedback.
    """

    def __init__(self, path: str | Path = "data/execution_logs/index_graph_runs.jsonl"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_plan(
        self,
        plan: RetrievalPlan,
        gold_tables: list[str] | None = None,
    ) -> str:
        run_id = plan.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        payload = {
            "event": "retrieval_plan",
            "timestamp": self._timestamp(),
            "run_id": run_id,
            "plan": plan.to_dict(),
            "gold_tables": list(gold_tables or []),
        }
        self._append(payload)
        logger.debug(f"ExecutionLogger: stored retrieval plan {run_id}")
        return run_id

    def log_feedback(self, feedback: ExecutionFeedback) -> None:
        payload = {
            "event": "execution_feedback",
            "timestamp": self._timestamp(),
            "feedback": feedback.to_dict(),
        }
        self._append(payload)
        logger.debug(f"ExecutionLogger: stored execution feedback {feedback.run_id}")

    def _append(self, payload: dict) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @property
    def path(self) -> str:
        return str(self._path)
