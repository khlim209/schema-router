"""
Neo4j schema initialisation script.

Run once before first use:
    python init_db.py

Creates constraints, indexes, and verifies connectivity.
"""

from __future__ import annotations

from loguru import logger
from graph_db.neo4j_client import Neo4jClient


def main():
    logger.info("Connecting to Neo4j…")
    client = Neo4jClient()

    logger.info("Creating constraints and indexes…")
    client.init_constraints()

    # Verify
    with client.session() as s:
        result = s.run("RETURN 1 AS ok")
        row = result.single()
        if row and row["ok"] == 1:
            logger.info("Neo4j connection verified ✓")
        else:
            logger.error("Neo4j connection verification failed!")

    client.close()
    logger.info("Database initialisation complete.")


if __name__ == "__main__":
    main()
