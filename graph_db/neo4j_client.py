"""
Neo4j client for the GraphRAG Query Router.

Node labels:
  - Database   { name, description }
  - Table      { name, db_name, description }
  - Column     { name, type, table_name, db_name }
  - Query      { id, text, count, last_accessed, community_id }
  - QueryCluster { id, summary, size }

Relationships:
  - (Database)-[:HAS_TABLE]->(Table)
  - (Table)-[:HAS_COLUMN]->(Column)
  - (Table)-[:JOINS_WITH { via_column }]->(Table)
  - (Query)-[:ACCESSED { count, last_accessed }]->(Table)
  - (Query)-[:ACCESSED { count, last_accessed }]->(Database)
  - (Query)-[:BELONGS_TO]->(QueryCluster)
  - (QueryCluster)-[:COVERS { access_count }]->(Table)
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from loguru import logger
from neo4j import GraphDatabase, Session

import config


class Neo4jClient:
    def __init__(
        self,
        uri: str = config.NEO4J_URI,
        user: str = config.NEO4J_USER,
        password: str = config.NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        self._driver.close()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with self._driver.session() as s:
            yield s

    # ------------------------------------------------------------------ #
    #  Schema initialization                                               #
    # ------------------------------------------------------------------ #

    def init_constraints(self):
        """Create uniqueness constraints and indexes."""
        stmts = [
            "CREATE CONSTRAINT db_name IF NOT EXISTS FOR (d:Database) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT table_key IF NOT EXISTS FOR (t:Table) REQUIRE (t.name, t.db_name) IS UNIQUE",
            "CREATE CONSTRAINT query_id IF NOT EXISTS FOR (q:Query) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT cluster_id IF NOT EXISTS FOR (c:QueryCluster) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX query_text IF NOT EXISTS FOR (q:Query) ON (q.text)",
        ]
        with self.session() as s:
            for stmt in stmts:
                s.run(stmt)
        logger.info("Neo4j constraints and indexes initialized.")

    # ------------------------------------------------------------------ #
    #  Schema node management                                              #
    # ------------------------------------------------------------------ #

    def upsert_database(self, name: str, description: str = "") -> None:
        with self.session() as s:
            s.run(
                """
                MERGE (d:Database {name: $name})
                ON CREATE SET d.description = $desc, d.created_at = $now
                ON MATCH  SET d.description = $desc
                """,
                name=name, desc=description, now=datetime.utcnow().isoformat(),
            )

    def upsert_table(self, db_name: str, table_name: str, description: str = "") -> None:
        with self.session() as s:
            s.run(
                """
                MERGE (t:Table {name: $tname, db_name: $dname})
                ON CREATE SET t.description = $desc, t.created_at = $now
                ON MATCH  SET t.description = $desc
                WITH t
                MATCH (d:Database {name: $dname})
                MERGE (d)-[:HAS_TABLE]->(t)
                """,
                tname=table_name, dname=db_name,
                desc=description, now=datetime.utcnow().isoformat(),
            )

    def upsert_column(
        self, db_name: str, table_name: str, col_name: str, col_type: str = ""
    ) -> None:
        with self.session() as s:
            s.run(
                """
                MERGE (c:Column {name: $cname, table_name: $tname, db_name: $dname})
                ON CREATE SET c.type = $ctype, c.created_at = $now
                ON MATCH  SET c.type = $ctype
                WITH c
                MATCH (t:Table {name: $tname, db_name: $dname})
                MERGE (t)-[:HAS_COLUMN]->(c)
                """,
                cname=col_name, tname=table_name, dname=db_name,
                ctype=col_type, now=datetime.utcnow().isoformat(),
            )

    def upsert_join(
        self, db_name: str, table_a: str, table_b: str, via_column: str = ""
    ) -> None:
        """Register a foreign-key / join relationship between two tables."""
        with self.session() as s:
            s.run(
                """
                MATCH (a:Table {name: $ta, db_name: $db})
                MATCH (b:Table {name: $tb, db_name: $db})
                MERGE (a)-[r:JOINS_WITH {via_column: $via}]->(b)
                """,
                ta=table_a, tb=table_b, db=db_name, via=via_column,
            )

    # ------------------------------------------------------------------ #
    #  Query node management                                               #
    # ------------------------------------------------------------------ #

    def upsert_query(self, query_id: str, text: str) -> None:
        with self.session() as s:
            s.run(
                """
                MERGE (q:Query {id: $qid})
                ON CREATE SET q.text = $text, q.count = 1,
                              q.last_accessed = $now, q.community_id = -1
                ON MATCH  SET q.count = q.count + 1,
                              q.last_accessed = $now
                """,
                qid=query_id, text=text, now=datetime.utcnow().isoformat(),
            )

    def record_access(
        self,
        query_id: str,
        db_name: str,
        table_name: str,
        access_count: int = 1,
    ) -> None:
        """
        Record that query_id accessed (db_name, table_name).
        Increments weight on the ACCESSED edge.
        """
        now = datetime.utcnow().isoformat()
        with self.session() as s:
            # Query → Table
            s.run(
                """
                MATCH (q:Query {id: $qid})
                MATCH (t:Table {name: $tname, db_name: $dname})
                MERGE (q)-[r:ACCESSED]->(t)
                ON CREATE SET r.count = $cnt, r.last_accessed = $now
                ON MATCH  SET r.count = r.count + $cnt,
                              r.last_accessed = $now
                """,
                qid=query_id, tname=table_name, dname=db_name,
                cnt=access_count, now=now,
            )
            # Query → Database
            s.run(
                """
                MATCH (q:Query {id: $qid})
                MATCH (d:Database {name: $dname})
                MERGE (q)-[r:ACCESSED]->(d)
                ON CREATE SET r.count = $cnt, r.last_accessed = $now
                ON MATCH  SET r.count = r.count + $cnt,
                              r.last_accessed = $now
                """,
                qid=query_id, dname=db_name, cnt=access_count, now=now,
            )

    # ------------------------------------------------------------------ #
    #  Retrieval helpers                                                   #
    # ------------------------------------------------------------------ #

    def get_accessed_paths(
        self, query_id: str
    ) -> list[dict[str, Any]]:
        """
        Return all (db, table, access_count) paths for a given query.
        """
        with self.session() as s:
            result = s.run(
                """
                MATCH (q:Query {id: $qid})-[r:ACCESSED]->(t:Table)
                RETURN t.db_name AS db, t.name AS table, r.count AS count
                ORDER BY r.count DESC
                """,
                qid=query_id,
            )
            return [dict(row) for row in result]

    def get_schema_paths_for_queries(
        self, query_ids: list[str]
    ) -> list[dict[str, Any]]:
        """
        Batch fetch (query_id, db, table, access_count) for a list of queries.
        Used by the retriever to aggregate evidence.
        """
        with self.session() as s:
            result = s.run(
                """
                MATCH (q:Query)-[r:ACCESSED]->(t:Table)
                WHERE q.id IN $ids
                RETURN q.id AS query_id, t.db_name AS db,
                       t.name AS table, r.count AS count
                ORDER BY r.count DESC
                """,
                ids=query_ids,
            )
            return [dict(row) for row in result]

    def get_neighboring_tables(
        self, db_name: str, table_name: str, max_hops: int = 2
    ) -> list[dict[str, Any]]:
        """
        BFS on JOINS_WITH edges from a seed table.
        Returns neighboring tables with hop distance.
        """
        with self.session() as s:
            result = s.run(
                f"""
                MATCH path = (seed:Table {{name: $tname, db_name: $dname}})
                             -[:JOINS_WITH*1..{max_hops}]-(neighbor:Table)
                RETURN neighbor.db_name AS db, neighbor.name AS table,
                       length(path) AS hops
                ORDER BY hops
                """,
                tname=table_name, dname=db_name,
            )
            return [dict(row) for row in result]

    def get_cluster_covered_tables(
        self, community_id: int
    ) -> list[dict[str, Any]]:
        """Tables covered by a QueryCluster."""
        with self.session() as s:
            result = s.run(
                """
                MATCH (c:QueryCluster {id: $cid})-[r:COVERS]->(t:Table)
                RETURN t.db_name AS db, t.name AS table, r.access_count AS count
                ORDER BY r.access_count DESC
                """,
                cid=community_id,
            )
            return [dict(row) for row in result]

    def get_all_queries(self) -> list[dict[str, Any]]:
        """Return all query nodes (id, text, community_id, count)."""
        with self.session() as s:
            result = s.run(
                "MATCH (q:Query) RETURN q.id AS id, q.text AS text, "
                "q.community_id AS community_id, q.count AS count"
            )
            return [dict(row) for row in result]

    def set_query_community(self, query_id: str, community_id: int) -> None:
        with self.session() as s:
            s.run(
                "MATCH (q:Query {id: $qid}) SET q.community_id = $cid",
                qid=query_id, cid=community_id,
            )

    def upsert_cluster(
        self, community_id: int, summary: str, size: int
    ) -> None:
        with self.session() as s:
            s.run(
                """
                MERGE (c:QueryCluster {id: $cid})
                SET c.summary = $summary, c.size = $size,
                    c.updated_at = $now
                """,
                cid=community_id, summary=summary,
                size=size, now=datetime.utcnow().isoformat(),
            )

    def update_cluster_coverage(
        self, community_id: int, db_name: str, table_name: str, access_count: int
    ) -> None:
        with self.session() as s:
            s.run(
                """
                MATCH (c:QueryCluster {id: $cid})
                MATCH (t:Table {name: $tname, db_name: $dname})
                MERGE (c)-[r:COVERS]->(t)
                ON CREATE SET r.access_count = $cnt
                ON MATCH  SET r.access_count = r.access_count + $cnt
                """,
                cid=community_id, tname=table_name,
                dname=db_name, cnt=access_count,
            )

    def get_dfs_schema_paths(self, db_name: str) -> list[str]:
        """
        DBCopilot-inspired: serialize the schema graph via DFS traversal.
        Returns a list of path strings like 'db.table.column'.
        """
        with self.session() as s:
            result = s.run(
                """
                MATCH (d:Database {name: $db})-[:HAS_TABLE]->(t:Table)
                OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                RETURN d.name AS db, t.name AS tbl, collect(c.name) AS cols
                ORDER BY t.name
                """,
                db=db_name,
            )
            paths: list[str] = []
            for row in result:
                for col in row["cols"]:
                    paths.append(f"{row['db']}.{row['tbl']}.{col}")
                if not row["cols"]:
                    paths.append(f"{row['db']}.{row['tbl']}")
            return paths
