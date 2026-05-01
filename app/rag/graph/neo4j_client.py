from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase

from app.core.config import settings
from app.core.logging import logger
from app.core.resilience import neo4j_breaker, retryable_external, with_breaker
from app.models.graph import GraphExtraction, GraphNode, GraphRelationship


class Neo4jClient:
    def __init__(self) -> None:
        self._driver = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._driver.verify_connectivity()
        await self._migrate_constraints()
        logger.info("Neo4j connected")

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def _migrate_constraints(self) -> None:
        """
        Drop any legacy UNIQUE(name) constraints and install composite
        UNIQUE(name, project) constraints. A node is uniquely identified by
        (name, project) so the same entity name can coexist across projects.
        """
        legacy = [
            "DROP CONSTRAINT module_name IF EXISTS",
            "DROP CONSTRAINT entity_name IF EXISTS",
            "DROP CONSTRAINT api_name IF EXISTS",
            "DROP CONSTRAINT uiscreen_name IF EXISTS",
        ]
        composite = [
            "CREATE CONSTRAINT module_name_project IF NOT EXISTS "
            "FOR (n:Module) REQUIRE (n.name, n.project) IS UNIQUE",
            "CREATE CONSTRAINT entity_name_project IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE (n.name, n.project) IS UNIQUE",
            "CREATE CONSTRAINT api_name_project IF NOT EXISTS "
            "FOR (n:API) REQUIRE (n.name, n.project) IS UNIQUE",
            "CREATE CONSTRAINT uiscreen_name_project IF NOT EXISTS "
            "FOR (n:UIScreen) REQUIRE (n.name, n.project) IS UNIQUE",
        ]
        backfill = (
            "MATCH (n) WHERE n.project IS NULL "
            "SET n.project = 'default'"
        )
        async with self._driver.session(database=settings.neo4j_database) as session:  # type: ignore[union-attr]
            for s in legacy:
                try:
                    await session.run(s)
                except Exception as exc:
                    logger.debug(f"legacy constraint drop skipped: {exc}")
            try:
                await session.run(backfill)
            except Exception as exc:
                logger.warning(f"Neo4j backfill of project='default' failed: {exc}")
            for s in composite:
                try:
                    await session.run(s)
                except Exception as exc:
                    logger.warning(f"Neo4j composite constraint failed: {exc}")

    @retryable_external()
    @with_breaker(neo4j_breaker, "Neo4j")
    async def upsert_extraction(
        self, extraction: GraphExtraction, project: str
    ) -> tuple[int, int]:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")

        nodes_count = 0
        rels_count = 0
        async with self._driver.session(database=settings.neo4j_database) as session:
            for node in extraction.nodes:
                await session.execute_write(self._merge_node, node, project)
                nodes_count += 1
            for rel in extraction.relationships:
                await session.execute_write(self._merge_relationship, rel, project)
                rels_count += 1
        return nodes_count, rels_count

    @staticmethod
    async def _merge_node(tx, node: GraphNode, project: str) -> None:
        props = dict(node.properties or {})
        props["project"] = project
        query = (
            f"MERGE (n:{node.label.value} {{name: $name, project: $project}}) "
            "SET n += $props"
        )
        await tx.run(query, name=node.name, project=project, props=props)

    @staticmethod
    async def _merge_relationship(tx, rel: GraphRelationship, project: str) -> None:
        props = dict(rel.properties or {})
        if rel.condition is not None:
            props["condition"] = rel.condition
        if rel.action is not None:
            props["action"] = rel.action
        if rel.trigger_point is not None:
            props["trigger_point"] = rel.trigger_point
        props["project"] = project

        query = (
            f"MERGE (a:{rel.from_label.value} {{name: $from_name, project: $project}}) "
            f"MERGE (b:{rel.to_label.value} {{name: $to_name, project: $project}}) "
            f"MERGE (a)-[r:{rel.type.value}]->(b) "
            "SET r += $props"
        )
        await tx.run(
            query,
            from_name=rel.from_name,
            to_name=rel.to_name,
            project=project,
            props=props,
        )

    @retryable_external()
    @with_breaker(neo4j_breaker, "Neo4j")
    async def delete_project(self, project: str) -> tuple[int, int]:
        """Cascade-delete all nodes (and their relationships) for a project."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")
        count_cypher = (
            "MATCH (n {project: $project}) "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS rels"
        )
        delete_cypher = "MATCH (n {project: $project}) DETACH DELETE n"
        async with self._driver.session(database=settings.neo4j_database) as session:
            res = await session.run(count_cypher, project=project)
            rec = await res.single()
            nodes = int(rec["nodes"]) if rec else 0
            rels = int(rec["rels"]) if rec else 0
            await session.run(delete_cypher, project=project)
        return nodes, rels

    @retryable_external()
    @with_breaker(neo4j_breaker, "Neo4j")
    async def fetch_dependencies(
        self,
        project: str,
        modules: list[str] | None = None,
        keywords: list[str] | None = None,
        depth: int = 2,
        limit: int = 50,
    ) -> dict[str, Any]:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")

        modules = [m for m in (modules or []) if m]
        keywords = [k for k in (keywords or []) if k]

        clauses = ["n.project = $project"]
        params: dict[str, Any] = {"limit": limit, "project": project}
        sub_clauses: list[str] = []
        if modules:
            sub_clauses.append("any(m IN $modules WHERE toLower(n.name) CONTAINS toLower(m))")
            params["modules"] = modules
        if keywords:
            sub_clauses.append("any(k IN $keywords WHERE toLower(n.name) CONTAINS toLower(k))")
            params["keywords"] = keywords
        if sub_clauses:
            clauses.append("(" + " OR ".join(sub_clauses) + ")")
        where = " AND ".join(clauses)
        max_depth = max(1, min(depth, 4))

        cypher = f"""
        MATCH (n)
        WHERE {where}
        WITH collect(DISTINCT n) AS seeds
        UNWIND seeds AS s
        OPTIONAL MATCH p = (s)-[*1..{max_depth}]-(m)
        WHERE all(x IN nodes(p) WHERE x.project = $project)
        WITH seeds, collect(DISTINCT p) AS paths
        WITH seeds, paths,
             [p IN paths | relationships(p)] AS rel_lists,
             [p IN paths | nodes(p)] AS node_lists
        WITH seeds,
             apoc.coll.toSet([n IN apoc.coll.flatten(node_lists + [seeds]) | n]) AS all_nodes,
             apoc.coll.toSet(apoc.coll.flatten(rel_lists)) AS all_rels
        RETURN all_nodes, all_rels
        LIMIT $limit
        """

        async with self._driver.session(database=settings.neo4j_database) as session:
            try:
                result = await session.run(cypher, params)
                record = await result.single()
                nodes = record["all_nodes"] if record else []
                rels = record["all_rels"] if record else []
            except Exception:
                plain = f"""
                MATCH (n)
                WHERE {where}
                OPTIONAL MATCH (n)-[r]-(m)
                WHERE m.project = $project
                RETURN n, r, m
                LIMIT $limit
                """
                res = await session.run(plain, params)
                nodes_map: dict[int, Any] = {}
                rels_out: list[dict[str, Any]] = []
                async for rec in res:
                    for key in ("n", "m"):
                        nd = rec.get(key)
                        if nd is not None:
                            nodes_map[nd.element_id] = nd  # type: ignore[attr-defined]
                    r = rec.get("r")
                    if r is not None:
                        rels_out.append(_serialize_rel(r))
                return {
                    "nodes": [_serialize_node(n) for n in nodes_map.values()],
                    "relationships": rels_out,
                }

            return {
                "nodes": [_serialize_node(n) for n in nodes],
                "relationships": [_serialize_rel(r) for r in rels],
            }


def _serialize_node(node) -> dict[str, Any]:
    return {
        "labels": list(node.labels),
        "name": node.get("name"),
        "properties": dict(node),
    }


def _serialize_rel(rel) -> dict[str, Any]:
    return {
        "type": rel.type,
        "start": rel.start_node.get("name") if rel.start_node else None,
        "end": rel.end_node.get("name") if rel.end_node else None,
        "properties": dict(rel),
    }


_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
