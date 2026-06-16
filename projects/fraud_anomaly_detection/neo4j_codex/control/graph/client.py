"""Small Neo4j query boundary used by active graph discovery."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class GraphQueryRunner(Protocol):
    """Runs one Cypher/GDS query and returns row dictionaries."""

    def run(self, query: str, params: Mapping[str, object]) -> Sequence[Mapping[str, Any]]:
        ...


class Neo4jClient:
    """Thin Neo4j driver wrapper.

    The `neo4j` package is optional for repo tests, but required when running
    the active graph discovery CLI against a live Neo4j mirror.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str | None = None,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # pyright: ignore[reportMissingImports]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Neo4j graph discovery requires the `neo4j` package. "
                "Run with `uv run --with neo4j --group fraud ...`."
            ) from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    @classmethod
    def from_env(
        cls,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> "Neo4jClient":
        resolved_password = password or os.environ.get("NEO4J_PASSWORD")
        if not resolved_password:
            raise ValueError(
                "Neo4j password is required via --neo4j-password or NEO4J_PASSWORD."
            )
        return cls(
            uri=uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=user or os.environ.get("NEO4J_USER", "neo4j"),
            password=resolved_password,
            database=database or os.environ.get("NEO4J_DATABASE"),
        )

    def run(self, query: str, params: Mapping[str, object]) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            return [dict(record) for record in session.run(query, dict(params))]

    def close(self) -> None:
        self._driver.close()
