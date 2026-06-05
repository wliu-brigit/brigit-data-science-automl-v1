"""Snowflake connection seam (sibling of gcs.py).

The single place that talks to the warehouse. Credentials come from the
environment only (.env via the project-config load_dotenv); this module
never logs or returns credential values. The connector choice
(snowflake-connector-python; Snowpark deliberately dropped — design §10)
is encapsulated here: if recipes ever need warehouse-side Python, swapping
is a one-file change.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd


REQUIRED_ENV = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
DEFAULT_WAREHOUSE = "DATA_SCIENCE_WH"
DEFAULT_ROLE = "DATA_SCIENCE_ROLE"


def missing_env() -> list[str]:
    """Names of required env vars that are unset/empty (for validate + errors)."""
    return [name for name in REQUIRED_ENV if not os.environ.get(name)]


def connection_params() -> dict[str, Any]:
    missing = missing_env()
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)} "
            "(set them in the repo-root .env)"
        )
    return {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE") or DEFAULT_WAREHOUSE,
        "role": os.environ.get("SNOWFLAKE_ROLE") or DEFAULT_ROLE,
    }


@contextmanager
def connect() -> Iterator[Any]:
    import snowflake.connector

    connection = snowflake.connector.connect(**connection_params())
    try:
        yield connection
    finally:
        connection.close()


def fetch_df(sql: str) -> pd.DataFrame:
    """Run a SELECT and return the result as pandas (Arrow-backed fetch)."""
    with connect() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetch_pandas_all()
        finally:
            cursor.close()


def fetch_one(sql: str) -> tuple | None:
    with connect() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetchone()
        finally:
            cursor.close()


def execute(sql: str) -> None:
    with connect() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
        finally:
            cursor.close()


def check_connection() -> None:
    """Cheapest possible live probe; driver errors propagate verbatim."""
    execute("SELECT 1")


__all__ = [
    "DEFAULT_ROLE",
    "DEFAULT_WAREHOUSE",
    "REQUIRED_ENV",
    "check_connection",
    "connect",
    "connection_params",
    "execute",
    "fetch_df",
    "fetch_one",
    "missing_env",
]
