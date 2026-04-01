"""Query helpers for graph + vector data stored in SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from cindex.services.embeddings import generate_embedding
from cindex.services.indexing.sqlite_store import _load_sqlite_vector_extension
from cindex.services.indexing.sqlite_store import vector_to_blob

def execute_sql_query(db_path: Path, sql: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Execute a SQL statement against a read-only connection and return column names and rows.

    The connection is opened with SQLite's ``mode=ro`` URI flag so the engine
    itself enforces the read-only constraint — no string-prefix heuristic is
    needed or used.
    """
    ro_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(ro_uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        columns = [description[0] for description in (cursor.description or [])]
        rows = [dict(row) for row in cursor.fetchall()]
    return columns, rows


def find_similar_vertices(
    db_path: Path,
    query_text: str,
    *,
    model_name: str,
    cache_folder: str | None = None,
    k: int = 10,
    label: str | None = None,
    exclude_external: bool = True,
    vector: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Return the nearest stored vertices to *query_text*.

    By default external import vertices are excluded from results since they
    carry no source location and are rarely useful in search output.  Pass
    ``exclude_external=False`` to include them.

    If *vector* is supplied it is used directly and the local embedding model
    is not loaded.  This allows callers to delegate embedding to a running
    ``cindex serve`` instance and avoid the cold-load penalty.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if not _load_sqlite_vector_extension(conn):
            raise RuntimeError("Failed to load the sqlite-vector extension.")

        dimension = _get_embedding_dimension(conn)
        if dimension is None:
            raise ValueError("No stored vertex embeddings found in the SQLite database.")

        _prepare_vector_search(conn, dimension)

        if vector is None:
            vector = generate_embedding(
                query_text,
                model_name=model_name,
                cache_folder=cache_folder,
            )
        query_blob = vector_to_blob(vector)

        clauses: list[str] = []
        params: list[Any] = [query_blob]
        if label:
            clauses.append("v.label = ?")
            params.append(label)
        if exclude_external:
            clauses.append("v.label != 'external'")
        where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        sql = f"""
        SELECT
          v.id AS vertex_id,
          v.label,
                    json_extract(v.properties_json, '$.name') AS name,
                    json_extract(v.properties_json, '$.path') AS path,
                    json_extract(v.properties_json, '$.line') AS line,
          json_extract(v.properties_json, '$.docstring') AS docstring,
          json_extract(v.properties_json, '$.source') AS source,
          ve.model_name,
          nn.distance
        FROM vector_full_scan('vertex_embeddings', 'embedding', ?, {k}) AS nn
        JOIN vertex_embeddings ve ON ve.rowid = nn.rowid
        JOIN vertices v ON v.id = ve.vertex_id
        {where_clause}
        ORDER BY nn.distance
        LIMIT {k}
        """
        return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]

def _get_embedding_dimension(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT dimension FROM vertex_embeddings LIMIT 1").fetchone()
    if row is None:
        return None
    return int(row[0])


def _prepare_vector_search(conn: sqlite3.Connection, dimension: int) -> None:
    config = f"type=FLOAT32,dimension={dimension},distance=COSINE"
    conn.execute("SELECT vector_init('vertex_embeddings', 'embedding', ?)", (config,))


