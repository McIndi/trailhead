"""SQLite persistence helpers for PropertyGraph.

This module lets cindex behave like a single-file database workflow:
index in memory, then save/load the graph from one sqlite file on disk.
"""
from __future__ import annotations

import importlib.resources
import json
import sqlite3
import struct
from pathlib import Path

from cindex.services.embeddings import generate_embeddings
from cindex.services.indexing.graph import PropertyGraph
from cindex.services.indexing.graph import Vertex

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vertices (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  properties_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  out_v_id TEXT NOT NULL,
  in_v_id TEXT NOT NULL,
  properties_json TEXT NOT NULL,
    FOREIGN KEY(out_v_id) REFERENCES vertices(id) ON DELETE CASCADE,
    FOREIGN KEY(in_v_id) REFERENCES vertices(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS vertex_embeddings (
    vertex_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    source_text TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY(vertex_id) REFERENCES vertices(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vertices_label ON vertices(label);
CREATE INDEX IF NOT EXISTS idx_edges_label ON edges(label);
CREATE INDEX IF NOT EXISTS idx_edges_out_v ON edges(out_v_id);
CREATE INDEX IF NOT EXISTS idx_edges_in_v ON edges(in_v_id);
CREATE INDEX IF NOT EXISTS idx_vertex_embeddings_model ON vertex_embeddings(model_name);
"""

SUPPORTED_VECTOR_VERTEX_LABELS: frozenset[str] = frozenset(
        {"module", "class", "function", "external"}
)


def persist_graph(
    graph: PropertyGraph,
    db_path: Path,
    *,
    append: bool = False,
) -> tuple[int, int]:
    """Persist *graph* to *db_path*.

    Returns a tuple: (vertex_count_written, edge_count_written).

    If append is False (default), existing rows in the graph tables are removed
    before writing, which makes each indexing run replace prior contents.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)

        if not append:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM vertices")

        conn.executemany(
            "INSERT OR REPLACE INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
            [
                (v.id, v.label, json.dumps(v.properties, sort_keys=True))
                for v in graph.vertices()
            ],
        )

        conn.executemany(
            """
            INSERT OR REPLACE INTO edges(id, label, out_v_id, in_v_id, properties_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    e.id,
                    e.label,
                    e.out_v.id,
                    e.in_v.id,
                    json.dumps(e.properties, sort_keys=True),
                )
                for e in graph.edges()
            ],
        )

        vertex_count = conn.execute("SELECT COUNT(*) FROM vertices").fetchone()[0]
        edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    return int(vertex_count), int(edge_count)


def persist_vertex_embeddings(
    graph: PropertyGraph,
    db_path: Path,
    *,
    model_name: str,
    cache_folder: str | None = None,
    append: bool = False,
    initialize_vector_extension: bool = True,
) -> tuple[int, int, bool]:
    """Persist vertex embeddings into the same SQLite file as the graph.

    Returns (embedding_row_count, vector_dimension, vector_extension_ready).
    """
    vertices = [
        vertex
        for vertex in graph.vertices()
        if vertex.label in SUPPORTED_VECTOR_VERTEX_LABELS
    ]

    if not vertices:
        return 0, 0, False

    texts = [_vertex_text(vertex) for vertex in vertices]
    embeddings = generate_embeddings(texts, model_name=model_name, cache_folder=cache_folder)
    dimension = len(embeddings[0]) if embeddings else 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)

        if not append:
            conn.execute("DELETE FROM vertex_embeddings")

        conn.executemany(
            """
            INSERT OR REPLACE INTO vertex_embeddings(
                vertex_id, model_name, source_text, dimension, embedding
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    vertices[i].id,
                    model_name,
                    texts[i],
                    len(embeddings[i]),
                    _vector_to_blob(embeddings[i]),
                )
                for i in range(len(vertices))
            ],
        )

        extension_ready = False
        if initialize_vector_extension and dimension > 0:
            extension_ready = _try_initialize_sqlite_vector(conn, dimension)

        row_count = conn.execute("SELECT COUNT(*) FROM vertex_embeddings").fetchone()[0]

    return int(row_count), int(dimension), extension_ready


def load_graph(db_path: Path) -> PropertyGraph:
    """Load a PropertyGraph from a SQLite file."""
    graph = PropertyGraph()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)

        vertex_rows = conn.execute(
            "SELECT id, label, properties_json FROM vertices"
        ).fetchall()
        vertices_by_id: dict[str, object] = {}

        for vertex_id, label, properties_json in vertex_rows:
            vertex = graph.add_vertex(
                label,
                vertex_id=str(vertex_id),
                **json.loads(properties_json),
            )
            vertices_by_id[str(vertex_id)] = vertex

        edge_rows = conn.execute(
            "SELECT id, label, out_v_id, in_v_id, properties_json FROM edges"
        ).fetchall()
        for edge_id, label, out_v_id, in_v_id, properties_json in edge_rows:
            out_v = vertices_by_id.get(str(out_v_id))
            in_v = vertices_by_id.get(str(in_v_id))
            if out_v is None or in_v is None:
                continue
            graph.add_edge(
                str(label),
                out_v,
                in_v,
                edge_id=str(edge_id),
                **json.loads(properties_json),
            )

    return graph


def _vertex_text(vertex: Vertex) -> str:
    name = str(vertex.properties.get("name", ""))
    path = str(vertex.properties.get("path", ""))
    line = vertex.properties.get("line")
    if line is None:
        return f"{vertex.label} {name} {path}".strip()
    return f"{vertex.label} {name} {path}:{line}".strip()


def _vector_to_blob(values: list[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"<{len(values)}f", *values)


def _try_initialize_sqlite_vector(conn: sqlite3.Connection, dimension: int) -> bool:
    if not _try_load_sqlite_vector_extension(conn):
        return False

    config = f"type=FLOAT32,dimension={dimension},distance=COSINE"
    try:
        conn.execute("SELECT vector_init('vertex_embeddings', 'embedding', ?)", (config,))
    except sqlite3.DatabaseError:
        # vector_init can fail if already initialized for the same table/column.
        return True
    return True


def _try_load_sqlite_vector_extension(conn: sqlite3.Connection) -> bool:
    try:
        ext_path = importlib.resources.files("sqlite_vector.binaries") / "vector"
    except (ModuleNotFoundError, FileNotFoundError):
        return False

    try:
        conn.enable_load_extension(True)
        conn.load_extension(str(ext_path))
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.DatabaseError, sqlite3.OperationalError):
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        return False

    return True
