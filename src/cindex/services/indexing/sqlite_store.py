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
from cindex.services.indexing.adapters import parse_file, supported_suffixes
from cindex.services.indexing.graph import PropertyGraph
from cindex.services.indexing.graph import Vertex

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vertices (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  properties_json TEXT NOT NULL,
  name TEXT GENERATED ALWAYS AS (json_extract(properties_json, '$.name')) VIRTUAL,
  path TEXT GENERATED ALWAYS AS (COALESCE(json_extract(properties_json, '$.path'), '')) VIRTUAL,
  line INTEGER GENERATED ALWAYS AS (json_extract(properties_json, '$.line')) VIRTUAL,
  complexity INTEGER GENERATED ALWAYS AS (json_extract(properties_json, '$.complexity')) VIRTUAL
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

CREATE TABLE IF NOT EXISTS indexed_files (
  path TEXT PRIMARY KEY,
  mtime_ns INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vertices_label ON vertices(label);
CREATE INDEX IF NOT EXISTS idx_vertices_label_name ON vertices(label, name);
CREATE INDEX IF NOT EXISTS idx_vertices_name ON vertices(name);
CREATE INDEX IF NOT EXISTS idx_vertices_path ON vertices(path);
CREATE INDEX IF NOT EXISTS idx_edges_label ON edges(label);
CREATE INDEX IF NOT EXISTS idx_edges_out_v ON edges(out_v_id);
CREATE INDEX IF NOT EXISTS idx_edges_in_v ON edges(in_v_id);
CREATE INDEX IF NOT EXISTS idx_vertex_embeddings_model ON vertex_embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_indexed_files_mtime ON indexed_files(mtime_ns);
"""

# Applied once to databases created before generated columns were added.
_MIGRATION_SQL = [
    "ALTER TABLE vertices ADD COLUMN name TEXT GENERATED ALWAYS AS (json_extract(properties_json, '$.name')) VIRTUAL",
    "ALTER TABLE vertices ADD COLUMN path TEXT GENERATED ALWAYS AS (COALESCE(json_extract(properties_json, '$.path'), '')) VIRTUAL",
    "ALTER TABLE vertices ADD COLUMN line INTEGER GENERATED ALWAYS AS (json_extract(properties_json, '$.line')) VIRTUAL",
    "ALTER TABLE vertices ADD COLUMN complexity INTEGER GENERATED ALWAYS AS (json_extract(properties_json, '$.complexity')) VIRTUAL",
    "CREATE INDEX IF NOT EXISTS idx_vertices_label_name ON vertices(label, name)",
    "CREATE INDEX IF NOT EXISTS idx_vertices_name ON vertices(name)",
    "CREATE INDEX IF NOT EXISTS idx_vertices_path ON vertices(path)",
]


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Apply schema migrations that cannot be expressed in CREATE TABLE IF NOT EXISTS."""
    for sql in _MIGRATION_SQL:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # column/index already exists

_META_EMBEDDING_MODEL = "embedding_model"

SUPPORTED_VECTOR_VERTEX_LABELS: frozenset[str] = frozenset(
    {"module", "class", "function", "external"}
)


def get_index_model(db_path: Path) -> str | None:
    """Return the embedding model name recorded in the index, or None if not set."""
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        _run_migrations(conn)
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (_META_EMBEDDING_MODEL,)
        ).fetchone()
    return str(row[0]) if row else None


def _set_index_model(conn: sqlite3.Connection, model_name: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (_META_EMBEDDING_MODEL, model_name),
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
        _run_migrations(conn)

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
        _run_migrations(conn)

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
                    vertices[index].id,
                    model_name,
                    texts[index],
                    len(embeddings[index]),
                    vector_to_blob(embeddings[index]),
                )
                for index in range(len(vertices))
            ],
        )

        _set_index_model(conn, model_name)

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
        _run_migrations(conn)

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


def persist_indexed_files(root: Path, db_path: Path, *, append: bool = False) -> int:
    """Persist a snapshot of indexed source file mtimes for *root*."""
    files = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in supported_suffixes()
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)
        _run_migrations(conn)
        if not append:
            conn.execute("DELETE FROM indexed_files")

        conn.executemany(
            "INSERT OR REPLACE INTO indexed_files(path, mtime_ns) VALUES (?, ?)",
            [(str(path), path.stat().st_mtime_ns) for path in files],
        )

        count = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]

    return int(count)


def get_indexed_files(db_path: Path) -> dict[str, int]:
    """Return the currently recorded file index state keyed by absolute path."""
    if not db_path.exists():
        return {}

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)
        _run_migrations(conn)
        rows = conn.execute("SELECT path, mtime_ns FROM indexed_files").fetchall()

    return {str(path): int(mtime_ns) for path, mtime_ns in rows}


def reindex_file(
    db_path: Path,
    path: Path,
    *,
    model_name: str | None = None,
    cache_folder: str | None = None,
    initialize_vector_extension: bool = True,
) -> tuple[int, int]:
    """Incrementally replace the indexed contents for a single file path."""
    resolved_path = path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)
        _run_migrations(conn)

        _delete_vertices_for_path(conn, str(resolved_path))

        embedding_candidates: list[tuple[str, str, str]] = []
        if resolved_path.exists() and resolved_path.is_file() and resolved_path.suffix in supported_suffixes():
            graph = PropertyGraph()
            parse_file(resolved_path, graph)
            embedding_candidates = _upsert_graph_slice(conn, graph)
            conn.execute(
                "INSERT OR REPLACE INTO indexed_files(path, mtime_ns) VALUES (?, ?)",
                (str(resolved_path), resolved_path.stat().st_mtime_ns),
            )
        else:
            conn.execute("DELETE FROM indexed_files WHERE path = ?", (str(resolved_path),))

        _delete_orphan_external_vertices(conn)

        embedding_rows = _persist_embedding_candidates(
            conn,
            embedding_candidates,
            model_name=model_name,
            cache_folder=cache_folder,
            initialize_vector_extension=initialize_vector_extension,
        )

        vertex_count = conn.execute(
            "SELECT COUNT(*) FROM vertices WHERE path = ?",
            (str(resolved_path),),
        ).fetchone()[0]

    return int(vertex_count), int(embedding_rows)


def _vertex_text(vertex: Vertex) -> str:
    name = str(vertex.properties.get("name", ""))
    path = str(vertex.properties.get("path", ""))
    line = vertex.properties.get("line")
    if line is None:
        return f"{vertex.label} {name} {path}".strip()
    return f"{vertex.label} {name} {path}:{line}".strip()


def vector_to_blob(values: list[float]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"<{len(values)}f", *values)


def _delete_vertices_for_path(conn: sqlite3.Connection, path: str) -> None:
    conn.execute(
        """
        DELETE FROM vertices
        WHERE label IN ('module', 'class', 'function')
          AND json_extract(properties_json, '$.path') = ?
        """,
        (path,),
    )


def _delete_orphan_external_vertices(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        DELETE FROM vertices
        WHERE label = 'external'
          AND NOT EXISTS (SELECT 1 FROM edges WHERE out_v_id = vertices.id OR in_v_id = vertices.id)
        """
    )


def _upsert_graph_slice(
    conn: sqlite3.Connection,
    graph: PropertyGraph,
) -> list[tuple[str, str, str]]:
    vertex_id_map: dict[str, str] = {}
    embedding_candidates: list[tuple[str, str, str]] = []

    for vertex in graph.vertices():
        properties_json = json.dumps(vertex.properties, sort_keys=True)
        if vertex.label == "external":
            existing_row = conn.execute(
                """
                SELECT id FROM vertices
                WHERE label = 'external'
                  AND json_extract(properties_json, '$.name') = ?
                LIMIT 1
                """,
                (vertex.properties.get("name"),),
            ).fetchone()
            if existing_row is not None:
                vertex_id_map[vertex.id] = str(existing_row[0])
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                    (vertex.id, vertex.label, properties_json),
                )
                vertex_id_map[vertex.id] = vertex.id
        else:
            conn.execute(
                "INSERT OR REPLACE INTO vertices(id, label, properties_json) VALUES (?, ?, ?)",
                (vertex.id, vertex.label, properties_json),
            )
            vertex_id_map[vertex.id] = vertex.id

        embedding_candidates.append((vertex_id_map[vertex.id], vertex.label, _vertex_text(vertex)))

    conn.executemany(
        """
        INSERT OR REPLACE INTO edges(id, label, out_v_id, in_v_id, properties_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                edge.id,
                edge.label,
                vertex_id_map[edge.out_v.id],
                vertex_id_map[edge.in_v.id],
                json.dumps(edge.properties, sort_keys=True),
            )
            for edge in graph.edges()
        ],
    )

    return embedding_candidates


def _persist_embedding_candidates(
    conn: sqlite3.Connection,
    embedding_candidates: list[tuple[str, str, str]],
    *,
    model_name: str | None,
    cache_folder: str | None,
    initialize_vector_extension: bool,
) -> int:
    if not model_name:
        return 0

    supported_candidates = [
        (vertex_id, label, text)
        for vertex_id, label, text in embedding_candidates
        if label in SUPPORTED_VECTOR_VERTEX_LABELS
    ]
    if not supported_candidates:
        return 0

    texts = [text for _, _, text in supported_candidates]
    embeddings = generate_embeddings(texts, model_name=model_name, cache_folder=cache_folder)
    if not embeddings:
        return 0

    conn.executemany(
        """
        INSERT OR REPLACE INTO vertex_embeddings(
            vertex_id, model_name, source_text, dimension, embedding
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                supported_candidates[index][0],
                model_name,
                supported_candidates[index][2],
                len(embeddings[index]),
                vector_to_blob(embeddings[index]),
            )
            for index in range(len(supported_candidates))
        ],
    )

    _set_index_model(conn, model_name)

    if initialize_vector_extension and embeddings:
        _try_initialize_sqlite_vector(conn, len(embeddings[0]))

    return len(supported_candidates)


def _try_initialize_sqlite_vector(conn: sqlite3.Connection, dimension: int) -> bool:
    if not _try_load_sqlite_vector_extension(conn):
        return False

    config = f"type=FLOAT32,dimension={dimension},distance=COSINE"
    try:
        conn.execute("SELECT vector_init('vertex_embeddings', 'embedding', ?)", (config,))
    except sqlite3.DatabaseError:
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
