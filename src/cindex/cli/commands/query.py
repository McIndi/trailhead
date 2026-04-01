"""``th query`` — query graph and vector data stored in SQLite."""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

from cindex.services.config import get_cache_dir
from cindex.services.indexing.query import execute_sql_query
from cindex.services.indexing.query import find_similar_vertices

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DB_SUBPATH = ".cindex/db.sqlite"
_DEFAULT_SERVER = "http://127.0.0.1:8000"

logger = logging.getLogger(__name__)


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "query",
        help="Query graph and vector data from a SQLite database.",
        description="Run SQL or semantic similarity queries against the trailhead SQLite database.",
    )
    query_subparsers = parser.add_subparsers(dest="query_command", required=True)

    sql_parser = query_subparsers.add_parser(
        "sql",
        help="Run a read-only SQL query.",
        description="Run a read-only SQL statement against the trailhead SQLite database.",
    )
    sql_parser.add_argument(
        "--sqlite-db",
        default=None,
        help=f"Path to the SQLite database file (default: ./{_DEFAULT_DB_SUBPATH}).",
    )
    sql_parser.add_argument("--sql", required=True, help="Read-only SQL statement to execute.")
    sql_parser.add_argument(
        "--output",
        choices=["json", "table"],
        default="table",
        help="Output format (default: table).",
    )
    sql_parser.set_defaults(func=run_sql)

    similar_parser = query_subparsers.add_parser(
        "similar",
        help="Find vertices semantically similar to a text query.",
        description="Embed a text query and search the stored vertex embeddings in SQLite.",
    )
    similar_parser.add_argument("text", help="Natural-language query text.")
    similar_parser.add_argument(
        "--sqlite-db",
        default=None,
        help=f"Path to the SQLite database file (default: ./{_DEFAULT_DB_SUBPATH}).",
    )
    similar_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformers model to use (default: {DEFAULT_MODEL}).",
    )
    similar_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded models. Overrides CINDEX_CACHE_DIR.",
    )
    similar_parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to return (default: 10).",
    )
    similar_parser.add_argument(
        "--label",
        default=None,
        help="Optional vertex label filter (for example: module, function, class, external).",
    )
    similar_parser.add_argument(
        "--server",
        default=_DEFAULT_SERVER,
        metavar="URL",
        help=(
            f"Base URL of a running th serve instance to use for embedding "
            f"(default: {_DEFAULT_SERVER}). Falls back to local model load if unreachable. "
            f"Pass an empty string to always use the local model."
        ),
    )
    similar_parser.add_argument(
        "--output",
        choices=["json", "table"],
        default="table",
        help="Output format (default: table).",
    )
    similar_parser.set_defaults(func=run_similar)


def run_sql(args: argparse.Namespace) -> int:
    db_path = Path(args.sqlite_db).resolve() if args.sqlite_db else Path.cwd() / _DEFAULT_DB_SUBPATH
    if not db_path.exists():
        logger.error("SQLite database not found: %s", db_path)
        return 1

    try:
        columns, rows = execute_sql_query(db_path, args.sql)
    except ValueError as exc:
        logger.error(str(exc))
        return 1
    _print_rows(columns, rows, output=args.output)
    return 0


def run_similar(args: argparse.Namespace) -> int:
    db_path = Path(args.sqlite_db).resolve() if args.sqlite_db else Path.cwd() / _DEFAULT_DB_SUBPATH
    if not db_path.exists():
        logger.error("SQLite database not found: %s", db_path)
        return 1

    cache_dir = args.cache_dir or get_cache_dir()
    if cache_dir:
        cache_dir = str(Path(cache_dir).resolve())

    server_url = args.server or ""
    vector = _try_embed_via_server(args.text, server_url) if server_url else None
    if vector is not None:
        logger.debug("Using embedding from server: %s", server_url)
    else:
        if server_url:
            logger.debug("Server unavailable (%s), falling back to local model.", server_url)

    try:
        rows = find_similar_vertices(
            db_path,
            args.text,
            model_name=args.model,
            cache_folder=cache_dir,
            k=args.k,
            label=args.label,
            vector=vector,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error(str(exc))
        return 1
    _print_rows(
        ["vertex_id", "label", "name", "path", "line", "model_name", "distance"],
        rows,
        output=args.output,
    )
    return 0


def _try_embed_via_server(text: str, server_url: str) -> list[float] | None:
    """POST *text* to a running th serve instance and return the embedding.

    Returns None on any failure (server unreachable, timeout, unexpected
    response), allowing the caller to fall back to a local model load.
    """
    url = server_url.rstrip("/") + "/api/embed"
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return data["embedding"]
    except Exception:
        return None


def _print_rows(columns: list[str], rows: list[dict[str, Any]], *, output: str) -> None:
    if output == "json":
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print("No rows returned.")
        return

    widths = {
        column: max(len(column), *(len(_format_cell(row.get(column))) for row in rows))
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(_format_cell(row.get(column)).ljust(widths[column]) for column in columns))


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)