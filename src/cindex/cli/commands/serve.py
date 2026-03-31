"""``cindex serve`` — run a warm-model FastAPI server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cindex.services.config import get_cache_dir
from cindex.services.indexing.sqlite_store import get_index_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Run the FastAPI server with warmed embedding models.",
        description=(
            "Start a long-lived API process so embedding models can stay loaded "
            "in memory across requests while a background indexer keeps the SQLite index fresh."
        ),
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to watch and index (default: current directory).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to preload on startup (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache downloaded models. Overrides CINDEX_CACHE_DIR.",
    )
    parser.add_argument(
        "--sqlite-db",
        required=True,
        help="SQLite database path used by query endpoints and the browser UI.",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Start the server without warming the default model during startup.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    import uvicorn

    cache_dir = args.cache_dir or get_cache_dir()
    if cache_dir:
        cache_dir = str(Path(cache_dir).resolve())

    watch_directory = str(Path(args.directory).resolve())

    db_path = Path(args.sqlite_db).resolve()
    existing_model = get_index_model(db_path)
    if existing_model is not None and existing_model != args.model:
        logger.error(
            "Model mismatch: server is configured with '%s' but the index was built with '%s'. "
            "Re-run 'cindex index --embed-model %s --sqlite-db %s' to regenerate embeddings, "
            "or pass '--model %s' to match the existing index.",
            args.model,
            existing_model,
            args.model,
            args.sqlite_db,
            existing_model,
        )
        return 1

    logger.info(
        "Starting cindex API server on %s:%d with model %s watching %s",
        args.host,
        args.port,
        args.model,
        watch_directory,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    from cindex.server.app import create_app

    app = create_app(
        default_model=args.model,
        cache_dir=cache_dir,
        sqlite_db=args.sqlite_db,
        watch_directory=watch_directory,
        preload_default_model=not args.no_preload,
        run_indexer=True,
    )

    uvicorn.run(app, host=args.host, port=args.port)
    return 0
