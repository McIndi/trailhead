"""``cindex serve`` — run a warm-model FastAPI server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cindex.services.config import get_cache_dir

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Run the FastAPI server with warmed embedding models.",
        description=(
            "Start a long-lived API process so embedding models can stay loaded "
            "in memory across requests."
        ),
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

    logger.info(
        "Starting cindex API server on %s:%d with model %s",
        args.host,
        args.port,
        args.model,
    )

    from cindex.server.app import create_app

    app = create_app(
        default_model=args.model,
        cache_dir=cache_dir,
        sqlite_db=args.sqlite_db,
        preload_default_model=not args.no_preload,
    )

    uvicorn.run(app, host=args.host, port=args.port)
    return 0
