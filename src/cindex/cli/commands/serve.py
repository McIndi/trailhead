"""``th serve`` — run a warm-model FastAPI server."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from cindex.services.config import ALLOWED_MODELS
from cindex.services.config import get_cache_dir
from cindex.services.config import is_model_allowed
from cindex.services.indexing.sqlite_store import get_index_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DB_SUBPATH = ".cindex/db.sqlite"


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
        default=None,
        help=(
            "SQLite database path used by query endpoints and the browser UI "
            f"(default: <directory>/{_DEFAULT_DB_SUBPATH})."
        ),
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Start the server without warming the default model during startup.",
    )
    parser.add_argument(
        "--allow-any-model",
        action="store_true",
        help=(
            "Allow any Hugging Face model ID for --model, bypassing the built-in "
            "allowlist. Can also be set via the CINDEX_ALLOW_ANY_MODEL environment variable."
        ),
    )
    parser.add_argument(
        "--cors-origins",
        default=None,
        metavar="ORIGINS",
        help=(
            "Comma-separated list of allowed CORS origins "
            "(e.g. http://localhost:3000,http://127.0.0.1:5173). "
            "Can also be set via CINDEX_CORS_ORIGINS. Unset disables CORS."
        ),
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=120,
        metavar="N",
        help="Max API requests per minute per IP (default: 120, 0 to disable).",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    import uvicorn

    if not is_model_allowed(args.model, allow_any=args.allow_any_model):
        logger.error(
            "Model '%s' is not in the allowlist. Allowed models: %s. "
            "Pass --allow-any-model or set CINDEX_ALLOW_ANY_MODEL=1 to use a custom model.",
            args.model,
            ", ".join(sorted(ALLOWED_MODELS)),
        )
        return 1

    cache_dir = args.cache_dir or get_cache_dir()
    if cache_dir:
        cache_dir = str(Path(cache_dir).resolve())

    watch_directory = str(Path(args.directory).resolve())

    db_path_for_checks = (
        Path(args.sqlite_db).resolve()
        if args.sqlite_db
        else Path(watch_directory) / _DEFAULT_DB_SUBPATH
    )
    existing_model = get_index_model(db_path_for_checks)
    if existing_model is not None and existing_model != args.model:
        logger.error(
            "Model mismatch: server is configured with '%s' but the index was built with '%s'. "
            "Re-run 'th index --embed-model %s --sqlite-db %s' to regenerate embeddings, "
            "or pass '--model %s' to match the existing index.",
            args.model,
            existing_model,
            args.model,
            db_path_for_checks,
            existing_model,
        )
        return 1

    sqlite_db_for_app = args.sqlite_db if args.sqlite_db else str(db_path_for_checks)

    logger.info(
        "Starting trailhead API server on %s:%d with model %s watching %s",
        args.host,
        args.port,
        args.model,
        watch_directory,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    cors_origins_raw = args.cors_origins or os.environ.get("CINDEX_CORS_ORIGINS", "")
    cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]

    from cindex.server.app import create_app

    app = create_app(
        default_model=args.model,
        cache_dir=cache_dir,
        sqlite_db=sqlite_db_for_app,
        watch_directory=watch_directory,
        preload_default_model=not args.no_preload,
        run_indexer=True,
        cors_origins=cors_origins,
        rate_limit=args.rate_limit,
    )

    uvicorn.run(app, host=args.host, port=args.port)
    return 0
