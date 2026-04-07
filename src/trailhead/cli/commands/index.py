"""``th index`` — recursively index a codebase into a property graph."""
from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path

from trailhead.services.config import ALLOWED_MODELS
from trailhead.services.config import get_cache_dir
from trailhead.services.config import is_model_allowed
from trailhead.services.indexing import discover_source_files
from trailhead.services.indexing import LiveIndexer
from trailhead.services.indexing.graph import PropertyGraph
from trailhead.services.indexing.sqlite_store import get_index_model
from trailhead.services.indexing.sqlite_store import persist_vertex_embeddings
from trailhead.services.indexing.walker import index_directory

logger = logging.getLogger(__name__)

_DEFAULT_DB_SUBPATH = ".trailhead/db.sqlite"


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "index",
        help="Index a codebase into a property graph.",
        description=(
            "Recursively parse source files under a directory and build a "
            "TinkerPop-compatible property graph of code relationships "
            "(modules, classes, functions, imports). "
            f"The graph is persisted to <directory>/{_DEFAULT_DB_SUBPATH} by default."
        ),
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to index (default: current directory).",
    )
    parser.add_argument(
        "--sqlite-db",
        default=None,
        help=(
            f"SQLite database file for graph persistence "
            f"(default: <directory>/{_DEFAULT_DB_SUBPATH}). "
            "Ignored when --in-memory is set."
        ),
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help=(
            "Build the graph in memory only — do not persist to SQLite. "
            "Outputs a summary (or JSON with --output json) and exits immediately. "
            "Incompatible with --watch."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "List the source files that would be indexed and exit without parsing "
            "or writing SQLite state. Incompatible with --watch, --in-memory, "
            "--sqlite-db, and --embed-model."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "After the initial sync, watch the directory for file changes and "
            "reindex incrementally. Incompatible with --in-memory. "
            "Press Ctrl-C to stop."
        ),
    )
    parser.add_argument(
        "--output",
        choices=["summary", "json"],
        default="summary",
        help="Output format when --in-memory or --dry-run is set (default: summary).",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help=(
            "Optional sentence-transformers model name. "
            "Stores vertex embeddings in the SQLite database. "
            "Incompatible with --in-memory."
        ),
    )
    parser.add_argument(
        "--embed-cache-dir",
        default=None,
        help=(
            "Optional cache directory for embedding models. "
            "Defaults to TRAILHEAD_CACHE_DIR if set, otherwise Hugging Face default."
        ),
    )
    parser.add_argument(
        "--allow-any-model",
        action="store_true",
        help=(
            "Allow any Hugging Face model ID for --embed-model, bypassing the built-in "
            "allowlist. Can also be set via the TRAILHEAD_ALLOW_ANY_MODEL environment variable."
        ),
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    root = Path(args.directory).resolve()

    if not root.exists():
        logger.error("Directory not found: %s", root)
        return 1
    if not root.is_dir():
        logger.error("Not a directory: %s", root)
        return 1

    if args.in_memory and args.watch:
        logger.error("--in-memory and --watch are mutually exclusive.")
        return 1

    if args.dry_run and args.watch:
        logger.error("--dry-run and --watch are mutually exclusive.")
        return 1

    if args.dry_run and args.in_memory:
        logger.error("--dry-run and --in-memory are mutually exclusive.")
        return 1

    if args.dry_run and args.sqlite_db:
        logger.error("--dry-run does not accept --sqlite-db.")
        return 1

    if args.embed_model and args.in_memory:
        logger.error("--embed-model requires SQLite persistence; remove --in-memory.")
        return 1

    if args.embed_model and args.dry_run:
        logger.error("--embed-model requires SQLite persistence; remove --dry-run.")
        return 1

    if args.embed_model and not args.sqlite_db:
        logger.error("--embed-model requires --sqlite-db.")
        return 1

    if args.embed_model and not is_model_allowed(args.embed_model, allow_any=args.allow_any_model):
        logger.error(
            "Model '%s' is not in the allowlist. Allowed models: %s. "
            "Pass --allow-any-model or set TRAILHEAD_ALLOW_ANY_MODEL=1 to use a custom model.",
            args.embed_model,
            ", ".join(sorted(ALLOWED_MODELS)),
        )
        return 1

    # ------------------------------------------------------------------
    # Dry-run path: enumerate files only, print preview, then exit.
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("Previewing index candidates under %s", root)
        files = discover_source_files(root)
        if args.output == "json":
            _print_dry_run_json(root, files)
        else:
            _print_dry_run_summary(root, files)
        return 0

    # ------------------------------------------------------------------
    # In-memory path: one-shot build, print summary/JSON, then exit.
    # ------------------------------------------------------------------
    if args.in_memory:
        logger.info("Indexing %s (in-memory)", root)
        graph = index_directory(root)
        if args.output == "json":
            _print_json(graph)
        else:
            _print_summary(graph)
        return 0

    # ------------------------------------------------------------------
    # SQLite path: use LiveIndexer for smart sync (and optionally watch).
    # This is the same code path used by ``th serve``.
    # ------------------------------------------------------------------
    db_path = Path(args.sqlite_db).resolve() if args.sqlite_db else root / _DEFAULT_DB_SUBPATH

    if args.embed_model:
        existing_model = get_index_model(db_path)
        if existing_model is not None and existing_model != args.embed_model:
            print(
                f"Warning: the existing index was built with model '{existing_model}', "
                f"but you are indexing with '{args.embed_model}'. "
                f"This will replace all existing embeddings."
            )
            response = input("Regenerate all embeddings? [y/N] ").strip().lower()
            if response != "y":
                logger.info("Aborted.")
                return 0

    cache_dir = args.embed_cache_dir or get_cache_dir()
    if cache_dir:
        cache_dir = str(Path(cache_dir).resolve())

    indexer = LiveIndexer(
        root=root,
        db_path=db_path,
        model_name=args.embed_model,
        cache_folder=cache_dir,
    )
    indexer.synchronize()

    if args.watch:
        logger.info("Watching %s for changes — press Ctrl-C to stop.", root)
        indexer.start()
        _stop = threading.Event()
        try:
            _stop.wait()
        except KeyboardInterrupt:
            pass
        finally:
            indexer.stop()

    return 0


def _print_json(graph: PropertyGraph) -> None:
    data = {
        "vertices": [
            {"id": v.id, "label": v.label, **v.properties}
            for v in graph.vertices()
        ],
        "edges": [
            {
                "id": e.id,
                "label": e.label,
                "from": e.out_v.id,
                "to": e.in_v.id,
                **e.properties,
            }
            for e in graph.edges()
        ],
    }
    print(json.dumps(data, indent=2))


def _print_summary(graph: PropertyGraph) -> None:
    modules = graph.vertices("module")
    classes = graph.vertices("class")
    functions = graph.vertices("function")
    externals = graph.vertices("external")

    print(
        f"Indexed {len(modules)} module(s), "
        f"{len(classes)} class(es), "
        f"{len(functions)} function(s)."
    )
    if externals:
        print(f"  {len(externals)} unique external import(s) referenced.")
    print()

    for mod in sorted(modules, key=lambda v: v.properties["name"]):
        print(f"  module: {mod.properties['name']}  ({mod.properties['path']})")

        for edge in graph.out_edges(mod, "defines"):
            v = edge.in_v
            line = v.properties.get("line", "?")
            print(f"    {v.label}: {v.properties['name']}  (line {line})")
            if v.label == "class":
                for e2 in graph.out_edges(v, "has_method"):
                    m = e2.in_v
                    print(
                        f"      method: {m.properties['name']}"
                        f"  (line {m.properties.get('line', '?')})"
                    )

        imports = graph.out_edges(mod, "imports")
        if imports:
            names = sorted(e.in_v.properties["name"] for e in imports)
            print(f"    imports: {', '.join(names)}")


def _print_dry_run_json(root: Path, files: list[Path]) -> None:
    data = {
        "root": str(root),
        "count": len(files),
        "files": [path.relative_to(root).as_posix() for path in files],
    }
    print(json.dumps(data, indent=2))


def _print_dry_run_summary(root: Path, files: list[Path]) -> None:
    print(f"Would index {len(files)} file(s) under {root}.")
    if not files:
        return
    print()
    for path in files:
        print(path.relative_to(root).as_posix())
