"""``cindex index`` — recursively index a codebase into a property graph."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cindex.services.config import ALLOWED_MODELS
from cindex.services.config import get_cache_dir
from cindex.services.config import is_model_allowed
from cindex.services.indexing.graph import PropertyGraph
from cindex.services.indexing.sqlite_store import get_index_model
from cindex.services.indexing.sqlite_store import persist_graph
from cindex.services.indexing.sqlite_store import persist_indexed_files
from cindex.services.indexing.sqlite_store import persist_vertex_embeddings
from cindex.services.indexing.walker import index_directory

logger = logging.getLogger(__name__)


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "index",
        help="Index a codebase into a property graph.",
        description=(
            "Recursively parse source files under a directory and build a "
            "TinkerPop-compatible property graph of code relationships "
            "(modules, classes, functions, imports)."
        ),
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to index (default: current directory).",
    )
    parser.add_argument(
        "--output",
        choices=["summary", "json"],
        default="summary",
        help="Output format (default: summary).",
    )
    parser.add_argument(
        "--sqlite-db",
        default=None,
        help=(
            "Optional SQLite database file to persist the indexed graph "
            "(for example: ./.cindex/graph.db)."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Append into an existing SQLite graph instead of replacing all rows. "
            "Only applies when --sqlite-db is provided."
        ),
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help=(
            "Optional sentence-transformers model name. If provided with --sqlite-db, "
            "stores vertex embeddings in the same SQLite file."
        ),
    )
    parser.add_argument(
        "--embed-cache-dir",
        default=None,
        help=(
            "Optional cache directory for embedding models. "
            "Defaults to CINDEX_CACHE_DIR if set, otherwise Hugging Face default."
        ),
    )
    parser.add_argument(
        "--allow-any-model",
        action="store_true",
        help=(
            "Allow any Hugging Face model ID for --embed-model, bypassing the built-in "
            "allowlist. Can also be set via the CINDEX_ALLOW_ANY_MODEL environment variable."
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

    if args.embed_model and not is_model_allowed(args.embed_model, allow_any=args.allow_any_model):
        logger.error(
            "Model '%s' is not in the allowlist. Allowed models: %s. "
            "Pass --allow-any-model or set CINDEX_ALLOW_ANY_MODEL=1 to use a custom model.",
            args.embed_model,
            ", ".join(sorted(ALLOWED_MODELS)),
        )
        return 1

    if args.embed_model and not args.sqlite_db:
        logger.error("--embed-model requires --sqlite-db so vectors can be persisted.")
        return 1

    if args.embed_model and args.sqlite_db:
        db_path_check = Path(args.sqlite_db).resolve()
        existing_model = get_index_model(db_path_check)
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

    logger.info("Indexing %s", root)
    graph = index_directory(root)

    if args.sqlite_db:
        db_path = Path(args.sqlite_db).resolve()
        v_count, e_count = persist_graph(graph, db_path, append=args.append)
        logger.info("Persisted graph to %s (%d vertices, %d edges)", db_path, v_count, e_count)

        if args.embed_model:
            cache_dir = args.embed_cache_dir or get_cache_dir()
            if cache_dir:
                cache_dir = str(Path(cache_dir).resolve())
            emb_count, dim, vector_ready = persist_vertex_embeddings(
                graph,
                db_path,
                model_name=args.embed_model,
                cache_folder=cache_dir,
                append=args.append,
                initialize_vector_extension=True,
            )
            logger.info(
                "Persisted %d embedding row(s) to %s (dimension=%d)",
                emb_count,
                db_path,
                dim,
            )
            if vector_ready:
                logger.info("sqlite-vector extension initialized for vertex_embeddings.embedding")
            else:
                logger.warning(
                    "Could not initialize sqlite-vector extension. "
                    "Embeddings were still persisted as FLOAT32 BLOBs."
                )

        indexed_file_count = persist_indexed_files(root, db_path, append=args.append)
        logger.info("Recorded %d indexed file snapshot row(s) in %s", indexed_file_count, db_path)

    if args.output == "json":
        _print_json(graph)
    else:
        _print_summary(graph)

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
