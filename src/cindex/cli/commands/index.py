"""``cindex index`` — recursively index a codebase into a property graph."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from cindex.services.indexing.graph import PropertyGraph
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
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    root = Path(args.directory).resolve()

    if not root.exists():
        logger.error("Directory not found: %s", root)
        return 1
    if not root.is_dir():
        logger.error("Not a directory: %s", root)
        return 1

    logger.info("Indexing %s", root)
    graph = index_directory(root)

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
