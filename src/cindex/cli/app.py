from __future__ import annotations

import argparse
from collections.abc import Sequence

from cindex.cli.commands.embed import configure_parser as configure_embed_parser
from cindex.cli.commands.index import configure_parser as configure_index_parser
from cindex.cli.commands.query import configure_parser as configure_query_parser
from cindex.cli.commands.serve import configure_parser as configure_serve_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cindex",
        description="Utilities for indexing and embeddings.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    configure_embed_parser(subparsers)
    configure_index_parser(subparsers)
    configure_query_parser(subparsers)
    configure_serve_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
