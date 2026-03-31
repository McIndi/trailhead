"""Directory walker — recursively indexes source files into a PropertyGraph."""
from __future__ import annotations

import logging
from pathlib import Path

from cindex.services.indexing.graph import PropertyGraph
from cindex.services.indexing.adapters import parse_file, supported_suffixes

logger = logging.getLogger(__name__)


def index_directory(root: Path, graph: PropertyGraph | None = None) -> PropertyGraph:
    """Recursively index all supported source files found under *root*.

    If *graph* is provided the vertices and edges are added to it in place and
    the same object is returned; otherwise a fresh PropertyGraph is created.
    """
    if graph is None:
        graph = PropertyGraph()

    suffixes = supported_suffixes()
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes
    )
    logger.info("Found %d file(s) to index under %s", len(files), root)

    failed: list[Path] = []
    for path in files:
        logger.debug("Indexing %s", path)
        try:
            parse_file(path, graph)
        except Exception:
            logger.warning("Failed to parse %s", path, exc_info=True)
            failed.append(path)

    if failed:
        logger.warning(
            "%d of %d file(s) could not be parsed and were skipped: %s",
            len(failed),
            len(files),
            ", ".join(p.name for p in failed),
        )

    return graph
