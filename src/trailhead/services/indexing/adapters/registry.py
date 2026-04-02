"""Global adapter registry — maps file extensions to LanguageAdapter instances."""
from __future__ import annotations

import logging
from pathlib import Path

from trailhead.services.indexing.graph import PropertyGraph, Vertex
from trailhead.services.indexing.adapters.base import LanguageAdapter

logger = logging.getLogger(__name__)

_registry: dict[str, LanguageAdapter] = {}


def register(adapter: LanguageAdapter) -> None:
    """Register *adapter* for all its declared extensions."""
    for ext in adapter.extensions:
        _registry[ext] = adapter
    logger.debug(
        "Registered %s for extensions: %s",
        type(adapter).__name__,
        ", ".join(sorted(adapter.extensions)),
    )


def get_adapter(path: Path) -> LanguageAdapter | None:
    """Return the adapter for *path*, or None if the extension is not registered."""
    return _registry.get(path.suffix)


def supported_suffixes() -> frozenset[str]:
    """Return all file extensions with a registered adapter."""
    return frozenset(_registry)


def parse_file(path: Path, graph: PropertyGraph) -> Vertex:
    """Parse *path* using the appropriate registered adapter.

    Raises :exc:`ValueError` if no adapter is registered for the file's extension.
    """
    adapter = get_adapter(path)
    if adapter is None:
        raise ValueError(
            f"No language adapter registered for {path.suffix!r} files ({path})"
        )
    return adapter.parse(path, graph)
