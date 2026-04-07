"""Shared source-file discovery for indexing workflows."""
from __future__ import annotations

import logging
from pathlib import Path

from pathspec import PathSpec

from trailhead.services.indexing.adapters import supported_suffixes

logger = logging.getLogger(__name__)

IGNORE_FILE_NAMES: tuple[str, str] = (".gitignore", ".trailheadignore")


def load_ignore_spec(root: Path) -> PathSpec:
    """Load gitignore-style patterns from ignore files under *root*."""
    root = root.resolve()
    patterns: list[str] = []
    for file_name in IGNORE_FILE_NAMES:
        ignore_path = root / file_name
        if not ignore_path.is_file():
            continue
        try:
            patterns.extend(ignore_path.read_text(encoding="utf-8").splitlines())
        except UnicodeDecodeError:
            logger.warning("Ignore file %s is not valid UTF-8 and will be skipped", ignore_path)
    return PathSpec.from_lines("gitignore", patterns)


def discover_source_files(root: Path) -> list[Path]:
    """Return supported, non-ignored source files under *root*."""
    root = root.resolve()
    ignore_spec = load_ignore_spec(root)
    suffixes = supported_suffixes()
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and should_index_path(path, root, spec=ignore_spec, suffixes=suffixes)
    )


def is_ignored_path(path: Path, root: Path, *, spec: PathSpec | None = None) -> bool:
    """Return True when *path* is excluded by an ignore file under *root*."""
    root = root.resolve()
    spec = spec or load_ignore_spec(root)
    relative_path = _relative_match_path(path, root)
    if relative_path is None:
        return False
    return spec.match_file(relative_path)


def should_index_path(
    path: Path,
    root: Path,
    *,
    spec: PathSpec | None = None,
    suffixes: frozenset[str] | None = None,
) -> bool:
    """Return True when *path* is a supported source file that is not ignored."""
    suffixes = suffixes or supported_suffixes()
    if path.suffix not in suffixes:
        return False
    return not is_ignored_path(path, root, spec=spec)


def _relative_match_path(path: Path, root: Path) -> str | None:
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative_path = candidate.resolve().relative_to(root)
    except ValueError:
        return None
    return relative_path.as_posix()