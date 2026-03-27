"""Cache configuration helpers."""

from __future__ import annotations

import os


def get_cache_dir() -> str | None:
    """Return cache directory from environment, if configured."""
    return os.environ.get("CINDEX_CACHE_DIR")
