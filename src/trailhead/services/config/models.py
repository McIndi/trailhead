"""Embedding model allowlist and validation helpers."""

from __future__ import annotations

import os

ALLOWED_MODELS: frozenset[str] = frozenset({
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
})

_ENV_VAR = "TRAILHEAD_ALLOW_ANY_MODEL"


def is_model_allowed(model_name: str, *, allow_any: bool = False) -> bool:
    """Return True if *model_name* is permitted for use.

    A model is permitted when any of the following is true:

    - It appears in ``ALLOWED_MODELS``.
    - *allow_any* is True (from ``--allow-any-model`` CLI flag).
    - The environment variable ``TRAILHEAD_ALLOW_ANY_MODEL`` is set to a
      truthy value (``1``, ``true``, or ``yes``, case-insensitive).
    """
    if model_name in ALLOWED_MODELS:
        return True
    if allow_any:
        return True
    return os.environ.get(_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}
