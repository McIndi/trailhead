"""Process-wide embedding model cache.

This lets a long-lived process such as the FastAPI server keep one or more
SentenceTransformer models resident in memory instead of reloading them for each
request.
"""

from __future__ import annotations

from threading import Lock

_model_cache: dict[tuple[str, str | None], object] = {}
_model_cache_lock = Lock()


def get_embedding_model(model_name: str, cache_folder: str | None = None):
    cache_key = (model_name, cache_folder)
    model = _model_cache.get(cache_key)
    if model is not None:
        return model

    with _model_cache_lock:
        model = _model_cache.get(cache_key)
        if model is not None:
            return model

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, cache_folder=cache_folder)
        _model_cache[cache_key] = model
        return model


def preload_embedding_model(model_name: str, cache_folder: str | None = None) -> None:
    get_embedding_model(model_name, cache_folder=cache_folder)


def get_loaded_model_names() -> list[str]:
    return sorted({model_name for model_name, _ in _model_cache})


def clear_embedding_model_cache() -> None:
    with _model_cache_lock:
        _model_cache.clear()
