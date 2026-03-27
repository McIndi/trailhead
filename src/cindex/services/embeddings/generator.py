"""Sentence embedding generation."""

from __future__ import annotations

from collections.abc import Sequence


def _normalize_embedding(embedding: object) -> list[float]:
    if hasattr(embedding, "tolist"):
        values = embedding.tolist()
    else:
        values = list(embedding)
    return [float(value) for value in values]


def generate_embedding(
    text: str, model_name: str, cache_folder: str | None = None
) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=cache_folder)
    embedding = model.encode(text)
    return _normalize_embedding(embedding)


def generate_embeddings(
    texts: Sequence[str],
    model_name: str,
    cache_folder: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts with a single model load."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=cache_folder)
    embeddings = model.encode(list(texts))

    if len(texts) == 1:
        return [_normalize_embedding(embeddings)]

    return [_normalize_embedding(embedding) for embedding in embeddings]
