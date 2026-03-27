"""Sentence embedding generation."""

from __future__ import annotations


def generate_embedding(
    text: str, model_name: str, cache_folder: str | None = None
) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=cache_folder)
    embedding = model.encode(text)
    if hasattr(embedding, "tolist"):
        values = embedding.tolist()
    else:
        values = list(embedding)
    return [float(value) for value in values]
