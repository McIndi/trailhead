from __future__ import annotations

import pytest

from cindex.services.embeddings import clear_embedding_model_cache


@pytest.fixture(autouse=True)
def clear_embedding_cache_between_tests():
    clear_embedding_model_cache()
    yield
    clear_embedding_model_cache()
