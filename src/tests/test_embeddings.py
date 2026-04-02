"""Tests for embedding generation service."""

from __future__ import annotations

import pytest


class TestGenerateEmbedding:
    """Test suite for generate_embedding functionality."""

    def test_generate_embedding_with_list_input(self, monkeypatch) -> None:
        """Test generate_embedding converts numpy array to list."""
        import numpy as np

        from trailhead.services.embeddings import generate_embedding

        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        class MockModel:
            def encode(self, text: str):
                return mock_embedding

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            assert model_name == "test-model"
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        result = generate_embedding("hello", "test-model")
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_generate_embedding_with_plain_list(self, monkeypatch) -> None:
        """Test generate_embedding handles plain lists (no .tolist() method)."""
        from trailhead.services.embeddings import generate_embedding

        plain_list = [0.5, 0.6, 0.7]

        class MockModel:
            def encode(self, text: str):
                return plain_list

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        result = generate_embedding("test", "model")
        assert result == [0.5, 0.6, 0.7]

    def test_generate_embedding_passes_cache_folder(self, monkeypatch) -> None:
        """Test that cache_folder is passed to SentenceTransformer."""
        from trailhead.services.embeddings import generate_embedding

        recorded = {}

        class MockModel:
            def encode(self, text: str):
                return [0.1]

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            recorded["model_name"] = model_name
            recorded["cache_folder"] = cache_folder
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        generate_embedding("text", "my-model", cache_folder="/custom/cache")

        assert recorded["model_name"] == "my-model"
        assert recorded["cache_folder"] == "/custom/cache"

    def test_generate_embedding_default_cache_folder(self, monkeypatch) -> None:
        """Test that cache_folder defaults to None."""
        from trailhead.services.embeddings import generate_embedding

        recorded = {}

        class MockModel:
            def encode(self, text: str):
                return [0.1]

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            recorded["cache_folder"] = cache_folder
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        generate_embedding("text", "model")

        assert recorded["cache_folder"] is None

    def test_generate_embedding_with_different_texts(self, monkeypatch) -> None:
        """Test generate_embedding encodes different texts."""
        from trailhead.services.embeddings import generate_embedding

        texts_encoded = []

        class MockModel:
            def encode(self, text: str):
                texts_encoded.append(text)
                return [0.1, 0.2]

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        generate_embedding("hello world", "model")
        generate_embedding("goodbye world", "model")

        assert texts_encoded == ["hello world", "goodbye world"]

    def test_generate_embedding_returns_floats(self, monkeypatch) -> None:
        """Test that all values in result are floats, even from integer arrays."""
        import numpy as np

        from trailhead.services.embeddings import generate_embedding

        mock_embedding = np.array([1, 2, 3], dtype=np.int32)

        class MockModel:
            def encode(self, text: str):
                return mock_embedding

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        result = generate_embedding("text", "model")

        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in result)

    def test_generate_embedding_empty_text(self, monkeypatch) -> None:
        """Test generate_embedding with empty text."""
        from trailhead.services.embeddings import generate_embedding

        class MockModel:
            def encode(self, text: str):
                return [0.0] * 384

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        result = generate_embedding("", "model")

        assert len(result) == 384
        assert all(v == 0.0 for v in result)

    def test_generate_embedding_long_text(self, monkeypatch) -> None:
        """Test generate_embedding with long text."""
        from trailhead.services.embeddings import generate_embedding

        long_text = "word " * 1000

        recorded_text = None

        class MockModel:
            def encode(self, text: str):
                nonlocal recorded_text
                recorded_text = text
                return [0.1, 0.2]

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        result = generate_embedding(long_text, "model")

        assert recorded_text == long_text
        assert result == [0.1, 0.2]

    def test_generate_embedding_special_characters(self, monkeypatch) -> None:
        """Test generate_embedding with special characters."""
        from trailhead.services.embeddings import generate_embedding

        special_text = "Hello 你好 مرحبا 🎉 @#$%^&*()"

        recorded_text = None

        class MockModel:
            def encode(self, text: str):
                nonlocal recorded_text
                recorded_text = text
                return [0.1]

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        result = generate_embedding(special_text, "model")

        assert recorded_text == special_text
        assert result == [0.1]

    def test_generate_embedding_different_model_names(self, monkeypatch) -> None:
        """Test generate_embedding with different model names."""
        from trailhead.services.embeddings import generate_embedding

        models_loaded = []

        class MockModel:
            def encode(self, text: str):
                return [0.1]

        def mock_sentence_transformer(model_name: str, cache_folder: str | None = None):
            models_loaded.append(model_name)
            return MockModel()

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_sentence_transformer,
        )

        generate_embedding("text", "sentence-transformers/all-MiniLM-L6-v2")
        generate_embedding("text", "sentence-transformers/all-mpnet-base-v2")
        generate_embedding("text", "my-custom-model")

        assert models_loaded == [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "my-custom-model",
        ]
