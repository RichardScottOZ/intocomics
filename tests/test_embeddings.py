"""Tests for the embeddings module."""

import numpy as np
import pytest

from src.intocomics.embeddings import TextEmbedder, cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        sim = cosine_similarity(a, b)
        assert sim.shape == (1, 1)
        assert np.isclose(sim[0, 0], 1.0)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        sim = cosine_similarity(a, b)
        assert np.isclose(sim[0, 0], 0.0)

    def test_opposite_vectors(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        sim = cosine_similarity(a, b)
        assert np.isclose(sim[0, 0], -1.0)

    def test_batch_similarity(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        sim = cosine_similarity(a, b)
        assert sim.shape == (2, 3)
        # a[0] should be most similar to b[0]
        assert np.isclose(sim[0, 0], 1.0)
        # a[1] should be most similar to b[1]
        assert np.isclose(sim[1, 1], 1.0)

    def test_zero_vector_handled(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 0.0]])
        sim = cosine_similarity(a, b)
        # Should not raise, zero vector gets clipped
        assert sim.shape == (1, 1)


class TestTextEmbedderBow:
    """Test the bag-of-words fallback embedder."""

    def test_encode_produces_correct_shape(self):
        embedder = TextEmbedder.__new__(TextEmbedder)
        texts = ["hello world", "foo bar baz"]
        result = embedder._encode_bow(texts, dim=100)
        assert result.shape == (2, 100)

    def test_encode_normalized(self):
        embedder = TextEmbedder.__new__(TextEmbedder)
        texts = ["hello world test"]
        result = embedder._encode_bow(texts, dim=100)
        norm = np.linalg.norm(result[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_similar_texts_more_similar(self):
        embedder = TextEmbedder.__new__(TextEmbedder)
        texts = [
            "batman dark knight comic superhero",
            "batman dark knight returns dc",
            "cooking recipe pasta italian food",
        ]
        embeddings = embedder._encode_bow(texts, dim=384)
        sim = cosine_similarity(embeddings[:1], embeddings[1:])
        # Batman texts should be more similar to each other than to cooking
        assert sim[0, 0] > sim[0, 1]

    def test_empty_text(self):
        embedder = TextEmbedder.__new__(TextEmbedder)
        texts = [""]
        result = embedder._encode_bow(texts, dim=50)
        assert result.shape == (1, 50)
        # Empty text should give zero vector
        assert np.allclose(result, 0.0)
