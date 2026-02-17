"""Embedding generation for comics using text and image encoders.

Provides a unified interface for generating embeddings from:
  - Text descriptions (using sentence-transformers)
  - Cover images (using CLIP or similar vision models)

These embeddings power the content-based recommender engine.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _text_hash(text: str) -> str:
    """Create a short hash for cache keys."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


class TextEmbedder:
    """Generate text embeddings using sentence-transformers.

    Falls back to a simple TF-IDF-based approach if sentence-transformers
    is not available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = False

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            self._fallback = True

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embedding vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._load_model()
        if self._fallback:
            return self._encode_tfidf(texts)
        return self._model.encode(texts, show_progress_bar=False)

    def _encode_tfidf(self, texts: List[str]) -> np.ndarray:
        """Simple TF-IDF fallback using sklearn."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(max_features=384)
            matrix = vectorizer.fit_transform(texts)
            return matrix.toarray().astype(np.float32)
        except ImportError:
            # Last resort: bag of words with fixed dimension
            return self._encode_bow(texts)

    @staticmethod
    def _encode_bow(texts: List[str], dim: int = 384) -> np.ndarray:
        """Ultra-simple bag-of-words embedding fallback."""
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                idx = hash(word) % dim
                embeddings[i, idx] += 1.0
            # L2 normalize
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        return embeddings


class ImageEmbedder:
    """Generate image embeddings using CLIP.

    Falls back to simple pixel-based features if transformers/CLIP
    is not available.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._fallback = False

    def _load_model(self):
        if self._model is not None or self._fallback:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
        except ImportError:
            self._fallback = True

    def encode_paths(self, image_paths: List[str]) -> np.ndarray:
        """Encode images from file paths into embedding vectors.

        Args:
            image_paths: List of paths to image files.

        Returns:
            numpy array of shape (len(image_paths), embedding_dim).
        """
        self._load_model()
        if self._fallback:
            return self._encode_pixel_fallback(image_paths)

        from PIL import Image

        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                # Create a blank image on failure
                images.append(Image.new("RGB", (224, 224)))

        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        import torch

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        embeddings = outputs.numpy()
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    @staticmethod
    def _encode_pixel_fallback(
        image_paths: List[str], dim: int = 512
    ) -> np.ndarray:
        """Simple pixel-based fallback for image embedding."""
        try:
            from PIL import Image
        except ImportError:
            return np.random.randn(len(image_paths), dim).astype(np.float32)

        embeddings = np.zeros((len(image_paths), dim), dtype=np.float32)
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB").resize((16, 16))
                pixels = np.array(img).flatten().astype(np.float32)
                # Pad or truncate to dim
                if len(pixels) < dim:
                    embeddings[i, : len(pixels)] = pixels
                else:
                    embeddings[i] = pixels[:dim]
                # L2 normalize
                norm = np.linalg.norm(embeddings[i])
                if norm > 0:
                    embeddings[i] /= norm
            except Exception:
                pass
        return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of vectors.

    Args:
        a: array of shape (n, d)
        b: array of shape (m, d)

    Returns:
        Similarity matrix of shape (n, m).
    """
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    return a_norm @ b_norm.T
