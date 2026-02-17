"""Content-based comic recommender engine.

Uses manifest data from Comic-Analysis and scraped data from comic websites
to recommend new comics. Supports both text-based and image-based similarity.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embeddings import TextEmbedder, cosine_similarity
from .manifest import ComicEntry
from .scrapers import ScrapedComic

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A recommended comic with a similarity score."""

    title: str
    score: float
    url: str = ""
    cover_image_url: Optional[str] = None
    description: str = ""
    source: str = ""
    genres: List[str] = field(default_factory=list)


class ComicRecommender:
    """Content-based recommender that matches user preferences to new comics.

    The recommender works by:
    1. Building text embeddings from the user's known/liked comics (from manifest).
    2. Building text embeddings from candidate comics (scraped from websites).
    3. Computing cosine similarity to rank candidates.

    This enables cross-source recommendations: use your existing comic collection
    to discover new works on neonichiban.com, globalcomix.com, or other platforms.
    """

    def __init__(self, text_embedder: Optional[TextEmbedder] = None):
        """Initialize the recommender.

        Args:
            text_embedder: Optional TextEmbedder instance. If None, a default
                           one will be created.
        """
        self.text_embedder = text_embedder or TextEmbedder()
        self._library_entries: List[ComicEntry] = []
        self._library_texts: List[str] = []
        self._library_embeddings: Optional[np.ndarray] = None

    def load_library(self, comics: Dict[str, ComicEntry]) -> None:
        """Load the user's comic library from manifest data.

        Args:
            comics: Dictionary of comic_id -> ComicEntry from manifest loader.
        """
        self._library_entries = list(comics.values())
        self._library_texts = [
            self._comic_entry_to_text(c) for c in self._library_entries
        ]
        if self._library_texts:
            self._library_embeddings = self.text_embedder.encode(
                self._library_texts
            )
        else:
            self._library_embeddings = None
        logger.info("Loaded %d comics into library", len(self._library_entries))

    def load_liked_titles(self, titles: List[str]) -> None:
        """Load a simple list of liked comic titles as the user profile.

        This is a convenience method for when full manifest data is not
        available - just provide titles of comics you like.

        Args:
            titles: List of comic title strings the user likes.
        """
        self._library_entries = [
            ComicEntry(comic_id=t, title=t, source="user_input")
            for t in titles
        ]
        self._library_texts = titles
        if self._library_texts:
            self._library_embeddings = self.text_embedder.encode(
                self._library_texts
            )
        else:
            self._library_embeddings = None

    def recommend(
        self,
        candidates: List[ScrapedComic],
        top_k: int = 10,
    ) -> List[Recommendation]:
        """Recommend comics from candidates based on the loaded library.

        Args:
            candidates: List of ScrapedComic objects (from scrapers).
            top_k: Number of top recommendations to return.

        Returns:
            List of Recommendation objects sorted by descending score.
        """
        if self._library_embeddings is None or len(self._library_entries) == 0:
            logger.warning("No library loaded; returning empty recommendations")
            return []

        if not candidates:
            return []

        candidate_texts = [self._scraped_to_text(c) for c in candidates]
        candidate_embeddings = self.text_embedder.encode(candidate_texts)

        # Compute similarity: each candidate vs all library items
        sim_matrix = cosine_similarity(candidate_embeddings, self._library_embeddings)

        # Score each candidate by its max similarity to any library item
        scores = sim_matrix.max(axis=1)

        # Rank by score
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for idx in ranked_indices:
            idx_int = int(idx)
            candidate = candidates[idx_int]
            recommendations.append(
                Recommendation(
                    title=candidate.title,
                    score=float(scores[idx_int]),
                    url=candidate.url,
                    cover_image_url=candidate.cover_image_url,
                    description=candidate.description,
                    source=candidate.source,
                    genres=candidate.genres,
                )
            )

        return recommendations

    @staticmethod
    def _comic_entry_to_text(entry: ComicEntry) -> str:
        """Convert a ComicEntry to a text representation for embedding."""
        parts = [entry.title]
        if entry.description:
            parts.append(entry.description)
        if entry.genres:
            parts.append("Genres: " + ", ".join(entry.genres))
        return " ".join(parts)

    @staticmethod
    def _scraped_to_text(comic: ScrapedComic) -> str:
        """Convert a ScrapedComic to a text representation for embedding."""
        parts = [comic.title]
        if comic.description:
            parts.append(comic.description)
        if comic.genres:
            parts.append("Genres: " + ", ".join(comic.genres))
        if comic.creators:
            parts.append("By: " + ", ".join(comic.creators))
        return " ".join(parts)


def recommend_from_manifest(
    manifest_comics: Dict[str, ComicEntry],
    candidates: List[ScrapedComic],
    top_k: int = 10,
) -> List[Recommendation]:
    """Convenience function to get recommendations in one call.

    Args:
        manifest_comics: Comics loaded from a Comic-Analysis manifest.
        candidates: Scraped comics from target websites.
        top_k: Number of recommendations to return.

    Returns:
        List of Recommendation objects.
    """
    recommender = ComicRecommender()
    recommender.load_library(manifest_comics)
    return recommender.recommend(candidates, top_k=top_k)
