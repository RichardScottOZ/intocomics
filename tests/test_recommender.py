"""Tests for the recommender module."""

import numpy as np
import pytest

from src.intocomics.manifest import ComicEntry, ComicPage
from src.intocomics.recommender import ComicRecommender, Recommendation, recommend_from_manifest
from src.intocomics.scrapers import ScrapedComic


class TestComicRecommender:
    def _make_library(self):
        """Create a small test library."""
        return {
            "superhero/batman": ComicEntry(
                comic_id="superhero/batman",
                title="Batman Dark Knight",
                description="A dark superhero story about a vigilante in Gotham City",
                genres=["superhero", "action", "noir"],
            ),
            "superhero/spiderman": ComicEntry(
                comic_id="superhero/spiderman",
                title="Spider-Man",
                description="A teenage superhero swinging through New York City",
                genres=["superhero", "action", "comedy"],
            ),
        }

    def _make_candidates(self):
        """Create test candidates."""
        return [
            ScrapedComic(
                title="Dark Vigilante Returns",
                url="https://example.com/dark-vigilante",
                description="A noir story about a dark vigilante fighting crime in a city",
                genres=["superhero", "noir"],
                source="test",
            ),
            ScrapedComic(
                title="Cooking Adventures",
                url="https://example.com/cooking",
                description="A fun story about learning to cook Italian pasta",
                genres=["cooking", "slice-of-life"],
                source="test",
            ),
            ScrapedComic(
                title="Space Hero",
                url="https://example.com/space-hero",
                description="A hero saves the galaxy with amazing powers",
                genres=["sci-fi", "action"],
                source="test",
            ),
        ]

    def test_recommend_returns_results(self):
        recommender = ComicRecommender()
        recommender.load_library(self._make_library())
        results = recommender.recommend(self._make_candidates(), top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, Recommendation) for r in results)

    def test_recommend_sorted_by_score(self):
        recommender = ComicRecommender()
        recommender.load_library(self._make_library())
        results = recommender.recommend(self._make_candidates(), top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_similar_comic_ranked_higher(self):
        recommender = ComicRecommender()
        recommender.load_library(self._make_library())
        results = recommender.recommend(self._make_candidates(), top_k=3)
        # The dark vigilante comic should rank higher than cooking
        titles = [r.title for r in results]
        vigilante_idx = titles.index("Dark Vigilante Returns")
        cooking_idx = titles.index("Cooking Adventures")
        assert vigilante_idx < cooking_idx

    def test_recommend_respects_top_k(self):
        recommender = ComicRecommender()
        recommender.load_library(self._make_library())
        results = recommender.recommend(self._make_candidates(), top_k=1)
        assert len(results) == 1

    def test_recommend_empty_library(self):
        recommender = ComicRecommender()
        results = recommender.recommend(self._make_candidates())
        assert results == []

    def test_recommend_empty_candidates(self):
        recommender = ComicRecommender()
        recommender.load_library(self._make_library())
        results = recommender.recommend([])
        assert results == []

    def test_load_liked_titles(self):
        recommender = ComicRecommender()
        recommender.load_liked_titles(["Batman", "Spider-Man", "X-Men"])
        results = recommender.recommend(self._make_candidates(), top_k=2)
        assert len(results) == 2

    def test_recommendation_has_metadata(self):
        recommender = ComicRecommender()
        recommender.load_library(self._make_library())
        results = recommender.recommend(self._make_candidates(), top_k=1)
        rec = results[0]
        assert rec.title != ""
        assert rec.url != ""
        assert rec.source == "test"
        assert isinstance(rec.score, float)


class TestRecommendFromManifest:
    def test_convenience_function(self):
        library = {
            "comic/hero": ComicEntry(
                comic_id="comic/hero",
                title="Superhero Adventures",
                description="A superhero saves the day",
            ),
        }
        candidates = [
            ScrapedComic(
                title="Hero Returns",
                url="https://example.com/hero",
                description="A hero returns to save the world",
                source="test",
            ),
        ]
        results = recommend_from_manifest(library, candidates, top_k=1)
        assert len(results) == 1
        assert results[0].title == "Hero Returns"
