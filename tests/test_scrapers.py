"""Tests for the scraper modules."""

import pytest

from src.intocomics.scrapers import BaseScraper, ScrapedComic
from src.intocomics.scrapers.neonichiban import NeonIchibanScraper
from src.intocomics.scrapers.globalcomix import GlobalComixScraper


class TestScrapedComic:
    def test_default_values(self):
        comic = ScrapedComic(title="Test", url="https://example.com")
        assert comic.title == "Test"
        assert comic.url == "https://example.com"
        assert comic.cover_image_url is None
        assert comic.description == ""
        assert comic.genres == []
        assert comic.creators == []
        assert comic.source == ""

    def test_full_values(self):
        comic = ScrapedComic(
            title="My Comic",
            url="https://example.com/my-comic",
            cover_image_url="https://example.com/cover.jpg",
            description="A great comic",
            genres=["action", "sci-fi"],
            creators=["Author A"],
            source="globalcomix",
        )
        assert comic.genres == ["action", "sci-fi"]
        assert comic.creators == ["Author A"]


class TestNeonIchibanScraper:
    def test_scraper_attributes(self):
        scraper = NeonIchibanScraper()
        assert scraper.source_name == "neonichiban"
        assert scraper.base_url == "https://neonichiban.com"

    def test_is_base_scraper(self):
        scraper = NeonIchibanScraper()
        assert isinstance(scraper, BaseScraper)

    def test_parse_card_empty(self):
        """Test that _parse_card handles element with no content gracefully."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        scraper = NeonIchibanScraper()
        soup = BeautifulSoup('<a href="/comic/test"></a>', "html.parser")
        card = soup.find("a")
        result = scraper._parse_card(card)
        # Empty text -> should return None
        assert result is None

    def test_parse_card_with_content(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        scraper = NeonIchibanScraper()
        html = """
        <a href="/comic/test-comic">
            <h3 class="title">Test Comic</h3>
            <img src="/covers/test.jpg" />
            <p class="description">A great test comic</p>
        </a>
        """
        soup = BeautifulSoup(html, "html.parser")
        card = soup.find("a")
        result = scraper._parse_card(card)
        assert result is not None
        assert result.title == "Test Comic"
        assert result.cover_image_url == "https://neonichiban.com/covers/test.jpg"
        assert result.description == "A great test comic"
        assert result.url == "https://neonichiban.com/comic/test-comic"


class TestGlobalComixScraper:
    def test_scraper_attributes(self):
        scraper = GlobalComixScraper()
        assert scraper.source_name == "globalcomix"
        assert scraper.base_url == "https://globalcomix.com"

    def test_is_base_scraper(self):
        scraper = GlobalComixScraper()
        assert isinstance(scraper, BaseScraper)

    def test_parse_card_with_content(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("beautifulsoup4 not installed")

        scraper = GlobalComixScraper()
        html = """
        <a href="/title/my-comic">
            <h3 class="title">My Comic</h3>
            <img src="https://cdn.globalcomix.com/cover.jpg" />
            <p>An amazing comic adventure</p>
        </a>
        """
        soup = BeautifulSoup(html, "html.parser")
        card = soup.find("a")
        result = scraper._parse_card(card)
        assert result is not None
        assert result.title == "My Comic"
        assert result.cover_image_url == "https://cdn.globalcomix.com/cover.jpg"
        assert result.description == "An amazing comic adventure"
        assert result.url == "https://globalcomix.com/title/my-comic"
