"""Base scraper interface for comic websites."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScrapedComic:
    """A comic discovered by scraping a website."""

    title: str
    url: str
    cover_image_url: Optional[str] = None
    description: str = ""
    genres: List[str] = field(default_factory=list)
    creators: List[str] = field(default_factory=list)
    source: str = ""


class BaseScraper(ABC):
    """Abstract base class for comic website scrapers."""

    source_name: str = ""
    base_url: str = ""

    @abstractmethod
    def scrape_listings(self, max_items: int = 50) -> List[ScrapedComic]:
        """Scrape comic listings from the website.

        Args:
            max_items: Maximum number of comics to scrape.

        Returns:
            List of ScrapedComic objects.
        """

    @abstractmethod
    def scrape_comic_detail(self, url: str) -> Optional[ScrapedComic]:
        """Scrape detailed info for a single comic.

        Args:
            url: URL of the comic detail page.

        Returns:
            ScrapedComic with full details, or None on failure.
        """
