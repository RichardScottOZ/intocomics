"""Scraper for globalcomix.com comic listings."""

import logging
from typing import List, Optional
from urllib.parse import urljoin

from . import BaseScraper, ScrapedComic

logger = logging.getLogger(__name__)


class GlobalComixScraper(BaseScraper):
    """Scrape comic listings from globalcomix.com.

    GlobalComix is a digital comics platform hosting a wide variety of
    independent and international comics. This scraper extracts cover images,
    titles, and descriptions from their public-facing catalog pages.
    """

    source_name = "globalcomix"
    base_url = "https://globalcomix.com"

    def __init__(self, session=None):
        """Initialize the scraper.

        Args:
            session: Optional requests.Session for HTTP requests.
                     If None, a new session will be created on first use.
        """
        self._session = session

    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; IntoComics/0.1; "
                        "+https://github.com/RichardScottOZ/intocomics)"
                    )
                }
            )
        return self._session

    def scrape_listings(self, max_items: int = 50) -> List[ScrapedComic]:
        """Scrape comic listings from the GlobalComix catalog.

        Args:
            max_items: Maximum number of comics to return.

        Returns:
            List of ScrapedComic objects with title, URL, cover image,
            and description where available.
        """
        comics: List[ScrapedComic] = []
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning(
                "beautifulsoup4 not installed; cannot scrape globalcomix.com"
            )
            return comics

        session = self._get_session()
        try:
            resp = session.get(
                urljoin(self.base_url, "/explore"), timeout=30
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error("Failed to fetch GlobalComix listings: %s", e)
            return comics

        soup = BeautifulSoup(resp.text, "html.parser")

        for card in soup.select(
            "a[href*='/comic'], a[href*='/title'], .comic-card, .title-card"
        )[:max_items]:
            try:
                comic = self._parse_card(card)
                if comic:
                    comics.append(comic)
            except Exception as e:
                logger.debug("Failed to parse card: %s", e)

        return comics[:max_items]

    def _parse_card(self, card) -> Optional[ScrapedComic]:
        """Parse a single comic card element into a ScrapedComic."""
        title_el = card.select_one(
            "h2, h3, h4, .title, .comic-title, [class*='title']"
        )
        title = title_el.get_text(strip=True) if title_el else card.get_text(strip=True)
        if not title:
            return None

        href = card.get("href", "")
        if href and not href.startswith("http"):
            href = urljoin(self.base_url, href)

        img_el = card.select_one("img")
        cover_url = None
        if img_el:
            cover_url = img_el.get("src") or img_el.get("data-src")
            if cover_url and not cover_url.startswith("http"):
                cover_url = urljoin(self.base_url, cover_url)

        desc_el = card.select_one(
            "p, .description, .synopsis, [class*='desc']"
        )
        description = desc_el.get_text(strip=True) if desc_el else ""

        return ScrapedComic(
            title=title,
            url=href,
            cover_image_url=cover_url,
            description=description,
            source=self.source_name,
        )

    def scrape_comic_detail(self, url: str) -> Optional[ScrapedComic]:
        """Scrape detailed information for a single comic.

        Args:
            url: URL of the comic detail page on globalcomix.com.

        Returns:
            ScrapedComic with full details, or None on failure.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4 not installed")
            return None

        session = self._get_session()
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Failed to fetch comic detail %s: %s", url, e)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        title_el = soup.select_one(
            "h1, .comic-title, .series-title, [class*='title']"
        )
        title = title_el.get_text(strip=True) if title_el else ""

        img_el = soup.select_one(
            "img.cover, .cover-image img, [class*='cover'] img, "
            "meta[property='og:image']"
        )
        cover_url = None
        if img_el:
            if img_el.name == "meta":
                cover_url = img_el.get("content")
            else:
                cover_url = img_el.get("src") or img_el.get("data-src")
            if cover_url and not cover_url.startswith("http"):
                cover_url = urljoin(self.base_url, cover_url)

        desc_el = soup.select_one(
            ".description, .synopsis, [class*='desc'], "
            "meta[property='og:description']"
        )
        description = ""
        if desc_el:
            if desc_el.name == "meta":
                description = desc_el.get("content", "")
            else:
                description = desc_el.get_text(strip=True)

        genres = []
        for genre_el in soup.select(
            ".genre, .tag, [class*='genre'], [class*='tag']"
        ):
            genre_text = genre_el.get_text(strip=True)
            if genre_text:
                genres.append(genre_text)

        creators = []
        for creator_el in soup.select(
            ".creator, .author, .artist, [class*='creator'], [class*='author']"
        ):
            creator_text = creator_el.get_text(strip=True)
            if creator_text:
                creators.append(creator_text)

        return ScrapedComic(
            title=title,
            url=url,
            cover_image_url=cover_url,
            description=description,
            genres=genres,
            creators=creators,
            source=self.source_name,
        )
