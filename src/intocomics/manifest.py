"""Load and manage Comic-Analysis manifest data.

The manifest CSV format from Comic-Analysis contains two columns:
  - canonical_id: a unique identifier for each comic page image
  - absolute_image_path: the path (local or S3) to the image

This module parses manifests and groups pages by comic title/series,
providing structured metadata for the recommender engine.
"""

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ComicPage:
    """A single comic page from the manifest."""

    canonical_id: str
    image_path: str
    page_number: int = 0


@dataclass
class ComicEntry:
    """A comic book entry derived from grouped manifest pages."""

    comic_id: str
    title: str
    pages: List[ComicPage] = field(default_factory=list)
    cover_image_path: Optional[str] = None
    description: str = ""
    source: str = ""
    genres: List[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def _extract_comic_id(canonical_id: str) -> str:
    """Extract comic identifier from a canonical_id by removing the page suffix.

    For example:
        'NeonIchiban/SomeComic/page_0001' -> 'NeonIchiban/SomeComic'
        'CalibreComics/Title Issue 1/page_0042' -> 'CalibreComics/Title Issue 1'
    """
    parts = canonical_id.rsplit("/", 1)
    if len(parts) == 2:
        return parts[0]
    return canonical_id


def _extract_page_number(canonical_id: str) -> int:
    """Extract page number from canonical_id."""
    basename = canonical_id.rsplit("/", 1)[-1]
    match = re.search(r"(\d+)", basename)
    if match:
        return int(match.group(1))
    return 0


def _derive_title(comic_id: str) -> str:
    """Derive a human-readable title from the comic_id path."""
    parts = comic_id.split("/")
    # Use the last meaningful directory as the title
    title = parts[-1] if parts else comic_id
    # Clean up common path artifacts
    title = title.replace("_", " ").replace("-", " ")
    return title.strip()


def load_manifest(manifest_path: str) -> Dict[str, ComicEntry]:
    """Load a Comic-Analysis manifest CSV and group pages into ComicEntry objects.

    Args:
        manifest_path: Path to the manifest CSV file with columns
                       (canonical_id, absolute_image_path).

    Returns:
        Dictionary mapping comic_id to ComicEntry.
    """
    comics: Dict[str, ComicEntry] = {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical_id = row["canonical_id"]
            image_path = row["absolute_image_path"]

            comic_id = _extract_comic_id(canonical_id)
            page_number = _extract_page_number(canonical_id)

            page = ComicPage(
                canonical_id=canonical_id,
                image_path=image_path,
                page_number=page_number,
            )

            if comic_id not in comics:
                comics[comic_id] = ComicEntry(
                    comic_id=comic_id,
                    title=_derive_title(comic_id),
                    pages=[],
                    source="manifest",
                )

            comics[comic_id].pages.append(page)

    # Sort pages and assign cover images
    for comic in comics.values():
        comic.pages.sort(key=lambda p: p.page_number)
        if comic.pages:
            comic.cover_image_path = comic.pages[0].image_path

    return comics


def load_manifest_from_rows(rows: List[Dict[str, str]]) -> Dict[str, ComicEntry]:
    """Load manifest data from a list of row dicts (for testing/in-memory use).

    Args:
        rows: List of dicts with 'canonical_id' and 'absolute_image_path' keys.

    Returns:
        Dictionary mapping comic_id to ComicEntry.
    """
    comics: Dict[str, ComicEntry] = {}

    for row in rows:
        canonical_id = row["canonical_id"]
        image_path = row["absolute_image_path"]

        comic_id = _extract_comic_id(canonical_id)
        page_number = _extract_page_number(canonical_id)

        page = ComicPage(
            canonical_id=canonical_id,
            image_path=image_path,
            page_number=page_number,
        )

        if comic_id not in comics:
            comics[comic_id] = ComicEntry(
                comic_id=comic_id,
                title=_derive_title(comic_id),
                pages=[],
                source="manifest",
            )

        comics[comic_id].pages.append(page)

    for comic in comics.values():
        comic.pages.sort(key=lambda p: p.page_number)
        if comic.pages:
            comic.cover_image_path = comic.pages[0].image_path

    return comics
