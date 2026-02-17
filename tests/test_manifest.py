"""Tests for the manifest loading module."""

import csv
import os
import tempfile

import pytest

from src.intocomics.manifest import (
    ComicEntry,
    ComicPage,
    _derive_title,
    _extract_comic_id,
    _extract_page_number,
    load_manifest,
    load_manifest_from_rows,
)


class TestExtractComicId:
    def test_simple_path(self):
        assert _extract_comic_id("NeonIchiban/SomeComic/page_0001") == "NeonIchiban/SomeComic"

    def test_deep_path(self):
        assert _extract_comic_id("CalibreComics/Author/Title Issue 1/page_0042") == "CalibreComics/Author/Title Issue 1"

    def test_no_separator(self):
        assert _extract_comic_id("single_page") == "single_page"

    def test_two_parts(self):
        assert _extract_comic_id("comic/page_0001") == "comic"


class TestExtractPageNumber:
    def test_page_with_numbers(self):
        assert _extract_page_number("comic/page_0042") == 42

    def test_page_zero(self):
        assert _extract_page_number("comic/page_0000") == 0

    def test_no_numbers(self):
        assert _extract_page_number("comic/cover") == 0

    def test_mixed_name(self):
        assert _extract_page_number("comic/img123abc") == 123


class TestDeriveTitle:
    def test_simple_title(self):
        assert _derive_title("NeonIchiban/My Comic") == "My Comic"

    def test_underscores(self):
        assert _derive_title("source/my_great_comic") == "my great comic"

    def test_dashes(self):
        assert _derive_title("source/my-great-comic") == "my great comic"

    def test_single_part(self):
        assert _derive_title("standalone") == "standalone"


class TestLoadManifest:
    def test_load_basic_manifest(self):
        rows = [
            {"canonical_id": "comics/batman/page_0000", "absolute_image_path": "/images/batman/page_0000.png"},
            {"canonical_id": "comics/batman/page_0001", "absolute_image_path": "/images/batman/page_0001.png"},
            {"canonical_id": "comics/batman/page_0002", "absolute_image_path": "/images/batman/page_0002.png"},
            {"canonical_id": "comics/spiderman/page_0000", "absolute_image_path": "/images/spiderman/page_0000.png"},
            {"canonical_id": "comics/spiderman/page_0001", "absolute_image_path": "/images/spiderman/page_0001.png"},
        ]
        comics = load_manifest_from_rows(rows)

        assert len(comics) == 2
        assert "comics/batman" in comics
        assert "comics/spiderman" in comics

        batman = comics["comics/batman"]
        assert batman.title == "batman"
        assert batman.page_count == 3
        assert batman.cover_image_path == "/images/batman/page_0000.png"
        assert batman.source == "manifest"

        spiderman = comics["comics/spiderman"]
        assert spiderman.page_count == 2

    def test_pages_sorted_by_number(self):
        rows = [
            {"canonical_id": "comic/page_0003", "absolute_image_path": "/p3.png"},
            {"canonical_id": "comic/page_0001", "absolute_image_path": "/p1.png"},
            {"canonical_id": "comic/page_0002", "absolute_image_path": "/p2.png"},
        ]
        comics = load_manifest_from_rows(rows)
        pages = comics["comic"].pages
        assert [p.page_number for p in pages] == [1, 2, 3]

    def test_cover_is_first_page(self):
        rows = [
            {"canonical_id": "comic/page_0005", "absolute_image_path": "/p5.png"},
            {"canonical_id": "comic/page_0001", "absolute_image_path": "/p1.png"},
        ]
        comics = load_manifest_from_rows(rows)
        assert comics["comic"].cover_image_path == "/p1.png"

    def test_empty_rows(self):
        comics = load_manifest_from_rows([])
        assert len(comics) == 0

    def test_load_from_csv_file(self, tmp_path):
        csv_path = tmp_path / "manifest.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["canonical_id", "absolute_image_path"])
            writer.writerow(["test/comic/page_0001", "/img/page_0001.png"])
            writer.writerow(["test/comic/page_0002", "/img/page_0002.png"])

        comics = load_manifest(str(csv_path))
        assert len(comics) == 1
        assert "test/comic" in comics
        assert comics["test/comic"].page_count == 2


class TestComicEntry:
    def test_page_count_property(self):
        entry = ComicEntry(
            comic_id="test",
            title="Test Comic",
            pages=[
                ComicPage(canonical_id="test/page_0001", image_path="/p1.png", page_number=1),
                ComicPage(canonical_id="test/page_0002", image_path="/p2.png", page_number=2),
            ],
        )
        assert entry.page_count == 2

    def test_default_values(self):
        entry = ComicEntry(comic_id="test", title="Test")
        assert entry.page_count == 0
        assert entry.cover_image_path is None
        assert entry.description == ""
        assert entry.source == ""
        assert entry.genres == []
