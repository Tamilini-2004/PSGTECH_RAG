"""Unit tests use local HTML and mocked HTTP responses only."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
import requests

from crawler import (
    CrawlConfig,
    Crawler,
    AccessRestrictionError,
    DomainVerificationError,
    KnowledgeBase,
    RobotsPolicyError,
    clean_text,
    content_hash,
    extract_page,
    is_asset_url,
    is_usable_content,
    normalize_url,
)


class FakeResponse:
    def __init__(
        self,
        body: bytes | str = b"",
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._body = body
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.closed = False

    @property
    def content(self) -> bytes:
        return self._body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8")

    def iter_content(self, chunk_size: int = 65536):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: dict[str, Any]):
        self.responses = responses
        self.requested: list[str] = []
        self.request_options: list[dict[str, Any]] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requested.append(url)
        self.request_options.append(kwargs)
        value = self.responses[url]
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self) -> None:
        self.closed = True


ROBOTS = FakeResponse(
    "User-agent: *\nAllow: /\n",
    headers={"Content-Type": "text/plain; charset=utf-8"},
)
HOME_HTML = """<!doctype html>
<html><head><title>PSG College of Technology</title>
<style>.hidden { display:none }</style><script>noise()</script></head>
<body><header>HEADER NOISE</header><nav>NAV NOISE</nav>
<div hidden>HIDDEN NOISE</div><div style="display: none">CSS HIDDEN NOISE</div>
<main><h1>Welcome to PSG College of Technology</h1>
<p>Quality engineering education for every learner.</p>
<a href="/about">About</a><a href="https://www.psgtech.edu/about#top">Alias</a>
<a href="/files/guide.pdf">PDF</a></main><footer>FOOTER NOISE</footer></body></html>"""


def missing_sitemap_responses(host: str) -> dict[str, FakeResponse]:
    """Mock conventional sitemap checks after a successful homepage fetch."""
    return {
        f"https://{host}{path}": FakeResponse(status_code=404)
        for path in ("/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml")
    }


def test_internal_url_is_accepted_and_www_alias_is_canonicalized():
    assert normalize_url("https://www.psgtech.edu/about") == "https://psgtech.edu/about"
    assert normalize_url("http://psgtech.edu/about") == "https://psgtech.edu/about"


def test_external_url_is_rejected():
    assert normalize_url("https://example.org/page") is None
    assert normalize_url("https://psgtech.edu.example.org/") is None


def test_encoded_authentication_route_is_rejected():
    from crawler import is_restricted_url

    assert is_restricted_url("https://psgtech.edu/%6cogin")


def test_fragment_and_tracking_parameters_are_removed_but_query_is_kept():
    assert normalize_url(
        "https://psgtech.edu/search?q=தமிழ்&utm_source=mail&page=2#results"
    ) == "https://psgtech.edu/search?q=%E0%AE%A4%E0%AE%AE%E0%AE%BF%E0%AE%B4%E0%AF%8D&page=2"


def test_path_dot_segments_are_normalized_and_spaces_are_encoded():
    assert normalize_url("https://psgtech.edu/one/../about us/") == (
        "https://psgtech.edu/about%20us/"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://psgtech.edu/files/file.pdf",
        "https://psgtech.edu/files/archive.zip",
        "https://psgtech.edu/media/photo.JPG",
        "https://psgtech.edu/files/report%2Epdf",
    ],
)
def test_file_and_media_assets_are_filtered(url: str):
    assert is_asset_url(url)


def test_html_extraction_removes_structural_noise_and_keeps_main_text():
    title, content = extract_page(HOME_HTML)
    assert title == "PSG College of Technology"
    assert "Welcome to PSG College of Technology" in content
    assert "HEADER NOISE" not in content
    assert "NAV NOISE" not in content
    assert "FOOTER NOISE" not in content
    assert "noise()" not in content
    assert ".hidden" not in content
    assert "HIDDEN NOISE" not in content
    assert "CSS HIDDEN NOISE" not in content


def test_whitespace_and_unicode_are_preserved_cleanly():
    assert clean_text("தமிழ்\u00a0 கல்வி\n\t பொறியியல்") == "தமிழ் கல்வி பொறியியல்"
    title, content = extract_page(
        "<html><head><title>தமிழ் கல்லூரி</title></head>"
        "<body><main><p>தமிழ் மொழியில் கல்வி மற்றும் பொறியியல்</p></main></body></html>"
    )
    assert title == "தமிழ் கல்லூரி"
    assert "தமிழ் மொழியில்" in content
    assert content_hash(content) == content_hash(content)


def test_empty_and_near_empty_content_is_skipped():
    assert extract_page("<html><body><main>   </main></body></html>") == (None, "")
    assert not is_usable_content("Only a few words.")
    assert is_usable_content("This page has enough readable text to be stored.")


def test_failed_request_is_counted_and_does_not_crash(tmp_path: Path, capsys):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": ROBOTS,
            "https://psgtech.edu/": requests.ConnectionError(
                "simulated connection failure"
            ),
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    stats = crawler.crawl()

    assert stats.urls_visited == 1
    assert stats.request_or_database_errors == 1
    assert "simulated connection failure" in capsys.readouterr().out


def test_external_redirect_is_rejected_without_requesting_destination(tmp_path: Path):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": ROBOTS,
            "https://psgtech.edu/": FakeResponse(
                status_code=302,
                headers={"Location": "https://evil.example/steal"},
            ),
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    stats = crawler.crawl()

    assert stats.request_or_database_errors == 1
    assert session.requested == ["https://psgtech.edu/robots.txt", "https://psgtech.edu/"]


def test_rate_limit_response_stops_without_retrying(tmp_path: Path):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": ROBOTS,
            "https://psgtech.edu/": FakeResponse(status_code=429),
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    with pytest.raises(AccessRestrictionError):
        crawler.crawl()

    assert session.requested == ["https://psgtech.edu/robots.txt", "https://psgtech.edu/"]


def test_redirect_to_approved_www_alias_checks_its_robots_and_deduplicates(tmp_path: Path):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": ROBOTS,
            "https://psgtech.edu/": FakeResponse(
                status_code=301,
                headers={"Location": "https://www.psgtech.edu/"},
            ),
            "https://www.psgtech.edu/robots.txt": FakeResponse(
                "User-agent: *\nAllow: /\n",
                headers={"Content-Type": "text/plain"},
            ),
            "https://www.psgtech.edu/": FakeResponse(
                "<html><head><title>PSG College of Technology</title></head>"
                "<body><main>Public information from PSG College of Technology "
                "for students, faculty, and the community.</main></body></html>"
            ),
            **missing_sitemap_responses("www.psgtech.edu"),
        }
    )
    database_path = tmp_path / "knowledge.db"
    crawler = Crawler(
        CrawlConfig(database_path=database_path),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    stats = crawler.crawl()

    connection = sqlite3.connect(database_path)
    rows = connection.execute("SELECT url FROM crawled_resources").fetchall()
    connection.close()
    assert stats.pages_stored_or_updated == 1
    assert rows == [("https://psgtech.edu/",)]
    assert session.requested == [
        "https://psgtech.edu/robots.txt",
        "https://psgtech.edu/",
        "https://www.psgtech.edu/robots.txt",
        "https://www.psgtech.edu/",
        "https://www.psgtech.edu/sitemap.xml",
        "https://www.psgtech.edu/sitemap_index.xml",
        "https://www.psgtech.edu/wp-sitemap.xml",
    ]


def test_unavailable_robots_file_stops_before_homepage(tmp_path: Path):
    session = FakeSession(
        {"https://psgtech.edu/robots.txt": requests.ConnectionError("offline")}
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    with pytest.raises(RobotsPolicyError):
        crawler.crawl()

    assert session.requested == ["https://psgtech.edu/robots.txt"]


def test_robots_http_404_uses_allow_all_policy_and_fetches_seed_page(tmp_path: Path):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": FakeResponse(status_code=404),
            "https://psgtech.edu/": FakeResponse(HOME_HTML),
            **missing_sitemap_responses("psgtech.edu"),
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db", page_limit=1),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    stats = crawler.crawl()

    assert stats.urls_visited == 1
    assert stats.pages_stored_or_updated == 1
    assert session.requested == [
        "https://psgtech.edu/robots.txt",
        "https://psgtech.edu/",
        "https://psgtech.edu/sitemap.xml",
        "https://psgtech.edu/sitemap_index.xml",
        "https://psgtech.edu/wp-sitemap.xml",
    ]


@pytest.mark.parametrize("status_code", [400, 403, 429, 500, 503])
def test_non_404_robots_http_errors_fail_closed(tmp_path: Path, status_code: int):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": FakeResponse(
                status_code=status_code
            )
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    with pytest.raises(RobotsPolicyError, match=f"HTTP {status_code}"):
        crawler.crawl()

    assert session.requested == ["https://psgtech.edu/robots.txt"]


def test_small_dh_key_failure_reports_scoped_tls_diagnostic_and_stops_before_homepage(
    tmp_path: Path,
):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": requests.exceptions.SSLError(
                "[SSL: DH_KEY_TOO_SMALL] dh key too small"
            )
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    with pytest.raises(RobotsPolicyError) as raised:
        crawler.crawl()

    message = str(raised.value)
    assert "DH_KEY_TOO_SMALL" in message
    assert "Diffie-Hellman parameters are too small" in message
    assert "session-scoped legacy TLS compatibility setting was applied" in message
    assert "certificate and hostname verification remained enabled" in message
    assert session.requested == ["https://psgtech.edu/robots.txt"]
    assert session.request_options[0]["verify"] is True
    assert session.request_options[0]["timeout"] == 10
    assert session.request_options[0]["allow_redirects"] is False


def test_other_tls_error_is_reported_and_stops_before_homepage(tmp_path: Path):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": requests.exceptions.SSLError(
                "certificate verify failed"
            )
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    with pytest.raises(RobotsPolicyError, match="TLS request.*certificate verify failed"):
        crawler.crawl()

    assert session.requested == ["https://psgtech.edu/robots.txt"]
    assert session.request_options[0]["verify"] is True


def test_unparseable_robots_response_stops_before_homepage(tmp_path: Path):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": FakeResponse(
                "temporary upstream failure",
                headers={"Content-Type": "text/plain"},
            )
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    with pytest.raises(RobotsPolicyError):
        crawler.crawl()

    assert session.requested == ["https://psgtech.edu/robots.txt"]


def test_seed_page_must_verify_the_expected_site_before_crawling(tmp_path: Path):
    session = FakeSession(
        {
            "https://psgtech.edu/robots.txt": ROBOTS,
            "https://psgtech.edu/": FakeResponse(
                "<html><head><title>Different Website</title></head>"
                "<body><main>Unrelated public information for visitors.</main></body></html>"
            ),
        }
    )
    crawler = Crawler(
        CrawlConfig(database_path=tmp_path / "knowledge.db"),
        session=session,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )

    with pytest.raises(DomainVerificationError):
        crawler.crawl()

    assert session.requested == ["https://psgtech.edu/robots.txt", "https://psgtech.edu/"]


def test_duplicate_urls_upsert_without_duplicate_rows_and_update_values(tmp_path: Path):
    database_path = tmp_path / "knowledge.db"
    database = KnowledgeBase(database_path)
    database.save("https://psgtech.edu/about", "About", "Original content for this page.")
    first_id = database.connection.execute(
        "SELECT id FROM crawled_resources WHERE url = ?",
        ("https://psgtech.edu/about",),
    ).fetchone()[0]
    database.save("https://psgtech.edu/about", "Updated about", "Updated content for this page.")
    rows = database.connection.execute(
        "SELECT id, url, page_title, sanitized_content, content_hash "
        "FROM crawled_resources"
    ).fetchall()
    database.close()

    assert len(rows) == 1
    row_id, url, title, content, digest = rows[0]
    assert row_id == first_id
    assert url == "https://psgtech.edu/about"
    assert title == "Updated about"
    assert content == "Updated content for this page."
    assert digest == content_hash(content)


def test_crawl_discovers_internal_pages_and_rerun_updates_database(tmp_path: Path):
    database_path = tmp_path / "knowledge.db"
    page_html = HOME_HTML
    about_html = (
        "<html><head><title>About the College</title></head><body><main>"
        "The college provides engineering education, research, and community service."
        "</main></body></html>"
    )

    def make_session(home: str, about: str) -> FakeSession:
        return FakeSession(
            {
                "https://psgtech.edu/robots.txt": FakeResponse(
                    "User-agent: *\nAllow: /\n",
                    headers={"Content-Type": "text/plain"},
                ),
                "https://psgtech.edu/": FakeResponse(home),
                "https://psgtech.edu/about": FakeResponse(about),
                **missing_sitemap_responses("psgtech.edu"),
            }
        )

    first = Crawler(
        CrawlConfig(database_path=database_path, page_limit=10),
        session=make_session(page_html, about_html),  # type: ignore[arg-type]
        sleep=lambda _: None,
    )
    first_stats = first.crawl()
    assert first_stats.pages_stored_or_updated == 2
    assert first_stats.skipped_assets_or_non_html == 1

    updated_about = (
        "<html><head><title>Updated About</title></head><body><main>"
        "Updated information about engineering programmes, research, and services."
        "</main></body></html>"
    )
    second = Crawler(
        CrawlConfig(database_path=database_path, page_limit=10),
        session=make_session(page_html, updated_about),  # type: ignore[arg-type]
        sleep=lambda _: None,
    )
    second_stats = second.crawl()

    connection = sqlite3.connect(database_path)
    rows = connection.execute(
        "SELECT url, page_title, sanitized_content FROM crawled_resources ORDER BY url"
    ).fetchall()
    connection.close()
    assert second_stats.pages_stored_or_updated == 2
    assert len(rows) == 2
    assert rows[0][0] == "https://psgtech.edu/"
    assert rows[0][1] == "PSG College of Technology"
    assert "Welcome to PSG College of Technology" in rows[0][2]
    assert rows[1][0] == "https://psgtech.edu/about"
    assert rows[1][1] == "Updated About"
    assert "Updated information" in rows[1][2]
