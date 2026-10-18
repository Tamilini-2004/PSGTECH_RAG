"""Polite, source-traceable crawler for the public PSG Tech website.

Importing this module does not start a crawl. Run python crawler.py to
create/update the local SQLite knowledge base.

"""


from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import posixpath
import re
import sqlite3
import ssl
import time
import urllib.robotparser
import zlib
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from requests.adapters import HTTPAdapter


START_URL = "https://psgtech.edu/"
ALLOWED_HOSTS = frozenset({"psgtech.edu", "www.psgtech.edu"})
CANONICAL_HOST = "psgtech.edu"
FOCUS_SEED_CANDIDATES = (
    ("Programme and degree catalogue", "https://psgtech.edu/programme.php"),
    ("Programmes offered", "https://psgtech.edu/programme_offered.php"),
    ("Department index", "https://psgtech.edu/index.php"),
    ("Department heads", "https://psgtech.edu/Heads.php"),
    ("Faculty directory", "https://psgtech.edu/facultyDetails.php"),
    ("Admissions information", "https://psgtech.edu/admission.php"),
    ("2026 postgraduate admissions", "https://psgtech.edu/pgadmission2026/"),
    ("2026 integrated programme admissions", "https://psgtech.edu/BScMScAdmission2026/"),
    ("Research directory", "https://psgtech.edu/Grd/Research.php"),
    ("Research and development", "https://psgtech.edu/Research/ird.html"),
    ("Research centres", "https://psgtech.edu/Research/coe.html"),
    ("Placement statistics", "https://psgtech.edu/placements/Statistics.php"),
    ("Placement and training", "https://psgtech.edu/placements/index.php"),
    ("Placement recruiters", "https://psgtech.edu/placements/recrs.php"),
    ("Placement programmes", "https://psgtech.edu/placements/programmes.php"),
    ("Hostel home", "https://psgtech.edu/hostel/"),
    ("Hostel rules", "https://psgtech.edu/hostel/general.php"),
    ("Hostel discipline rules", "https://psgtech.edu/hostel/discipline.php"),
    ("Hostel mess information", "https://psgtech.edu/hostel/mess.php"),
    ("Student coding club", "https://psgtech.edu/placements/codingclub.php"),
    ("Official PSG Tech website directory", "https://psgtech.edu/websiteUrls.php"),
)
SITEMAP_FALLBACK_PATHS = ("/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml")
MAX_SITEMAP_DOCUMENTS = 25
MAX_SITEMAP_LOCATIONS = 50_000
USER_AGENT = "PSGTech-Academic-Crawler/1.0 (educational research)"
DEFAULT_DATABASE = Path(__file__).with_name("psgtech_knowledge_base.db")
TRACKING_PARAMETERS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "fbclid", "gclid", "dclid", "msclkid",
    }
)
ASSET_EXTENSIONS = frozenset(
    {
        ".7z", ".aac", ".apk", ".avi", ".bmp", ".css", ".csv", ".dat",
        ".deb", ".dmg",
        ".doc", ".docx", ".eot", ".eps", ".exe", ".flac", ".flv",
        ".gif", ".gz", ".ico", ".jpeg", ".jpg", ".js", ".json", ".m4a",
        ".ics", ".iso", ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".odp",
        ".ods", ".odt", ".ogg",
        ".otf", ".pdf", ".png", ".ppt", ".pptx", ".ps", ".psd", ".rar",
        ".rss", ".sh", ".svg",
        ".swf", ".tar", ".tif", ".tiff", ".ttf", ".txt", ".wav", ".webm", ".webp",
        ".webmanifest", ".woff", ".woff2", ".xls", ".xlsx", ".xml", ".yaml",
        ".yml", ".zip",
    }
)
RESTRICTED_SEGMENTS = frozenset(
    {
        "admin", "administrator", "auth", "account", "dashboard", "login",
        "logout", "signin", "sign-in", "wp-admin", "wp-login.php",
    }
)
NOISE_TAGS = (
    "script", "style", "nav", "footer", "header", "noscript", "svg",
    "iframe", "form", "button", "input", "select", "textarea", "template",
)
MIN_CONTENT_CHARS = 20
HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class CrawlError(Exception):
    """Base class for crawl failures that should be reported clearly."""


class RobotsPolicyError(CrawlError):
    """Raised when robots.txt cannot be retrieved or safely interpreted."""


class LegacyTlsAdapter(HTTPAdapter):
    """Use a session-scoped TLS compatibility level while verifying certificates.

    OpenSSL security level 1 accepts smaller DH parameters than the default
    level 2. This weakens TLS negotiation for this adapter, but does not change
    process-wide OpenSSL settings or disable certificate/hostname checks.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED
        self.ssl_context.check_hostname = True
        self.ssl_context.set_ciphers("DEFAULT@SECLEVEL=1")
        super().__init__(*args, **kwargs)

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:
        pool_kwargs["ssl_context"] = self.ssl_context
        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        proxy_kwargs.setdefault("ssl_context", self.ssl_context)
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def _tls_failure_message(url: str, error: requests.exceptions.SSLError) -> str:
    """Explain a verified TLS failure after the scoped compatibility attempt."""
    if "DH_KEY_TOO_SMALL" in str(error).upper():
        return (
            f"TLS handshake for {url} failed with DH_KEY_TOO_SMALL: the server's "
            "Diffie-Hellman parameters are too small for this connection. "
            "The session-scoped legacy TLS compatibility setting was applied, while "
            "certificate and hostname verification remained enabled. Robots policy "
            "could not be verified, so the crawl stopped before fetching pages. Ask "
            "the site administrator to update TLS (for example, use modern ECDHE or "
            "sufficiently strong DH parameters), then retry."
        )
    return (
        f"TLS request for {url} failed: {error}. Certificate and hostname "
        "verification remain enabled; robots policy could not be verified, so the "
        "crawl stopped before fetching pages. Resolve the TLS or certificate "
        "problem and retry."
    )


class ResponseTooLargeError(CrawlError):
    """Raised when a response exceeds the configured byte limit."""


class UnsafeRedirectError(CrawlError):
    """Raised when a redirect leaves the approved host set."""


class AccessRestrictionError(CrawlError):
    """Raised when the server signals access denial or rate limiting."""


class DomainVerificationError(CrawlError):
    """Raised when the seed page does not identify the expected institution."""


@dataclass
class CrawlConfig:
    """Options that bound and identify a crawl."""

    start_url: str = START_URL
    database_path: Path = DEFAULT_DATABASE
    page_limit: int = 100
    request_timeout: float = 10.0
    min_request_interval: float = 1.0
    max_response_bytes: int = 2_000_000
    max_pdf_bytes: int = 20_000_000
    max_pdf_pages: int = 500
    max_pdf_text_chars: int = 10_000_000
    max_sitemap_bytes: int = 10_000_000
    include_pdfs: bool = False
    include_focus_seeds: bool = False
    user_agent: str = USER_AGENT
    max_redirects: int = 5

    def __post_init__(self) -> None:
        if self.page_limit < 1:
            raise ValueError("page_limit must be at least 1")
        if not 0 < self.request_timeout <= 10:
            raise ValueError("request_timeout must be greater than 0 and at most 10 seconds")
        if self.min_request_interval < 1:
            raise ValueError("min_request_interval must be at least 1 second")
        if self.max_response_bytes < 1:
            raise ValueError("max_response_bytes must be at least 1 byte")
        if self.max_pdf_bytes < 1:
            raise ValueError("max_pdf_bytes must be at least 1 byte")
        if self.max_pdf_pages < 1:
            raise ValueError("max_pdf_pages must be at least 1")
        if self.max_pdf_text_chars < 1:
            raise ValueError("max_pdf_text_chars must be at least 1 character")
        if self.max_sitemap_bytes < 1:
            raise ValueError("max_sitemap_bytes must be at least 1 byte")
        if self.max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
        if not self.user_agent.strip():
            raise ValueError("user_agent must not be empty")


@dataclass
class CrawlStats:
    """Counters printed at the end of a crawl."""

    urls_visited: int = 0
    html_pages_fetched: int = 0
    pdf_documents_fetched: int = 0
    pages_stored_or_updated: int = 0
    skipped_assets_or_non_html: int = 0
    robots_disallowed: int = 0
    request_or_database_errors: int = 0


def is_approved_host(hostname: str | None) -> bool:
    """Return whether a hostname is one of the explicitly approved hosts."""
    return (hostname or "").rstrip(".").lower() in ALLOWED_HOSTS


def normalize_url(
    url: str,
    base_url: str | None = None,
    *,
    canonicalize_host: bool = True,
) -> str | None:
    """Resolve and canonicalize an approved URL, or return None if unsafe.

    HTTP links are upgraded to HTTPS, www is collapsed to the approved
    canonical hostname, fragments and common tracking parameters are removed,
    and dot segments in paths are normalized. Meaningful query parameters are
    retained in their original order.
    """
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        absolute_url = urljoin(base_url, url.strip()) if base_url else url.strip()
        parsed = urlsplit(absolute_url)
        hostname = parsed.hostname
        if parsed.scheme.lower() not in {"http", "https"} or not is_approved_host(hostname):
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        # Accessing .port validates malformed port values.
        port = parsed.port
        if port not in (None, 80, 443):
            return None
        path = parsed.path or "/"
        had_trailing_slash = path.endswith("/")
        normalized_path = posixpath.normpath(path)
        if not normalized_path.startswith("/"):
            normalized_path = "/" + normalized_path
        if had_trailing_slash and normalized_path != "/" and not normalized_path.endswith("/"):
            normalized_path += "/"
        if normalized_path == ".":
            normalized_path = "/"
        normalized_path = quote(
            normalized_path,
            safe="/%:@!$&'()*+,;=-._~",
        )

        query_parts = parse_qsl(parsed.query, keep_blank_values=True)
        filtered_query = [
            (key, value)
            for key, value in query_parts
            if key.casefold() not in TRACKING_PARAMETERS
        ]
        query = urlencode(filtered_query, doseq=True)
        normalized_host = CANONICAL_HOST if canonicalize_host else hostname.lower()
        return urlunsplit(("https", normalized_host, normalized_path, query, ""))
    except (TypeError, ValueError):
        return None


def is_asset_url(url: str) -> bool:
    """Check the final path suffix against known non-HTML assets."""
    try:
        path = unquote(urlsplit(url).path).rstrip("/").lower()
        return Path(path).suffix in ASSET_EXTENSIONS
    except ValueError:
        return True


def is_pdf_url(url: str) -> bool:
    """Recognize PDF paths and download endpoints with a PDF-valued query."""
    try:
        parsed = urlsplit(url)
        candidates = [parsed.path, *(value for _key, value in parse_qsl(parsed.query))]
        return any(unquote(value).strip().casefold().endswith(".pdf") for value in candidates)
    except ValueError:
        return False


def is_restricted_url(url: str) -> bool:
    """Reject common authentication, administration, and private routes."""
    try:
        segments = {
            segment.casefold()
            for segment in unquote(urlsplit(url).path).split("/")
            if segment
        }
        return bool(segments & RESTRICTED_SEGMENTS)
    except ValueError:
        return True


def clean_text(text: str) -> str:
    """Collapse Unicode whitespace, including tabs and non-breaking spaces."""
    return " ".join(text.replace("\xa0", " ").split())


def extract_page(html: str | bytes) -> tuple[str | None, str]:
    """Extract a document title and readable page text from HTML safely."""
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else None
    title = title or None

    for element in soup.find_all(NOISE_TAGS):
        element.decompose()
    for element in soup.find_all(True):
        if element.attrs is None:
            continue
        inline_style = element.get("style", "")
        if (
            element.has_attr("hidden")
            or str(element.get("aria-hidden", "")).casefold() == "true"
            or re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", inline_style, re.I)
        ):
            element.decompose()

    content_root = soup.find("main") or soup.find("article") or soup.body
    if content_root is None:
        return title, ""
    content = clean_text(content_root.get_text(" ", strip=True))
    return title, content


def is_usable_content(content: str) -> bool:
    """Ignore blank and near-empty pages."""
    return len(content.strip()) >= MIN_CONTENT_CHARS


def content_hash(content: str) -> str:
    """Return a stable SHA-256 digest of UTF-8 cleaned text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class KnowledgeBase:
    """Small SQLite persistence layer with URL-keyed upserts."""

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS crawled_resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                page_title TEXT,
                sanitized_content TEXT NOT NULL,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_hash TEXT
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_crawled_resources_url "
            "ON crawled_resources(url)"
        )
        self.connection.commit()

    def save(self, url: str, title: str | None, content: str) -> None:
        """Insert or update a page while preserving its existing row ID."""
        digest = content_hash(content)
        try:
            self.connection.execute(
                """
                INSERT INTO crawled_resources
                    (url, page_title, sanitized_content, content_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    page_title = excluded.page_title,
                    sanitized_content = excluded.sanitized_content,
                    extracted_at = CURRENT_TIMESTAMP,
                    content_hash = excluded.content_hash
                """,
                (url, title, content, digest),
            )
            self.connection.commit()
        except sqlite3.Error:
            self.connection.rollback()
            raise

    def close(self) -> None:
        self.connection.close()


class RobotsManager:
    """Fetch and cache strict per-host robots policies."""

    def __init__(
        self,
        session: requests.Session,
        config: CrawlConfig,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.session = session
        self.config = config
        self.sleep = sleep
        self.parsers: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.crawl_delays: dict[str, float] = {}

    def _pace(self, hostname: str) -> None:
        """Wait after each request, including unsuccessful requests."""
        interval = max(
            self.config.min_request_interval,
            self.crawl_delays.get(hostname, 0.0),
        )
        self.sleep(interval)

    def _fetch_text(self, url: str, hostname: str) -> str:
        response: Any = None
        try:
            response = self.session.get(
                url,
                headers={"User-Agent": self.config.user_agent},
                timeout=min(10.0, self.config.request_timeout),
                verify=True,
                allow_redirects=False,
                stream=True,
            )
            if response.status_code in REDIRECT_STATUSES:
                location = response.headers.get("Location")
                redirect_target = normalize_url(
                    location or "", base_url=url, canonicalize_host=False
                )
                if redirect_target is None:
                    raise RobotsPolicyError(
                        f"robots.txt for {hostname} redirected outside the approved hosts"
                    )
                target_host = urlsplit(redirect_target).hostname or hostname
                if target_host != hostname:
                    raise RobotsPolicyError(
                        f"robots.txt for {hostname} redirected to another host"
                    )
                response.close()
                response = None
                self._pace(hostname)
                response = self.session.get(
                    redirect_target,
                    headers={"User-Agent": self.config.user_agent},
                    timeout=min(10.0, self.config.request_timeout),
                    verify=True,
                    allow_redirects=False,
                    stream=True,
                )
            if response.status_code == 404:
                return "User-agent: *\nAllow: /\n"
            if response.status_code != 200:
                raise RobotsPolicyError(
                    f"robots.txt request for {hostname} returned HTTP {response.status_code}"
                )
            try:
                media_type = response.headers.get("Content-Type", "")
                if media_type and media_type.split(";", 1)[0].strip().lower() != "text/plain":
                    raise RobotsPolicyError(
                        f"Unexpected robots.txt content type for {hostname}: {media_type}"
                    )
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self.config.max_response_bytes:
                        raise RobotsPolicyError(
                            f"robots.txt for {hostname} exceeds the configured size limit"
                        )
                    chunks.append(chunk)
                raw = b"".join(chunks)
                return raw.decode("utf-8-sig")
            except (AttributeError, requests.RequestException, UnicodeError) as exc:
                raise RobotsPolicyError(
                    f"Could not read robots.txt for {hostname}: {exc}"
                ) from exc
        except requests.exceptions.SSLError as exc:
            raise RobotsPolicyError(_tls_failure_message(url, exc)) from exc
        except (requests.RequestException, OSError) as exc:
            raise RobotsPolicyError(
                f"Could not retrieve robots.txt for {hostname}: {exc}"
            ) from exc
        finally:
            if response is not None:
                response.close()
            self._pace(hostname)

    def ensure_loaded(self, hostname: str) -> None:
        """Load rules once; fail closed on any unavailable or ambiguous file."""
        if hostname in self.parsers:
            return
        if not is_approved_host(hostname):
            raise RobotsPolicyError(f"Host is not approved: {hostname}")
        robots_url = f"https://{hostname}/robots.txt"
        body = self._fetch_text(robots_url, hostname)
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            for line_number, line in enumerate(body.splitlines(), start=1):
                directive = line.split("#", 1)[0].strip()
                if directive and not re.match(r"^[A-Za-z][A-Za-z0-9_-]*\s*:", directive):
                    raise RobotsPolicyError(
                        f"Unrecognized robots.txt line {line_number} for {hostname}"
                    )
            parser.parse(body.splitlines())
        except RobotsPolicyError:
            raise
        except Exception as exc:
            raise RobotsPolicyError(
                f"Could not parse robots.txt for {hostname}: {exc}"
            ) from exc
        crawl_delay = parser.crawl_delay(self.config.user_agent)
        if crawl_delay is None:
            crawl_delay = parser.crawl_delay("*")
        if crawl_delay is not None:
            self.crawl_delays[hostname] = max(0.0, float(crawl_delay))
        self.parsers[hostname] = parser

    def allowed(self, url: str) -> bool:
        """Return robots permission for a normalized URL."""
        hostname = urlsplit(url).hostname
        if not hostname or not is_approved_host(hostname):
            return False
        self.ensure_loaded(hostname)
        return self.parsers[hostname].can_fetch(self.config.user_agent, url)

    def pace_after_page_request(self, hostname: str) -> None:
        """Apply the configured and robots-provided delay after a page request."""
        self._pace(hostname)


class Crawler:
    """Breadth-first crawler restricted to PSG Tech's approved hostnames."""

    def __init__(
        self,
        config: CrawlConfig | None = None,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config or CrawlConfig()
        self.session = session or requests.Session()
        self._owns_session = session is None
        mount = getattr(self.session, "mount", None)
        if callable(mount):
            mount("https://", LegacyTlsAdapter())
        self.sleep = sleep
        self.robots = RobotsManager(self.session, self.config, sleep=sleep)
        self.stats = CrawlStats()
        self.visited: set[str] = set()
        self.queued: set[str] = set()
        self.considered: set[str] = set()
        configured_start = normalize_url(self.config.start_url)
        if configured_start == normalize_url(START_URL):
            seed_urls = [configured_start]
            if self.config.include_focus_seeds:
                seed_urls.extend(
                    normalize_url(url) for _, url in FOCUS_SEED_CANDIDATES
                )
            self.initial_seeds = tuple(dict.fromkeys(seed for seed in seed_urls if seed))
        else:
            self.initial_seeds = (configured_start,) if configured_start else ()
        self._sitemap_discovery_done = False
        self.database: KnowledgeBase | None = None

    def close(self) -> None:
        if self.database is not None:
            self.database.close()
            self.database = None
        if self._owns_session:
            self.session.close()

    def _log(self, message: str) -> None:
        print(message)

    def _eligible(self, url: str) -> str | None:
        normalized = normalize_url(url)
        if normalized is None or is_restricted_url(normalized):
            return None
        if normalized in self.considered:
            return None
        self.considered.add(normalized)
        is_pdf = is_pdf_url(normalized)
        if (is_asset_url(normalized) or is_pdf) and not (
            self.config.include_pdfs and is_pdf
        ):
            self.stats.skipped_assets_or_non_html += 1
            return None
        if not self.robots.allowed(normalized):
            self.stats.robots_disallowed += 1
            self._log(f"[robots] Disallowed: {normalized}")
            return None
        return normalized

    def _read_limited_response(
        self, response: Any, *, max_bytes: int | None = None
    ) -> bytes:
        limit = self.config.max_response_bytes if max_bytes is None else max_bytes
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > limit:
                    raise ResponseTooLargeError("Response exceeds configured size limit")
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        iterator: Iterable[bytes] = response.iter_content(chunk_size=64 * 1024)
        for chunk in iterator:
            if not chunk:
                continue
            total += len(chunk)
            if total > limit:
                raise ResponseTooLargeError("Response exceeds configured size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    def _fetch_html(self, start_url: str) -> tuple[str, bytes] | None:
        """Fetch HTML manually following only safe, robots-allowed redirects."""
        current = start_url
        chain: set[str] = set()
        redirects = 0
        while True:
            if current in chain:
                raise CrawlError(f"Redirect loop detected at {current}")
            chain.add(current)
            host = urlsplit(current).hostname or ""
            if not self.robots.allowed(current):
                self.stats.robots_disallowed += 1
                self._log(f"[robots] Disallowed: {current}")
                return None
            response: Any = None
            request_already_paced = False
            try:
                response = self.session.get(
                    current,
                    headers={
                        "User-Agent": self.config.user_agent,
                        "Accept": "text/html,application/xhtml+xml",
                    },
                    timeout=min(10.0, self.config.request_timeout),
                    verify=True,
                    allow_redirects=False,
                    stream=True,
                )
                if response.status_code in REDIRECT_STATUSES:
                    if redirects >= self.config.max_redirects:
                        raise CrawlError(f"Too many redirects from {start_url}")
                    location = response.headers.get("Location")
                    target = normalize_url(
                        location or "", base_url=current, canonicalize_host=False
                    )
                    if target is None:
                        raise UnsafeRedirectError(
                            f"Rejected redirect outside approved hosts from {current}"
                        )
                    if is_asset_url(target) or is_restricted_url(target):
                        raise CrawlError(
                            f"Rejected redirect to an asset or restricted route: {target}"
                        )
                    response.close()
                    response = None
                    # The redirect response is a completed request. Pace before
                    # fetching another host's robots file or the redirect target.
                    self.robots.pace_after_page_request(host)
                    request_already_paced = True
                    if not self.robots.allowed(target):
                        self.stats.robots_disallowed += 1
                        self._log(f"[robots] Redirect target disallowed: {target}")
                        return None
                    redirects += 1
                    current = target
                    continue
                if response.status_code in {401, 403, 429}:
                    raise AccessRestrictionError(
                        f"HTTP {response.status_code} for {current}; stopping without retrying"
                    )
                if response.status_code != 200:
                    raise CrawlError(f"HTTP {response.status_code} for {current}")
                content_type = response.headers.get("Content-Type", "")
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type not in HTML_CONTENT_TYPES:
                    self.stats.skipped_assets_or_non_html += 1
                    self._log(f"[skip] Non-HTML response ({media_type or 'unknown'}): {current}")
                    return None
                body = self._read_limited_response(response)
                self.stats.html_pages_fetched += 1
                return current, body
            finally:
                if response is not None:
                    response.close()
                if not request_already_paced:
                    self.robots.pace_after_page_request(host)

    def _fetch_pdf(self, start_url: str) -> tuple[str, bytes] | None:
        """Fetch an explicitly enabled PDF after robots, host, and size checks."""
        current = start_url
        chain: set[str] = set()
        redirects = 0
        while True:
            if current in chain:
                raise CrawlError(f"PDF redirect loop detected at {current}")
            chain.add(current)
            host = urlsplit(current).hostname or ""
            if not self.robots.allowed(current):
                self.stats.robots_disallowed += 1
                self._log(f"[robots] Disallowed PDF: {current}")
                return None

            response: Any = None
            request_already_paced = False
            try:
                response = self.session.get(
                    current,
                    headers={
                        "User-Agent": self.config.user_agent,
                        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
                    },
                    timeout=min(10.0, self.config.request_timeout),
                    verify=True,
                    allow_redirects=False,
                    stream=True,
                )
                if response.status_code in REDIRECT_STATUSES:
                    if redirects >= self.config.max_redirects:
                        raise CrawlError(f"Too many PDF redirects from {start_url}")
                    location = response.headers.get("Location")
                    target = normalize_url(
                        location or "", base_url=current, canonicalize_host=False
                    )
                    if (
                        target is None
                        or is_restricted_url(target)
                        or (is_asset_url(target) and not is_pdf_url(target))
                    ):
                        raise UnsafeRedirectError(
                            f"Rejected PDF redirect outside the approved document scope: {location}"
                        )
                    response.close()
                    response = None
                    self.robots.pace_after_page_request(host)
                    request_already_paced = True
                    if not self.robots.allowed(target):
                        self.stats.robots_disallowed += 1
                        self._log(f"[robots] PDF redirect target disallowed: {target}")
                        return None
                    redirects += 1
                    current = target
                    continue
                if response.status_code in {401, 403, 429}:
                    raise AccessRestrictionError(
                        f"HTTP {response.status_code} for PDF {current}; stopping without retrying"
                    )
                if response.status_code != 200:
                    raise CrawlError(f"PDF request returned HTTP {response.status_code} for {current}")
                media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if media_type not in {"application/pdf", "application/octet-stream"}:
                    self.stats.skipped_assets_or_non_html += 1
                    self._log(
                        f"[skip] PDF URL returned non-PDF content ({media_type or 'unknown'}): {current}"
                    )
                    return None
                body = self._read_limited_response(
                    response, max_bytes=self.config.max_pdf_bytes
                )
                if b"%PDF-" not in body[:1024]:
                    raise CrawlError(f"The response at {current} is not a valid PDF file")
                self.stats.pdf_documents_fetched += 1
                return current, body
            finally:
                if response is not None:
                    response.close()
                if not request_already_paced:
                    self.robots.pace_after_page_request(host)

    def _extract_pdf(self, body: bytes, source_url: str) -> tuple[str, str]:
        """Extract bounded text from a readable PDF without changing the schema."""
        try:
            reader = PdfReader(io.BytesIO(body), strict=False)
            if reader.is_encrypted:
                raise CrawlError(f"The PDF is encrypted: {source_url}")
            page_count = len(reader.pages)
            if page_count < 1:
                raise CrawlError(f"The PDF contains no pages: {source_url}")
            if page_count > self.config.max_pdf_pages:
                raise CrawlError(
                    f"The PDF has {page_count} pages, above the configured limit "
                    f"of {self.config.max_pdf_pages}: {source_url}"
                )
            extracted_pages: list[str] = []
            total_chars = 0
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = clean_text(page.extract_text() or "")
                except Exception as exc:
                    raise CrawlError(
                        f"Text extraction failed on PDF page {page_number} at {source_url}: {exc}"
                    ) from exc
                if page_text:
                    total_chars += len(page_text)
                    if total_chars > self.config.max_pdf_text_chars:
                        raise CrawlError(
                            f"Extracted PDF text exceeds the configured character limit: {source_url}"
                        )
                    extracted_pages.append(page_text)
        except CrawlError:
            raise
        except Exception as exc:
            raise CrawlError(
                f"Could not parse PDF at {source_url} ({type(exc).__name__}): {exc}"
            ) from exc

        content = clean_text("\n".join(extracted_pages))
        if not is_usable_content(content):
            raise CrawlError(
                f"No selectable text was found in PDF {source_url}; scanned files need OCR"
            )
        try:
            metadata_title = (
                clean_text(str(reader.metadata.title or ""))
                if reader.metadata
                else ""
            )
        except Exception:
            metadata_title = ""
        filename = posixpath.splitext(posixpath.basename(unquote(urlsplit(source_url).path)))[0]
        fallback_title = clean_text(filename.replace("_", " ").replace("-", " "))
        return metadata_title or fallback_title or "PSG Tech PDF document", content

    def _discover_links(self, html: bytes, page_url: str, queue: list[str]) -> None:
        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.find_all("a", href=True):
            candidate = normalize_url(anchor.get("href", ""), base_url=page_url)
            if candidate is None or candidate in self.queued or candidate in self.visited:
                continue
            eligible = self._eligible(candidate)
            if eligible is not None:
                queue.append(eligible)
                self.queued.add(eligible)

    @staticmethod
    def _sitemap_loc_values(document: bytes) -> tuple[bool, list[str]]:
        """Parse a sitemap or sitemap index without resolving external entities."""
        lowered = document.lower()
        if b"<!doctype" in lowered or b"<!entity" in lowered:
            raise ET.ParseError(
                "DTD and entity declarations are not accepted in sitemaps"
            )
        root = ET.fromstring(document)
        if not isinstance(root.tag, str):
            raise ET.ParseError("Sitemap XML has no valid document element")
        root_name = root.tag.rsplit("}", 1)[-1].casefold()
        locations = [
            (element.text or "").strip()
            for element in root.iter()
            if isinstance(element.tag, str)
            and element.tag.rsplit("}", 1)[-1].casefold() == "loc"
            and (element.text or "").strip()
        ]
        return root_name == "sitemapindex", locations

    def _read_sitemap_response(self, response: Any) -> bytes:
        """Read a bounded sitemap body, including a directly gzipped sitemap."""
        raw = self._read_limited_response(
            response, max_bytes=self.config.max_sitemap_bytes
        )
        if raw.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
                document = compressed.read(self.config.max_sitemap_bytes + 1)
            if len(document) > self.config.max_sitemap_bytes:
                raise ResponseTooLargeError(
                    "Decompressed sitemap exceeds configured size limit"
                )
            return document
        return raw

    def _fetch_sitemap(self, sitemap_url: str) -> bytes | None:
        """Fetch a same-site sitemap with robots checks and manually vetted redirects."""
        current = sitemap_url
        visited_redirects: set[str] = set()
        redirects = 0
        while True:
            if current in visited_redirects:
                self._log(f"[sitemap] Redirect loop rejected: {current}")
                self.stats.request_or_database_errors += 1
                return None
            visited_redirects.add(current)
            host = urlsplit(current).hostname or ""
            if not is_approved_host(host):
                self._log(f"[sitemap] Unapproved sitemap host rejected: {current}")
                return None
            if not self.robots.allowed(current):
                self.stats.robots_disallowed += 1
                self._log(f"[robots] Sitemap disallowed: {current}")
                return None

            response: Any = None
            request_already_paced = False
            try:
                response = self.session.get(
                    current,
                    headers={
                        "User-Agent": self.config.user_agent,
                        "Accept": "application/xml,text/xml,application/gzip,*/*;q=0.5",
                    },
                    timeout=min(10.0, self.config.request_timeout),
                    verify=True,
                    allow_redirects=False,
                    stream=True,
                )
                if response.status_code in REDIRECT_STATUSES:
                    if redirects >= self.config.max_redirects:
                        raise CrawlError(f"Too many sitemap redirects from {sitemap_url}")
                    location = response.headers.get("Location")
                    target = normalize_url(
                        location or "", base_url=current, canonicalize_host=False
                    )
                    if target is None:
                        raise UnsafeRedirectError(
                            f"Sitemap redirect outside approved hosts from {current}"
                        )
                    response.close()
                    response = None
                    self.robots.pace_after_page_request(host)
                    request_already_paced = True
                    redirects += 1
                    current = target
                    continue
                if response.status_code == 404:
                    self._log(f"[sitemap] Not found: {current}")
                    return None
                if response.status_code != 200:
                    raise CrawlError(
                        f"Sitemap request for {current} returned HTTP {response.status_code}"
                    )
                return self._read_sitemap_response(response)
            except (
                requests.RequestException,
                CrawlError,
                OSError,
                EOFError,
                zlib.error,
            ) as exc:
                self.stats.request_or_database_errors += 1
                self._log(f"[sitemap] Could not safely retrieve {current}: {exc}")
                return None
            finally:
                if response is not None:
                    response.close()
                if not request_already_paced:
                    self.robots.pace_after_page_request(host)

    def _discover_sitemap_urls(self, queue: list[str], homepage_url: str) -> None:
        """Inject bounded, robots-allowed internal URLs from the site's sitemap."""
        if self._sitemap_discovery_done:
            return
        self._sitemap_discovery_done = True

        hostname = urlsplit(homepage_url).hostname or ""
        parser = self.robots.parsers.get(hostname)
        advertised = parser.site_maps() if parser is not None else None
        candidates: list[str] = []
        for advertised_url in advertised or []:
            sitemap_url = normalize_url(
                advertised_url, canonicalize_host=False
            )
            if sitemap_url is not None and sitemap_url not in candidates:
                candidates.append(sitemap_url)
        fallback_candidates = [
            f"https://{hostname}{path}" for path in SITEMAP_FALLBACK_PATHS
        ]
        sitemap_queue = list(candidates or fallback_candidates)
        try_fallbacks = bool(candidates)
        usable_sitemap_found = False
        sitemap_seen: set[str] = set()
        locations_seen = 0
        while len(sitemap_seen) < MAX_SITEMAP_DOCUMENTS:
            if not sitemap_queue:
                if try_fallbacks and not usable_sitemap_found:
                    # An advertised sitemap was absent or invalid. Try the
                    # standard paths on the already verified homepage origin.
                    sitemap_queue = [
                        url for url in fallback_candidates if url not in sitemap_seen
                    ]
                    try_fallbacks = False
                    continue
                break
            sitemap_url = sitemap_queue.pop(0)
            if sitemap_url in sitemap_seen:
                continue
            sitemap_seen.add(sitemap_url)
            document = self._fetch_sitemap(sitemap_url)
            if document is None:
                continue
            try:
                is_index, locations = self._sitemap_loc_values(document)
            except ET.ParseError as exc:
                self.stats.request_or_database_errors += 1
                self._log(f"[sitemap] Invalid XML at {sitemap_url}: {exc}")
                continue
            usable_sitemap_found = True

            if is_index:
                for location in locations:
                    nested = normalize_url(location, canonicalize_host=False)
                    if (
                        nested is not None
                        and nested not in sitemap_seen
                        and nested not in sitemap_queue
                    ):
                        sitemap_queue.append(nested)
                continue

            for location in locations:
                locations_seen += 1
                if locations_seen > MAX_SITEMAP_LOCATIONS:
                    self._log(
                        f"[sitemap] Stopped after {MAX_SITEMAP_LOCATIONS} locations."
                    )
                    return
                candidate = normalize_url(location)
                if (
                    candidate is None
                    or candidate in self.visited
                    or candidate in self.queued
                ):
                    continue
                eligible = self._eligible(candidate)
                if eligible is not None:
                    queue.append(eligible)
                    self.queued.add(eligible)
            self._log(
                f"[sitemap] Discovered URLs from {sitemap_url}; "
                f"{len(queue)} URL(s) are queued."
            )

    def _verify_homepage(self, url: str, title: str | None, content: str) -> None:
        """Confirm the seed response belongs to the requested official site."""
        host = urlsplit(url).hostname
        if not is_approved_host(host):
            raise DomainVerificationError(f"Unexpected homepage host: {host}")
        branding = f"{title or ''} {content}".casefold()
        if "psg college of technology" not in branding:
            raise DomainVerificationError(
                "The start page did not identify itself as PSG College of Technology; "
                "stopping before following any links."
            )

    def crawl(self) -> CrawlStats:
        """Run a bounded BFS crawl and persist each usable HTML page."""
        start = normalize_url(self.config.start_url)
        if start is None:
            raise CrawlError("The configured start URL is outside the approved HTTPS domain.")
        if is_asset_url(start) or is_restricted_url(start):
            raise CrawlError("The configured start URL is not an eligible public HTML page.")

        try:
            # Strictly load policy before the first page request.
            self.robots.ensure_loaded(urlsplit(start).hostname or "")
            if not self.robots.allowed(start):
                self.stats.robots_disallowed += 1
                self._log(f"[robots] Disallowed start URL: {start}")
                raise CrawlError(f"robots.txt disallows the start URL: {start}")
            try:
                self.database = KnowledgeBase(self.config.database_path)
            except (sqlite3.Error, OSError) as exc:
                self.stats.request_or_database_errors += 1
                self._log(f"[database error] {self.config.database_path}: {exc}")
                return self.stats

            queue = list(self.initial_seeds or (start,))
            for seed in queue:
                self.queued.add(seed)
            head = 0
            while head < len(queue) and self.stats.urls_visited < self.config.page_limit:
                requested_url = queue[head]
                head += 1
                self.queued.discard(requested_url)
                if requested_url in self.visited:
                    continue
                self.visited.add(requested_url)
                self.stats.urls_visited += 1
                self._log(
                    f"[visit {self.stats.urls_visited}/{self.config.page_limit}] {requested_url}"
                )
                try:
                    is_pdf = self.config.include_pdfs and is_pdf_url(requested_url)
                    if is_pdf:
                        fetched_pdf = self._fetch_pdf(requested_url)
                        if fetched_pdf is None:
                            continue
                        final_url, pdf_body = fetched_pdf
                        title, content = self._extract_pdf(pdf_body, final_url)
                        html: bytes | None = None
                    else:
                        fetched_html = self._fetch_html(requested_url)
                        if fetched_html is None:
                            continue
                        final_url, html = fetched_html
                        title, content = extract_page(html)
                        if requested_url == start:
                            self._verify_homepage(final_url, title, content)

                    canonical_final_url = normalize_url(final_url) or final_url
                    self.visited.add(canonical_final_url)

                    if is_usable_content(content):
                        assert self.database is not None
                        self.database.save(canonical_final_url, title, content)
                        self.stats.pages_stored_or_updated += 1
                        self._log(
                            f"[stored] {title or '(no title)'} | {canonical_final_url}"
                        )
                    else:
                        self._log(
                            f"[skip] Empty or near-empty page: {canonical_final_url}"
                        )
                    if html is not None:
                        self._discover_links(html, final_url, queue)
                        if requested_url == start:
                            self._discover_sitemap_urls(queue, final_url)
                except RobotsPolicyError:
                    # Never continue if a newly discovered hostname's policy
                    # cannot be verified.
                    raise
                except DomainVerificationError:
                    raise
                except AccessRestrictionError as exc:
                    self.stats.request_or_database_errors += 1
                    raise
                except (requests.RequestException, CrawlError, OSError, sqlite3.Error) as exc:
                    self.stats.request_or_database_errors += 1
                    label = "database error" if isinstance(exc, sqlite3.Error) else "error"
                    self._log(f"[{label}] {requested_url}: {exc}")
            if head < len(queue):
                self._log(
                    f"[limit] Page limit reached; {len(queue) - head} queued URL(s) remain."
                )
            return self.stats
        except RobotsPolicyError:
            self.stats.request_or_database_errors += 1
            raise
        except DomainVerificationError:
            self.stats.request_or_database_errors += 1
            raise
        finally:
            self.close()


def main(argv: list[str] | None = None) -> None:
    """Run a crawl with optional bounds and database path overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-limit", type=int, default=100)
    parser.add_argument("--max-response-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-pdf-bytes", type=int, default=20_000_000)
    parser.add_argument("--max-pdf-pages", type=int, default=500)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--include-pdfs",
        action="store_true",
        help="Fetch robots-allowed PDFs from approved hosts and extract their text",
    )
    parser.add_argument(
        "--focus-seeds",
        action="store_true",
        help="Prioritize verified programme, faculty, placement, hostel, and directory pages",
    )
    args = parser.parse_args(argv)
    crawler = Crawler(
        CrawlConfig(
            page_limit=args.page_limit,
            max_response_bytes=args.max_response_bytes,
            max_pdf_bytes=args.max_pdf_bytes,
            max_pdf_pages=args.max_pdf_pages,
            include_pdfs=args.include_pdfs,
            include_focus_seeds=args.focus_seeds,
            min_request_interval=args.request_delay,
            database_path=args.database,
        )
    )
    try:
        crawler.crawl()
    except RobotsPolicyError as exc:
        print(f"[stopped] Robots policy could not be verified: {exc}")
        raise SystemExit(2) from exc
    except CrawlError as exc:
        print(f"[stopped] {exc}")
        raise SystemExit(2) from exc
    finally:
        stats = crawler.stats
        print("\nCrawl summary")
        print(f"  URLs visited:                    {stats.urls_visited}")
        print(f"  HTML pages successfully fetched: {stats.html_pages_fetched}")
        print(f"  PDF documents successfully fetched: {stats.pdf_documents_fetched}")
        print(f"  Pages stored or updated:         {stats.pages_stored_or_updated}")
        print(f"  Assets/non-HTML skipped:          {stats.skipped_assets_or_non_html}")
        print(f"  Robots-disallowed URLs:           {stats.robots_disallowed}")
        print(f"  Request/database errors:          {stats.request_or_database_errors}")
        print(f"Database: {crawler.config.database_path}")
        crawler.close()


if __name__ == "__main__":
    main()
