"""Polite, source-traceable crawler for the public PSG Tech website.

Importing this module does not start a crawl. Run python crawler.py to
create/update the local SQLite knowledge base.

"""


from __future__ import annotations

import argparse
import hashlib
import posixpath
import re
import sqlite3
import ssl
import time
import urllib.robotparser
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
from requests.adapters import HTTPAdapter


START_URL = "https://psgtech.edu/"
ALLOWED_HOSTS = frozenset({"psgtech.edu", "www.psgtech.edu"})
CANONICAL_HOST = "psgtech.edu"
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
        if self.max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")
        if not self.user_agent.strip():
            raise ValueError("user_agent must not be empty")


@dataclass
class CrawlStats:
    """Counters printed at the end of a crawl."""

    urls_visited: int = 0
    html_pages_fetched: int = 0
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
        if is_asset_url(normalized):
            self.stats.skipped_assets_or_non_html += 1
            return None
        if not self.robots.allowed(normalized):
            self.stats.robots_disallowed += 1
            self._log(f"[robots] Disallowed: {normalized}")
            return None
        return normalized

    def _read_limited_response(self, response: Any) -> bytes:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.config.max_response_bytes:
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
            if total > self.config.max_response_bytes:
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

            queue = [start]
            self.queued.add(start)
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
                    fetched = self._fetch_html(requested_url)
                    if fetched is None:
                        continue
                    final_url, html = fetched
                    canonical_final_url = normalize_url(final_url) or final_url
                    self.visited.add(canonical_final_url)
                    title, content = extract_page(html)
                    if requested_url == start:
                        self._verify_homepage(final_url, title, content)

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
                    self._discover_links(html, final_url, queue)
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
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args(argv)
    crawler = Crawler(
        CrawlConfig(
            page_limit=args.page_limit,
            max_response_bytes=args.max_response_bytes,
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
        print(f"  Pages stored or updated:         {stats.pages_stored_or_updated}")
        print(f"  Assets/non-HTML skipped:          {stats.skipped_assets_or_non_html}")
        print(f"  Robots-disallowed URLs:           {stats.robots_disallowed}")
        print(f"  Request/database errors:          {stats.request_or_database_errors}")
        print(f"Database: {crawler.config.database_path}")
        crawler.close()


if __name__ == "__main__":
    main()
