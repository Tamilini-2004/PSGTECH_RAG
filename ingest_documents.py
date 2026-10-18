"""Extract text from a local PDF and upsert it into the PSG Tech RAG database."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # Keep a useful diagnostic if dependencies were not installed.
    PdfReader = None  # type: ignore[assignment,misc]

from crawler import DEFAULT_DATABASE, content_hash, clean_text, normalize_url


def inject_pdf_to_rag_db(
    pdf_path: str,
    source_url: str,
    page_title: str,
    database_path: str | Path = DEFAULT_DATABASE,
) -> bool:
    """Extract a local PDF text layer and upsert it using its official source URL.

    Returns ``True`` after a successful commit. Errors are reported to stderr
    and return ``False``; the existing database schema is never modified.
    """
    print(f"[pdf] Starting ingestion: {pdf_path}")
    if PdfReader is None:
        print(
            "[error] pypdf is not installed. Run `python -m pip install pypdf` "
            "or install the project requirements.",
            file=sys.stderr,
        )
        return False

    try:
        pdf_file = Path(pdf_path).expanduser()
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF path does not exist: {pdf_file}")
        if not pdf_file.is_file():
            raise ValueError(f"PDF path is not a file: {pdf_file}")
        if pdf_file.suffix.casefold() != ".pdf":
            raise ValueError(f"Expected a .pdf file, received: {pdf_file.name}")
        if pdf_file.stat().st_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_file}")

        normalized_url = normalize_url(source_url)
        if normalized_url is None:
            raise ValueError(
                "source_url must point to an approved PSG Tech hostname "
                "(psgtech.edu or www.psgtech.edu); stored URLs are normalized to HTTPS."
            )
        title = page_title.strip()
        if not title:
            raise ValueError("page_title must not be empty.")

        print(f"[pdf] Reading pages from {pdf_file.resolve()}")
        reader = PdfReader(str(pdf_file), strict=False)
        if reader.is_encrypted:
            raise ValueError(
                "The PDF is encrypted. Provide a readable, unencrypted local file."
            )

        page_texts: list[str] = []
        page_count = len(reader.pages)
        if page_count == 0:
            raise ValueError("The PDF contains no pages.")
        for page_number, page in enumerate(reader.pages, start=1):
            print(f"[pdf] Extracting page {page_number}/{page_count}")
            try:
                extracted = page.extract_text()
            except Exception as exc:
                raise ValueError(
                    f"Text extraction failed on PDF page {page_number}: {exc}"
                ) from exc
            if extracted:
                cleaned_page = clean_text(extracted)
                if cleaned_page:
                    page_texts.append(cleaned_page)

        sanitized_content = clean_text("\n".join(page_texts))
        if not sanitized_content:
            raise ValueError(
                "No selectable text was found. This may be a scanned PDF; "
                "OCR is required before ingestion."
            )

        database = Path(database_path).expanduser()
        if not database.is_file():
            raise FileNotFoundError(
                f"Knowledge-base database does not exist: {database}. "
                "Run the crawler first or pass the correct --database path."
            )

        digest = content_hash(sanitized_content)
        print(
            f"[db] Upserting {len(sanitized_content):,} text characters "
            f"under {normalized_url}"
        )
        connection = sqlite3.connect(database)
        try:
            with connection:
                connection.execute(
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
                    (normalized_url, title, sanitized_content, digest),
                )
        finally:
            connection.close()

        print(f"[done] PDF text stored successfully in {database.resolve()}")
        return True
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"[error] PDF ingestion failed: {exc}", file=sys.stderr)
        return False
    except Exception as exc:
        # pypdf can raise format-specific exceptions for damaged documents.
        print(
            f"[error] Could not read or store this PDF ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a local PDF into the PSG Tech RAG SQLite database."
    )
    parser.add_argument("pdf_path", help="Path to a local PDF file")
    parser.add_argument("source_url", help="Official PSG Tech URL for this document")
    parser.add_argument("page_title", help="Title to store with the source record")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args(argv)
    return 0 if inject_pdf_to_rag_db(
        args.pdf_path,
        args.source_url,
        args.page_title,
        database_path=args.database,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
