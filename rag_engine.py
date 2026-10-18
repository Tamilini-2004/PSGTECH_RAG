"""Build a persistent Chroma index from the PSG Tech crawler SQLite database.

Run once after crawling with:
    python rag_engine.py

SQLite retains each complete page. Chroma stores small CPU-embedded windows,
and every window maps back to its original source URL.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterator, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = BASE_DIR / "psgtech_knowledge_base.db"
DEFAULT_OUTPUT_DIR = BASE_DIR
CHROMA_DIRECTORY = "psgtech_chroma_db"
COLLECTION_NAME = "psgtech_knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 1_000
CHUNK_OVERLAP = 150
EMBEDDING_WINDOW_WORDS = 120
EMBEDDING_WINDOW_OVERLAP = 20
CONTENT_MARKER = " | Content: "
STRUCTURED_PAGE_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:faculty[a-z0-9_-]*|rosters?|programmes?|programs?|"
    r"courses?|syllabus|syllabi)(?![a-z0-9])",
    re.IGNORECASE,
)


class IndexBuildError(RuntimeError):
    """Raised when source data cannot be converted into a consistent index."""


def load_database_rows(database_path: Path) -> list[tuple[str, str, str]]:
    """Read URL, title and cleaned content without modifying the SQLite schema."""
    if not database_path.is_file():
        raise IndexBuildError(f"SQLite database does not exist: {database_path}")
    try:
        read_only_uri = f"{database_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(read_only_uri, uri=True) as connection:
            rows = connection.execute(
                """
                SELECT url, page_title, sanitized_content
                FROM crawled_resources
                ORDER BY url
                """
            ).fetchall()
    except sqlite3.Error as exc:
        raise IndexBuildError(
            f"Could not read crawled_resources from {database_path}: {exc}"
        ) from exc

    normalized_rows: list[tuple[str, str, str]] = []
    for row_number, (url, title, content) in enumerate(rows, start=1):
        if not isinstance(url, str) or not url.strip():
            print(f"[skip] Database row {row_number} has no URL.")
            continue
        if not isinstance(content, str) or not content.strip():
            print(f"[skip] {url}: no sanitized content.")
            continue
        normalized_rows.append(
            (url.strip(), (title or "Untitled").strip(), content.strip())
        )
    return normalized_rows


def chunk_page(url: str, title: str, content: str) -> list[str]:
    """Create context-enriched logical chunks, keeping structured pages intact."""
    header = (
        "College: PSG College of Technology"
        f" | Resource: {title}"
        f" | Source: {url}"
        f"{CONTENT_MARKER}"
    )
    if STRUCTURED_PAGE_PATTERN.search(f"{url}\n{title}\n{content}"):
        # Keep the complete source in SQLite as one logical page; shorter
        # embedding windows below make all of its entries searchable.
        return [header + content]

    words = content.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [
        header + " ".join(words[start : start + CHUNK_WORDS])
        for start in range(0, len(words), step)
        if words[start : start + CHUNK_WORDS]
    ]


def embedding_windows(chunk: str) -> Iterator[str]:
    """Split page chunks into token-safe inputs for the CPU sentence embedder."""
    prefix, marker, body = chunk.partition(CONTENT_MARKER)
    if not marker:
        raise IndexBuildError("A page chunk is missing its content marker.")
    words = body.split()
    if not words:
        return
    step = EMBEDDING_WINDOW_WORDS - EMBEDDING_WINDOW_OVERLAP
    for start in range(0, len(words), step):
        window = words[start : start + EMBEDDING_WINDOW_WORDS]
        if window:
            yield prefix + CONTENT_MARKER + " ".join(window)


def stable_document_id(url: str, chunk_number: int, window_number: int, text: str) -> str:
    """Generate deterministic IDs so reindexing does not duplicate documents."""
    identity = f"{url}\0{chunk_number}\0{window_number}\0{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def iter_documents(
    rows: Sequence[tuple[str, str, str]],
) -> Iterator[tuple[Document, str]]:
    """Yield model-sized Chroma documents mapped to their source page."""
    for page_number, (url, title, content) in enumerate(rows, start=1):
        chunks = chunk_page(url, title, content)
        print(
            f"[chunk] Page {page_number}/{len(rows)}: {title!r} "
            f"({len(chunks)} logical chunk(s))"
        )
        for chunk_number, chunk in enumerate(chunks):
            for window_number, text in enumerate(embedding_windows(chunk)):
                metadata = {
                    "url": url,
                    "page_title": title,
                    "chunk_number": chunk_number,
                    "window_number": window_number,
                }
                yield (
                    Document(page_content=text, metadata=metadata),
                    stable_document_id(url, chunk_number, window_number, text),
                )


def build_index(
    database_path: Path,
    output_dir: Path,
    batch_size: int = 32,
) -> tuple[int, int]:
    """Build or replace the persistent Chroma collection with CPU embeddings."""
    if batch_size < 1:
        raise IndexBuildError("Batch size must be a positive integer.")

    rows = load_database_rows(database_path)
    print(f"[data] Loaded {len(rows)} non-empty pages from {database_path}")
    if not rows:
        raise IndexBuildError("No non-empty crawled pages are available to index.")

    output_dir.mkdir(parents=True, exist_ok=True)
    chroma_path = output_dir / CHROMA_DIRECTORY
    print(f"[model] Loading {EMBEDDING_MODEL} on CPU...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(chroma_path),
        )
    except Exception as exc:
        raise IndexBuildError(
            "Could not initialize the CPU embedding model or Chroma store. "
            f"Details: {exc}"
        ) from exc

    try:
        old_ids = vector_store.get(include=["metadatas"]).get("ids", [])
        for start in range(0, len(old_ids), batch_size):
            vector_store.delete(ids=old_ids[start : start + batch_size])
        if old_ids:
            print(f"[clear] Removed {len(old_ids)} previous Chroma documents.")

        documents: list[Document] = []
        document_ids: list[str] = []
        indexed_count = 0
        for document, document_id in iter_documents(rows):
            documents.append(document)
            document_ids.append(document_id)
            if len(documents) >= batch_size:
                vector_store.add_documents(documents, ids=document_ids)
                indexed_count += len(documents)
                print(f"[embed] Indexed {indexed_count} embedding windows...")
                documents.clear()
                document_ids.clear()

        if documents:
            vector_store.add_documents(documents, ids=document_ids)
            indexed_count += len(documents)
            print(f"[embed] Indexed {indexed_count} embedding windows...")

        stored_count = vector_store._collection.count()
        if stored_count != indexed_count:
            raise IndexBuildError(
                f"Chroma contains {stored_count} documents; expected {indexed_count}."
            )
    except IndexBuildError:
        raise
    except Exception as exc:
        raise IndexBuildError(f"Chroma ingestion failed: {exc}") from exc

    print(f"[done] Pages indexed: {len(rows)}")
    print(f"[done] Searchable embedding windows: {indexed_count}")
    print(f"[done] Persistent Chroma collection: {chroma_path}")
    return len(rows), indexed_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the PSG Tech Chroma semantic index from SQLite."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"SQLite database path (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Documents embedded per CPU batch (default: 32)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_index(args.database, args.output_dir, args.batch_size)
    except IndexBuildError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
