"""Build a local FAISS index from the PSG Tech crawler SQLite database.

Run once after crawling with:
    python rag_engine.py

The generated arrays use Unicode dtypes rather than pickle-backed objects, so
they can be loaded safely with ``allow_pickle=False`` in the Streamlit app.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = BASE_DIR / "psgtech_knowledge_base.db"
DEFAULT_OUTPUT_DIR = BASE_DIR
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIMENSION = 384
CHUNK_WORDS = 300
CHUNK_OVERLAP = 50
FACULTY_DEPARTMENT_PATTERN = re.compile(
    r"Department of (.+?)\s+S\.No\.",
    re.IGNORECASE,
)


class IndexBuildError(RuntimeError):
    """Raised when source data cannot be converted into a consistent index."""


def load_database_rows(database_path: Path) -> list[tuple[str, str, str]]:
    """Read URL, title, and cleaned body text without modifying the database."""
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
        normalized_rows.append((url.strip(), (title or "Untitled").strip(), content.strip()))
    return normalized_rows


def chunk_page(url: str, title: str, content: str) -> list[str]:
    """Chunk text with source headers, keeping faculty department lists intact."""
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks: list[str] = []

    def append_windows(text: str, department: str | None = None) -> None:
        words = text.split()
        for start in range(0, len(words), step):
            word_window = words[start : start + CHUNK_WORDS]
            if not word_window:
                continue
            body = " ".join(word_window)
            department_header = f" | Department: {department}" if department else ""
            chunks.append(
                "College: PSG College of Technology"
                f" | Resource: {title}"
                f"{department_header}"
                f" | Source: {url}"
                f" | Content: {body}"
            )

    faculty_sections = list(FACULTY_DEPARTMENT_PATTERN.finditer(content))
    if len(faculty_sections) >= 2 and (
        "facultydetails" in url.casefold()
        or "department-wise faculty" in content.casefold()
    ):
        # Keep each department name in its own embedding chunk. This makes a
        # department's roster retrievable independently of the entire page.
        if faculty_sections[0].start() > 0:
            append_windows(content[: faculty_sections[0].start()].strip())
        for section_number, match in enumerate(faculty_sections):
            section_end = (
                faculty_sections[section_number + 1].start()
                if section_number + 1 < len(faculty_sections)
                else len(content)
            )
            department = " ".join(match.group(1).split())
            section_text = content[match.start() : section_end].strip()
            append_windows(section_text, department)
    else:
        append_windows(content)
    return chunks


def build_index(
    database_path: Path,
    output_dir: Path,
    batch_size: int = 32,
) -> tuple[int, int]:
    """Embed the database and write an aligned FAISS index and NumPy matrices."""
    if batch_size < 1:
        raise IndexBuildError("Batch size must be a positive integer.")

    rows = load_database_rows(database_path)
    print(f"[data] Loaded {len(rows)} non-empty pages from {database_path}")
    if not rows:
        raise IndexBuildError("No non-empty crawled pages are available to index.")

    enriched_chunks: list[str] = []
    metadata_rows: list[tuple[str, str]] = []
    for page_number, (url, title, content) in enumerate(rows, start=1):
        page_chunks = chunk_page(url, title, content)
        enriched_chunks.extend(page_chunks)
        metadata_rows.extend((title, url) for _ in page_chunks)
        print(
            f"[chunk] Page {page_number}/{len(rows)}: {title!r} "
            f"({len(page_chunks)} chunks)"
        )

    if not enriched_chunks:
        raise IndexBuildError("No text chunks were produced from the database.")
    if len(enriched_chunks) != len(metadata_rows):
        raise IndexBuildError("Internal error: chunk and metadata counts differ.")

    print(f"[model] Loading {EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        dimension = model.get_sentence_embedding_dimension()
        if dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise IndexBuildError(
                f"Expected {EXPECTED_EMBEDDING_DIMENSION}-dimensional embeddings; "
                f"the model reports {dimension!r}."
            )

        print(
            f"[embed] Encoding {len(enriched_chunks)} chunks "
            f"(batch size {batch_size}; dimension {dimension})..."
        )
        vectors = model.encode(
            enriched_chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    except IndexBuildError:
        raise
    except Exception as exc:
        raise IndexBuildError(f"Embedding model failed: {exc}") from exc

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (
        len(enriched_chunks),
        EXPECTED_EMBEDDING_DIMENSION,
    ):
        raise IndexBuildError(
            "The embedding model returned an unexpected shape: "
            f"{vectors.shape}; expected "
            f"({len(enriched_chunks)}, {EXPECTED_EMBEDDING_DIMENSION})."
        )
    if not np.isfinite(vectors).all():
        raise IndexBuildError("Embeddings contain NaN or infinite values.")

    print("[faiss] Creating IndexFlatL2 and adding vectors...")
    index = faiss.IndexFlatL2(EXPECTED_EMBEDDING_DIMENSION)
    index.add(vectors)
    if index.ntotal != len(enriched_chunks):
        raise IndexBuildError(
            f"FAISS stored {index.ntotal} vectors for {len(enriched_chunks)} chunks."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "psgtech_vector_index.faiss"
    chunks_path = output_dir / "psgtech_chunks.npy"
    metadata_path = output_dir / "psgtech_metadata.npy"

    # Fixed-width Unicode arrays avoid pickle payloads and stay compatible with
    # numpy.load(..., allow_pickle=False). Metadata columns are title, then URL.
    chunk_array = np.asarray(enriched_chunks, dtype=str)
    metadata_array = np.asarray(metadata_rows, dtype=str)
    if metadata_array.ndim != 2 or metadata_array.shape[1] != 2:
        raise IndexBuildError("Could not create the expected two-column metadata array.")

    print(f"[save] Writing index and aligned arrays to {output_dir}...")
    try:
        faiss.write_index(index, str(index_path))
        np.save(chunks_path, chunk_array, allow_pickle=False)
        np.save(metadata_path, metadata_array, allow_pickle=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise IndexBuildError(f"Could not write index artifacts: {exc}") from exc

    print(f"[done] Pages indexed: {len(rows)}")
    print(f"[done] Chunks/vectors: {len(enriched_chunks)}")
    print(f"[done] Vector index: {index_path}")
    print(f"[done] Chunk strings: {chunks_path}")
    print(f"[done] Metadata (title, URL): {metadata_path}")
    return len(rows), len(enriched_chunks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the PSG Tech semantic search index from the crawler database."
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
        help=f"Directory for FAISS/NumPy artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
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
