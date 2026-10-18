"""Bounded retry helper for streamed Gemini responses, with a local mock demo."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from typing import Any


def http_status_code(error: BaseException) -> int | None:
    """Read an HTTP status from common Google GenAI and HTTP client errors."""
    response = getattr(error, "response", None)
    candidates = (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(response, "status_code", None),
    )
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        try:
            status_code = int(candidate)
        except (TypeError, ValueError):
            continue
        if 100 <= status_code <= 599:
            return status_code
    return None


def generate_gemini_stream_with_retry(
    request_stream: Callable[[], Iterable[Any]],
    *,
    on_partial: Callable[[str], None] | None = None,
    on_retry: Callable[[int, float], None] | None = None,
    max_attempts: int = 5,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> str:
    """Consume a Gemini stream, retrying only HTTP 503 errors up to five times.

    The request callable is invoked again for each attempt. Exceptions for any
    non-503 status, and the final 503, are re-raised unchanged with their payload.
    """
    if not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be between 1 and 5")
    if base_delay_seconds < 0 or max_delay_seconds < 0:
        raise ValueError("retry delay values must not be negative")

    for attempt in range(1, max_attempts + 1):
        response_parts: list[str] = []
        try:
            for chunk in request_stream():
                text = getattr(chunk, "text", None)
                if not isinstance(text, str) or not text:
                    continue
                response_parts.append(text)
                if on_partial is not None:
                    on_partial("".join(response_parts))
            return "".join(response_parts)
        except Exception as error:
            status_code = http_status_code(error)
            if status_code != 503 or attempt == max_attempts:
                raise

            # Discard partial output from the failed stream before retrying.
            if on_partial is not None:
                on_partial("")
            delay_limit = min(
                max_delay_seconds,
                base_delay_seconds * (2 ** (attempt - 1)),
            )
            delay = min(max(0.0, jitter(0.0, delay_limit)), delay_limit)
            print(
                f"[Gemini retry] HTTP 503 on attempt {attempt}/{max_attempts}; "
                f"retrying in {delay:.2f} seconds."
            )
            if on_retry is not None:
                on_retry(attempt, delay)
            sleep(delay)

    raise RuntimeError("Gemini retry loop ended unexpectedly")


class MockGeminiError(RuntimeError):
    """Small HTTP-like exception used by the executable mock example."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


class MockChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class MockGeminiModels:
    """Fail once with 503, then return a short response without network access."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_content_stream(self) -> Iterable[MockChunk]:
        self.calls += 1
        if self.calls == 1:
            raise MockGeminiError(503, "temporary mock service outage")
        return iter((MockChunk('{"answer":"'), MockChunk('Mock answer."}')))


if __name__ == "__main__":
    mock_models = MockGeminiModels()
    mock_answer = generate_gemini_stream_with_retry(
        mock_models.generate_content_stream,
        # Avoid waiting in this demonstration; production uses real sleep/jitter.
        sleep=lambda _seconds: None,
        jitter=lambda _low, _high: 0.0,
    )
    assert mock_answer == '{"answer":"Mock answer."}'
    assert mock_models.calls == 2
    print(f"Mock succeeded after {mock_models.calls} calls: {mock_answer}")

    try:
        generate_gemini_stream_with_retry(
            lambda: (_ for _ in ()).throw(
                MockGeminiError(401, "mock credential rejected")
            ),
            sleep=lambda _seconds: None,
        )
    except MockGeminiError as error:
        print(f"Non-retryable HTTP {error.status_code} propagated immediately.")
