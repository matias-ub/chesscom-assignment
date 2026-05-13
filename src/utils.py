"""Shared helpers for the chess prediction pipeline."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "ChessPrediction/1.0 (take-home exercise; m.uncalbressi@gmail.com)"
REQUEST_DELAY = 0.5  # seconds between API calls


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get(url: str, session: requests.Session, retries: int = 3) -> dict[str, Any]:
    """GET a URL with retries and the required User-Agent header."""
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(REQUEST_DELAY * attempt)
            else:
                raise
    raise RuntimeError("Unreachable")  # pragma: no cover


def load_cache(path: Path) -> dict[str, Any] | None:
    """Return parsed JSON from cache file, or None if absent."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cache(path: Path, data: dict[str, Any]) -> None:
    """Persist data as JSON to cache file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
