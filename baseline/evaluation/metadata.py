"""Exact requirements metadata cache with bounded retries and atomic writes."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import requests

from core.io import hash_json


def make_cached_get(original_get, cache: Path, timeout: int, attempts=3):
    cache.mkdir(parents=True, exist_ok=True)

    def get(url, *args, **kwargs):
        if not url.startswith("https://raw.githubusercontent.com/"):
            return original_get(url, *args, **kwargs)
        path = cache / (hash_json(url) + ".txt")
        if path.exists():
            response = requests.Response()
            response.status_code = 200
            response._content = path.read_bytes()
            response.url = url
            return response
        parts = url.removeprefix("https://raw.githubusercontent.com/").split("/", 3)
        candidates = [url]
        if len(parts) == 4:
            owner, repo, revision, resource = parts
            candidates.append(f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{revision}/{resource}")
        kwargs.setdefault("timeout", timeout)
        error = None
        for attempt in range(attempts):
            for candidate in candidates:
                try:
                    response = original_get(candidate, *args, **kwargs)
                    if response.status_code in (404, 410):
                        # Preserve official harness candidate-path fallback.
                        return response
                    response.raise_for_status()
                except requests.RequestException as exc:
                    error = exc
                    continue
                fd, temporary = tempfile.mkstemp(dir=cache, prefix=".metadata-")
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(response.content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
                return response
            if attempt + 1 < attempts:
                time.sleep(min(0.5 * 2**attempt, 2))
        raise error

    return get
