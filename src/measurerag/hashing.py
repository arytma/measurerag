"""SHA-256 of everything that could drift between runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a JSON-serialisable object, key order independent."""
    return sha256_bytes(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def sha256_records(records: Iterable[Any]) -> str:
    """Streaming hash of an iterable of JSON-serialisable records.

    This is what fingerprints a dataset *slice*: the exact documents and
    queries that were used, not the file they happened to come from.
    """
    h = hashlib.sha256()
    for rec in records:
        h.update(
            json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        h.update(b"\n")
    return h.hexdigest()


def dir_size_bytes(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
