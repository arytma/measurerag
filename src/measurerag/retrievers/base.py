"""Retriever contract.

Every retriever must be able to answer a *single* query, because that is how a
deployed system is used and the only honest way to measure latency. Batch
search exists for throughput, never for the latency numbers in the result table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from ..datasets.base import Doc


@dataclass
class BuildStats:
    build_seconds: float = 0.0
    index_bytes: int = 0
    model_bytes: int = 0
    n_docs: int = 0
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    name: str

    def build(self, docs: Sequence[Doc], workdir: Path) -> BuildStats: ...

    def search_one(self, query: str, top_k: int) -> list[tuple[str, float]]: ...

    def close(self) -> None: ...


class RetrieverLoadError(RuntimeError):
    """Retriever could not be loaded within its time box; log and skip."""
