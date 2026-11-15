"""BM25 - the baseline everything else is measured against.

It has no model weights, needs no GPU, and on an air-gapped box it is the thing
that definitely runs. If a dense retriever cannot beat this inside the same RAM
and latency budget, it has not earned its place on the machine.
"""

from __future__ import annotations

import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..datasets.base import Doc
from ..hashing import dir_size_bytes
from .base import BuildStats

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _strip(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


@lru_cache(maxsize=1 << 20)
def _lemma(token: str, lang: str) -> str:
    import simplemma

    return simplemma.lemmatize(token, lang=lang).lower()


def tokenize(
    text: str, lowercase: bool = True, strip_diacritics: bool = False, lemmatize: str | None = None
) -> list[str]:
    """Unicode word tokenisation.

    Diacritic stripping is off by default and exposed as a knob: for Czech it is
    a real, measurable trade-off (it recovers sloppy-typing recall and costs
    precision), and the kit's job is to measure such knobs, not to pick for you.

    `lemmatize` names a language for simplemma:
    each word becomes its lemma, lowercased, and diacritics are stripped only
    after that, because the lemma tables are written with them. Without it the
    tokens are exactly what they always were.
    """
    if lowercase:
        text = text.lower()
    if lemmatize is None:
        if strip_diacritics:
            text = _strip(text)
        return _TOKEN.findall(text)
    tokens = [_lemma(t, lemmatize) for t in _TOKEN.findall(text)]
    return [_strip(t) for t in tokens] if strip_diacritics else tokens


class BM25Retriever:
    """Wraps bm25s (fast, memory-mapped) and falls back to rank_bm25."""

    def __init__(
        self,
        k1: float = 0.9,
        b: float = 0.4,
        lowercase: bool = True,
        strip_diacritics: bool = False,
        lemmatize: str | None = None,
        **_,
    ) -> None:
        self.name = "bm25"
        self.k1 = k1
        self.b = b
        self.lowercase = lowercase
        self.strip_diacritics = strip_diacritics
        self.lemmatize = lemmatize
        self._impl = None
        self._backend = "none"
        self._doc_ids: list[str] = []

    def _tok(self, text: str) -> list[str]:
        return tokenize(text, self.lowercase, self.strip_diacritics, self.lemmatize)

    def build(self, docs: Sequence[Doc], workdir: Path) -> BuildStats:
        t0 = time.perf_counter()
        self._doc_ids = [d.doc_id for d in docs]
        corpus_tokens = [self._tok(d.indexable()) for d in docs]

        try:
            import bm25s

            retriever = bm25s.BM25(k1=self.k1, b=self.b, method="lucene")
            retriever.index(corpus_tokens, show_progress=False)
            self._impl = retriever
            self._backend = "bm25s"
        except Exception:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._impl = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)
            self._backend = "rank_bm25"

        build_seconds = time.perf_counter() - t0

        index_dir = workdir / "bm25_index"
        index_bytes = 0
        if self._backend == "bm25s":
            try:
                self._impl.save(str(index_dir))  # type: ignore[union-attr]
                index_bytes = dir_size_bytes(index_dir)
            except Exception:
                index_bytes = 0
        if index_bytes == 0:
            # rank_bm25 has no on-disk form; estimate from the posting data we hold
            index_bytes = sum(len(t) * 8 for t in corpus_tokens)

        return BuildStats(
            build_seconds=build_seconds,
            index_bytes=index_bytes,
            model_bytes=0,
            n_docs=len(docs),
            extra={"backend": self._backend, "k1": self.k1, "b": self.b,
                   **({"lemmatize": f"simplemma:{self.lemmatize}"} if self.lemmatize else {})},
        )

    def search_one(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = self._tok(query)
        if not tokens:
            return []
        if self._backend == "bm25s":
            k = min(top_k, len(self._doc_ids))
            idx, scores = self._impl.retrieve(  # type: ignore[union-attr]
                [tokens], k=k, show_progress=False, n_threads=1
            )
            return [
                (self._doc_ids[int(i)], float(s)) for i, s in zip(idx[0], scores[0])
            ]
        scores = np.asarray(self._impl.get_scores(tokens))  # type: ignore[union-attr]
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self._doc_ids[int(i)], float(scores[int(i)])) for i in top]

    def close(self) -> None:
        self._impl = None
