"""A dense retriever with no downloads: TF-IDF projected by truncated SVD (LSA).

Why it is here: `measurerag smoke` must produce a *multi-row* budget table on a
machine that has never seen the network, and a one-row table teaches a newcomer
nothing. This is also a genuinely useful floor - it is the cheapest thing that
is "dense", so any neural encoder should beat it by enough to justify its RAM.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..datasets.base import Doc
from ..seeds import GLOBAL_SEED
from .base import BuildStats
from .bm25 import tokenize


class TfidfSvdRetriever:
    def __init__(
        self,
        n_components: int = 128,
        min_df: int = 1,
        strip_diacritics: bool = False,
        **_,
    ) -> None:
        self.name = f"tfidf-svd{n_components}"
        self.n_components = n_components
        self.min_df = min_df
        self.strip_diacritics = strip_diacritics
        self._vectorizer = None
        self._svd = None
        self._matrix: np.ndarray | None = None
        self._doc_ids: list[str] = []

    def _analyzer(self, text: str) -> list[str]:
        return tokenize(text, lowercase=True, strip_diacritics=self.strip_diacritics)

    def build(self, docs: Sequence[Doc], workdir: Path) -> BuildStats:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        t0 = time.perf_counter()
        self._doc_ids = [d.doc_id for d in docs]
        self._vectorizer = TfidfVectorizer(analyzer=self._analyzer, min_df=self.min_df)
        tfidf = self._vectorizer.fit_transform([d.indexable() for d in docs])

        k = max(2, min(self.n_components, min(tfidf.shape) - 1))
        # randomized SVD with a fixed seed: same input -> same components, always
        self._svd = TruncatedSVD(n_components=k, random_state=GLOBAL_SEED, algorithm="randomized")
        mat = self._svd.fit_transform(tfidf).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.maximum(norms, 1e-12)
        build_seconds = time.perf_counter() - t0

        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / "tfidf_svd.npy"
        np.save(path, self._matrix)
        return BuildStats(
            build_seconds=build_seconds,
            index_bytes=path.stat().st_size,
            model_bytes=0,  # the "model" is the index; nothing extra on disk
            n_docs=len(docs),
            extra={"n_components": int(k), "vocab": int(len(self._vectorizer.vocabulary_))},
        )

    def search_one(self, query: str, top_k: int) -> list[tuple[str, float]]:
        vec = self._svd.transform(self._vectorizer.transform([query])).astype(np.float32)
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        scores = (self._matrix @ vec.T).ravel()
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self._doc_ids[int(i)], float(scores[int(i)])) for i in top]

    def close(self) -> None:
        self._matrix = None
