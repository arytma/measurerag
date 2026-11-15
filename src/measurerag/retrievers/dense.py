"""Dense retrievers: a sentence-transformers encoder plus a FAISS index.

Encoders are declared in MODELS below. The declaration carries the prompt
prefixes the model was trained with - getting e5's "query: "/"passage: " wrong
silently costs several nDCG points, which is exactly the kind of quiet error a
benchmark must not make on its user's behalf.

It carries the pooling for the same reason, and that one has already bitten.
Neither Seznam model ships a `modules.json`, so sentence-transformers falls back
to mean pooling without saying so, while both model cards specify the CLS token
(`last_hidden_state[:, 0]`). retromae-cs scored 0.2778 nDCG@10 on CsFEVER under
mean pooling - a number about the loader, not about the model. Pooling is now
declared per model and enforced against whatever the repo configures.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..datasets.base import Doc

from .base import BuildStats, RetrieverLoadError


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    pooling: str          # "cls" or "mean"; no default, so every model must say
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int | None = None
    note: str = ""
    revision: str | None = None   # a pinned commit; None loads the repository's main


MODELS: dict[str, ModelSpec] = {
    "e5-small": ModelSpec(
        "e5-small",
        "intfloat/multilingual-e5-small",
        pooling="mean",           # repo ships 1_Pooling: mean
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_seq_length=512,
        note="118M params, 384 dim. The working horse of the small-budget tier.",
    ),
    "e5-base": ModelSpec(
        "e5-base",
        "intfloat/multilingual-e5-base",
        pooling="mean",           # repo ships 1_Pooling: mean
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_seq_length=512,
        note="278M params, 768 dim.",
    ),
    "retromae-cs": ModelSpec(
        "retromae-cs",
        "Seznam/retromae-small-cs",
        pooling="cls",            # model card: last_hidden_state[:, 0]. Ships no
                                  # modules.json, so ST would mean-pool silently.
        max_seq_length=512,
        note="Czech-only RetroMAE small. No instruction prefix. CLS pooling.",
        # main on 14 Nov 2026, and the commit every result used: its files re-hash to the build's cache and its
        # byte count is the one the results record
        revision="c9696feffb314a7f104b4be71e46dc197c193232",
    ),
    "simcse-cs-en": ModelSpec(
        "simcse-cs-en",
        "Seznam/simcse-dist-mpnet-paracrawl-cs-en",
        pooling="cls",            # model card: last_hidden_state[:, 0]. Ships no
                                  # modules.json, so ST would mean-pool silently.
        max_seq_length=512,
        note="Czech-English bilingual SimCSE distillation. CLS pooling.",
    ),
    # v0.4.0: read from each card and repository at download
    "granite-107m": ModelSpec(
        "granite-107m",
        "ibm-granite/granite-embedding-107m-multilingual",
        pooling="cls",            # repo ships 1_Pooling: cls; the card pools the CLS token
        max_seq_length=512,       # the card's context length
        note="IBM, 107M params, 384 dim, Apache-2.0. No prefix. Czech is one of its 12 languages.",
    ),
    "granite-278m": ModelSpec(
        "granite-278m",
        "ibm-granite/granite-embedding-278m-multilingual",
        pooling="cls",            # repo ships 1_Pooling: cls; the card pools the CLS token
        max_seq_length=512,
        note="IBM, 278M params, 768 dim, Apache-2.0. No prefix. Czech is one of its 12 languages.",
    ),
    "embeddinggemma": ModelSpec(
        "embeddinggemma",
        "google/embeddinggemma-300m",
        pooling="mean",           # repo ships 1_Pooling: mean, then 2_Dense, 3_Dense and 4_Normalize, all loaded
        query_prefix="task: search result | query: ",   # the repo's "query" prompt; no default prompt is set
        passage_prefix="title: none | text: ",          # the repo's "document" prompt
        max_seq_length=512,       # e5-small's, for like-with-like latency; the model takes 2048
        note="Google, 308M params, 768 dim, Gemma Terms of Use (gated). float32: the card says no float16.",
    ),
    "bge-m3": ModelSpec(
        "bge-m3",
        "BAAI/bge-m3",
        pooling="cls",            # repo ships 1_Pooling: cls
        max_seq_length=1024,
        note="Ceiling reference only: 568M params, 1024 dim, will not fit 4 GB.",
    ),
    "bge-m3-512": ModelSpec(
        "bge-m3-512",
        "BAAI/bge-m3",
        pooling="cls",
        max_seq_length=512,
        note=(
            "The same weights as bge-m3 at the e5 models' sequence length. Exists "
            "so the latency column compares like with like: at its native 1024 "
            "bge-m3 encodes twice the text per document, so part of its p95 is "
            "text, not model."
        ),
    ),
}


# Formats a sentence-transformers load never touches; shipping them to an
# air-gapped box would be waste, so they are excluded from "model size".
_UNUSED_DIRS = {"onnx", "openvino", "coreml", "tflite", ".git"}


def _model_disk_bytes(hf_id: str, revision: str | None = None) -> int:
    """What you would actually have to copy onto the customer's machine.

    Not the whole Hugging Face blob store: a repo often carries a .bin and a
    .safetensors of the same weights plus ONNX exports, and counting all of them
    would overstate a model by two or three times. Duplicate weight formats are
    counted once, with safetensors preferred.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(hf_id, revision=revision, local_files_only=True))
    except Exception:
        return 0

    files: list[Path] = []
    for f in snapshot.rglob("*"):
        if not f.is_file():
            continue
        if any(part in _UNUSED_DIRS for part in f.relative_to(snapshot).parts):
            continue
        files.append(f)

    has_safetensors = any(f.suffix == ".safetensors" for f in files)
    total = 0
    for f in files:
        if has_safetensors and f.suffix in {".bin", ".h5", ".msgpack", ".ckpt"}:
            continue  # same weights in another format
        if f.suffix == ".onnx":
            continue  # an export beside the weights (granite ships one at the top level), never loaded
        try:
            total += f.resolve().stat().st_size
        except OSError:
            pass
    return total


def _pooling_mode(encoder) -> str | None:
    """The pooling a loaded sentence-transformers model is actually using."""
    for module in encoder:
        mode = getattr(module, "pooling_mode", None)
        if isinstance(mode, str):
            return mode
        if isinstance(mode, (tuple, list)) and len(mode) == 1:
            return str(mode[0])
    return None


def _build_with_pooling(spec: ModelSpec, device: str):
    """Assemble the encoder by hand so the pooling is the declared one."""
    import warnings

    from sentence_transformers import SentenceTransformer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer

    pin = {"revision": spec.revision} if spec.revision else None
    body = Transformer(spec.hf_id, model_kwargs=pin, processor_kwargs=pin, config_kwargs=pin)
    if spec.max_seq_length:
        body.max_seq_length = spec.max_seq_length
    pool = Pooling(body.get_word_embedding_dimension(), pooling_mode=spec.pooling)
    return SentenceTransformer(modules=[body, pool], device=device)


class DenseRetriever:
    """Encoder + FAISS. `index_type` is `flat` (exact, the reference) or `hnsw`."""

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        batch_size: int = 32,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        load_timeout_s: float = 600.0,
        **_,
    ) -> None:
        if model not in MODELS:
            raise KeyError(f"unknown dense model {model!r}; known: {', '.join(MODELS)}")
        self.spec = MODELS[model]
        self.name = f"{model}+faiss-{index_type}"
        self.device = device
        self.batch_size = batch_size
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.load_timeout_s = load_timeout_s
        self._encoder = None
        self._index = None
        self._doc_ids: list[str] = []
        self._dim = 0
        self.pooling_source = "unloaded"

    # -- encoder ----------------------------------------------------------
    def _load_encoder(self):
        if self._encoder is not None:
            return self._encoder
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        t0 = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer

            enc = SentenceTransformer(self.spec.hf_id, device=self.device, revision=self.spec.revision)
            actual = _pooling_mode(enc)
            # A repo with no modules.json gets mean pooling from
            # sentence-transformers with no warning. When the model card says
            # otherwise, rebuild it rather than measure the wrong thing.
            if actual != self.spec.pooling:
                enc = _build_with_pooling(self.spec, self.device)
                rebuilt = _pooling_mode(enc)
                if rebuilt != self.spec.pooling:  # pragma: no cover - defensive
                    raise RetrieverLoadError(
                        f"{self.spec.hf_id}: asked for {self.spec.pooling!r} pooling, "
                        f"got {rebuilt!r}"
                    )
                self.pooling_source = f"declared ({actual!r} would have been used)"
            else:
                self.pooling_source = "repo"
        except RetrieverLoadError:
            raise
        except Exception as exc:  # pragma: no cover - network / missing weights
            raise RetrieverLoadError(f"{self.spec.hf_id}: {exc}") from exc
        elapsed = time.perf_counter() - t0
        if elapsed > self.load_timeout_s:
            raise RetrieverLoadError(
                f"{self.spec.hf_id} took {elapsed:.0f}s to load, over the "
                f"{self.load_timeout_s:.0f}s box"
            )
        if self.spec.max_seq_length:
            enc.max_seq_length = self.spec.max_seq_length
        self._encoder = enc
        return enc

    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int) -> np.ndarray:
        enc = self._load_encoder()
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vecs = enc.encode(
            payload,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine == inner product
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    # -- index ------------------------------------------------------------
    def build(self, docs: Sequence[Doc], workdir: Path) -> BuildStats:
        import faiss

        t0 = time.perf_counter()
        self._doc_ids = [d.doc_id for d in docs]
        vecs = self._encode(
            [d.indexable() for d in docs], self.spec.passage_prefix, self.batch_size
        )
        self._dim = int(vecs.shape[1])

        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self._dim)
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self._dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
        else:
            raise ValueError(f"index_type must be flat or hnsw, got {self.index_type!r}")
        index.add(vecs)
        self._index = index

        build_seconds = time.perf_counter() - t0

        workdir.mkdir(parents=True, exist_ok=True)
        index_path = workdir / f"faiss_{self.index_type}.index"
        faiss.write_index(index, str(index_path))

        return BuildStats(
            build_seconds=build_seconds,
            index_bytes=index_path.stat().st_size,
            model_bytes=_model_disk_bytes(self.spec.hf_id, self.spec.revision),
            n_docs=len(docs),
            extra={
                "model": self.spec.hf_id,
                "dim": self._dim,
                "pooling": self.spec.pooling,
                "pooling_source": self.pooling_source,
                "index_type": self.index_type,
                "device": self.device,
            },
        )

    def search_one(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """One query, encoded and searched - encoding cost included on purpose."""
        vec = self._encode([query], self.spec.query_prefix, batch_size=1)
        k = min(top_k, len(self._doc_ids))
        scores, idx = self._index.search(vec, k)  # type: ignore[union-attr]
        return [
            (self._doc_ids[int(i)], float(s))
            for i, s in zip(idx[0], scores[0])
            if int(i) >= 0
        ]

    def close(self) -> None:
        self._encoder = None
        self._index = None
