"""Retriever registry.

Adding a retriever: write a class with `build`, `search_one` and `close`, then
add one line to `_REGISTRY`. The runner, the budget harness and the plots need
no changes - see README, "Adding a retriever".
"""

from __future__ import annotations

from typing import Any

from .base import BuildStats, Retriever, RetrieverLoadError  # noqa: F401

_REGISTRY: dict[str, tuple[str, str]] = {
    "bm25": ("measurerag.retrievers.bm25", "BM25Retriever"),
    "tfidf-svd": ("measurerag.retrievers.tfidf_svd", "TfidfSvdRetriever"),
    "dense": ("measurerag.retrievers.dense", "DenseRetriever"),
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def dense_models() -> list[str]:
    from .dense import MODELS

    return sorted(MODELS)


def describe_retriever(kind: str, **params: Any) -> dict:
    """The declarations that decide what a configuration measures.

    This is folded into the run spec so that resume re-measures a configuration
    whose *declaration* changed even though its config line did not. It exists
    because it was needed: correcting retromae-cs and simcse-cs-en from mean to
    CLS pooling changed their numbers while leaving their specs byte-identical,
    and resume would have served the stale results back forever.
    """
    if kind == "dense":
        from dataclasses import asdict

        from .dense import MODELS

        model = params.get("model")
        if model in MODELS:
            spec = asdict(MODELS[model])
            # A revision only where one is pinned: an empty key would change the
            # spec of every unpinned dense configuration, and resume would
            # re-measure results whose model did not change.
            if spec.get("revision") is None:
                spec.pop("revision", None)
            return {"model_spec": spec}
    return {}


def build_retriever(kind: str, **kwargs: Any):
    import importlib

    if kind not in _REGISTRY:
        raise KeyError(f"unknown retriever {kind!r}; known: {', '.join(available())}")
    module_name, class_name = _REGISTRY[kind]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**kwargs)
