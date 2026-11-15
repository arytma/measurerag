"""Dataset registry.

Adding a dataset means writing a module with a `load(cache_dir, **kwargs) ->
Dataset` function and adding one line here. Nothing else in the kit changes.
"""

from __future__ import annotations

from typing import Callable

from .base import (  # noqa: F401  (re-exported for loader authors)
    Dataset,
    DatasetBlocked,
    DatasetUnavailable,
    Doc,
    apply_caps,
    cache_root,
)

_REGISTRY: dict[str, str] = {
    "toy": "measurerag.datasets.toy",
    "scifact": "measurerag.datasets.beir",
    "nfcorpus": "measurerag.datasets.beir",
    "fiqa": "measurerag.datasets.beir",
    "hotpotqa": "measurerag.datasets.beir",
    "hotpotqa-rag": "measurerag.datasets.hotpotqa",
    "nq-rag": "measurerag.datasets.nq",
    "csfever": "measurerag.datasets.csfever",
    "cwrczech": "measurerag.datasets.cwrczech",
    "dareczech": "measurerag.datasets.dareczech",
    "sqad": "measurerag.datasets.sqad",
    "sqad-rag": "measurerag.datasets.sqad",
    "sqad32": "measurerag.datasets.sqad32",
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_loader(name: str) -> Callable[..., Dataset]:
    import importlib

    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(available())}")
    module = importlib.import_module(_REGISTRY[key])
    loader = getattr(module, "load")

    def _call(**kwargs) -> Dataset:
        # BEIR-family modules serve several datasets from one loader.
        if _REGISTRY[key].endswith(".beir"):
            kwargs.setdefault("subset", key)
        return loader(**kwargs)

    return _call


def load_dataset(
    name: str,
    cache_dir=None,
    max_docs: int | None = None,
    max_queries: int | None = None,
    **kwargs,
) -> Dataset:
    # loaders that can cap while streaming take the caps directly; the rest
    # ignore them and get capped afterwards by apply_caps
    ds = get_loader(name)(
        cache_dir=cache_dir, max_docs=max_docs, max_queries=max_queries, **kwargs
    )
    return apply_caps(ds, max_docs=max_docs, max_queries=max_queries)
