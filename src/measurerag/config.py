"""Config loading with explicit defaults.

Every knob a run depends on ends up in the manifest, so the resolved config -
not the file on disk - is what gets hashed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "unnamed",
    "device": "cpu",  # the target box is CPU-only; never default to a GPU
    "seed": 20260101,
    "query_threads": 1,   # the deployed path: one user, one query, fixed threads
    "build_threads": None,  # None = every core; indexing is a one-off offline job
    "dataset": {
        "name": "toy",
        "fallbacks": [],
        "max_docs": 100_000,
        "max_queries": 2_000,
        "options": {},
    },
    "retrieval": {"top_k": 100, "ks": [10, 100], "context_k": 5, "quality_key": "ndcg@10"},
    "budgets": {"ram_gb": [4, 8, 16], "retrieval_ms": [200, 500], "e2e_s": None},
    "retrievers": [{"id": "bm25", "kind": "bm25"}],
    "rag": {"enabled": False, "generators": [], "max_tokens": 256, "n_ctx": 4096},
    "timeout_s": 3600,
    "out_dir": None,
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _merge(DEFAULTS, raw)
    cfg.setdefault("name", Path(path).stem)
    if cfg["name"] == "unnamed":
        cfg["name"] = Path(path).stem
    if not cfg.get("out_dir"):
        cfg["out_dir"] = f"results/{cfg['name']}"
    return cfg
