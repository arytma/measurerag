"""results/manifest.json - the reproducibility record.

It holds the SHA-256 of every dataset slice and every result file, plus the
machine, the package versions and the config. Two runs that agree on this file
(minus the timing fields, which are physics, not logic) produced the same
numbers for the same reasons.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_json

# Anything below is expected to differ between two identical runs; the repro
# check compares everything else byte for byte.
VOLATILE_KEYS = {
    "created_utc",
    "wall_seconds",
    "build_seconds",
    "total_seconds",
    "seconds",
    "retrieval_latency_ms",
    "e2e_latency_s",
    "retrieval_ms_p50",
    "retrieval_ms_p95",
    "e2e_s_p50",
    "e2e_s_p95",
    "gen_tokens_per_s",
    "tokens_per_s",
    "peak_rss_gb",
    "model_path",
    "verdicts",  # derived from latency and RSS, so equally volatile
    "measurement",
}


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "measurerag", "numpy", "scipy", "scikit-learn", "torch",
        "sentence-transformers", "transformers", "faiss-cpu", "bm25s",
        "llama-cpp-python", "datasets", "huggingface-hub",
    ]
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = "not installed"
    return out


def build_manifest(
    *,
    config: dict,
    config_path: str | Path | None,
    dataset_fingerprints: list[dict],
    results: list[dict],
    result_files: list[str | Path],
    machine: dict,
    wall_seconds: float,
    seed: int,
) -> dict[str, Any]:
    files = {}
    for f in result_files:
        p = Path(f)
        if p.exists():
            files[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    # The scientific content of a result: metrics and identity, no timings.
    stable_results = [
        {
            "config_id": r.get("config_id"),
            "retriever_id": r.get("retriever_id"),
            "retriever_name": r.get("retriever_name"),
            "status": r.get("status"),
            "metrics": r.get("metrics"),
            "answers": r.get("answers"),
            "dataset": r.get("dataset"),
            "n_docs": r.get("n_docs"),
            "n_queries": r.get("n_queries"),
        }
        for r in results
    ]

    return {
        "schema": "measurerag/manifest/1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "config_path": str(config_path) if config_path else None,
        "config": config,
        "config_sha256": sha256_json(config),
        "datasets": dataset_fingerprints,
        "machine": machine,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "wall_seconds": round(wall_seconds, 2),
        "results_stable_sha256": sha256_json(stable_results),
        "results_stable": stable_results,
        "files": files,
    }


def strip_volatile(obj: Any) -> Any:
    """Recursively drop the fields that legitimately move between runs."""
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def compare_manifests(a: dict, b: dict) -> dict:
    """Did two runs agree where they had to?"""
    sa, sb = strip_volatile(a), strip_volatile(b)
    identical = sha256_json(sa) == sha256_json(sb)
    diffs: list[str] = []
    if not identical:
        diffs = _diff(sa, sb, "")
    return {
        "identical_ignoring_timings": identical,
        "results_stable_sha256_a": a.get("results_stable_sha256"),
        "results_stable_sha256_b": b.get("results_stable_sha256"),
        "metrics_identical": a.get("results_stable_sha256") == b.get("results_stable_sha256"),
        "differences": diffs[:50],
    }


def _diff(a: Any, b: Any, path: str) -> list[str]:
    out: list[str] = []
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing in first")
            elif k not in b:
                out.append(f"{path}.{k}: missing in second")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
