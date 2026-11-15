"""Orchestrates a run: resolve the data, launch one child per configuration,
write the table, the curves and the manifest.

Nothing heavy is imported here on purpose - the parent stays small so that its
own memory never confuses a child's measurement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from . import report
from .budget import budget_grid
from .datasets import load_dataset
from .datasets.base import Dataset, DatasetBlocked, DatasetUnavailable, cache_root
from .hardware import describe_machine, resolve_device
from .hashing import sha256_bytes
from .manifest import build_manifest
from .openmp import preflight as openmp_preflight
from .rag.generator import load_prompt_template
from .retrievers import describe_retriever


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# data resolution, with the fallback chain the specification requires
# --------------------------------------------------------------------------
def resolve_dataset(cfg: dict, cache: Path, log: Callable[[str], None] = _log) -> tuple[Dataset, list[dict]]:
    """Try the primary dataset, then each fallback, recording why each failed.

    A dataset that needs a form or a login is BLOCKED, never worked around.
    """
    dcfg = cfg["dataset"]
    chain = [dcfg["name"], *dcfg.get("fallbacks", [])]
    attempts: list[dict] = []

    for i, name in enumerate(chain):
        role = "primary" if i == 0 else "fallback"
        try:
            log(f"  dataset: trying {name} ({role}) ...")
            ds = load_dataset(
                name,
                cache_dir=cache,
                max_docs=dcfg.get("max_docs"),
                max_queries=dcfg.get("max_queries"),
                **(dcfg.get("options") or {}),
            )
            if role == "fallback":
                ds.notes.append(
                    f"FALLBACK: the primary dataset {chain[0]!r} was not obtainable "
                    f"({attempts[0].get('error', 'see manifest')}). Numbers are on {name}."
                )
            attempts.append({"dataset": name, "role": role, "status": "ok"})
            log(f"  dataset: {ds.name} - {len(ds.docs)} docs, {len(ds.queries)} queries")
            return ds, attempts
        except DatasetBlocked as exc:
            log(f"  dataset: BLOCKED {name}: {exc}")
            attempts.append({"dataset": name, "role": role, "status": "blocked", "error": str(exc)})
        except (DatasetUnavailable, Exception) as exc:
            log(f"  dataset: unavailable {name}: {type(exc).__name__}: {exc}")
            attempts.append(
                {"dataset": name, "role": role, "status": "unavailable",
                 "error": f"{type(exc).__name__}: {exc}"}
            )

    raise RuntimeError(
        "no dataset in the chain could be obtained: "
        + json.dumps(attempts, ensure_ascii=False)
    )


# --------------------------------------------------------------------------
# the run
def wanted(retriever_id: str, generator: str | None, only: list[str] | None) -> bool:
    """`only` names retriever ids ("bm25") or single configurations
    ("bm25+qwen2.5-1.5b"), so a timed queue can measure one at a time.
    With `generator=None` the question is whether any job of the retriever is wanted."""
    if not only or retriever_id in only:
        return True
    if generator is None:
        return any(o.startswith(retriever_id + "+") for o in only)
    return f"{retriever_id}+{generator}" in only


# --------------------------------------------------------------------------
def run(
    cfg: dict,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    only: list[str] | None = None,
    resume: bool = True,
    log: Callable[[str], None] = _log,
) -> dict:
    t_start = time.perf_counter()
    # Before anything expensive: an environment that would segfault halfway
    # through is not worth four hours of wall-clock to discover.
    openmp_preflight()
    out = Path(out_dir or cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cache = cache_root(cfg.get("cache_dir"))
    device = resolve_device(cfg.get("device", "cpu"))
    if device != (cfg.get("device") or "cpu"):
        log(f"  device: requested {cfg.get('device')!r}, using {device!r}")

    log(f"[{cfg['name']}] device={device} seed={cfg['seed']}")

    ds, attempts = resolve_dataset(cfg, cache, log)
    slice_dir = out / "slice"
    ds.save(slice_dir)
    fingerprints = [ds.fingerprint() | {"attempts": attempts}]

    rcfg = cfg["retrieval"]
    bcfg = cfg["budgets"]
    rag_cfg = cfg.get("rag") or {}
    rag_enabled = bool(rag_cfg.get("enabled")) and bool(ds.answers)
    budgets = budget_grid(
        [float(x) for x in bcfg["ram_gb"]],
        [float(x) for x in bcfg["retrieval_ms"]],
        [float(x) for x in bcfg["e2e_s"]] if (rag_enabled and bcfg.get("e2e_s")) else None,
    )
    budget_dicts = [b.as_dict() for b in budgets]

    # one job per (retriever) or (retriever x generator) when RAG is on
    jobs: list[dict] = []
    for r in cfg["retrievers"]:
        if not wanted(r["id"], None, only):
            continue
        if rag_enabled and rag_cfg.get("generators"):
            for gen in rag_cfg["generators"]:
                if not wanted(r["id"], gen, only):
                    continue
                jobs.append(
                    {
                        "config_id": f"{r['id']}+{gen}",
                        "retriever": dict(r),
                        "rag": {
                            "enabled": True,
                            "generator": gen,
                            "max_tokens": rag_cfg.get("max_tokens", 256),
                            "n_ctx": rag_cfg.get("n_ctx"),
                            "n_threads": rag_cfg.get("n_threads"),
                            # only when set, so that every spec written before this key existed stays the same
                            **({"device": rag_cfg["device"]} if rag_cfg.get("device") else {}),
                            # the same: the GGUF is memory-mapped (llama-cpp-python's default) unless a config says
                            **({"use_mmap": bool(rag_cfg["use_mmap"])} if "use_mmap" in rag_cfg else {}),
                        },
                    }
                )
        else:
            jobs.append({"config_id": r["id"], "retriever": dict(r), "rag": {"enabled": False}})

    results: list[dict] = []
    specs_dir = out / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        spec = {
            "config_id": job["config_id"],
            "slice_dir": str(slice_dir),
            "workdir": str(out / "work" / job["config_id"].replace("/", "_")),
            "retriever": job["retriever"],
            "rag": job["rag"],
            "budgets": budget_dicts,
            "top_k": rcfg["top_k"],
            "ks": rcfg["ks"],
            "context_k": rcfg["context_k"],
            "device": device,
            "seed": cfg["seed"],
            "query_threads": cfg.get("query_threads", cfg.get("torch_threads", 1)),
            "build_threads": cfg.get("build_threads"),
        }
        # Only when there is one: an empty key would change every lexical spec
        # too, and re-measure BM25 for nothing.
        declaration = describe_retriever(
            job["retriever"]["kind"],
            **{k: v for k, v in job["retriever"].items() if k not in ("id", "kind")},
        )
        if declaration:
            spec["retriever_declaration"] = declaration
        # The prompt depends on the language, so the spec has to say so:
        # otherwise a Czech result measured under the old prompt would pass as finished.
        # Only where it matters, for the reason above: English specs stay byte-identical.
        if job["rag"].get("enabled") and ds.language != "en":
            spec["prompt_language"] = ds.language
            spec["prompt_sha256"] = sha256_bytes(load_prompt_template(ds.language).encode("utf-8"))

        safe = job["config_id"].replace("/", "_")
        spec_path = specs_dir / f"{safe}.spec.json"
        res_path = specs_dir / f"{safe}.result.json"
        spec_json = json.dumps(spec, indent=2, sort_keys=True)

        # Resume. A configuration that already finished under a byte-identical
        # spec is not re-measured: an unattended queue that loses a machine at
        # 03:00 should cost the hour it was in, not the six behind it. The spec
        # carries the slice path, the caps, the seed and the thread counts, so
        # "identical spec" is the same comparison the manifest makes.
        if resume and res_path.exists() and spec_path.exists():
            if spec_path.read_text(encoding="utf-8") == spec_json:
                try:
                    prev = json.loads(res_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    prev = None
                if prev and prev.get("status") == "ok":
                    prev["resumed"] = True
                    log(f"  {job['config_id']}: reusing the result already on disk")
                    results.append(prev)
                    continue

        # A rebuild of a sweep's files from finished results must never measure:
        # a spec that drifted would re-measure an accepted configuration and
        # replace its timings. Nothing is written before this point.
        if os.environ.get("MEASURERAG_REUSE_ONLY"):
            raise RuntimeError(
                f"{job['config_id']}: no finished result under an identical spec, "
                "and MEASURERAG_REUSE_ONLY forbids measuring it"
            )

        spec_path.write_text(spec_json, encoding="utf-8")

        log(f"  running {job['config_id']} (isolated process) ...")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "measurerag.worker", "--spec", str(spec_path), "--out", str(res_path)],
            capture_output=True,
            text=True,
            timeout=cfg.get("timeout_s", 3600),
            check=False,
        )
        elapsed = time.perf_counter() - t0

        if res_path.exists():
            res = json.loads(res_path.read_text(encoding="utf-8"))
        else:
            detail = (proc.stderr or proc.stdout or "").strip()
            if proc.returncode < 0:
                import signal as _signal

                name = _signal.Signals(-proc.returncode).name
                detail = (
                    f"child killed by {name} ({proc.returncode}) without writing a result"
                    + (f"; last output: {detail[-600:]}" if detail else "")
                )
            res = {
                "config_id": job["config_id"],
                "retriever_id": job["retriever"]["id"],
                "status": "failed",
                "error": (detail or "child produced no result")[-1000:],
            }
        res["wall_seconds"] = round(elapsed, 2)
        if res.get("status") == "ok":
            q = res.get("metrics", {}).get(rcfg["quality_key"])
            log(
                f"    ok in {elapsed:.1f}s - {rcfg['quality_key']}="
                f"{q:.4f} peak={res['measurement']['peak_rss_gb']:.2f} GB "
                f"p95={res['measurement']['retrieval_ms_p95']:.1f} ms"
                if q is not None
                else f"    ok in {elapsed:.1f}s"
            )
        else:
            log(f"    FAILED: {res.get('error', '')[:200]}")
        results.append(res)

    # ---- outputs ---------------------------------------------------------
    rows = [report.flatten(r) for r in results]
    csv_path = report.write_csv(rows, out / "results.csv")

    # predictions are bulky; keep them beside the results, not inside them
    slim = []
    for r in results:
        r = dict(r)
        preds = r.pop("predictions", None)
        if preds:
            report.write_json(preds, out / f"predictions_{r['config_id'].replace('/', '_')}.json")
        slim.append(r)

    payload = {
        "config": cfg,
        "budgets": budget_dicts,
        "datasets": fingerprints,
        "machine": describe_machine().as_dict(),
        "results": slim,
    }
    json_path = report.write_json(payload, out / "results.json")

    qk = rcfg["quality_key"]
    mem_png = report.curve_quality_vs_memory(results, budget_dicts, out / "curve_quality_vs_memory.png", qk)
    lat_png = report.curve_quality_vs_latency(results, budget_dicts, out / "curve_quality_vs_latency.png", qk)

    table = report.budget_table(results, budget_dicts, qk)
    (out / "results_table.txt").write_text(table + "\n", encoding="utf-8")

    wall = time.perf_counter() - t_start
    manifest = build_manifest(
        config=cfg,
        config_path=config_path,
        dataset_fingerprints=fingerprints,
        results=results,
        result_files=[p for p in [csv_path, json_path, mem_png, lat_png] if p],
        machine=describe_machine().as_dict(),
        wall_seconds=wall,
        seed=cfg["seed"],
    )
    report.write_json(manifest, out / "manifest.json")

    log("")
    log(table)
    log("")
    log(f"  wrote {out}/results.csv, results.json, manifest.json, 2 curves  ({wall:.1f}s)")
    return {"out_dir": str(out), "results": results, "budgets": budget_dicts, "table": table,
            "manifest": manifest, "datasets": fingerprints}
