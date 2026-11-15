"""Runs exactly one configuration, in its own process.

Isolation is the whole point: peak RSS only means something if nothing else is
resident. If four retrievers share a process, the fourth one inherits the
high-water mark of the first three and every RAM verdict is wrong. The parent
(runner.py) launches this module once per configuration and reads back a JSON
result.

    python -m measurerag.worker --spec spec.json --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def run_spec(spec: dict) -> dict:
    from .budget import Budget, Measurement, check_all
    from .datasets.base import Dataset
    from .hardware import describe_machine, resolve_device
    from .memory import PeakRSS
    from .metrics import evaluate_run, latency_summary
    from .retrievers import build_retriever
    from .seeds import seed_everything

    seed_everything(int(spec.get("seed", 20260101)))
    device = resolve_device(spec.get("device", "cpu"))

    # Two thread counts, on purpose. Indexing is a one-off, offline, do-it-on-
    # the-big-machine job, so it gets every core. Serving one query is the thing
    # the budget is about and the thing that must be reproducible, so it is
    # pinned to a fixed (default: single) thread count regardless of the box.
    build_threads = spec.get("build_threads")
    query_threads = int(spec.get("query_threads", spec.get("torch_threads", 1)))
    if build_threads is None:
        import os as _os

        build_threads = _os.cpu_count() or 1
    build_threads = int(build_threads)

    def set_threads(n: int) -> None:
        try:
            import torch

            torch.set_num_threads(max(1, n))
        except Exception:
            pass

    slice_dir = Path(spec["slice_dir"])
    ds = Dataset.load_slice(slice_dir)
    workdir = Path(spec["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)

    top_k = int(spec.get("top_k", 100))
    ks = [int(k) for k in spec.get("ks", [10, 100])]
    context_k = int(spec.get("context_k", 5))
    rcfg = dict(spec["retriever"])
    kind = rcfg.pop("kind")
    rid = rcfg.pop("id", kind)
    rcfg.setdefault("device", device)

    result: dict = {
        "config_id": spec.get("config_id", rid),
        "retriever_id": rid,
        "retriever_kind": kind,
        "retriever_params": dict(rcfg),
        "dataset": ds.name,
        "language": ds.language,
        "device": device,
        "n_docs": len(ds.docs),
        "n_queries": len(ds.queries),
        "status": "ok",
    }

    with PeakRSS() as peak:
        t_start = time.perf_counter()
        retriever = build_retriever(kind, **rcfg)
        result["retriever_name"] = getattr(retriever, "name", rid)

        set_threads(build_threads)
        stats = retriever.build(ds.docs, workdir)
        set_threads(query_threads)
        result["threads"] = {"build": build_threads, "query": query_threads}
        result["build"] = {
            "build_seconds": stats.build_seconds,
            "index_bytes": stats.index_bytes,
            "model_bytes": stats.model_bytes,
            "n_docs": stats.n_docs,
            **stats.extra,
        }

        # ---- retrieval, one query at a time: that is how it is deployed ----
        run: dict[str, list[str]] = {}
        scores: dict[str, list[float]] = {}
        latencies: list[float] = []
        for qid in sorted(ds.queries):
            t0 = time.perf_counter()
            hits = retriever.search_one(ds.queries[qid], top_k)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            run[qid] = [doc_id for doc_id, _ in hits]
            scores[qid] = [s for _, s in hits]

        lat = latency_summary(latencies)
        result["retrieval_latency_ms"] = lat
        result["metrics"] = evaluate_run(run, ds.qrels, ks=ks, context_k=context_k)

        # ---- optional end-to-end RAG stage --------------------------------
        rag_cfg = spec.get("rag") or {}
        if rag_cfg.get("enabled") and ds.answers:
            result.update(
                _run_rag_stage(ds, run, rag_cfg, device, workdir, context_k, int(spec.get("seed", 20260101)))
            )

        total_seconds = time.perf_counter() - t_start
        retriever.close()

    measurement = Measurement(
        peak_rss_gb=round(peak.peak_gb, 4),
        index_bytes=int(result["build"].get("index_bytes", 0)),
        model_bytes=int(result["build"].get("model_bytes", 0))
        + int(result.get("generator", {}).get("model_bytes", 0)),
        retrieval_ms_p50=lat["p50"],
        retrieval_ms_p95=lat["p95"],
        e2e_s_p50=result.get("e2e_latency_s", {}).get("p50"),
        e2e_s_p95=result.get("e2e_latency_s", {}).get("p95"),
        gen_tokens_per_s=result.get("generator", {}).get("tokens_per_s"),
    )
    budgets = [Budget.from_dict(b) for b in spec.get("budgets", [])]
    result["measurement"] = measurement.as_dict()
    result["verdicts"] = [v.as_dict() for v in check_all(measurement, budgets)]
    result["total_seconds"] = total_seconds
    result["machine"] = describe_machine().as_dict()
    return result


def _run_rag_stage(ds, run, rag_cfg, device, workdir, context_k, seed) -> dict:
    """Generate an answer per question from the passages this retriever found."""
    from .metrics import evaluate_answers, latency_summary
    from .rag.generator import GeneratorUnavailable, LlamaGenerator, build_prompt

    gen_key = rag_cfg["generator"]
    out: dict = {}
    try:
        gen = LlamaGenerator(
            gen_key,
            device=rag_cfg.get("device") or device,     # a config may generate on another device than it retrieves on
            seed=seed,
            max_tokens=int(rag_cfg.get("max_tokens", 256)),
            n_ctx=rag_cfg.get("n_ctx"),
            n_threads=rag_cfg.get("n_threads"),
            use_mmap=bool(rag_cfg.get("use_mmap", True)),
        )
    except (GeneratorUnavailable, Exception) as exc:
        return {
            "generator": {"key": gen_key, "status": "unavailable", "error": str(exc)[:400]},
            "rag_status": "skipped",
        }

    by_id = {d.doc_id: d for d in ds.docs}
    preds: dict[str, str] = {}
    e2e: list[float] = []
    tok_rates: list[float] = []
    n_tokens = 0
    gen_seconds = 0.0

    # An unattended queue watches this file: answers are only written when a configuration ends.
    heartbeat = os.environ.get("MEASURERAG_HEARTBEAT")
    for qid in sorted(ds.answers):
        if qid not in run:
            continue
        t0 = time.perf_counter()
        passages = [by_id[d].indexable() for d in run[qid][:context_k] if d in by_id]
        prompt = build_prompt(ds.queries[qid], passages, language=ds.language)
        text, info = gen.generate(prompt)
        e2e.append(time.perf_counter() - t0)
        preds[qid] = text
        if heartbeat:
            Path(heartbeat).write_text(str(len(preds)), encoding="utf-8")
        n_tokens += info["completion_tokens"]
        gen_seconds += info["seconds"]
        if info["tokens_per_s"]:
            tok_rates.append(info["tokens_per_s"])

    gen.close()
    out["answers"] = evaluate_answers(preds, ds.answers)
    out["e2e_latency_s"] = latency_summary(e2e)
    out["generator"] = {
        "key": gen_key,
        "status": "ok",
        "model_bytes": gen.model_bytes,
        "model_path": str(gen.model_path),
        "tokens_per_s": (n_tokens / gen_seconds) if gen_seconds > 0 else 0.0,
        "completion_tokens": n_tokens,
        "use_mmap": gen.use_mmap,      # whether the model file was memory-mapped: peak RSS depends on it
    }
    out["rag_status"] = "ok"
    out["predictions"] = preds
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    try:
        result = run_spec(spec)
    except Exception as exc:
        result = {
            "config_id": spec.get("config_id", "?"),
            "retriever_id": spec.get("retriever", {}).get("id", "?"),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
