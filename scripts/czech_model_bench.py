"""A short benchmark of Western model families against Qwen2.5, on the dev set only.

    .venv/bin/python scripts/czech_model_bench.py --download            # fetch the pinned GGUFs (network)
    .venv/bin/python scripts/czech_model_bench.py --run                 # every model, each in its own process, GPU then CPU
    .venv/bin/python scripts/czech_model_bench.py                       # read the answers: writes the records and the result
    .venv/bin/python scripts/czech_model_bench.py --check               # recompute the result from the shipped records

Through the release's own prompt, generation and scoring code. Writes answers/sqad32/answers_model_bench.jsonl.gz (ids,
answers, scores) and results/czech/model_bench/bench.json. Prints no question and no passage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_CACHE", str(REPO / "data/hf-hub"))

from measurerag.crossover.stats import bootstrap_counts, established, interval  # noqa: E402
from measurerag.czech_sensitivity import YES_NO_CLASSES, compliance, yes_no_class  # noqa: E402
from measurerag.czech_store import load_labels, read_jsonl_gz, write_jsonl_gz  # noqa: E402

K, MT, SEED = 3, 64, 20260101
SLICE = REPO / "results/crossover/slices/complete_dev"
RANKS = REPO / "results/crossover/cache/v030_32"
WORK = REPO / "results/crossover/cache/model_bench"
ROWS = REPO / "answers/sqad32/answers_model_bench.jsonl.gz"
OUT = REPO / "results/czech/model_bench/bench.json"
CPU_QUESTIONS = 20
REFERENCE = "qwen2.5"

# family -> (smaller, larger); a model is key -> (repo, revision, file). Qwen's come from the release's registry.
FAMILIES = {
    "qwen2.5": ("qwen2.5-1.5b", "qwen2.5-3b"),
    "gemma-4": ("gemma-4-e2b", "gemma-4-e4b"),
    "granite-4.0": ("granite-4.0-1b", "granite-4.0-micro"),
    "ministral-3": ("ministral-3-3b", "ministral-3-8b"),
    "qwen3.5": ("qwen3.5-2b", "qwen3.5-4b"),
}
WESTERN = ("gemma-4", "granite-4.0", "ministral-3")          # the families the rule can add
COMPARISON = "qwen3.5"                                        # Qwen's current generation: reported, not in the rule
PINNED = {
    "gemma-4-e2b": ("unsloth/gemma-4-E2B-it-GGUF", "0314792d7f1f7e229411f620751375812bb9faf2", "gemma-4-E2B-it-Q4_K_M.gguf"),
    "gemma-4-e4b": ("unsloth/gemma-4-E4B-it-GGUF", "bfc15c382204943c3a8fff0c750b94ae2364d7a3", "gemma-4-E4B-it-Q4_K_M.gguf"),
    "granite-4.0-1b": ("ibm-granite/granite-4.0-1b-GGUF", "b27c2fe3f211b7f44e80fa620177aea371099aaa", "granite-4.0-1b-Q4_K_M.gguf"),
    "granite-4.0-micro": ("ibm-granite/granite-4.0-micro-GGUF", "ec48475f0c811d812fbfb61975717a9c36eeb652", "granite-4.0-micro-Q4_K_M.gguf"),
    "ministral-3-3b": ("mistralai/Ministral-3-3B-Instruct-2512-GGUF", "eb599d408350ea2bb60452cb86be7c7b2fc28227", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
    "qwen3.5-2b": ("unsloth/Qwen3.5-2B-GGUF", "f6d5376be1edb4d416d56da11e5397a961aca8ae", "Qwen3.5-2B-Q4_K_M.gguf"),
    "qwen3.5-4b": ("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q4_K_M.gguf"),
    "ministral-3-8b": ("mistralai/Ministral-3-8B-Instruct-2512-GGUF", "0102285ad796bd99af90f58de616092e5630e970", "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"),
}
MODELS = [m for pair in FAMILIES.values() for m in pair]


def model_path(key: str) -> Path:
    from huggingface_hub import hf_hub_download

    from measurerag.rag import generator as gm

    if key not in PINNED:
        return gm.resolve_model_path(key)
    repo, rev, name = PINNED[key]
    return Path(hf_hub_download(repo, name, revision=rev))


def register(key: str) -> None:
    """Make a pinned model loadable through the release's LlamaGenerator, in this process only."""
    from measurerag.rag import generator as gm

    # The benchmark's own pinned file wins, in this process: since v0.3.1 the release's registry has keys of the same name for
    # Unsloth's Ministral 3 files, and here "ministral-3-3b" means Mistral AI's own GGUF.
    if key in PINNED:
        repo, _, name = PINNED[key]
        gm.GENERATORS[key] = gm.GeneratorSpec(key, repo, name)
        path = model_path(key)
        original = gm.resolve_model_path
        gm.resolve_model_path = lambda k, cache_dir=None: path if k == key else original(k, cache_dir)


def prompts() -> list[tuple[str, str, list[str]]]:
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset
    from measurerag.rag.generator import build_prompt

    ds = Dataset.load_slice(SLICE)
    by_id = {d.doc_id: d for d in ds.docs}
    run = load_run(RANKS, "e5-small")["run"]
    out = []
    for q in sorted(ds.queries):
        ctx = list(run[q][:K])
        out.append((q, build_prompt(ds.queries[q], [by_id[d].indexable() for d in ctx if d in by_id], language="cs", variant="v1"), ctx))
    return out


def generate(key: str, device: str, limit: int | None) -> int:
    """One model, one process, an empty start, sorted question ids. Appends one JSON line per answer."""
    from measurerag.rag.generator import LlamaGenerator

    register(key)
    out = WORK / f"{key}_{device}.jsonl"
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    items = prompts()[:limit] if limit else prompts()
    t0 = time.perf_counter()
    gen = LlamaGenerator(key, device=device, seed=SEED, max_tokens=MT, n_ctx=4096)
    load_s = time.perf_counter() - t0
    with open(out, "w", encoding="utf-8") as fh:
        for q, prompt, ctx in items:
            text, info = gen.generate(prompt)
            fh.write(json.dumps({"q": q, "model": key, "device": device, "context": ctx, "answer": text, "seconds": round(info["seconds"], 4),
                                 "completion_tokens": info["completion_tokens"], "prompt_tokens": info["prompt_tokens"], "load_seconds": round(load_s, 2)}, ensure_ascii=False) + "\n")
    gen.close()
    print(key, device, len(items), "answers", flush=True)
    return 0


def run_all(models: list[str] | None = None, script: str = __file__, failures: str = "failures.json") -> int:
    """Each model in a process of its own: on the GPU for every question, then on the CPU for the first ones."""
    failed = {}
    for device, limit in (("cuda", None), ("cpu", CPU_QUESTIONS)):
        for key in models or MODELS:
            cmd = [sys.executable, script, "--one", key, "--device", device, *(["--limit", str(limit)] if limit else [])]
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                failed[f"{key} {device}"] = (r.stdout + r.stderr)[-1500:]
                print(f"{key} {device}: FAILED", flush=True)
            else:
                print(r.stdout.strip(), flush=True)
    (WORK / failures).write_text(json.dumps(failed, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def export(models: list[str] | None = None) -> list[dict]:
    from czech_records import scores
    from measurerag.datasets.base import Dataset

    ds = Dataset.load_slice(SLICE)
    rows = []
    for key in models or MODELS:
        for device in ("cuda", "cpu"):
            path = WORK / f"{key}_{device}.jsonl"
            if not path.exists():
                continue
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                rows.append({"set": "model_bench", "q": r["q"], "model": key, "device": device, "config": f"e5-small+{key}", "kind": "model_bench", "p": 0.0,
                             "max_tokens": MT, "seed": SEED, "context": r["context"], "answer": r["answer"], "seconds": r["seconds"],
                             "completion_tokens": r["completion_tokens"], "load_seconds": r["load_seconds"], **scores(r["answer"], ds.answers[r["q"]])})
    rows.sort(key=lambda r: (r["model"], r["device"], r["q"]))
    return rows


def ci(point: float, boot: np.ndarray) -> dict:
    lo, hi = interval(boot)
    return {"point": float(point), "lo": lo, "hi": hi, "established": established(lo, hi)}


def analyse(rows: list[dict], labels: dict, sizes: dict[str, float], failures: dict) -> dict:
    gpu = {m: {r["q"]: r for r in rows if r["model"] == m and r["device"] == "cuda"} for m in MODELS}
    cpu = {m: [r for r in rows if r["model"] == m and r["device"] == "cpu"] for m in MODELS}
    qids = sorted(next(v for v in gpu.values() if v))
    is_yn = np.array([labels[q]["type"] == "yes/no" for q in qids])
    out: dict = {"questions": len(qids), "yes_no_questions": int(is_yn.sum()), "models": {}, "families": {}, "failures": sorted(failures)}
    for m in MODELS:
        out["models"][m] = model_block(gpu[m], cpu[m], qids, is_yn, sizes.get(m))
    return families_and_rule(out, gpu, qids)


def model_block(g: dict, cpu_rows: list[dict], qids: list[str], is_yn: np.ndarray, size: float | None) -> dict:
    """One model's measures on the dev set."""
    if len(g) != len(qids):
        return {"answered": False}
    rs = [g[q] for q in qids]
    yn = [r for r, y in zip(rs, is_yn) if y]
    span = [r for r, y in zip(rs, is_yn) if not y]
    classes = {c: sum(yes_no_class(r["answer"]) == c for r in yn) for c in YES_NO_CLASSES}
    return {
        "answered": True, "gguf_gb": size,
        "em_yes_no_mapped": float(np.mean([r["em_yes_no_mapped"] for r in rs])), "em": float(np.mean([r["em"] for r in rs])), "f1": float(np.mean([r["f1"] for r in rs])),
        "compliance": compliance(classes), "strict_share": classes["ano or ne"] / len(yn), "yes_no_classes": classes,
        "em_on_yes_no": float(np.mean([r["em"] for r in yn])), "em_yes_no_mapped_on_yes_no": float(np.mean([r["em_yes_no_mapped"] for r in yn])),
        "em_on_span": float(np.mean([r["em"] for r in span])), "f1_on_span": float(np.mean([r["f1"] for r in span])),
        "span_declined": sum(yes_no_class(r["answer"]) == "unanswerable" for r in span),
        "answers_cut_at_the_cap": sum(r["completion_tokens"] >= MT for r in rs),
        "gpu_seconds_per_answer": float(np.mean([r["seconds"] for r in rs])),
        "cpu_seconds_per_answer": float(np.mean([r["seconds"] for r in cpu_rows])) if cpu_rows else None, "cpu_questions": len(cpu_rows),
    }


def families_and_rule(out: dict, gpu: dict, qids: list[str]) -> dict:
    counts = bootstrap_counts(len(qids), 10000, "czech-v030", "model-bench").astype(float)

    def family_scores(f: str) -> np.ndarray | None:
        a, b = FAMILIES[f]
        if not (out["models"][a].get("answered") and out["models"][b].get("answered")):
            return None
        return np.array([(gpu[a][q]["em_yes_no_mapped"] + gpu[b][q]["em_yes_no_mapped"]) / 2 for q in qids], dtype=float)

    ref, newer = family_scores(REFERENCE), family_scores(COMPARISON)
    for f in FAMILIES:
        s = family_scores(f)
        block: dict = {"eligible": s is not None}
        if s is not None:
            block["score"] = mean_ci(s, counts)
            if f != REFERENCE:
                block["minus_qwen"] = ci(float((s - ref).mean()), (counts @ (s - ref)) / len(qids))
            if f != COMPARISON and newer is not None:
                block["minus_qwen3_5"] = ci(float((s - newer).mean()), (counts @ (s - newer)) / len(qids))
        out["families"][f] = block
    western = [f for f in WESTERN if out["families"][f]["eligible"]]
    best = max(western, key=lambda f: out["families"][f]["score"]["point"]) if western else None
    wins = bool(best) and out["families"][best]["minus_qwen"]["established"] and out["families"][best]["minus_qwen"]["point"] > 0
    out["rule"] = {"best_western_family": best, "joins_beside_qwen": best if wins else None,
                   "outcome": f"{best} joins every Czech run beside Qwen" if wins else "Qwen alone; the benchmark is reported"}
    return out


def mean_ci(values: np.ndarray, counts: np.ndarray) -> dict:
    return ci(float(values.mean()), (counts @ values) / counts.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--one", default=None)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.chdir(REPO)
    if args.download:
        for key in MODELS:
            print(key, model_path(key), flush=True)
        return 0
    if args.one:
        return generate(args.one, args.device, args.limit)
    if args.run:
        return run_all()
    labels = load_labels(REPO / "results/czech/records", "sqad32_test")
    if not args.check:
        write_jsonl_gz(ROWS, export())
    rows = read_jsonl_gz(ROWS)
    failures = json.loads((WORK / "failures.json").read_text(encoding="utf-8")) if (WORK / "failures.json").exists() else {}
    if args.check and OUT.exists():
        failures = {k: "" for k in json.loads(OUT.read_text(encoding="utf-8"))["failures"]}
        sizes = json.loads(OUT.read_text(encoding="utf-8"))["gguf_gb"]
    else:
        sizes = {m: round(model_path(m).stat().st_size / 1e9, 2) for m in MODELS if not any(k.startswith(m + " ") for k in failures)}
    result = analyse(rows, labels, sizes, failures)
    result["gguf_gb"] = sizes
    result["pinned"] = {k: {"repo": v[0], "revision": v[1], "file": v[2]} for k, v in PINNED.items()}
    text = json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if args.check:
        same = OUT.read_text(encoding="utf-8") == text
        print("czech_model_bench:", "OK" if same else "DIFFERENT")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({"rule": result["rule"], "families": result["families"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
