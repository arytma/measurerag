"""Ministral 3 on the benchmark's dev set, through Unsloth's build. Exploratory.

    .venv/bin/python scripts/czech_model_bench_mistral.py --download
    .venv/bin/python scripts/czech_model_bench_mistral.py --run
    .venv/bin/python scripts/czech_model_bench_mistral.py            # writes the records and the result
    .venv/bin/python scripts/czech_model_bench_mistral.py --check    # recompute the result from the shipped records

Mistral AI's own GGUF files do not load in the release's llama.cpp build, and Unsloth's build of the same weights does.
Added after the benchmark's result was read, so it changes nothing that the benchmark decided. The same procedure,
prompt, generation and scoring as scripts/czech_model_bench.py. Writes answers/sqad32/answers_model_bench_mistral.jsonl.gz
and results/czech/model_bench/mistral.json. Prints no question and no passage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import czech_model_bench as mb  # noqa: E402
from measurerag.crossover.stats import bootstrap_counts  # noqa: E402
from measurerag.czech_store import load_labels, read_jsonl_gz, write_jsonl_gz  # noqa: E402

MISTRAL = {
    "ministral-3-3b-unsloth": ("unsloth/Ministral-3-3B-Instruct-2512-GGUF", "7564922f37fa5bbb62b87f09a55c12f1f91d7a6a", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
    "ministral-3-8b-unsloth": ("unsloth/Ministral-3-8B-Instruct-2512-GGUF", "3731507ec3e867db16d620f73e14d689125758f4", "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"),
    "ministral-3-14b-unsloth": ("unsloth/Ministral-3-14B-Instruct-2512-GGUF", "9b948841b705cc11285ff274b84ba6c885908b19", "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf"),
}
mb.PINNED.update(MISTRAL)
MODELS = list(MISTRAL)
PAIR = ("ministral-3-3b-unsloth", "ministral-3-8b-unsloth")     # the two sizes the benchmark's pair would have been
ROWS = REPO / "answers/sqad32/answers_model_bench_mistral.jsonl.gz"
OUT = REPO / "results/czech/model_bench/mistral.json"
FAILURES = "failures_mistral.json"


def analyse(rows: list[dict], bench_rows: list[dict], labels: dict, sizes: dict) -> dict:
    gpu = {m: {r["q"]: r for r in rows if r["model"] == m and r["device"] == "cuda"} for m in MODELS}
    cpu = {m: [r for r in rows if r["model"] == m and r["device"] == "cpu"] for m in MODELS}
    other = {m: {r["q"]: r for r in bench_rows if r["model"] == m and r["device"] == "cuda"} for m in mb.MODELS}
    qids = sorted(next(iter(other.values())))
    is_yn = np.array([labels[q]["type"] == "yes/no" for q in qids])
    out: dict = {"questions": len(qids), "models": {m: mb.model_block(gpu[m], cpu[m], qids, is_yn, sizes.get(m)) for m in MODELS}}
    counts = bootstrap_counts(len(qids), 10000, "czech-v030", "model-bench").astype(float)   # the benchmark's resamples, so the scores compare

    def score(a: dict, b: dict) -> np.ndarray:
        return np.array([(a[q]["em_yes_no_mapped"] + b[q]["em_yes_no_mapped"]) / 2 for q in qids], dtype=float)

    if all(out["models"][m].get("answered") for m in PAIR):
        mine = score(gpu[PAIR[0]], gpu[PAIR[1]])
        block: dict = {"score": mb.mean_ci(mine, counts), "minus": {}}
        for family, (a, b) in mb.FAMILIES.items():
            if other[a] and other[b] and len(other[a]) == len(qids) == len(other[b]):
                theirs = score(other[a], other[b])
                block["minus"][family] = mb.ci(float((mine - theirs).mean()), (counts @ (mine - theirs)) / len(qids))
        out["family_3b_and_8b"] = block
    big = gpu["ministral-3-14b-unsloth"]
    if len(big) == len(qids):
        v = np.array([big[q]["em_yes_no_mapped"] for q in qids], dtype=float)
        out["14b_minus"] = {}
        for m in ("gemma-4-e4b", "qwen3.5-4b", "qwen2.5-3b"):
            if len(other.get(m, {})) == len(qids):
                d = v - np.array([other[m][q]["em_yes_no_mapped"] for q in qids], dtype=float)
                out["14b_minus"][m] = mb.ci(float(d.mean()), (counts @ d) / len(qids))
    return out


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
            print(key, mb.model_path(key), flush=True)
        return 0
    if args.one:
        return mb.generate(args.one, args.device, args.limit)
    if args.run:
        return mb.run_all(MODELS, __file__, FAILURES)
    labels = load_labels(REPO / "results/czech/records", "sqad32_test")
    if not args.check:
        write_jsonl_gz(ROWS, mb.export(MODELS))
    if args.check:
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        sizes, failures = prev["gguf_gb"], prev["failures"]
    else:
        fpath = mb.WORK / FAILURES
        failures = sorted(json.loads(fpath.read_text(encoding="utf-8"))) if fpath.exists() else []
        sizes = {m: round(mb.model_path(m).stat().st_size / 1e9, 2) for m in MODELS if not any(f.startswith(m + " ") for f in failures)}
    result = analyse(read_jsonl_gz(ROWS), read_jsonl_gz(mb.ROWS), labels, sizes)
    result.update({"gguf_gb": sizes, "failures": failures, "pinned": {k: {"repo": v[0], "revision": v[1], "file": v[2]} for k, v in MISTRAL.items()}})
    text = json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if args.check:
        same = OUT.read_text(encoding="utf-8") == text
        print("czech_model_bench_mistral:", "OK" if same else "DIFFERENT")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("family_3b_and_8b", "14b_minus") if k in result}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
