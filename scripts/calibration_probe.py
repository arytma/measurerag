"""Calibration probe: is the machine as fast now as when the baseline was taken?

The specification's measurement hygiene (Rule 6): before every timed batch, time 20
fixed generations with the 1.5B model on bm25 contexts at k = 5 and compare the
median with the baseline recorded at preflight. More than 10% drift (heat,
background load) means wait and probe again; a batch that ran under drift is
rerun.

The 20 questions are the first 20 HotpotQA slice ids (sorted); the contexts are
bm25's cached top 5; the generator settings are the harness's own (temperature
0, seed 20260101, max_tokens 64, n_ctx 4096, llama.cpp's default threads). One
untimed warm-up generation precedes the 20, so model load and first-call costs
are not in the median. The answers are hashed: the probe also checks that
generation is still deterministic.

    uv run python scripts/calibration_probe.py --baseline     # once, at preflight
    uv run python scripts/calibration_probe.py --label <batch>  # exit 0 ok, 3 drift

Every probe is appended to results/crossover/calibration/probes.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "results" / "crossover" / "calibration"
BASELINE = CAL / "baseline.json"
PROBES = CAL / "probes.jsonl"
SLICE = ROOT / "results" / "en_rag" / "slice"
CACHE = ROOT / "results" / "crossover" / "cache" / "hotpotqa"

N, K, GEN, SEED, MAX_TOKENS, N_CTX = 20, 5, "qwen2.5-1.5b", 20260101, 64, 4096
TOLERANCE = 0.10


def probe_prompts() -> list[str]:
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset
    from measurerag.rag.generator import build_prompt

    ds = Dataset.load_slice(SLICE)
    run = load_run(CACHE, "bm25")["run"]
    by_id = {d.doc_id: d for d in ds.docs}
    qids = sorted(q for q in ds.answers if q in run)[:N]
    return [build_prompt(ds.queries[q], [by_id[d].indexable() for d in run[q][:K]]) for q in qids]


def drift(median: float, base: float) -> float:
    return (median - base) / base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="record this probe as the baseline")
    ap.add_argument("--label", default="", help="what the probe is for (batch name)")
    ap.add_argument("--log", default=None, help="append the probe here instead of probes.jsonl (v0.4.0's timed rows keep their own)")
    ap.add_argument("--baseline-file", default=None, help="the baseline to record or compare with, instead of baseline.json "
                    "(Ministral 3's English sweeps keep their own)")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from measurerag.openmp import preflight
    from measurerag.rag.generator import LlamaGenerator

    preflight()
    prompts = probe_prompts()
    gen = LlamaGenerator(GEN, device="cpu", seed=SEED, max_tokens=MAX_TOKENS, n_ctx=N_CTX)
    gen.generate(prompts[0])  # warm-up, untimed
    secs, answers = [], []
    for p in prompts:
        t0 = time.perf_counter()
        text, _ = gen.generate(p)
        secs.append(time.perf_counter() - t0)
        answers.append(text)
    gen.close()

    med = statistics.median(secs)
    row = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "baseline" if args.baseline else args.label,
        "median_s": round(med, 4),
        "seconds": [round(s, 4) for s in secs],
        "answers_sha256": hashlib.sha256("\n".join(answers).encode("utf-8")).hexdigest(),
        "loadavg_1m": round(os.getloadavg()[0], 2),
        "macos": platform.mac_ver()[0],
    }
    baseline = Path(args.baseline_file) if args.baseline_file else BASELINE
    baseline.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        baseline.write_text(json.dumps(row, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        status = 0
    else:
        base = json.loads(baseline.read_text(encoding="utf-8"))
        row["drift"] = round(drift(med, base["median_s"]), 4)
        row["answers_match_baseline"] = row["answers_sha256"] == base["answers_sha256"]
        row["ok"] = abs(row["drift"]) <= TOLERANCE and row["answers_match_baseline"]
        status = 0 if row["ok"] else 3
    log_path = Path(args.log) if args.log else PROBES
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "seconds"}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
