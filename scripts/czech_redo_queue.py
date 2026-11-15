"""Every Czech generation again, once, with the fixed prompt. One heavy process at a time.

The first v0.3.0 cut's queue, kept as the record of how that cut's answers were made (answers/<source>/first_cut/,
results/czech/compute.json). The complete run replaced it: scripts/czech_complete_queue.py.

    caffeinate -dimsu .venv/bin/python scripts/czech_redo_queue.py

Restartable: answers are cached per prompt, and the bench reuses a finished configuration.
A dial starts from a cache with no copied answer in it and walks its own order, as v0.2.0's did, and
must then agree with the bench.
Stops cleanly at either cap, and a stop keeps every answer already made:
    MAX_GENERATIONS  3,200
    MAX_SECONDS      12 hours of wall clock over all restarts
Priority order: the released 200, SQAD 3.2 full, SQAD 3.2 debiased, the SQAD 3.0 copy debiased,
then the 0.10 extras. A stage that would pass the generation cap does not start, and the queue
records why. Rankings are reused: no retrieval is re-run. Writes results/czech/compute.json.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
MAX_GENERATIONS = 3200
MAX_SECONDS = 43200
PY = ".venv/bin/python"
LOGS = REPO / "logs/v030"
STATE = REPO / "results/czech/compute.json"
B, G = "qwen2.5-1.5b", "qwen2.5-3b"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_CACHE": str(REPO / "data/hf-hub"), "TOKENIZERS_PARALLELISM": "false"}
COPY = "results/crossover/cache/v030_copy"
C32 = "results/crossover/cache/v030_32"
S = "results/crossover/slices"
ALL_LEVELS = "0,0.2,0.4,0.6,0.8,1.0"
SMOKE_GENERATIONS = 40   # scripts/czech_prompt_smoke.py
# The first attempt. Its spot check regenerated ten answers and one differed; finding out why took 37
# more. These count against the cap, and the wall clock counts from the first start.
FIRST_START = "2026-11-08 13:40:44"
FAILED_CHECK_AND_DIAGNOSIS = 47


def account_for_the_first_attempt(st: dict) -> dict:
    """Once: the stage that failed its check moves aside, and what it and its diagnosis cost is counted."""
    failed = st["stages"].pop("released_backfill", None)
    if failed:
        st["abandoned"] = {"stage": "released_backfill", "exit": failed["exit"], "finished": failed["finished"],
                           "why": "a spot check in random order",
                           "generations_failed_check_and_diagnosis": FAILED_CHECK_AND_DIAGNOSIS}
        st["generations"] += FAILED_CHECK_AND_DIAGNOSIS
        st["not_run"] = []
        st.pop("stopped", None)
        st["wall_seconds"] = round(time.time() - time.mktime(time.strptime(FIRST_START, "%Y-%m-%d %H:%M:%S")), 1)
    return st


def load_state() -> dict:
    if STATE.exists():
        return account_for_the_first_attempt(json.loads(STATE.read_text(encoding="utf-8")))
    return {"max_generations": MAX_GENERATIONS, "max_seconds": MAX_SECONDS, "wall_seconds": 0.0,
            "generations": SMOKE_GENERATIONS, "generations_smoke_test": SMOKE_GENERATIONS, "stages": {}, "not_run": []}


def save(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, sort_keys=True) + "\n", encoding="utf-8")


class Queue:
    def __init__(self) -> None:
        self.st = load_state()
        self.st.pop("stopped", None)
        self.t0 = time.time()
        self.base = float(self.st.get("wall_seconds", 0.0))
        LOGS.mkdir(parents=True, exist_ok=True)

    def elapsed(self) -> float:
        return self.base + time.time() - self.t0

    def done(self, name: str) -> bool:
        return (self.st["stages"].get(name) or {}).get("exit") == 0

    def stop(self, name: str, why: str) -> None:
        self.st.setdefault("not_run", []).append({"stage": name, "why": why})
        self.st["stopped"] = {"stage": name, "why": why}
        self.st["wall_seconds"] = round(self.elapsed(), 1)
        save(self.st)
        print(f"== STOPPED at {name}: {why}", flush=True)

    def run(self, name: str, cmd: list[str], gens: int = 0) -> bool:
        if self.done(name):
            print(f"== {name}: done before, skipped", flush=True)
            return True
        left = MAX_SECONDS - self.elapsed()
        if left <= 0:
            self.stop(name, "the 12-hour cap")
            return False
        if self.st["generations"] + gens > MAX_GENERATIONS:
            self.stop(name, f"the generation cap ({self.st['generations']} used, {gens} needed)")
            return False
        print(f"== {time.strftime('%H:%M:%S')} {name}: {gens} generations planned, {left / 3600:.2f} h left", flush=True)
        t = time.time()
        rc = 124
        with open(LOGS / f"{name}.log", "w", encoding="utf-8") as fh:
            p = subprocess.Popen(cmd, cwd=REPO, env=ENV, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=left)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 15)
                p.wait()
        secs = time.time() - t
        self.st["stages"][name] = {"exit": rc, "seconds": round(secs, 1), "generations": gens if rc == 0 else 0,
                                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if rc == 0:
            self.st["generations"] += gens
        self.st["wall_seconds"] = round(self.elapsed(), 1)
        save(self.st)
        print(f"== {time.strftime('%H:%M:%S')} {name}: exit {rc} after {secs:.0f} s; {self.st['generations']} generations so far", flush=True)
        self.export()
        if rc == 124:
            self.stop(name, "the 12-hour cap, during this stage")
            return False
        if rc != 0:
            self.stop(name, f"exit {rc}; see logs/v030/{name}.log")
            return False
        return True

    def export(self) -> None:
        """Save the per-question records as we go. A failure here must not stop the generations."""
        if (REPO / "scripts/czech_records.py").exists():
            with open(LOGS / "records.log", "a", encoding="utf-8") as fh:
                subprocess.run([PY, "scripts/czech_records.py", "--partial"], cwd=REPO, env=ENV, stdout=fh, stderr=subprocess.STDOUT)

    def planned(self, slice_: str, cache: str, tag: str, retriever: str, gen: str, levels: str) -> int:
        out = subprocess.run([PY, "scripts/dial_generate.py", "--dataset", tag, "--slice", slice_, "--cache", cache,
                              "--retriever", retriever, "--k", "3", "--levels", levels, "--generator", gen, "--dry-run"],
                             cwd=REPO, env=ENV, capture_output=True, text=True)
        m = re.search(r"(\d+) to generate", out.stdout)
        if not m:
            raise SystemExit(f"dry run failed: {out.stdout} {out.stderr[-500:]}")
        return int(m.group(1))

    def dial(self, name: str, slice_: str, cache: str, tag: str, retriever: str, gen: str, levels: str = ALL_LEVELS) -> bool:
        if self.done(name):
            print(f"== {name}: done before, skipped", flush=True)
            return True
        n = self.planned(slice_, cache, tag, retriever, gen, levels)
        return self.run(name, [PY, "scripts/dial_generate.py", "--dataset", tag, "--slice", slice_, "--cache", cache, "--retriever",
                               retriever, "--k", "3", "--levels", levels, "--generator", gen], gens=n)

    def no_copied_answers(self, name: str, cache: str, gen: str) -> bool:
        """A dial must not start from answers another process made: they would change its order."""
        path = REPO / cache / f"gen_{gen}.jsonl"
        copied = sum('"copied_from"' in line for line in path.open(encoding="utf-8")) if path.exists() else 0
        if copied:
            self.stop(name, f"{copied} copied answers in {cache} before the dial")
        return not copied

    def whole_set(self, key: str, config: str, bench_dir: str, slice_: str, cache: str, tag: str) -> bool:
        """The bench; the dial from an empty cache; do they agree; then the bench's other answers."""
        common = ["--from", bench_dir, "--slice", slice_, "--cache", cache]
        backfill = [PY, "scripts/sqad32_backfill.py", *common]
        natural_done = self.done(f"{key}_copy_natural")
        return (self.run(f"{key}_bench", [".venv/bin/measurerag", "run", "--config", config], gens=600)
                and (natural_done or self.no_copied_answers(f"{key}_dial_1.5b", cache, B))
                and self.dial(f"{key}_dial_1.5b", slice_, cache, tag, "e5-small", B)
                and self.run(f"{key}_agree", [*backfill, "--agree", "--report", f"results/czech/agree_{key}.json"])
                and self.dial(f"{key}_dial_3b", slice_, cache, tag, "e5-small", G)
                and self.run(f"{key}_copy_natural", [*backfill, "--retrievers", "bm25,retromae-cs", "--report", f"results/czech/backfill_{key}.json"]))

    def question_set(self, key: str, slice_: str, cache: str, tag: str) -> bool:
        """A set over a corpus already measured: the dial and the two other retrievers at p = 0."""
        return (self.dial(f"{key}_dial_1.5b", slice_, cache, tag, "e5-small", B)
                and self.dial(f"{key}_dial_3b", slice_, cache, tag, "e5-small", G)
                and self.dial(f"{key}_bm25_1.5b", slice_, cache, tag, "bm25", B, "natural")
                and self.dial(f"{key}_retromae_1.5b", slice_, cache, tag, "retromae-cs", B, "natural"))


def main() -> int:
    from measurerag.datasets.base import Dataset
    from measurerag.rag.generator import build_prompt

    if "ano or ne" not in build_prompt("q", ["p"], language="cs"):
        raise SystemExit("the Czech prompt is not the fixed one")
    for s in ("results/cz_rag/slice", f"{S}/czech32_full", f"{S}/czech32_archive_rule", f"{S}/czech30_archive_rule"):
        if Dataset.load_slice(REPO / s).language != "cs":
            raise SystemExit(f"{s} is not a Czech slice")

    q = Queue()
    print("queue: start", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    steps = (
        lambda: q.whole_set("released", "configs/cz_rag.yaml", "results/cz_rag", "results/cz_rag/slice", COPY, "czech"),
        lambda: q.whole_set("sqad32_full", "configs/cz_rag_32.yaml", "results/cz_rag_32", f"{S}/czech32_full", C32, "czech32"),
        lambda: q.question_set("sqad32_debiased", f"{S}/czech32_archive_rule", C32, "czech32"),
        lambda: q.question_set("copy_debiased", f"{S}/czech30_archive_rule", COPY, "czech"),
        lambda: q.question_set("extra010_sqad32", f"{S}/czech32_debiased", C32, "czech32"),
        lambda: q.question_set("extra010_copy", f"{S}/czech30_debiased", COPY, "czech"),
    )
    for step in steps:
        if not step():
            print("queue: stopped", q.st["generations"], "generations", round(q.elapsed() / 3600, 2), "h", flush=True)
            return 1
    q.st["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    q.st["wall_seconds"] = round(q.elapsed(), 1)
    save(q.st)
    print("queue: all stages done", q.st["generations"], "generations", round(q.st["wall_seconds"] / 3600, 2), "h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
