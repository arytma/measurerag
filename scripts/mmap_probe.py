"""Peak RSS of a generator loaded with its model file memory-mapped and not, as the bench loads it.

    .venv/bin/python scripts/mmap_probe.py --out results/crossover/mmap_probe.jsonl   # the record that ships
    .venv/bin/python scripts/mmap_probe.py --one ministral-3-3b 1                     # one load (a child)

llama-cpp-python maps a GGUF by default (`use_mmap=True`), and every published run loaded its generator so. On the
CPU, llama.cpp also copies most weights into its own buffers, so the file's pages stay resident beside that copy
and peak RSS counts that part of the weights twice. This probe measures by how much.

Each load runs in its own process, as the bench runs each configuration (src/measurerag/worker.py): the bench's
LlamaGenerator, on the CPU, with the Ministral 3 sweeps' settings (n_ctx 4096, max_tokens 64, the seed), inside the
bench's PeakRSS, answering one prompt built by the bench's build_prompt from five passages of the bench's maximum
length, cut from the repository's LICENSE. Repeat i loads each model twice, back to back: mapping on then off for
odd i, off then on for even i. Nothing else is loaded, so the probe measures the generator alone.

The record is JSON lines: the machine and the library versions; a snapshot of the machine's load before the first load
and after the last (the share of a core all other processes used over a short window, the busiest one's share, the free
memory); then one line per load. No path, host, user, process or application name is written. scripts/crossover.py takes
the surplus per model from it (the median over the repeats of on minus off; docs/LIMITATIONS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

MODELS = ("ministral-3-3b", "ministral-3-8b")
REPEATS = 5
SEED = 20260101
MAX_TOKENS = 64          # configs/crossover/*_ministral.yaml
N_CTX = 4096
CPU_WINDOW_S = 30
QUESTION = "Under what conditions may the work be redistributed?"



def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def prompt() -> str:
    from measurerag.rag.generator import build_prompt

    text = " ".join((REPO / "LICENSE").read_text(encoding="utf-8").split())
    passages = [text[i * 1200:(i + 1) * 1200] for i in range(5)]
    return build_prompt(QUESTION, passages)


# --------------------------------------------------------------------------
# one load, in its own process
# --------------------------------------------------------------------------
def mapped_file_row(pid: int) -> dict | None:
    """vmmap's summary row for mapped files: their virtual and resident sizes, as vmmap prints them."""
    out = subprocess.run(["vmmap", "--summary", str(pid)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("mapped file"):
            parts = line.split()
            return {"virtual": parts[2], "resident": parts[3]}
    return None


def one(key: str, mmap: bool) -> dict:
    from measurerag.memory import PeakRSS, current_rss_gb
    from measurerag.rag.generator import GENERATORS, LlamaGenerator

    spec = GENERATORS[key]
    p = prompt()
    start = utc()
    before = current_rss_gb()
    with PeakRSS() as peak:
        t0 = time.perf_counter()
        gen = LlamaGenerator(key, device="cpu", seed=SEED, max_tokens=MAX_TOKENS, n_ctx=N_CTX, use_mmap=mmap)
        load_s = time.perf_counter() - t0
        after_load = current_rss_gb()
        text, info = gen.generate(p)
        after_answer = current_rss_gb()
        mapped = mapped_file_row(os.getpid())
    file_bytes = gen.model_bytes
    gen.close()
    return {"record": "load", "model": key, "use_mmap": mmap, "utc_start": start, "utc_end": utc(),
            "model_file": f"hf://{spec.repo_id}@{spec.revision}/{spec.filename}", "file_bytes": file_bytes,
            "peak_rss_gb": round(peak.peak_gb, 4), "rss_before_gb": round(before, 4),
            "rss_after_load_gb": round(after_load, 4), "rss_after_answer_gb": round(after_answer, 4),
            "load_s": round(load_s, 2), "answer_s": round(info["seconds"], 2), "prompt_tokens": info["prompt_tokens"],
            "completion_tokens": info["completion_tokens"], "answer": text, "vmmap_mapped_file": mapped}


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------
def snapshot(label: str) -> dict:
    import psutil
    from sweep_queue import ps_table

    me = os.getpid()
    first = {pid: secs for pid, _, _, secs, _ in ps_table()}
    time.sleep(CPU_WINDOW_S)
    rows = ps_table()
    use: dict[int, float] = {}
    for pid, _, _, secs, comm in rows:
        if pid == me or comm == "ps":
            continue
        share = 100 * (secs - first.get(pid, secs)) / CPU_WINDOW_S
        if share > 0:
            use[pid] = share
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    # shares only: the names of other processes describe the machine's owner, not the measurement
    return {"record": "snapshot", "when": label, "utc": utc(), "cpu_window_s": CPU_WINDOW_S,
            "cpu_all_others_pct_of_a_core": round(sum(use.values()), 1),
            "memory_gb": {"total": round(vm.total / 1024**3, 2), "available": round(vm.available / 1024**3, 2),
                          "swap_used": round(sw.used / 1024**3, 2)},
            "cpu_busiest_other_pct_of_a_core": round(max(use.values(), default=0.0), 1)}


def machine() -> dict:
    import llama_cpp
    import psutil

    from measurerag.hardware import describe_machine

    return {"record": "machine", "utc": utc(), "machine": describe_machine().as_dict(),
            "macos": platform.mac_ver()[0], "llama_cpp_python": llama_cpp.__version__, "psutil": psutil.__version__,
            "python": platform.python_version(), "models": list(MODELS), "repeats": REPEATS,
            "order": "repeat i: mapping on then off for odd i, off then on for even i; each load in its own process",
            "settings": {"device": "cpu", "n_ctx": N_CTX, "max_tokens": MAX_TOKENS, "seed": SEED,
                         "prompt": "build_prompt(QUESTION, five 1,200-character passages of LICENSE)"}}


def run(out: Path) -> int:
    import psutil

    lines = [machine(), snapshot("before the first load")]
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    print(f"mmap_probe: {lines[1]['utc']} snapshot: other processes {lines[1]['cpu_all_others_pct_of_a_core']}% of a core, "
          f"{lines[1]['memory_gb']['available']} GB available", flush=True)
    for key in MODELS:
        for i in range(1, REPEATS + 1):
            for mmap in ((True, False) if i % 2 else (False, True)):
                free = round(psutil.virtual_memory().available / 1024**3, 2)
                p = subprocess.run([sys.executable, __file__, "--one", key, "1" if mmap else "0"],
                                   capture_output=True, text=True, cwd=REPO)
                if p.returncode != 0:
                    print(p.stdout + p.stderr)
                    return 1
                row = {**json.loads(p.stdout.strip().splitlines()[-1]), "repeat": i, "available_before_gb": free}
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"mmap_probe: {row['utc_end']} {key} repeat {i} use_mmap={mmap}: peak {row['peak_rss_gb']} GB", flush=True)
    end = snapshot("after the last load")
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(end, ensure_ascii=False) + "\n")
    print(f"mmap_probe: {end['utc']} done: {summary(out)}")
    return 0


def surplus(path: Path) -> dict:
    """The surplus per model: the median over the repeats of (peak RSS mapped) minus (peak RSS not mapped)."""
    loads = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    loads = [x for x in loads if x.get("record") == "load"]
    out = {}
    for key in sorted({x["model"] for x in loads}):
        by = {(x["repeat"], x["use_mmap"]): x for x in loads if x["model"] == key}
        reps = sorted({r for r, _ in by})
        diffs = [by[(r, True)]["peak_rss_gb"] - by[(r, False)]["peak_rss_gb"] for r in reps]
        out[key] = {"surplus_gb": statistics.median(diffs), "differences_gb": diffs, "repeats": len(reps),
                    "peak_on_gb": [by[(r, True)]["peak_rss_gb"] for r in reps],
                    "peak_off_gb": [by[(r, False)]["peak_rss_gb"] for r in reps],
                    "answers_identical": len({x["answer"] for x in loads if x["model"] == key}) == 1}
    return out


def summary(path: Path) -> str:
    return "; ".join(f"{k}: surplus {v['surplus_gb']:.3f} GB (differences {min(v['differences_gb']):.3f} to "
                     f"{max(v['differences_gb']):.3f}), answers identical: {v['answers_identical']}"
                     for k, v in surplus(path).items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/crossover/mmap_probe.jsonl")
    ap.add_argument("--one", nargs=2, metavar=("MODEL", "MMAP"))
    ap.add_argument("--summary", action="store_true", help="print the surplus per model from --out and stop")
    args = ap.parse_args()
    # the bench's queue's environment (scripts/sweep_queue.py): the pinned files from the repository's cache, offline
    for k, v in (("HF_HUB_CACHE", str(REPO / "data" / "hf-hub")), ("HF_HUB_OFFLINE", "1"), ("HF_HUB_DISABLE_TELEMETRY", "1")):
        os.environ.setdefault(k, v)
    if args.one:
        print(json.dumps(one(args.one[0], args.one[1] == "1"), ensure_ascii=False))
        return 0
    out = REPO / args.out
    if args.summary:
        print(summary(out))
        return 0
    return run(out)


if __name__ == "__main__":
    sys.exit(main())
