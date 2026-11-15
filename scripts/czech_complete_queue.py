"""The complete Czech run: every generation of the Czech study, one heavy process at a time.

    caffeinate -dimsu .venv/bin/python scripts/czech_complete_queue.py --phase dev
    caffeinate -dimsu .venv/bin/python scripts/czech_complete_queue.py --phase main
    .venv/bin/python scripts/czech_complete_queue.py --machine gx10 --phase pilot   # on the GX10
    .venv/bin/python scripts/czech_complete_queue.py --machine gx10 --phase main

No clock and no generation cap: completeness is the stage list. In their place, these guards:

    a checkpoint     every configuration is a stage, and a finished stage is never run again. Answers are
                     cached one by one, and a resumed stage keeps its order (scripts/dial_generate.py);
    free disk        below 50 GB the queue stops the stage and reports;
    no progress      45 minutes with no new answer or output: the stage restarts once, and then the queue reports;
    PROGRESS.md      in the report folder, rewritten every 5 minutes;
    the stop switch  a file named STOP in the report folder ends the queue after the current configuration.

Exit 0 when every stage is done, 1 when stopped by a guard or a failure, 3 when stopped by the switch. Run it
again to resume where it stopped. Writes results/czech/complete/compute.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PY = ".venv/bin/python"
B, G = "qwen2.5-1.5b", "qwen2.5-3b"
E2, E4 = "gemma-4-e2b", "gemma-4-e4b"
# The Czech runs' two families: the release's Qwen2.5, and Gemma 4 beside it.
PAIRS = ((B, G), (E2, E4))
# v0.3.1's third family, run alone in its own phase on the GX10
M3, M8 = "ministral-3-3b", "ministral-3-8b"
MINISTRAL_PAIRS = ((M3, M8),)
# v0.4.0's fourth family, Qwen's current generation, run alone in its own phase on the GX10
Q2, Q4 = "qwen3.5-2b", "qwen3.5-4b"
QWEN35_PAIRS = ((Q2, Q4),)
N2, N4 = "gemma-3n-e2b", "gemma-3n-e4b"          # v0.4.2: Gemma 4's previous generation
GEMMA3N_PAIRS = ((N2, N4),)
SHORT = {B: "1.5b", G: "3b", E2: E2, E4: E4, M3: M3, M8: M8, Q2: Q2, Q4: Q4, N2: N2, N4: N4}       # in stage names; Qwen2.5's keep the names they had
KIND = {B: "1.5b", G: "3b", E2: "e2b", E4: "e4b", M3: "m3b", M8: "m8b", Q2: "q2b", Q4: "q4b", N2: "n2b", N4: "n4b"}  # what a stage's rate is measured under
K, MT = 3, 64
C = "results/crossover/cache"
COMPLETE = f"{C}/complete"
SL = "results/crossover/slices"
RANK_COPY, RANK_32 = f"{C}/v030_copy", f"{C}/v030_32"
LEVELS = "0,0.2,0.4,0.6,0.8,1.0"
# A bench's finish stage only rebuilds its files from the configurations its bench stages measured: under this the
# runner reuses each result measured under an identical spec and refuses to measure (src/measurerag/runner.py), so a
# spec that changed since stops the stage instead of measuring that configuration again
REUSE_ONLY = {"MEASURERAG_REUSE_ONLY": "1"}
STATE = "results/czech/complete/compute.json"
LOGS = "logs/complete"
ENV = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
       "HF_HUB_CACHE": str(REPO / "data/hf-hub"), "TOKENIZERS_PARALLELISM": "false"}
MIN_FREE_GB = 50
NO_PROGRESS_S = 45 * 60
RESTARTS = 1
POLL_S = 15
PROGRESS_EVERY_S = 300
TOKEN_RATE_KINDS = ("bench_1.5b", "dial_1.5b", "dial_3b", "natural_1.5b", "bench_e2b", "dial_e2b", "dial_e4b", "natural_e2b", "bench_m3b", "dial_m3b", "dial_m8b", "natural_m3b",
                    "bench_q2b", "dial_q2b", "dial_q4b", "natural_q2b", "bench_n2b", "dial_n2b", "dial_n4b", "natural_n2b")
# Gemma 4's starting rates, seconds per answer, from the dev-set benchmark on the GX10's GPU (results/czech/model_bench/bench.json),
# until a stage of that kind finishes; a bench also embeds, so it starts higher
GEMMA_START_RATES = {"dial_e2b": 0.15, "natural_e2b": 0.15, "dial_e4b": 0.25, "bench_e2b": 0.5}
# Ministral 3's, from the same benchmark (results/czech/model_bench/mistral.json)
MINISTRAL_START_RATES = {"dial_m3b": 0.25, "natural_m3b": 0.25, "dial_m8b": 0.47, "bench_m3b": 0.7}
# Qwen3.5's, from the same benchmark (results/czech/model_bench/bench.json)
QWEN35_START_RATES = {"dial_q2b": 0.13, "natural_q2b": 0.13, "dial_q4b": 0.27, "bench_q2b": 0.5}
# Gemma 3n's, as Gemma 4's until a stage of that kind finishes (no GX10 benchmark ran it)
GEMMA3N_START_RATES = {"dial_n2b": 0.15, "natural_n2b": 0.15, "dial_n4b": 0.25, "bench_n2b": 0.5}
# Where a run's state and logs go, and where its generators run. The GX10 keeps its own record, so that no stage the Mac
# finished counts as finished there.
MACHINES = {"mac": {"state": STATE, "logs": LOGS, "device": "cpu"},
            "gx10": {"state": "results/czech/complete/compute_gx10.json", "logs": "logs/complete_gx10", "device": "cuda"}}
GEN_DEVICE = "cpu"
PILOT_RANKINGS = {"released": "results/cz_rag/slice", "sqad32_full": f"{SL}/complete_sqad32_full"}
PILOT_RETRIEVERS = ("bm25:bm25", "e5-small:dense:e5-small", "retromae-cs:dense:retromae-cs")

QUESTION_SETS = {
    "sqad32_debiased": {"slice": f"{SL}/complete_sqad32_debiased", "rank": RANK_32, "tag": "czech32"},
    "copy_debiased": {"slice": f"{SL}/complete_copy_debiased", "rank": RANK_COPY, "tag": "czech"},
    "extra010_copy": {"slice": f"{SL}/complete_extra010_copy", "rank": RANK_COPY, "tag": "czech"},
    "extra010_sqad32": {"slice": f"{SL}/complete_extra010_sqad32", "rank": RANK_32, "tag": "czech32"},
}
BENCH_SETS = {
    "released": {"slice": "results/cz_rag/slice", "rank": RANK_COPY, "tag": "czech", "config": "configs/cz_rag.yaml", "bench": "results/cz_rag"},
    "sqad32_full": {"slice": f"{SL}/complete_sqad32_full", "rank": RANK_32, "tag": "czech32", "config": "configs/cz_rag_32.yaml", "bench": "results/cz_rag_32"},
}
CROSS = {"cross_copy": {"slice": f"{SL}/complete_cross_copy", "tag": "czech"}, "cross_32": {"slice": f"{SL}/complete_cross_32", "tag": "czech32"}}


@dataclass
class Stage:
    name: str
    cmd: list[str]
    kind: str = "none"                         # what its rate is measured under
    planned: int = 0                           # generations, from the plan until the stage starts
    watch: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    fatal: bool = True                         # a check that fails is reported, and does not end the run
    set_key: str | None = None                 # whose records to save afterwards
    dry_run: list[str] | None = None
    prepare: str | None = None                 # a bench directory whose first-cut files move aside first
    count_stage: str | None = None             # counts rows of this stage in watch[0]; else reads a heartbeat


def dial_stage(name: str, set_key: str, slice_: str, cache: str, rank: str, tag: str, retriever: str, gen: str, levels: str, kind: str, planned: int, variant: str | None = None) -> Stage:
    cmd = [PY, "scripts/dial_generate.py", "--dataset", tag, "--slice", slice_, "--cache", cache, "--rank-cache", rank, "--retriever", retriever, "--k", str(K),
           "--levels", levels, "--generator", gen, "--max-tokens", str(MT), "--stage", name]
    if variant:
        cmd += ["--variant", variant]
    if GEN_DEVICE != "cpu":
        cmd += ["--device", GEN_DEVICE]
    cache_file = f"{cache}/gen_{gen}.jsonl"
    return Stage(name, cmd, kind, planned, [cache_file], set_key=set_key, dry_run=[*cmd, "--dry-run"], count_stage=name)


def question_set_stages(key: str, spec: dict, plan: dict, pairs: tuple = PAIRS) -> list[Stage]:
    cache = f"{COMPLETE}/{key}"
    by = plan["compute_estimate"]["by_kind"][key]
    n = plan["sets"][key]
    args = (key, spec["slice"], cache, spec["rank"], spec["tag"])
    stages = []
    for small, big in pairs:
        stages += [dial_stage(f"{key}_dial_{SHORT[small]}", *args, "e5-small", small, LEVELS, f"dial_{KIND[small]}", by["dial_1.5b"]),
                   dial_stage(f"{key}_dial_{SHORT[big]}", *args, "e5-small", big, LEVELS, f"dial_{KIND[big]}", by["dial_3b"]),
                   dial_stage(f"{key}_bm25_{SHORT[small]}", *args, "bm25", small, "natural", f"natural_{KIND[small]}", n),
                   dial_stage(f"{key}_retromae_{SHORT[small]}", *args, "retromae-cs", small, "natural", f"natural_{KIND[small]}", n)]
    return stages


def bench_set_stages(key: str, spec: dict, plan: dict, pairs: tuple = PAIRS, first_run: bool = True) -> list[Stage]:
    """first_run: the bench folder's earlier outputs move aside before its first stage. A later family's phase keeps them,
    and its finish reuses every configuration already measured under an unchanged spec."""
    cache = f"{COMPLETE}/{key}"
    by = plan["compute_estimate"]["by_kind"][key]
    n = plan["sets"][key]
    stages = []
    for small, _big in pairs:
        for rid in ("bm25", "e5-small", "retromae-cs"):
            beat = f"{spec['bench']}/work/heartbeat_{rid}+{small}.txt"
            name = f"{key}_bench_{rid}" if small == B else f"{key}_bench_{rid}_{SHORT[small]}"
            stages.append(Stage(name, [".venv/bin/measurerag", "run", "--config", spec["config"], "--only", f"{rid}+{small}"], f"bench_{KIND[small]}", n, [beat],
                                env={"MEASURERAG_HEARTBEAT": beat}, set_key=None, prepare=spec["bench"] if first_run and not stages else None))
    stages.append(Stage(f"{key}_bench_finish" if first_run else f"{key}_bench_finish_{SHORT[pairs[0][0]]}", [".venv/bin/measurerag", "run", "--config", spec["config"]],
                        env=dict(REUSE_ONLY), set_key=key))
    args = (key, spec["slice"], cache, spec["rank"], spec["tag"])
    for small, big in pairs:
        stages.append(dial_stage(f"{key}_dial_{SHORT[small]}", *args, "e5-small", small, LEVELS, f"dial_{KIND[small]}", by["dial_1.5b"]))
        agree = f"{key}_agree" if small == B else f"{key}_agree_{SHORT[small]}"
        stages.append(Stage(agree, [PY, "scripts/sqad32_backfill.py", "--agree", "--from", spec["bench"], "--slice", spec["slice"], "--cache", cache, "--rank-cache", spec["rank"],
                                    "--generator", small, "--report", f"results/czech/complete/agree_{key}{'' if small == B else '_' + SHORT[small]}.json"], fatal=False))
        stages.append(dial_stage(f"{key}_dial_{SHORT[big]}", *args, "e5-small", big, LEVELS, f"dial_{KIND[big]}", by["dial_3b"]))
    return stages


def cross_stages(plan: dict, pairs: tuple = PAIRS, rankings: bool = True) -> list[Stage]:
    n = plan["sets"]["cross_version"]
    stages = []
    for key, spec in CROSS.items() if rankings else ():
        cache = f"{COMPLETE}/{key}"
        stages.append(Stage(f"{key}_ranking", [PY, "scripts/retrieval_cache.py", "--slice", spec["slice"], "--cache", cache, "--retriever", "e5-small:dense:e5-small"], watch=[f"{cache}/work"]))
    if rankings:
        stages.append(Stage("cross_ranking_check", [PY, "scripts/czech_complete_queue.py", "--check-cross-rankings"], fatal=False))
    for gen in (g for pair in pairs for g in pair):
        for key, spec in CROSS.items():
            cache = f"{COMPLETE}/{key}"
            stages.append(dial_stage(f"{key}_{SHORT[gen]}", "cross_version", spec["slice"], cache, cache, spec["tag"], "e5-small", gen, "natural", f"dial_{KIND[gen]}", n))
    return stages


def dev_stages() -> list[Stage]:
    stages = []
    for variant in ("v1", "v2", "v3", "v4"):
        for gen, kind in ((B, "dial_1.5b"), (G, "dial_3b")):
            short = "1.5b" if gen == B else "3b"
            stages.append(dial_stage(f"dev_{variant}_{short}", None, f"{SL}/complete_dev", f"{COMPLETE}/dev_{variant}", RANK_32, "czech32", "e5-small", gen, "natural", kind, 200, variant=variant))
    return stages


def pilot_stages() -> list[Stage]:
    """The GX10's pilot: the dev set's eight configurations, V1 again from a new empty cache, and the benches' rankings on
    its CPU. Then the check of the move between machines, which generates nothing."""
    stages = dev_stages()
    for gen, kind in ((B, "dial_1.5b"), (G, "dial_3b")):
        short = "1.5b" if gen == B else "3b"
        stages.append(dial_stage(f"dev_v1_{short}_repeat", None, f"{SL}/complete_dev", f"{COMPLETE}/dev_v1_repeat", RANK_32, "czech32", "e5-small", gen, "natural", kind, 200, variant="v1"))
    for key, slice_ in PILOT_RANKINGS.items():
        for r in PILOT_RETRIEVERS:
            cache = f"{COMPLETE}/pilot_rank_{key}"
            stages.append(Stage(f"pilot_rank_{key}_{r.split(':')[0]}", [PY, "scripts/retrieval_cache.py", "--slice", slice_, "--cache", cache, "--retriever", r], watch=[f"{cache}/work"]))
    stages.append(Stage("pilot_check", [PY, "scripts/czech_pilot_check.py"], fatal=False))
    return stages


def main_stages(plan: dict) -> list[Stage]:
    return [*bench_set_stages("released", BENCH_SETS["released"], plan),
            *question_set_stages("sqad32_debiased", QUESTION_SETS["sqad32_debiased"], plan),
            *question_set_stages("copy_debiased", QUESTION_SETS["copy_debiased"], plan),
            *bench_set_stages("sqad32_full", BENCH_SETS["sqad32_full"], plan),
            *cross_stages(plan),
            *question_set_stages("extra010_copy", QUESTION_SETS["extra010_copy"], plan),
            *question_set_stages("extra010_sqad32", QUESTION_SETS["extra010_sqad32"], plan)]


def ministral_stages(plan: dict, p: tuple = MINISTRAL_PAIRS) -> list[Stage]:
    """v0.3.1: Ministral 3 alone, every stage Gemma 4 ran, in the same order. The cross-version
    rankings exist already, and the benches keep their outputs. v0.4.0's Qwen3.5 runs the same stages."""
    return [*bench_set_stages("released", BENCH_SETS["released"], plan, p, first_run=False),
            *question_set_stages("sqad32_debiased", QUESTION_SETS["sqad32_debiased"], plan, p),
            *question_set_stages("copy_debiased", QUESTION_SETS["copy_debiased"], plan, p),
            *bench_set_stages("sqad32_full", BENCH_SETS["sqad32_full"], plan, p, first_run=False),
            *cross_stages(plan, p, rankings=False),
            *question_set_stages("extra010_copy", QUESTION_SETS["extra010_copy"], plan, p),
            *question_set_stages("extra010_sqad32", QUESTION_SETS["extra010_sqad32"], plan, p)]


def gemma3n_stages(plan: dict) -> list[Stage]:
    """v0.4.2: Gemma 3n alone, every stage Ministral 3 ran."""
    return ministral_stages(plan, GEMMA3N_PAIRS)


def qwen35_stages(plan: dict) -> list[Stage]:
    """v0.4.0: Qwen3.5 alone, every stage Ministral 3 ran."""
    return ministral_stages(plan, QWEN35_PAIRS)


def human(hours: float) -> str:
    return f"{hours:.1f} h ({hours / 24:.1f} days)"


class Queue:
    def __init__(self, stages: list[Stage], report_dir: Path, plan_rates: dict, phase: str, root: Path = REPO, estimate: str = "", env: dict | None = None,
                 min_free_gb: float = MIN_FREE_GB, no_progress_s: float = NO_PROGRESS_S, poll_s: float = POLL_S, export: bool = True,
                 state: str = STATE, logs: str = LOGS, resume_flags: str = "") -> None:
        self.stages, self.report_dir, self.plan_rates, self.phase, self.root, self.estimate = stages, report_dir, plan_rates, phase, root, estimate
        self.env = {**os.environ, **(ENV if env is None else env)}
        self.min_free_gb, self.no_progress_s, self.poll_s, self.export = min_free_gb, no_progress_s, poll_s, export
        self.state_path = root / state
        self.logs = root / logs
        self.logs_rel = logs
        self.resume_flags = resume_flags
        self.logs.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.st = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {"stages": {}, "warnings": [], "restarts": []}
        self.st.pop("stopped", None)
        self.current: Stage | None = None
        self.current_started = 0.0
        self.last_progress_write = 0.0

    # ---- state ------------------------------------------------------------------------------------
    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.st, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.state_path)

    def done(self, name: str) -> bool:
        s = self.st["stages"].get(name) or {}
        return s.get("exit") == 0 or (s.get("exit") is not None and s.get("non_fatal") is True)

    def warn(self, text: str) -> None:
        self.st.setdefault("warnings", []).append(f"{time.strftime('%Y-%m-%d %H:%M')}: {text}")
        self.save()

    def free_gb(self) -> float:
        return shutil.disk_usage(self.root).free / 1e9

    def stop_file(self) -> Path:
        return self.report_dir / "STOP"

    def stopped(self, why: str, code: int) -> int:
        self.st["stopped"] = {"why": why, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code}
        self.save()
        self.write_progress(force=True)
        print(f"== STOPPED: {why}", flush=True)
        return code

    # ---- counting -----------------------------------------------------------------------------------
    def count(self, stage: Stage) -> int:
        """Generations this stage has made, from what it wrote."""
        if not stage.watch:
            return 0
        path = self.root / stage.watch[0]
        if not path.exists() or path.is_dir():
            return 0
        if stage.count_stage:
            tag = f'"stage": "{stage.count_stage}"'
            with open(path, encoding="utf-8") as fh:
                return sum(tag in line for line in fh)
        text = path.read_text(encoding="utf-8").strip()
        return int(text) if text.isdigit() else 0

    def signature(self, stage: Stage, log: Path) -> tuple:
        sig = []
        for p in [*(self.root / w for w in stage.watch), log]:
            try:
                s = p.stat()
                sig.append((str(p), s.st_size, s.st_mtime_ns))
            except OSError:
                sig.append((str(p), None, None))
        return tuple(sig)

    # ---- one attempt, under the guards ---------------------------------------------------------------
    def attempt(self, stage: Stage, log: Path) -> tuple[str, int]:
        """('exit', code), ('stalled', 0) or ('disk', 0). The monotonic clock does not tick while the Mac sleeps."""
        with open(log, "a", encoding="utf-8") as fh:
            p = subprocess.Popen(stage.cmd, cwd=self.root, env={**self.env, **stage.env}, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            last_sig, last_change = self.signature(stage, log), time.monotonic()
            while p.poll() is None:
                time.sleep(self.poll_s)
                sig = self.signature(stage, log)
                if sig != last_sig:
                    last_sig, last_change = sig, time.monotonic()
                self.write_progress()
                if self.free_gb() < self.min_free_gb:
                    self.kill(p)
                    return "disk", 0
                if time.monotonic() - last_change > self.no_progress_s:
                    self.kill(p)
                    return "stalled", 0
            return "exit", p.returncode

    @staticmethod
    def kill(p: subprocess.Popen) -> None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()

    def move_first_cut_aside(self, bench: str) -> None:
        """The first cut's bench outputs leave before the bench runs. Its answers are in the records."""
        src, dst = self.root / bench, self.root / ".local/data/v030_first_cut" / Path(bench).name
        if dst.exists():
            return
        dst.mkdir(parents=True)
        for item in list(src.iterdir()) if src.exists() else []:
            if item.name == "slice":
                continue
            if item.name.startswith("predictions_") or item.name in ("specs", "work"):
                shutil.move(str(item), str(dst / item.name))
            elif item.is_file():
                shutil.copy2(item, dst / item.name)

    def planned_for(self, stage: Stage) -> int:
        if not stage.dry_run:
            return stage.planned
        out = subprocess.run(stage.dry_run, cwd=self.root, env={**self.env, **stage.env}, capture_output=True, text=True)
        m = re.search(r"(\d+) to generate", out.stdout)
        return (int(m.group(1)) + self.count(stage)) if m else stage.planned

    def run_stage(self, stage: Stage) -> int | None:
        """None to go on, or the queue's exit code."""
        if self.done(stage.name):
            return None
        if self.stop_file().exists():
            return self.stopped("the stop switch", 3)
        if self.free_gb() < self.min_free_gb:
            return self.stopped(f"free disk is {self.free_gb():.0f} GB, below {self.min_free_gb} GB, before {stage.name}", 1)
        if stage.prepare:
            self.move_first_cut_aside(stage.prepare)
        for w in stage.watch:
            (self.root / w).parent.mkdir(parents=True, exist_ok=True)
        stage.planned = self.planned_for(stage)
        self.current, self.current_started = stage, time.monotonic()
        log = self.logs / f"{stage.name}.log"
        print(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} {stage.name}: {stage.planned} generations planned", flush=True)
        self.write_progress(force=True)
        previous = self.st["stages"].get(stage.name, {})
        seconds_before = float(previous.get("seconds", 0.0))
        restarts = int(previous.get("restarts", 0))
        while True:
            t = time.monotonic()
            outcome, code = self.attempt(stage, log)
            seconds_before += time.monotonic() - t
            if outcome == "disk":
                self.record(stage, None, seconds_before, restarts)
                return self.stopped(f"free disk fell below {self.min_free_gb} GB during {stage.name}", 1)
            if outcome == "stalled":
                restarts += 1
                self.record(stage, None, seconds_before, restarts)
                if restarts > RESTARTS:
                    return self.stopped(f"{stage.name} made no progress for {self.no_progress_s / 60:.0f} minutes, after {RESTARTS} restart", 1)
                self.st.setdefault("restarts", []).append({"stage": stage.name, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "why": "no progress"})
                self.warn(f"{stage.name} made no progress for {self.no_progress_s / 60:.0f} minutes and was restarted once")
                continue
            break
        self.record(stage, code, seconds_before, restarts)
        print(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} {stage.name}: exit {code} after {seconds_before:.0f} s, {self.count(stage)} generations", flush=True)
        if code != 0 and stage.fatal:
            return self.stopped(f"{stage.name} exited {code}; see {self.logs_rel}/{stage.name}.log", 1)
        if code != 0:
            self.warn(f"{stage.name} exited {code}: the check did not pass; see {self.logs_rel}/{stage.name}.log")
        self.current = None
        self.save_records(stage)
        self.write_progress(force=True)
        return None

    def record(self, stage: Stage, code: int | None, seconds: float, restarts: int) -> None:
        self.st["stages"][stage.name] = {"exit": code, "seconds": round(seconds, 1), "generations": self.count(stage), "kind": stage.kind, "planned": stage.planned, "restarts": restarts,
                                         "non_fatal": not stage.fatal, "finished": time.strftime("%Y-%m-%d %H:%M:%S") if code is not None else None}
        self.save()

    def save_records(self, stage: Stage) -> None:
        """The per-question records as we go. A failure here is reported and stops nothing."""
        if not (self.export and stage.set_key):
            return
        with open(self.logs / "records.log", "a", encoding="utf-8") as fh:
            r = subprocess.run([PY, "scripts/czech_records.py", "--partial", "--sets", stage.set_key], cwd=self.root, env=self.env, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            self.warn(f"saving the records of {stage.set_key} exited {r.returncode}; see {self.logs_rel}/records.log")

    # ---- PROGRESS.md -----------------------------------------------------------------------------------
    def rates(self) -> dict[str, float]:
        """Seconds per generation by kind: measured in this run where there is a finished stage, else the plan's."""
        got = dict(self.plan_rates)
        seconds: dict[str, float] = {}
        gens: dict[str, int] = {}
        mine = {stage.name for stage in self.stages}     # the dev set's other prompts have other rates
        for name, s in self.st["stages"].items():
            if name in mine and s.get("generations") and s.get("kind") in TOKEN_RATE_KINDS:
                seconds[s["kind"]] = seconds.get(s["kind"], 0.0) + s["seconds"]
                gens[s["kind"]] = gens.get(s["kind"], 0) + s["generations"]
        for k in seconds:
            got[k] = seconds[k] / gens[k]
        return got

    def write_progress(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_progress_write < PROGRESS_EVERY_S:
            return
        self.last_progress_write = now
        rates = self.rates()
        done_gens = left_gens = 0
        left_s = 0.0
        done_names, left_names = [], []
        for stage in self.stages:
            rec = self.st["stages"].get(stage.name) or {}
            if self.done(stage.name):
                done_names.append(stage.name)
                done_gens += int(rec.get("generations", 0))
                continue
            left_names.append(stage.name)
            made = self.count(stage) if stage is self.current or rec else 0
            done_gens += made
            todo = max(stage.planned - made, 0)
            left_gens += todo
            left_s += todo * rates.get(stage.kind, 0.0)
        finish = time.strftime("%a %d %b %Y, %H:%M", time.localtime(time.time() + left_s))
        lines = [
            "# measureRAG v0.3.0, complete run: progress", "",
            self.estimate, "",
            f"Updated: {time.strftime('%a %d %b %Y, %H:%M:%S')}",
            f"Phase: {self.phase}",
            f"Running now: {self.current.name if self.current else 'nothing'}",
            f"Configurations: {len(done_names)} done, {len(left_names)} left, of {len(self.stages)}",
            f"Generations in this phase: {done_gens:,} done, {left_gens:,} left",
            f"Projected finish of this phase: {finish} ({human(left_s / 3600)} from now, at the measured rates below)",
            f"Free disk: {self.free_gb():.0f} GB (the queue stops below {self.min_free_gb:.0f} GB)", "",
            "Measured rate, seconds per generation (the plan's, until a stage of that kind finishes):", "",
            *[f"- {k}: {rates[k]:.2f}" for k in TOKEN_RATE_KINDS if k in rates], "",
            f"Stop switch: create the file `{self.stop_file().name}` beside this file; the queue stops after the current configuration.",
            f"To resume after a crash or a reboot, from the repository: `caffeinate -dimsu {PY} scripts/czech_complete_queue.py{self.resume_flags} --phase {self.phase}`", "",
        ]
        if self.st.get("stopped"):
            lines += [f"**STOPPED** at {self.st['stopped']['when']}: {self.st['stopped']['why']}", ""]
        if self.st.get("warnings"):
            lines += ["Warnings:", "", *[f"- {w}" for w in self.st["warnings"]], ""]
        lines += ["Configurations left:", "", *[f"- {n}" for n in left_names], "", "Configurations done:", "", *[f"- {n}: {self.st['stages'][n].get('generations', 0)} generations, {self.st['stages'][n].get('seconds', 0):.0f} s" for n in done_names], ""]
        path = self.report_dir / "PROGRESS.md"
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        tmp.replace(path)

    def run(self) -> int:
        self.write_progress(force=True)
        for stage in self.stages:
            code = self.run_stage(stage)
            if code is not None:
                return code
        self.st.setdefault("finished_phases", {})[self.phase] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.current = None
        self.save()
        self.write_progress(force=True)
        print(f"== {self.phase}: every stage is done", flush=True)
        return 0


def check_cross_rankings() -> int:
    """A ranking made here must be the first cut's, on every question both have. No generation."""
    from measurerag.crossover.retrieval import load_run

    report = {}
    failed = False
    for key, old in (("cross_copy", RANK_COPY), ("cross_32", RANK_32)):
        new_run = load_run(REPO / f"{COMPLETE}/{key}", "e5-small")["run"]
        old_run = load_run(REPO / old, "e5-small")["run"]
        both = sorted(set(new_run) & set(old_run))
        differ = [q for q in both if new_run[q][:K] != old_run[q][:K]]
        report[key] = {"questions": len(new_run), "also_ranked_in_the_first_cut": len(both), "top_3_differs": len(differ)}
        failed = failed or bool(differ)
    out = REPO / "results/czech/complete/cross_ranking_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(report, flush=True)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "main", "pilot", "ministral", "qwen35", "gemma3n"))
    ap.add_argument("--machine", choices=tuple(MACHINES), default="mac")
    ap.add_argument("--report-dir", default=str(REPO / "logs" / "czech_progress"))   # inside the git-ignored logs/
    ap.add_argument("--check-cross-rankings", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    args = ap.parse_args()
    os.chdir(REPO)
    if args.check_cross_rankings:
        return check_cross_rankings()
    global GEN_DEVICE
    machine = MACHINES[args.machine]
    GEN_DEVICE = machine["device"]
    if args.phase in ("pilot", "ministral", "qwen35", "gemma3n") and args.machine != "gx10":
        raise SystemExit(f"the {args.phase} phase runs on the GX10")
    if args.phase == "ministral":
        # its own record, so that v0.3.0's stays as it was
        machine = {**machine, "state": "results/czech/complete/compute_gx10_ministral.json", "logs": "logs/complete_gx10_ministral"}
    if args.phase == "qwen35":
        # its own record too
        machine = {**machine, "state": "results/czech/complete/compute_gx10_qwen35.json", "logs": "logs/complete_gx10_qwen35"}
    if args.phase == "gemma3n":
        # its own record too
        machine = {**machine, "state": "results/czech/complete/compute_gx10_gemma3n.json", "logs": "logs/complete_gx10_gemma3n"}
    plan = json.loads((REPO / "results/czech/plan/plan.json").read_text(encoding="utf-8"))
    ce = plan["compute_estimate"]
    if args.phase in ("main", "ministral", "qwen35", "gemma3n"):
        from measurerag.rag.generator import CZECH_PROMPT

        chosen = json.loads((REPO / "results/czech/dev/selection.json").read_text(encoding="utf-8"))["selection"]["chosen"]
        if chosen != CZECH_PROMPT:
            raise SystemExit(f"the dev set chose {chosen}, and the code has {CZECH_PROMPT} frozen")
    stages = {"dev": dev_stages, "pilot": pilot_stages, "ministral": lambda: ministral_stages(plan), "qwen35": lambda: qwen35_stages(plan), "gemma3n": lambda: gemma3n_stages(plan)}.get(args.phase, lambda: main_stages(plan))()
    if args.list:
        for s in stages:
            print(s.name, s.kind, s.planned)
        return 0
    estimate = (f"Estimate at the start, from Phase 0: {ce['total_generations']:,} generations and {human(ce['total_hours'])}, one process at a time, "
                f"at the first v0.3.0 cut's rates ({Path('results/czech/plan/plan.json')}).")
    if args.phase == "main":
        estimate += f" This phase runs under the frozen prompt {CZECH_PROMPT}, from empty caches."
    if args.machine != "mac":
        estimate += f" Machine: {args.machine}, generators on {GEN_DEVICE}. Rates start from the Mac's until a stage here finishes."
    q = Queue(stages, Path(args.report_dir), {**ce["seconds_per_generation"], **GEMMA_START_RATES, **MINISTRAL_START_RATES, **QWEN35_START_RATES, **GEMMA3N_START_RATES}, args.phase, estimate=estimate, export=args.phase == "main",
              state=machine["state"], logs=machine["logs"], resume_flags="" if args.machine == "mac" else f" --machine {args.machine}")
    return q.run()


if __name__ == "__main__":
    sys.exit(main())
