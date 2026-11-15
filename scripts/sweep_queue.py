"""Run 2: the timed harness batches, one at a time, under the specification's
measurement hygiene (Rule 6).

A timed batch is one harness configuration (one retriever x one generator at
one k). For each:

1. wait while anyone holds the queue (scripts/quiet.sh, for heavy work that must
   not overlap a timed batch), then take logs/sweep/RUNNING;
2. calibration probe; if the median drifts more than 10% from the preflight
   baseline, wait 10 minutes and probe again (up to 4 h);
3. run the configuration (`measurerag run --only <config>`), sampling every
   process outside the harness every 30 s;
4. probe again (recorded; on this fanless MacBook Air a long batch heats the
   machine itself, which is the same for every configuration, so the post-probe
   is information, not a discard criterion);
5. accept the batch unless the harness failed or another process contended for
   the CPU; a rejected attempt's result is moved to `<out>/discarded/` and the
   batch is rerun, at most three attempts (Rule 4's failure rule);
6. reassemble the sweep's results.json from every finished configuration and
   commit it with the calibration log.

Every attempt is a row in results/crossover/calibration/batches.jsonl.
Restartable: a configuration with an accepted batch on record is never re-run;
a result on disk without one (an interrupted batch) is discarded and re-measured.

    nohup .venv/bin/python scripts/sweep_queue.py > logs/sweep/queue_stdout.log 2>&1 &
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "sweep"
CAL = ROOT / "results" / "crossover" / "calibration"
BATCHES = CAL / "batches.jsonl"
RUNNING = LOGS / "RUNNING"
HOLD_DIR = LOGS / "hold.d"
STATE = LOGS / "queue_state.json"
QLOG = LOGS / "queue.log"
PY = str(ROOT / ".venv" / "bin" / "python")
BIN = str(ROOT / ".venv" / "bin" / "measurerag")

ENV = dict(
    os.environ,
    HF_HUB_CACHE=str(ROOT / "data" / "hf-hub"),
    HF_HUB_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    TOKENIZERS_PARALLELISM="false",
)

# A rebuild of a bench's or a sweep's files from accepted results runs under this: the runner then reuses a result
# measured under an identical spec and refuses to measure anything (src/measurerag/runner.py). This queue's rebuild,
# scripts/czech_timed_queue.py's and the finish stages of scripts/czech_complete_queue.py all use it.
REUSE_ONLY = {"MEASURERAG_REUSE_ONLY": "1"}

HYGIENE = True            # --no-hygiene (the author's choice): no probes, no gate, no contention rejection
ADOPT: set[str] = set()   # --adopt: stages whose result a stopped queue's harness finished
MAX_ATTEMPTS = 3          # Rule 4: the same step failing three times is marked failed
MAX_CONTENDED = 6         # rejections for outside contention are not failures;
                          # they are retried after a wait, up to this many per pass
DRIFT_WAIT_S = 600        # Rule 6: drift -> wait 10 minutes and probe again
MAX_DRIFT_WAITS = 24      # 4 h, the stall guard's horizon
STUCK_S = 45 * 60         # Rule 4: no progress for 45 minutes
SAMPLE_S = 30
CONTENDED_MAX = 80.0      # one foreign process at >= 80% of a core, or
CONTENDED_SUM = 150.0     # foreign processes together >= 1.5 cores (idle: 30-55%)
CONTENDED_FRAC = 0.05     # a batch with contended samples in at least 5% of samples,
CONTENDED_MIN = 3         # and at least 3, is rejected
MIN_FREE_GB = 8.0         # Rule 4's disk guard
# an optional trailer for the queue's result commits (one line of text after a blank line)
COMMIT_TRAILER = os.environ.get("MEASURERAG_COMMIT_TRAILER", "")

G15, G3 = "qwen2.5-1.5b", "qwen2.5-3b"
K5, K3, K10, NQ = (f"configs/crossover/{x}.yaml" for x in ("hotpotqa_k5", "hotpotqa_k3", "hotpotqa_k10", "nq_k5"))


def T(cfg: str, rid: str, gen: str, step: int) -> dict:
    return {"kind": "timed", "config": cfg, "config_id": f"{rid}+{gen}", "step": step,
            "id": f"{Path(cfg).stem}:{rid}+{gen}"}


def R(dataset: str, slice_: str, cache: str, spec: str, reuse_cfg: str, reuse_cid: str, step: int) -> dict:
    """A cached ranking (untimed, but heavy, so it also runs alone). It reuses the
    flat index a harness configuration built on a byte-identical slice."""
    return {"kind": "ranking", "dataset": dataset, "slice": slice_, "cache": cache, "spec": spec,
            "reuse_cfg": reuse_cfg, "reuse_cid": reuse_cid, "step": step,
            "id": f"ranking:{dataset}:{spec.split(':')[0]}"}


# Order of work (specification, RULE 3), from where run 2 stopped: step 1 (HotpotQA
# dial) and step 3 (NQ dial) are complete; HotpotQA k = 3 is complete.
STAGES = [
    # step 2: HotpotQA natural sweep, k = 5 first (re-measured), then k = 10's missing configuration
    T(K5, "bm25", G15, 2), T(K5, "e5-small", G15, 2), T(K5, "e5-base", G15, 2), T(K5, "e5-base", G3, 2),
    T(K5, "bm25", G3, 2), T(K5, "e5-small", G3, 2), T(K5, "tfidf-svd", G15, 2), T(K5, "tfidf-svd", G3, 2),
    T(K10, "e5-base", G3, 2),
    # step 4: NQ natural sweep at k = 5
    T(NQ, "bm25", G15, 4), T(NQ, "e5-small", G15, 4), T(NQ, "e5-base", G15, 4), T(NQ, "e5-base", G3, 4),
    T(NQ, "bm25", G3, 4), T(NQ, "e5-small", G3, 4), T(NQ, "tfidf-svd", G15, 4), T(NQ, "tfidf-svd", G3, 4),
    # step 5: bge-m3 rungs
    T(K5, "bge-m3", G15, 5),
    R("hotpotqa", "results/en_rag/slice", "results/crossover/cache/hotpotqa", "bge-m3:dense:bge-m3",
      K5, f"bge-m3+{G15}", 5),
    T(K5, "bge-m3", G3, 5),
    T(NQ, "bge-m3", G15, 5),
    R("nq", "results/crossover/slices/nq", "results/crossover/cache/nq", "bge-m3:dense:bge-m3",
      NQ, f"bge-m3+{G15}", 5),
    T(NQ, "bge-m3", G3, 5),
    T(K3, "bge-m3", G15, 5), T(K3, "bge-m3", G3, 5),
    T(K10, "bge-m3", G15, 5), T(K10, "bge-m3", G3, 5),
]


# --------------------------------------------------------------------------
def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{utc()} {msg}"
    print(line, flush=True)
    with open(QLOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def out_dir(cfg: str) -> Path:
    return ROOT / yaml.safe_load((ROOT / cfg).read_text(encoding="utf-8"))["out_dir"]


def result_path(cfg: str, cid: str) -> Path:
    return out_dir(cfg) / "specs" / f"{cid}.result.json"


def read_result(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def batch_rows() -> list[dict]:
    if not BATCHES.exists():
        return []
    rows = []
    for line in BATCHES.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def record(row: dict) -> None:
    CAL.mkdir(parents=True, exist_ok=True)
    with open(BATCHES, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def accepted_ids() -> set[str]:
    """Stages with an accepted batch on record; a later "superseded" row takes a stage's accepted batch back."""
    done: set[str] = set()
    for r in batch_rows():
        if r.get("superseded"):
            done.discard(r["stage"])
        elif r.get("accepted"):
            done.add(r["stage"])
    return done


def supersede(st: dict, reason: str) -> None:
    """Take a stage's accepted batch back so that it is measured again: its result moves to `<out>/superseded/`
    (kept and committed, without per-question outputs) and a "superseded" row joins the batch record."""
    sid, cfg, cid = st["id"], st["config"], st["config_id"]
    src = result_path(cfg, cid)
    dst = out_dir(cfg) / "superseded" / f"{cid}.{utc().replace(':', '')}.result.json"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        res = read_result(src) or {}
        res.pop("predictions", None)
        dst.write_text(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        src.unlink()
    record({"stage": sid, "step": st["step"], "config": cfg, "config_id": cid, "superseded": True, "utc": utc(),
            "reason": reason, "kept_as": str(dst.relative_to(ROOT))})
    log(f"[{sid}] superseded ({reason}); its result is kept as {dst.relative_to(ROOT)}")


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def holders() -> list[Path]:
    HOLD_DIR.mkdir(parents=True, exist_ok=True)
    live = []
    for f in HOLD_DIR.iterdir():
        try:
            pid = int(f.name)
        except ValueError:
            continue
        if pid_alive(pid):
            live.append(f)
        else:
            f.unlink(missing_ok=True)
    return live


def acquire() -> None:
    announced = False
    while True:
        while holders():
            if not announced:
                log("held: waiting for scripts/quiet.sh holders to finish")
                announced = True
            time.sleep(15)
        RUNNING.write_text(json.dumps({"pid": os.getpid(), "since": utc()}), encoding="utf-8")
        if holders():          # someone took a hold in between: back off
            RUNNING.unlink(missing_ok=True)
            continue
        return


def release() -> None:
    RUNNING.unlink(missing_ok=True)


# --------------------------------------------------------------------------
def probe(label: str) -> dict:
    extra = ["--log", str(CAL / "probes.jsonl"), "--baseline-file", str(CAL / "baseline.json")] if PHASE != "run2" else []
    p = subprocess.run([PY, "scripts/calibration_probe.py", "--label", label, *extra], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    row: dict = {"exit": p.returncode}
    try:
        row.update(json.loads(p.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError):
        row["error"] = (p.stderr or p.stdout)[-400:]
        row["ok"] = False
    return row


def slim_probe(r: dict | None) -> dict | None:
    if not r:
        return None
    return {k: r.get(k) for k in ("median_s", "drift", "ok", "answers_match_baseline", "loadavg_1m", "error")
            if r.get(k) is not None}


COOL_C = 42.0             # Ministral 3's sweeps: probe once the hottest CPU die sensor is at or below this,
SETTLE_S = 60             # and a minute after that (the author, 11 Nov 2026)
COOL_MAX_S = 20 * 60      # not cool after 20 minutes: probe anyway, and let the probe decide


QUARTER = 0.25           # with hygiene off: a batch more than this share of whose samples were contended is re-run


def chip_c() -> float | None:
    """The hottest CPU die sensor now, for the record (Ministral 3's sweeps); None for run 2 or where it cannot be read."""
    if PHASE == "run2":
        return None
    from mac_temps import die_max

    t = die_max()
    return round(t, 1) if t is not None else None


def wait_cool() -> None:
    """Until the chip has cooled, read from its sensors, then a minute more. Run 2 waited ten minutes instead."""
    from mac_temps import die_max

    t0, t = time.monotonic(), die_max()
    if t is None:
        return
    while t > COOL_C and time.monotonic() - t0 < COOL_MAX_S:
        time.sleep(15)
        t = die_max() or 0.0
    log(f"  chip at {t:.1f} C after {(time.monotonic() - t0) / 60:.1f} min" + ("" if t <= COOL_C else ", not cooled: probing anyway"))
    time.sleep(SETTLE_S)


def gated_probe(sid: str, attempt: int) -> dict | None:
    """Probe until the machine is within 10% of the baseline. None = gave up."""
    for i in range(MAX_DRIFT_WAITS + 1):
        if PHASE != "run2":
            wait_cool()
        r = probe(f"pre:{sid}#{attempt}")
        if r.get("ok"):
            return r
        if r.get("error") or r.get("answers_match_baseline") is False:
            log(f"  probe FAILED ({r.get('error') or 'answers differ from the baseline'}); not waiting it out")
            return {**r, "fatal": True}
        if PHASE != "run2":
            log(f"  probe drift {r.get('drift')} > 10%: waiting for the chip to cool ({i + 1}/{MAX_DRIFT_WAITS})")
            continue
        log(f"  probe drift {r.get('drift')} > 10%: waiting {DRIFT_WAIT_S // 60} min ({i + 1}/{MAX_DRIFT_WAITS})")
        time.sleep(DRIFT_WAIT_S)
    return None


def ps_table() -> list[tuple[int, int, float, float, str]]:
    """(pid, ppid, %cpu, cpu seconds, command basename) for every process."""
    p = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,pcpu=,time=,comm="], capture_output=True, text=True)
    rows = []
    for line in p.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            t = parts[3]
            secs = 0.0
            if "-" in t:
                d, t = t.split("-", 1)
                secs += 86400 * int(d)
            for x in t.split(":"):
                secs = secs * 60 + float(x) if x else secs
            rows.append((int(parts[0]), int(parts[1]), float(parts[2]), secs, os.path.basename(parts[4].strip())))
        except ValueError:
            continue
    return rows


def wait_for_quiet(windows: int = 4, window_s: int = 30, total_max: float = 60.0, one_max: float = 40.0) -> None:
    """Until every other process together used under `total_max`% of a core, and none over `one_max`%, in `windows`
    windows of `window_s` seconds in a row: the Mac as the batches run on it, an editor open and nobody working."""
    quiet, waited, me = 0, 0, os.getpid()
    before = {r[0]: r[3] for r in ps_table()}
    while quiet < windows:
        time.sleep(window_s)
        waited += window_s
        rows = ps_table()
        use = {pid: 100 * (secs - before.get(pid, secs)) / window_s for pid, _, _, secs, _ in rows if pid != me}
        before = {r[0]: r[3] for r in rows}
        total, top = sum(use.values()), max(use.values(), default=0.0)
        quiet = quiet + 1 if total < total_max and top < one_max else 0
        if waited % 600 == 0:
            log(f"  waiting for a quiet Mac: other processes at {total:.0f}% of a core, the busiest at {top:.0f}%")


class Sampler(threading.Thread):
    """Watches everything outside the harness, and whether the harness progresses."""

    def __init__(self, root_pid: int) -> None:
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.stop = threading.Event()
        self.samples: list[dict] = []
        self.last_progress = time.monotonic()
        self.last_cpu = -1.0
        self.stuck = False

    def run(self) -> None:
        me = os.getpid()
        while not self.stop.wait(SAMPLE_S):
            rows = ps_table()
            children: dict[int, list[int]] = {}
            for pid, ppid, *_ in rows:
                children.setdefault(ppid, []).append(pid)
            tree, todo = set(), [self.root_pid]
            while todo:
                x = todo.pop()
                if x not in tree:
                    tree.add(x)
                    todo += children.get(x, [])
            foreign = [(pc, cm) for pid, _, pc, _, cm in rows if pid not in tree and pid != me and cm != "ps"]
            top = max(foreign, default=(0.0, ""))
            s = sum(pc for pc, _ in foreign)
            self.samples.append({"max": top[0], "max_comm": top[1], "sum": round(s, 1),
                                 "contended": top[0] >= CONTENDED_MAX or s >= CONTENDED_SUM})
            cpu = sum(t for pid, _, _, t, _ in rows if pid in tree)
            if cpu > self.last_cpu + 1.0:
                self.last_cpu, self.last_progress = cpu, time.monotonic()
            elif time.monotonic() - self.last_progress > STUCK_S:
                self.stuck = True
                return

    def summary(self) -> dict:
        n = len(self.samples)
        c = [s for s in self.samples if s["contended"]]
        worst = max(self.samples, key=lambda s: s["sum"], default=None)
        return {"n_samples": n, "n_contended": len(c),
                "contaminated": len(c) >= max(CONTENDED_MIN, math.ceil(CONTENDED_FRAC * n)),
                "worst_foreign": worst and {k: worst[k] for k in ("max", "max_comm", "sum")},
                "contended_comms": sorted({s["max_comm"] for s in c})[:8]}


# --------------------------------------------------------------------------
def git(*args: str) -> subprocess.CompletedProcess:
    for _ in range(20):
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        if "index.lock" not in (p.stderr or ""):
            return p
        time.sleep(5)
    return p


def commit(paths: list[Path], msg: str) -> None:
    rel = [str(p.relative_to(ROOT)) for p in paths if p.exists()]
    git("add", "--", *rel)
    p = git("commit", "-q", "-m", f"{msg}\n\n{COMMIT_TRAILER}" if COMMIT_TRAILER else msg, "--", *rel)
    if p.returncode != 0 and "nothing" not in (p.stdout + p.stderr):
        log(f"  commit failed: {(p.stderr or p.stdout)[-300:]}")


def assemble(cfg: str) -> None:
    """Rewrite the sweep's results.json from every finished configuration."""
    specs = out_dir(cfg) / "specs"
    done = sorted(f.name[: -len(".result.json")] for f in specs.glob("*.result.json")
                  if (read_result(f) or {}).get("status") == "ok")
    if not done:
        # nothing accepted yet: no sweep files at all, rather than a rejected batch's
        for f in list(out_dir(cfg).glob("results.*")) + list(out_dir(cfg).glob("results_table.txt")) + \
                list(out_dir(cfg).glob("manifest.json")) + list(out_dir(cfg).glob("curve_*.png")) + \
                list(out_dir(cfg).glob("predictions_*.json")):
            f.unlink()
        return
    # reuse only: a spec that changed since a batch was accepted must fail here, not measure again
    p = subprocess.run([BIN, "run", "--config", cfg, "--only", ",".join(done)], cwd=ROOT,
                       env={**ENV, **REUSE_ONLY}, capture_output=True, text=True)
    if p.returncode != 0:
        log(f"  assemble {cfg} failed: {(p.stderr or p.stdout)[-300:]}")


def discard(cfg: str, cid: str, reason: str) -> None:
    src = result_path(cfg, cid)
    if src.exists():
        dst = out_dir(cfg) / "discarded" / f"{cid}.{utc().replace(':', '')}.result.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        # the discarded record is committed; per-question outputs never are
        res = read_result(src) or {}
        res.pop("predictions", None)
        dst.write_text(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        src.unlink()
        log(f"  discarded {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)} ({reason})")


def run_timed(st: dict) -> str:
    sid, cfg, cid = st["id"], st["config"], st["config_id"]
    if result_path(cfg, cid).exists() and sid in ADOPT:
        res = read_result(result_path(cfg, cid)) or {}
        if res.get("status") == "ok":
            m = res.get("measurement", {})
            record({"stage": sid, "step": st["step"], "config": cfg, "config_id": cid,
                    "attempt": len([r for r in batch_rows() if r.get("stage") == sid]) + 1,
                    "utc_end": utc(), "accepted": True, "hygiene": "off",
                    "reason": "adopted: started under the probe gate by the previous queue process, which was "
                              "stopped mid-batch when hygiene was switched off; its contention record was lost",
                    "n_answered": (res.get("answers") or {}).get("n_answered"), "em": (res.get("answers") or {}).get("em"),
                    "peak_rss_gb": m.get("peak_rss_gb"), "e2e_s_p50": m.get("e2e_s_p50"), "e2e_s_p95": m.get("e2e_s_p95"),
                    "macos": (res.get("machine") or {}).get("os_version"), "result_status": "ok"})
            log(f"[{sid}] adopted the result the stopped queue's harness finished")
            assemble(cfg)
            commit([out_dir(cfg), CAL], f"{COMMIT_PREFIX} {st['step']}: {cid} at {Path(cfg).stem} adopted "
                   f"(EM {res['answers']['em']}, e2e p50 {m.get('e2e_s_p50', 0):.2f} s)")
            return "ok"
    if result_path(cfg, cid).exists():
        # on disk but no accepted batch on record: an interrupted run, provenance unknown
        discard(cfg, cid, "result without an accepted batch record")
    prior = [r for r in batch_rows() if r.get("stage") == sid]
    failures = contended = 0
    attempt = len(prior)
    while failures < MAX_ATTEMPTS and contended < MAX_CONTENDED:
        attempt += 1
        pause = False
        if free_gb() < MIN_FREE_GB:
            log(f"DISK GUARD: {free_gb():.1f} GB free < {MIN_FREE_GB} GB; stopping")
            return "disk"
        acquire()
        t0, start = time.monotonic(), utc()
        row = {"stage": sid, "step": st["step"], "config": cfg, "config_id": cid, "attempt": attempt,
               "utc_start": start}
        try:
            row["hygiene"] = "on" if HYGIENE else "off"
            pre = gated_probe(sid, attempt) if HYGIENE else {"skipped": True}
            row["pre_probe"] = slim_probe(pre) if HYGIENE else None
            if pre is None or pre.get("fatal"):
                row.update(accepted=False, reason="probe never within 10%" if pre is None else "probe failed",
                           utc_end=utc())
                record(row)
                if pre is None:
                    return "drift"
                failures += 1
                continue
            log(f"[{sid}] attempt {attempt}: " + (f"probe {pre['median_s']} s ({pre['drift']:+.1%}); running"
                                                   if HYGIENE else "hygiene off; running"))
            chip0 = chip_c()
            lf = open(LOGS / f"{sid.replace(':', '__')}.{attempt}.log", "w", encoding="utf-8")
            # the worker writes how many questions it has answered to this file after each answer, so a batch's progress
            # can be read while it runs (its answers and timings are written only when it ends)
            beat = LOGS / f"{sid.replace(':', '__')}.{attempt}.heartbeat"
            proc = subprocess.Popen([BIN, "run", "--config", cfg, "--only", cid], cwd=ROOT,
                                    env={**ENV, "MEASURERAG_HEARTBEAT": str(beat)}, stdout=lf, stderr=subprocess.STDOUT)
            sampler = Sampler(proc.pid)
            sampler.start()
            while proc.poll() is None:
                if sampler.stuck:
                    log(f"  STUCK: no CPU progress for {STUCK_S // 60} min; killing")
                    proc.terminate()
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(10)
            sampler.stop.set()
            lf.close()
            post = probe(f"post:{sid}#{attempt}") if HYGIENE else {}
            chip1 = chip_c()
            res = read_result(result_path(cfg, cid)) or {}
            cont = sampler.summary()
            # with hygiene off, Ministral 3's sweeps still re-run a batch more than a quarter of whose samples were contended
            heavy = (not HYGIENE and PHASE != "run2" and cont["n_samples"] > 0
                     and cont["n_contended"] > QUARTER * cont["n_samples"])
            ok = (proc.returncode == 0 and res.get("status") == "ok" and not sampler.stuck
                  and (not cont["contaminated"] or not HYGIENE) and not heavy)
            reason = ("stuck" if sampler.stuck else
                      f"harness exit {proc.returncode}, status {res.get('status')}" if res.get("status") != "ok" else
                      "contention" if (cont["contaminated"] and HYGIENE) or heavy else
                      "ok, hygiene off (contention recorded, not enforced)" if not HYGIENE else "ok")
            m = res.get("measurement", {})
            row.update({
                "utc_end": utc(), "wall_s": round(time.monotonic() - t0, 1),
                "post_probe": slim_probe(post) if HYGIENE else None, "contention": cont, "chip_c": [chip0, chip1],
                "harness_exit": proc.returncode, "result_status": res.get("status"),
                "n_answered": (res.get("answers") or {}).get("n_answered"),
                "em": (res.get("answers") or {}).get("em"),
                "peak_rss_gb": m.get("peak_rss_gb"), "e2e_s_p50": m.get("e2e_s_p50"),
                "e2e_s_p95": m.get("e2e_s_p95"),
                "macos": (res.get("machine") or {}).get("os_version"),
                "accepted": ok, "reason": reason,
            })
            record(row)
            log(f"  -> {reason}; EM {row['em']}, p50 {row['e2e_s_p50']}, peak {row['peak_rss_gb']} GB, "
                f"post-probe {(post or {}).get('drift')}, contended samples {cont['n_contended']}/{cont['n_samples']}, "
                f"chip {chip0} -> {chip1} C")
            if ok:
                assemble(cfg)
                commit([out_dir(cfg), CAL],
                       f"{COMMIT_PREFIX} {st['step']}: {cid} at {Path(cfg).stem} measured "
                       f"{'under the probe' if HYGIENE else 'with hygiene off'} "
                       f"(EM {row['em']}, e2e p50 {row['e2e_s_p50']:.2f} s, peak {row['peak_rss_gb']} GB)")
                return "ok"
            discard(cfg, cid, reason)
            assemble(cfg)           # the sweep's files show accepted batches only
            commit([out_dir(cfg), CAL], f"{COMMIT_PREFIX} {st['step']}: {cid} at {Path(cfg).stem} attempt "
                   f"{attempt} discarded ({reason})")
            if sampler.stuck:
                return "stuck"
            if reason == "contention":
                contended += 1
                pause = True
            else:
                failures += 1
        finally:
            release()
            write_state()
        if pause:
            # outside the RUNNING marker, so quiet-window work can use the wait
            log(f"  contention is not a failure; waiting {DRIFT_WAIT_S // 60} min before attempt {attempt + 1}")
            time.sleep(DRIFT_WAIT_S)
    return "contended" if contended >= MAX_CONTENDED else "failed"


def run_ranking(st: dict) -> str:
    sid = st["id"]
    spec_id = st["spec"].split(":")[0]
    out = ROOT / st["cache"] / f"retrieval_{spec_id}.json"
    if out.exists():
        return "ok"
    reuse = out_dir(st["reuse_cfg"]) / "work" / st["reuse_cid"] / "faiss_flat.index"
    harness_slice = out_dir(st["reuse_cfg"]) / "slice" / "corpus.jsonl"
    same = (harness_slice.exists() and
            harness_slice.read_bytes() == (ROOT / st["slice"] / "corpus.jsonl").read_bytes())
    args = [PY, "scripts/retrieval_cache.py", "--slice", st["slice"], "--cache", st["cache"], "--retriever", st["spec"]]
    if reuse.exists() and same:
        args += ["--reuse-index", str(reuse)]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        acquire()
        try:
            log(f"[{sid}] attempt {attempt}: {'reusing ' + str(reuse.relative_to(ROOT)) if '--reuse-index' in args else 'building'}")
            with open(LOGS / f"{sid.replace(':', '__')}.{attempt}.log", "w", encoding="utf-8") as lf:
                p = subprocess.run(args, cwd=ROOT, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
            if p.returncode == 0 and out.exists():
                log(f"  -> ok")
                return "ok"
            log(f"  -> exit {p.returncode}")
        finally:
            release()
            write_state()
    return "failed"


# --------------------------------------------------------------------------
STATUS: dict[str, str] = {}


def write_state() -> None:
    rows = batch_rows()
    acc = [r for r in rows if r.get("accepted")]
    STATE.write_text(json.dumps({
        "utc": utc(), "pid": os.getpid(),
        "stages": [{"id": s["id"], "step": s["step"], "status": STATUS.get(s["id"], "pending")} for s in STAGES],
        "batches_accepted": len(acc), "batches_rejected": sum(1 for r in rows if r.get("accepted") is False),
        "answers_accepted": sum(int(r.get("n_answered") or 0) for r in acc),
        "answers_all_attempts": sum(int(r.get("n_answered") or 0) for r in rows),
        "free_gb": round(free_gb(), 1),
    }, indent=1), encoding="utf-8")


# Ministral 3's English sweeps (v0.4.2): the two main sweeps of run 2, HotpotQA and NQ at k = 5, with Ministral 3 3B and 8B
# in Qwen2.5 1.5B's and 3B's places, under the same hygiene. Own result folders and calibration record, so that run 2's stay
# byte for byte as they are; the e5-base pair first, the configurations the budgets price, so a stopped run still has them.
M3, M8 = "ministral-3-3b", "ministral-3-8b"
K5M, NQM = (f"configs/crossover/{x}_ministral.yaml" for x in ("hotpotqa_k5", "nq_k5"))
STAGES_MINISTRAL = [T(cfg, rid, gen, step) for cfg, step in ((K5M, 1), (NQM, 2))
                    for rid in ("e5-base", "bm25", "e5-small", "bge-m3", "tfidf-svd") for gen in (M3, M8)]
PHASE = "run2"
COMMIT_PREFIX = "Run 2, step"
PHASES = {"run2": {"stages": STAGES, "cal": CAL, "logs": LOGS, "commit": "Run 2, step"},
          "ministral": {"stages": STAGES_MINISTRAL, "cal": ROOT / "results" / "crossover" / "calibration_ministral",
                        "logs": ROOT / "logs" / "sweep_ministral", "commit": "Ministral 3's English sweeps, step"}}


def use_phase(name: str) -> None:
    """Point the queue's paths at one phase's stages, calibration record and logs."""
    global PHASE, STAGES, CAL, BATCHES, LOGS, RUNNING, HOLD_DIR, STATE, QLOG, COMMIT_PREFIX
    ph = PHASES[name]
    PHASE, STAGES, CAL, LOGS = name, ph["stages"], ph["cal"], ph["logs"]
    BATCHES, RUNNING, HOLD_DIR = CAL / "batches.jsonl", LOGS / "RUNNING", LOGS / "hold.d"
    STATE, QLOG, COMMIT_PREFIX = LOGS / "queue_state.json", LOGS / "queue.log", ph["commit"]


def main() -> int:
    import argparse

    global HYGIENE
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hygiene", action="store_true",
                    help="no probes, no gate, no contention rejection (contention is still recorded)")
    ap.add_argument("--wait-pid", type=int, default=None, help="wait for this process (a running harness) first")
    ap.add_argument("--adopt", default="", help="comma-separated stage ids whose finished result to accept")
    ap.add_argument("--phase", choices=sorted(PHASES), default="run2", help="run 2's stages, or Ministral 3's English sweeps")
    ap.add_argument("--retime", default="", help="comma-separated stage ids whose accepted batch to supersede and measure again")
    ap.add_argument("--retime-reason", default="", help="why, written into the batch record")
    ap.add_argument("--baseline-after", type=float, default=2.0,
                    help="a phase with no baseline yet takes it itself, this many minutes after starting and once the Mac is quiet")
    args = ap.parse_args()
    use_phase(args.phase)
    HYGIENE = not args.no_hygiene
    ADOPT.update(x for x in args.adopt.split(",") if x)
    LOGS.mkdir(parents=True, exist_ok=True)
    HOLD_DIR.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        log(f"waiting for pid {args.wait_pid} (the batch the previous queue started) to finish")
        while pid_alive(args.wait_pid):
            time.sleep(15)
    if subprocess.run([BIN, "doctor"], cwd=ROOT, env=ENV, capture_output=True).returncode != 0:
        log("ABORTED: measurerag doctor is unhappy (run `measurerag doctor --fix`)")
        return 1
    if PHASE != "run2" and HYGIENE and not (CAL / "baseline.json").exists():
        # the probe runs on every core, so it is about a fifth slower while someone works on the Mac: the baseline is taken
        # by the queue itself, once the Mac is quiet, the state its batches run in
        log(f"no baseline yet: waiting {args.baseline_after:g} min, then for two quiet minutes in a row, then taking it")
        time.sleep(args.baseline_after * 60)
        wait_for_quiet()
        wait_cool()                 # the state every gate probe is taken in: quiet, the chip cooled, a minute more
        p = subprocess.run([PY, "scripts/calibration_probe.py", "--baseline", "--baseline-file", str(CAL / "baseline.json"),
                            "--log", str(CAL / "probes.jsonl")], cwd=ROOT, env=ENV, capture_output=True, text=True)
        log(f"baseline: {(p.stdout.strip().splitlines() or [p.stderr[-300:]])[-1]}")
        if p.returncode != 0:
            log("ABORTED: the baseline probe failed")
            return 1
    log(f"queue start, pid {os.getpid()}, {len(STAGES)} stages, {free_gb():.0f} GB free, "
        f"hygiene {'on' if HYGIENE else 'OFF'}")
    retime = [x for x in args.retime.split(",") if x]
    unknown = [x for x in retime if x not in {st["id"] for st in STAGES}]
    if unknown or (retime and not args.retime_reason):
        log(f"ABORTED: --retime needs known stage ids and a --retime-reason ({unknown or 'no reason'})")
        return 1
    for st in STAGES:
        # once per stage: a restart finds the superseded row and goes on measuring, it never supersedes the new batch
        if st["id"] in retime and not any(r.get("superseded") and r["stage"] == st["id"] for r in batch_rows()):
            supersede(st, args.retime_reason)
    done = accepted_ids()
    retry: list[dict] = []
    for st in STAGES:
        if st["kind"] == "timed" and st["id"] in done and result_path(st["config"], st["config_id"]).exists():
            STATUS[st["id"]] = "done"
            continue
        STATUS[st["id"]] = "running"
        write_state()
        outcome = run_timed(st) if st["kind"] == "timed" else run_ranking(st)
        STATUS[st["id"]] = outcome
        write_state()
        if outcome == "disk":
            return 2
        if outcome in ("stuck", "drift", "contended", "failed"):
            retry.append(st)      # Rule 4: move on, come back once at the end
    for st in retry:
        log(f"coming back once to {st['id']}")
        outcome = run_timed(st)
        STATUS[st["id"]] = outcome if outcome == "ok" else f"{outcome} (twice)"
        write_state()
    log("queue done: " + json.dumps({k: v for k, v in STATUS.items() if v != "done"}))
    (LOGS / "QUEUE_DONE").write_text(utc() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
