"""v0.4.0's timed rows.

    nohup .venv/bin/python scripts/czech_timed_queue.py > logs/v040_timed/stdout.log 2>&1 &

Every BM25 and every dense retriever of configs/cz_v040_timed.yaml runs on SQAD 3.2's corpus, retrieval only, one
configuration per process (`measurerag run --only <id>`), under the measurement hygiene of docs/METHOD.md,
section 4: the hold of scripts/quiet.sh, the calibration probe before each batch with its 10% gate, the contention
sampler, rejection and rerun. It reuses scripts/sweep_queue.py's pieces and keeps its own records, so Run 2's
calibration log does not change:

    results/czech/v040/timed/probes.jsonl    every probe
    results/czech/v040/timed/batches.jsonl   every attempt, accepted or not
    results/cz_v040_timed/                   the bench's results, accepted configurations only

The queue commits nothing. Restartable: a retriever with an accepted batch on record is skipped, and a result on
disk without one is discarded and measured again.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_queue as sq  # noqa: E402

ROOT = sq.ROOT
CFG = "configs/cz_v040_timed.yaml"
OUT = ROOT / "results" / "cz_v040_timed"
TIMED = ROOT / "results" / "czech" / "v040" / "timed"
LOGS = ROOT / "logs" / "v040_timed"

# sweep_queue's functions read these module globals: point them at this queue's own records
sq.CAL, sq.BATCHES, sq.QLOG = TIMED, TIMED / "batches.jsonl", LOGS / "queue.log"


def probe(label: str) -> dict:
    p = subprocess.run([sq.PY, "scripts/calibration_probe.py", "--label", label, "--log", str(TIMED / "probes.jsonl")],
                       cwd=ROOT, env=sq.ENV, capture_output=True, text=True)
    row: dict = {"exit": p.returncode}
    try:
        row.update(json.loads(p.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError):
        row["error"] = (p.stderr or p.stdout)[-400:]
        row["ok"] = False
    return row


sq.probe = probe          # gated_probe calls it by this name


def retrievers() -> list[str]:
    return [r["id"] for r in yaml.safe_load((ROOT / CFG).read_text(encoding="utf-8"))["retrievers"]]


def result_path(rid: str) -> Path:
    return OUT / "specs" / f"{rid}.result.json"


def discard(rid: str, reason: str) -> None:
    src = result_path(rid)
    if src.exists():
        dst = OUT / "discarded" / f"{rid}.{sq.utc().replace(':', '')}.result.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        sq.log(f"  discarded {src.relative_to(ROOT)} ({reason})")


def accepted_ids() -> set[str]:
    """The retrievers with an accepted row on record, in record order: a later "superseded" row takes an accepted row back
, so the retriever is timed again; the superseded row's result is kept under discarded/."""
    ids: set[str] = set()
    for r in sq.batch_rows():
        if r.get("superseded"):
            ids.discard(r["stage"])
        elif r.get("accepted"):
            ids.add(r["stage"])
    return ids


def supersede(rid: str, reason: str) -> None:
    sq.record({"stage": rid, "config": CFG, "config_id": rid, "superseded": True, "accepted": False, "utc_start": sq.utc(),
               "reason": reason})
    sq.log(f"[{rid}] accepted row superseded: {reason}; it is timed again")
    discard(rid, "superseded")


def assemble() -> None:
    """The bench's results.json and table from the accepted configurations only. Reuse only: a spec that changed since a
    batch was accepted stops the rebuild instead of measuring that retriever again."""
    done = [rid for rid in retrievers() if rid in accepted_ids()]
    if done:
        p = subprocess.run([sq.BIN, "run", "--config", CFG, "--only", ",".join(done)], cwd=ROOT, env={**sq.ENV, **sq.REUSE_ONLY},
                           capture_output=True, text=True)
        if p.returncode != 0:
            sq.log(f"  assemble failed: {(p.stderr or p.stdout)[-300:]}")


def run_one(rid: str) -> str:
    if result_path(rid).exists():
        discard(rid, "result without an accepted batch record")
    attempt = len([r for r in sq.batch_rows() if r.get("stage") == rid])
    failures = contended = 0
    while failures < sq.MAX_ATTEMPTS and contended < sq.MAX_CONTENDED:
        attempt += 1
        if sq.free_gb() < sq.MIN_FREE_GB:
            sq.log(f"DISK GUARD: {sq.free_gb():.1f} GB free; stopping")
            return "disk"
        sq.acquire()
        pause = False
        row = {"stage": rid, "config": CFG, "config_id": rid, "attempt": attempt, "utc_start": sq.utc(), "hygiene": "on"}
        try:
            t0 = time.monotonic()
            pre = sq.gated_probe(rid, attempt)
            row["pre_probe"] = sq.slim_probe(pre)
            if pre is None or pre.get("fatal"):
                row.update(accepted=False, reason="probe never within 10%" if pre is None else "probe failed", utc_end=sq.utc())
                sq.record(row)
                if pre is None:
                    return "drift"
                failures += 1
                continue
            sq.log(f"[{rid}] attempt {attempt}: probe {pre['median_s']} s ({pre['drift']:+.1%}); running")
            with open(LOGS / f"{rid}.{attempt}.log", "w", encoding="utf-8") as lf:
                proc = subprocess.Popen([sq.BIN, "run", "--config", CFG, "--only", rid], cwd=ROOT, env=sq.ENV,
                                        stdout=lf, stderr=subprocess.STDOUT)
                sampler = sq.Sampler(proc.pid)
                sampler.start()
                while proc.poll() is None:
                    if sampler.stuck:
                        sq.log(f"  STUCK: no CPU progress for {sq.STUCK_S // 60} min; killing")
                        proc.terminate()
                        try:
                            proc.wait(60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                    time.sleep(10)
                sampler.stop.set()
            post = probe(f"post:{rid}#{attempt}")
            res = sq.read_result(result_path(rid)) or {}
            cont = sampler.summary()
            ok = proc.returncode == 0 and res.get("status") == "ok" and not sampler.stuck and not cont["contaminated"]
            reason = ("stuck" if sampler.stuck else
                      f"harness exit {proc.returncode}, status {res.get('status')}" if res.get("status") != "ok" else
                      "contention" if cont["contaminated"] else "ok")
            m = res.get("measurement", {})
            row.update({"utc_end": sq.utc(), "wall_s": round(time.monotonic() - t0, 1), "post_probe": sq.slim_probe(post),
                        "contention": cont, "harness_exit": proc.returncode, "result_status": res.get("status"),
                        "peak_rss_gb": m.get("peak_rss_gb"), "retrieval_ms_p50": m.get("retrieval_ms_p50"),
                        "retrieval_ms_p95": m.get("retrieval_ms_p95"), "index_bytes": m.get("index_bytes"),
                        "model_bytes": m.get("model_bytes"), "macos": (res.get("machine") or {}).get("os_version"),
                        "accepted": ok, "reason": reason})
            sq.record(row)
            sq.log(f"  -> {reason}; peak {row['peak_rss_gb']} GB, p95 {row['retrieval_ms_p95']} ms, "
                   f"contended samples {cont['n_contended']}/{cont['n_samples']}")
            if ok:
                return "ok"
            discard(rid, reason)
            if sampler.stuck:
                return "stuck"
            if reason == "contention":
                contended += 1
                pause = True
            else:
                failures += 1
        finally:
            sq.release()
        if pause:
            sq.log(f"  contention is not a failure; waiting {sq.DRIFT_WAIT_S // 60} min")
            time.sleep(sq.DRIFT_WAIT_S)
    return "contended" if contended >= sq.MAX_CONTENDED else "failed"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--remeasure", default="", help="comma-separated retrievers whose accepted row is superseded and timed again")
    ap.add_argument("--reason", default="", help="why, written into the record (required with --remeasure)")
    args = ap.parse_args()
    if args.remeasure and not args.reason:
        ap.error("--remeasure needs --reason")
    for rid in filter(None, args.remeasure.split(",")):
        if rid in accepted_ids():
            supersede(rid, args.reason)
    LOGS.mkdir(parents=True, exist_ok=True)
    TIMED.mkdir(parents=True, exist_ok=True)
    outcome = {}
    for rid in retrievers():
        if rid in accepted_ids():
            sq.log(f"[{rid}] accepted earlier; skipped")
            outcome[rid] = "ok"
            continue
        outcome[rid] = run_one(rid)
        if outcome[rid] in ("drift", "disk"):
            break
    assemble()
    sq.log(f"done: {json.dumps(outcome)}")
    return 0 if all(v == "ok" for v in outcome.values()) and len(outcome) == len(retrievers()) else 1


if __name__ == "__main__":
    sys.exit(main())
