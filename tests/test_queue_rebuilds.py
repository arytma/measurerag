"""Every queue's rebuild of a bench's or a sweep's files runs reuse-only, so it can never measure again.

The runner's side, that MEASURERAG_REUSE_ONLY reuses a result measured under an identical spec and refuses to
measure, is tested in tests/test_resume.py. These check that each rebuild sets it: the English sweeps'
(scripts/sweep_queue.py), the Czech timed retrievers' (scripts/czech_timed_queue.py) and every finish stage of the
complete Czech run (scripts/czech_complete_queue.py), while the stages meant to measure do not. No model, no data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import czech_complete_queue as cq  # noqa: E402
import sweep_queue as sq  # noqa: E402

FLAG = "MEASURERAG_REUSE_ONLY"


class Calls:
    """Stands in for subprocess.run: records each command and its environment, and succeeds."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd, **kw):
        self.calls.append((list(cmd), dict(kw.get("env") or {})))
        return subprocess.CompletedProcess(cmd, 0, "", "")


def test_the_three_queues_and_the_runner_name_the_same_guard():
    assert sq.REUSE_ONLY == cq.REUSE_ONLY == {FLAG: "1"}
    assert f'os.environ.get("{FLAG}")' in (REPO / "src/measurerag/runner.py").read_text(encoding="utf-8")


def test_the_english_rebuild_is_reuse_only(tmp_path, monkeypatch):
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "bm25+ministral-3-3b.result.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (specs / "e5-small+ministral-3-3b.result.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    monkeypatch.setattr(sq, "out_dir", lambda cfg: tmp_path)
    calls = Calls()
    monkeypatch.setattr(subprocess, "run", calls)
    sq.assemble("configs/crossover/hotpotqa_k5_ministral.yaml")
    [(cmd, env)] = calls.calls
    assert cmd[cmd.index("--only") + 1] == "bm25+ministral-3-3b"      # accepted results only
    assert env.get(FLAG) == "1"


def test_the_czech_timed_rebuild_is_reuse_only(monkeypatch):
    # importing the timed queue points sweep_queue's record paths at its own; keep sweep_queue's and restore them after
    for name in ("CAL", "BATCHES", "QLOG"):
        monkeypatch.setattr(sq, name, getattr(sq, name))
    import czech_timed_queue as ctq

    accepted = {"bm25", "retromae-cs"}
    monkeypatch.setattr(ctq, "accepted_ids", lambda: accepted)
    calls = Calls()
    monkeypatch.setattr(subprocess, "run", calls)
    ctq.assemble()
    [(cmd, env)] = calls.calls
    assert cmd[cmd.index("--only") + 1].split(",") == [r for r in ctq.retrievers() if r in accepted]
    assert env.get(FLAG) == "1"


def test_every_czech_finish_stage_is_reuse_only_and_every_bench_stage_may_measure():
    """A finish stage runs the whole bench configuration to rebuild its files from the configurations its bench stages
    measured; a spec that changed since (a pinned model's, say) must stop it, not measure that configuration again."""
    plan = json.loads((REPO / "results/czech/plan/plan.json").read_text(encoding="utf-8"))
    phases = [cq.main_stages(plan), cq.ministral_stages(plan), cq.gemma3n_stages(plan), cq.qwen35_stages(plan)]
    runs = [s for phase in phases for s in phase if s.cmd[:2] == [".venv/bin/measurerag", "run"]]
    finishes = [s for s in runs if "--only" not in s.cmd]
    benches = [s for s in runs if "--only" in s.cmd]
    assert len(finishes) == 2 * len(phases) and benches
    assert all("_bench_finish" in s.name for s in finishes)
    assert all(s.env.get(FLAG) == "1" for s in finishes)
    assert not any(FLAG in s.env for s in benches)
