"""English results and CsFEVER are frozen at v0.2.0: not re-run, changed or rounded.

The hashes and the token values are in tests/frozen_v020.json, read from v0.2.0's commit by
scripts/freeze_english.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = json.loads((REPO / "tests/frozen_v020.json").read_text(encoding="utf-8"))


def _block_hash(block: dict) -> str:
    return hashlib.sha256(json.dumps(block, sort_keys=True).encode("utf-8")).hexdigest()


def test_every_frozen_results_file_is_byte_identical():
    assert FROZEN["files"]
    changed = [rel for rel, digest in FROZEN["files"].items() if hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != digest]
    assert not changed


def test_the_english_blocks_of_the_crossover_are_unchanged():
    crossover = json.loads((REPO / "results/crossover.json").read_text(encoding="utf-8"))
    for key, digest in FROZEN["crossover_blocks"].items():
        # v0.4.2 adds other families' English dials beside Qwen2.5's, under their own key;
        # every reading of Qwen2.5's stays as v0.2.0 froze it
        block = dict(crossover["datasets"][key])
        dials = block.pop("dial_families", {})
        assert set(dials) <= {"gemma-3n", "gemma-4", "ministral-3", "qwen2.5"}, key
        # Qwen2.5's own entry there is the frozen dial read again, level for level, for the per-family tables
        qwen = dials.pop("qwen2.5", None)
        assert qwen is None or (qwen["dial"]["levels"] == block["dial"]["levels"]
                                and qwen["dial"]["roots"]["raw"] == block["dial"]["roots"]["raw"]), key
        assert not any(g.startswith("qwen") for d in dials.values() for g in d["generators"]), key
        assert _block_hash(block) == digest, key
    if FROZEN["agreement"]:
        assert _block_hash(crossover["agreement"]) == FROZEN["agreement"]


def test_no_token_outside_sqad_changed():
    tokens = json.loads((REPO / "results/stats.json").read_text(encoding="utf-8"))["tokens"]
    changed = {k: (v, tokens.get(k)) for k, v in FROZEN["tokens"].items() if tokens.get(k) != v}
    assert not changed


def test_ministral_3s_sweeps_write_nothing_run_2_froze():
    """Ministral 3's timed sweeps have their own configurations, result folders and calibration
    record, run 2's ten configurations of each k = 5 sweep with Ministral 3 in Qwen2.5's places."""
    import sys

    import yaml

    sys.path.insert(0, str(REPO / "scripts"))
    import sweep_queue as sq

    sq.use_phase("ministral")
    try:
        frozen = set(FROZEN["files"])
        run2_cal = sq.PHASES["run2"]["cal"]
        assert sq.CAL != run2_cal and not any(run2_cal in p.parents or p == run2_cal for p in (sq.CAL, sq.BATCHES))
        assert len(sq.STAGES) == 20 and {s["config"] for s in sq.STAGES} == {sq.K5M, sq.NQM}
        for cfg in (sq.K5M, sq.NQM):
            new = yaml.safe_load((REPO / cfg).read_text(encoding="utf-8"))
            old = yaml.safe_load((REPO / cfg.replace("_ministral", "")).read_text(encoding="utf-8"))
            assert new["rag"]["generators"] == ["ministral-3-3b", "ministral-3-8b"]
            assert not any(f.startswith(new["out_dir"] + "/") for f in frozen)
            for k in ("dataset", "retrieval", "budgets", "retrievers", "seed", "device", "query_threads"):
                assert new[k] == old[k], (cfg, k)
            assert {k: v for k, v in new["rag"].items() if k != "generators"} == {k: v for k, v in old["rag"].items() if k != "generators"}
            assert {s["config_id"] for s in sq.STAGES if s["config"] == cfg} == {f"{r['id']}+{g}" for r in old["retrievers"] for g in new["rag"]["generators"]}
    finally:
        sq.use_phase("run2")
