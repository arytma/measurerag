"""The shipped run records keep every count and name no other process or application (scripts/redact_records.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from redact_records import NEUTRAL, RECORDS, names_left, redact_line, redact_token  # noqa: E402


def test_a_contention_record_keeps_its_counts_and_loses_its_names():
    line = json.dumps({"stage": "s", "contention": {"n_samples": 10, "n_contended": 3, "contaminated": False,
                       "worst_foreign": {"max": 90.0, "max_comm": "Some App Helper (Renderer)", "sum": 120.5},
                       "contended_comms": ["Some App", "Other (Plugin)"]}})
    d = json.loads(redact_line(line))
    c = d["contention"]
    assert c["worst_foreign"] == {"max": 90.0, "max_comm": NEUTRAL, "sum": 120.5}
    assert c["contended_comms"] == [NEUTRAL] and c["n_samples"] == 10 and c["n_contended"] == 3
    empty = json.dumps({"contention": {"contended_comms": []}})
    assert redact_line(empty) == empty


def test_a_probe_snapshot_keeps_its_shares_and_memory_and_loses_its_lists():
    line = json.dumps({"record": "snapshot", "cpu_all_others_pct_of_a_core": 50.0, "memory_gb": {"available": 17.0},
                       "applications_open": ["Some App"], "remote_desktop_processes": ["x"],
                       "cpu_by_process": [["a", 30.0], ["b", 20.0]]})
    d = json.loads(redact_line(line))
    assert d == {"record": "snapshot", "cpu_all_others_pct_of_a_core": 50.0, "memory_gb": {"available": 17.0},
                 "cpu_busiest_other_pct_of_a_core": 30.0}


def test_a_token_keeps_its_share_and_loses_the_name():
    assert redact_token("k10 x+1.5b (83% of samples, mostly A (B)); k10 y (59% of samples, mostly z)") == \
        "k10 x+1.5b (83% of samples); k10 y (59% of samples)"


def test_no_shipped_record_or_token_names_another_process():
    for rel in RECORDS:
        p = REPO / rel
        if p.exists():
            assert not names_left(p.read_text(encoding="utf-8")), rel
    tokens = json.loads((REPO / "results/stats.json").read_text(encoding="utf-8"))["tokens"]
    assert not [k for k, v in tokens.items() if "% of samples, mostly" in v]
