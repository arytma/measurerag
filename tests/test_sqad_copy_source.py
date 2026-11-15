"""Where the SQAD 3.0 copy comes from: the row-by-row comparison, on made-up rows, and the shipped record against the
copy's own counts. No network, no SQAD text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import sqad_copy_source as cs  # noqa: E402


def row(q: str, c: str, a: list[str]) -> dict:
    return {"question": q, "context": c, "answers": sorted(a)}


def test_identical_rows_compare_identical_and_any_difference_shows():
    copy = {"1": row("q1", "c1", ["a"]), "2": row("q2", "c2", ["b", "c"])}
    assert cs.identical(cs.compare(copy, dict(copy)))
    changed = dict(copy, **{"2": row("q2", "c2 edited", ["b", "c"])})
    c = cs.compare(copy, changed)
    assert not cs.identical(c) and c["same_context"] == 1 and c["same_question"] == 2
    extra = dict(copy, **{"3": row("q3", "c3", ["d"])})
    c = cs.compare(copy, extra)
    assert not cs.identical(c) and c["only_in_source"] == 1 and c["ids_in_common"] == 2


def test_the_shipped_record_covers_every_row_of_the_copy():
    rec = json.loads((REPO / "results/czech/copy_source.json").read_text(encoding="utf-8"))
    checks = json.loads((REPO / "results/czech_checks.json").read_text(encoding="utf-8"))
    assert rec["identical_row_for_row"] is True
    assert all(cs.identical(c) for c in rec["splits"].values())
    assert sum(c["copy_rows"] for c in rec["splits"].values()) == checks["copy"]["rows"]
    assert rec["source"] == {"repo_id": cs.SOURCE, "revision": cs.SOURCE_REVISION}
