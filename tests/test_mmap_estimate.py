"""Peak RSS without the double-counted mapped pages: the probe's surplus, the estimate beside the measured values,
and that no measured value moved.

llama-cpp-python maps a GGUF by default, and on the CPU llama.cpp also copies most weights into its own buffers, so
peak RSS counts that part of the weights twice. scripts/mmap_probe.py measured the surplus per model;
scripts/crossover.py subtracts it from every English Ministral 3 row and writes results/crossover/mmap_estimate.json
and the en_m3.est.* tokens (docs/LIMITATIONS.md)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

PROBE = REPO / "results/crossover/mmap_probe.jsonl"
ESTIMATE = REPO / "results/crossover/mmap_estimate.json"
STATS = REPO / "results/stats.json"
CROSSOVER = REPO / "results/crossover.json"

# The measured tokens and crossover.json as they stood before the estimate was added: neither may move. The tokens
# are pinned as they stand after other processes' names were removed from the shipped records (scripts/redact_records.py),
# which dropped one token that counted batches by a process's name and reworded two; no number changed.
MEASURED_TOKENS_SHA256 = "b0a9289ed743f5abb251ad33825306423ae915fdab0487a23fb487a0320cc834"
MEASURED_TOKENS_N = 2798
CROSSOVER_SHA256 = "49ed93b69a58c29223e2746b45f701f4632f997f5cd5fab4249c6a0ef8e1332f"


def _load(model, mmap, repeat, peak, answer="x"):
    return {"record": "load", "model": model, "use_mmap": mmap, "repeat": repeat, "peak_rss_gb": peak, "answer": answer}


# -- the rules, on made-up numbers ------------------------------------------
def test_the_surplus_is_the_median_of_the_paired_differences(tmp_path):
    from mmap_probe import surplus

    rows = [{"record": "machine"}]
    on = [4.5, 4.6, 4.4, 4.5, 9.0]          # one outlier: the median ignores it
    off = [3.2, 3.2, 3.2, 3.3, 3.2]
    for i, (a, b) in enumerate(zip(on, off), 1):
        rows += [_load("m-3b", True, i, a), _load("m-3b", False, i, b)]
    p = tmp_path / "probe.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    s = surplus(p)["m-3b"]
    diffs = sorted(round(a - b, 6) for a, b in zip(on, off))
    assert s["repeats"] == 5 and round(s["surplus_gb"], 6) == diffs[2]
    assert s["answers_identical"] is True


def test_the_guard_keeps_the_estimate_above_the_bound_and_the_higher_end_below_it():
    from crossover import guarded_estimate

    assert guarded_estimate(10.0, 3.0, 5.0) == (7.0, 7.0)          # above the bound: measured minus surplus
    assert guarded_estimate(6.0, 3.0, 5.0) == (3.0, 5.0)           # below it: the bound, the higher end
    assert guarded_estimate(6.0, 3.0, 7.0) == (3.0, 6.0)           # never above the measured peak


def test_a_merged_table_doubles_only_the_columns_that_differ():
    from crossover import ESTIMATE, MEASURED, md_merge, md_parse, md_table

    m = md_table(["p", "raw", "per GB"], [["0.2", "0.1", "0.02"], ["0.4", "0.2", "0.03"]])
    e = md_table(["p", "raw", "per GB"], [["0.2", "0.1", "0.04"], ["0.4", "0.2", "0.06"]])
    h, rows = md_parse(md_merge(m, e))
    assert h == ["p", "raw", f"per GB, {MEASURED}", f"per GB, {ESTIMATE}"]
    assert rows == [["0.2", "0.1", "0.02", "0.04"], ["0.4", "0.2", "0.03", "0.06"]]
    with pytest.raises(AssertionError):
        md_merge(m, md_table(["p", "raw"], [["0.2", "0.1"]]))


def test_the_sensitivity_table_gives_the_per_gb_rows_twice_and_the_per_ms_rows_once():
    from crossover import ESTIMATE, MEASURED, md_parse, md_table, sensitivity_merge

    h = ["pricing", "retriever cost", "generator cost", "g* isotonic", "g* linear"]
    m = md_table(h, [["per GB of peak RSS: a → b", "0.95", "5.07", "x", "y"], ["per ms of added latency: a → b", "9", "10", "z", "w"]])
    e = md_table(h, [["per GB of peak RSS: a → b", "0.95", "2.75", "x2", "y2"], ["per ms of added latency: a → b", "9", "10", "z", "w"]])
    _, rows = md_parse(sensitivity_merge(m, e))
    assert [r[0] for r in rows] == [f"per GB of peak RSS, {MEASURED}: a → b", f"per GB of peak RSS, {ESTIMATE}: a → b",
                                    "per ms of added latency: a → b"]


# -- the shipped record and estimate ----------------------------------------
def _rows():
    return json.loads(ESTIMATE.read_text(encoding="utf-8"))


def test_the_probe_record_ships_with_every_load_and_no_path_host_or_user():
    lines = [json.loads(x) for x in PROBE.read_text(encoding="utf-8").splitlines() if x.strip()]
    loads = [x for x in lines if x["record"] == "load"]
    assert {x["record"] for x in lines} == {"machine", "snapshot", "load"}
    assert len([x for x in lines if x["record"] == "snapshot"]) == 2
    assert sorted((x["model"], x["repeat"], x["use_mmap"]) for x in loads) == sorted(
        (m, i, b) for m in ("ministral-3-3b", "ministral-3-8b") for i in range(1, 6) for b in (True, False))
    text = PROBE.read_text(encoding="utf-8")
    assert "/Users/" not in text and "/home/" not in text and "/private/" not in text


def test_every_estimate_is_the_measured_peak_less_its_generators_surplus():
    from crossover import guarded_estimate
    from mmap_probe import surplus

    doc = _rows()
    s = surplus(PROBE)
    assert {k: v["surplus_gb"] for k, v in doc["probe"]["surplus"].items()} == pytest.approx({k: v["surplus_gb"] for k, v in s.items()})
    assert len(doc["rows"]) == 20
    for r in doc["rows"]:
        res = json.loads((REPO / r["sweep"] / "results.json").read_text(encoding="utf-8"))
        x = next(y for y in res["results"] if y["config_id"] == r["config_id"])
        assert r["measured_peak_rss_gb"] == x["measurement"]["peak_rss_gb"]
        est, used = guarded_estimate(r["measured_peak_rss_gb"], s[r["generator"]]["surplus_gb"], r["bound_peak_rss_gb"])
        assert r["measured_minus_surplus_gb"] == pytest.approx(est) and r["estimate_gb"] == pytest.approx(used)


def test_the_retriever_steps_gb_does_not_move_and_the_generator_steps_does():
    doc = _rows()
    measured = json.loads(CROSSOVER.read_text(encoding="utf-8"))["headline_families"]["ministral-3"]["datasets"]
    for name, d in doc["datasets"].items():
        m = measured[name]["costs"]
        assert [s["gb"] for s in d["costs"]["retriever_steps"]] == pytest.approx([s["gb"] for s in m["retriever_steps"]], abs=1e-6)
        assert d["costs"]["priced_retriever_step"]["gb"] == pytest.approx(m["priced_retriever_step"]["gb"], abs=1e-6)
        assert d["costs"]["generator_step"]["gb"] < m["generator_step"]["gb"]


def test_no_measured_value_moved():
    """crossover.json is byte-identical, and every token but the estimates keeps its value."""
    assert hashlib.sha256(CROSSOVER.read_bytes()).hexdigest() == CROSSOVER_SHA256
    tokens = json.loads(STATS.read_text(encoding="utf-8"))["tokens"]
    measured = {k: v for k, v in tokens.items() if not k.startswith("en_m3.est.")}
    assert len(measured) == MEASURED_TOKENS_N
    assert hashlib.sha256(json.dumps(measured, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == MEASURED_TOKENS_SHA256


def test_every_estimate_token_is_labelled_where_the_documents_use_it():
    """No document shows an affected figure's estimate without its label, or the measured value alone where an
    estimate exists: each template line or paragraph that uses an en_m3.est token also says what it is."""
    import re

    sys.path.insert(0, str(REPO / "scripts"))
    from fill import docs_config

    cfg = docs_config(REPO)
    tokens = json.loads(STATS.read_text(encoding="utf-8"))["tokens"]
    tables = {k for k, v in tokens.items() if k.startswith("en_m3.est.") and v.startswith("| ")}
    for rel in cfg["documents"]:
        path = REPO / cfg["templates"] / rel
        if not path.exists():
            continue
        for para in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
            used = re.findall(r"\{\{\s*(en_m3\.est\.[^}|\s]+)", para)
            prose = [u for u in used if u not in tables and not u.startswith("en_m3.est.probe.")
                     and not u.startswith("en_m3.est.guard.") and not u.endswith("_measured")]
            if prose:
                assert "estimate" in para and "double-counted" in para, (rel, prose)
