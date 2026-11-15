"""The GX10's pilot, read by three conditions fixed before it ran. Generates nothing.

    .venv/bin/python scripts/czech_pilot_check.py            # writes results/czech/complete/pilot_gx10.json; exit 0 when all three hold
    .venv/bin/python scripts/czech_pilot_check.py --export   # the repeat's answers and the pilot's rankings, as records
    .venv/bin/python scripts/czech_pilot_check.py --check    # recompute both results from the shipped records alone, byte for byte
    .venv/bin/python scripts/czech_pilot_check.py --from-records   # the same, and write them

Reads the GX10's dev caches (V1 twice, each from an empty cache), its compute record, the Mac's dev answers and compute
record, the pilot rankings and the shipped rankings. Also writes the GX10's dev answers beside the Mac's
(answers/sqad32/answers_dev_gx10.jsonl.gz, results/czech/dev/selection_gx10.json). `--check` reads instead
answers/sqad32/answers_dev_gx10.jsonl.gz, answers/sqad32/answers_dev_gx10_repeat.jsonl.gz, the pilot's top 10 in
results/czech/records/pilot_rankings_gx10.jsonl.gz (ids only) and the shipped rankings in results/czech/records/. The
seconds per answer are each machine's record of its own run. Prints no question and no passage.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import czech_dev_prompts as dev  # noqa: E402
from czech_complete_queue import COMPLETE, MACHINES, PILOT_RANKINGS, RANK_32, RANK_COPY, STATE  # noqa: E402
from measurerag.crossover.stats import bootstrap_counts, established, interval  # noqa: E402
from measurerag.czech_store import load_labels, read_jsonl_gz, write_jsonl_gz  # noqa: E402

OUT = REPO / "results/czech/complete/pilot_gx10.json"
REPEAT_ROWS = REPO / "answers/sqad32/answers_dev_gx10_repeat.jsonl.gz"
PILOT_RANKS = REPO / "results/czech/records/pilot_rankings_gx10.jsonl.gz"
POPULATION = {"released": "copy_test", "sqad32_full": "sqad32_test"}
MAX_SHARE = 0.2          # condition 2: at most a fifth of the Mac's seconds per answer
TOP = 10
SHIPPED = {"released": RANK_COPY, "sqad32_full": RANK_32}


def by_key(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["q"], r["variant"], r["config"]): r for r in rows}


def repeat_check(first: list[dict], again: list[dict]) -> dict:
    """Condition 1: the same string for every answer, with each generator."""
    a, b = by_key([r for r in first if r["variant"] == "v1"]), by_key(again)
    out = {}
    for g in dev.GENERATORS:
        keys = sorted(k for k in a if k[2] == f"e5-small+{g}")
        out[g] = {"answers": len(keys), "same_string": sum(a[k]["answer"] == b[k]["answer"] for k in keys)}
    return out


def rate(state: dict) -> dict[str, float]:
    """Seconds per answer on the dev set, by generator kind."""
    sums: dict[str, list[float]] = {}
    for name, s in state["stages"].items():
        if name.startswith("dev_") and s.get("generations") and s.get("kind") in ("dial_1.5b", "dial_3b"):
            x = sums.setdefault(s["kind"], [0.0, 0])
            x[0] += float(s["seconds"])
            x[1] += int(s["generations"])
    return {k: v[0] / v[1] for k, v in sums.items()}


def pilot_rows() -> list[dict]:
    """The pilot's rankings, as records: each question's top 10 per retriever. Ids only."""
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset

    rows = []
    for key, slice_ in PILOT_RANKINGS.items():
        qids = sorted(Dataset.load_slice(REPO / slice_).queries)
        for rid in ("bm25", "e5-small", "retromae-cs"):
            run = load_run(REPO / COMPLETE / f"pilot_rank_{key}", rid)["run"]
            rows += [{"set": key, "retriever": rid, "q": q, "top": list(run[q][:TOP])} for q in qids]
    return rows


def ranking_check_from_records() -> dict:
    """Condition 3 from the shipped records: the pilot's top 10 against the shipped rankings' top 10."""
    from measurerag.czech_store import load_rankings

    rows = read_jsonl_gz(PILOT_RANKS)
    out: dict = {}
    for key in PILOT_RANKINGS:
        shipped = load_rankings(REPO / "results/czech/records", POPULATION[key])
        out[key] = {}
        for rid in ("bm25", "e5-small", "retromae-cs"):
            mine = [r for r in rows if r["set"] == key and r["retriever"] == rid]
            out[key][rid] = {"questions": len(mine), "top_10_differs": sum(r["top"] != shipped[r["q"]][rid][:TOP] for r in mine),
                             "top_3_differs": sum(r["top"][:3] != shipped[r["q"]][rid][:3] for r in mine)}
    return out


def ranking_check() -> dict:
    """Condition 3: every pilot ranking's top 10 is the shipped one, on every question of the two sets."""
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset

    out: dict = {}
    for key, slice_ in PILOT_RANKINGS.items():
        qids = sorted(Dataset.load_slice(REPO / slice_).queries)
        out[key] = {}
        for rid in ("bm25", "e5-small", "retromae-cs"):
            pilot, shipped = load_run(REPO / COMPLETE / f"pilot_rank_{key}", rid)["run"], load_run(REPO / SHIPPED[key], rid)["run"]
            out[key][rid] = {"questions": len(qids), "top_10_differs": sum(pilot[q][:TOP] != shipped[q][:TOP] for q in qids),
                             "top_3_differs": sum(pilot[q][:3] != shipped[q][:3] for q in qids)}
    return out


def ci(point: float, boot: np.ndarray) -> dict:
    lo, hi = interval(boot)
    return {"point": float(point), "lo": lo, "hi": hi, "established": established(lo, hi)}


def agreement(gx10: list[dict], mac: list[dict]) -> dict:
    """Reported, not a condition: the same prompts on both machines, paired."""
    a, b = by_key(gx10), by_key(mac)
    out: dict = {}
    for v in dev.VARIANTS:
        for g in dev.GENERATORS:
            keys = sorted(k for k in a if k[1] == v and k[2] == f"e5-small+{g}")
            if not keys or any(k not in b for k in keys):
                continue
            counts = bootstrap_counts(len(keys), 10000, "czech-v030", "pilot", v, g).astype(float)
            block: dict = {"answers": len(keys), "same_string": sum(a[k]["answer"] == b[k]["answer"] for k in keys)}
            for field in ("em_yes_no_mapped", "em"):
                d = np.array([a[k][field] - b[k][field] for k in keys], dtype=float)
                block[field] = {"gx10": float(np.mean([a[k][field] for k in keys])), "mac": float(np.mean([b[k][field] for k in keys])), "gx10_minus_mac": ci(float(d.mean()), (counts @ d) / len(keys))}
            out[f"{v} {g}"] = block
    return out


def dumped(obj: dict) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="write the repeat's answers and the pilot's rankings as records, from the caches")
    ap.add_argument("--check", action="store_true", help="recompute from the shipped records alone and compare byte for byte")
    ap.add_argument("--from-records", action="store_true", help="recompute from the shipped records alone and write the results")
    args = ap.parse_args()
    records = args.check or args.from_records
    os.chdir(REPO)
    if args.export:
        again = dev.export_rows(caches=lambda v: dev.cache_of("v1_repeat"), variants=("v1",))
        write_jsonl_gz(REPEAT_ROWS, again)
        write_jsonl_gz(PILOT_RANKS, pilot_rows())
        print("wrote", REPEAT_ROWS.relative_to(REPO), "and", PILOT_RANKS.relative_to(REPO))
        return 0
    gx10_state = json.loads((REPO / MACHINES["gx10"]["state"]).read_text(encoding="utf-8"))
    mac_state = json.loads((REPO / STATE).read_text(encoding="utf-8"))
    rows_path, sel_path = dev.paths("gx10")
    if records:
        gx10, again = read_jsonl_gz(rows_path), read_jsonl_gz(REPEAT_ROWS)
    else:
        gx10 = dev.export_rows()
        again = dev.export_rows(caches=lambda v: dev.cache_of("v1_repeat"), variants=("v1",))
        write_jsonl_gz(rows_path, gx10)
    labels = load_labels(REPO / "results/czech/records", "sqad32_test")
    selection = dev.analyse(gx10, labels)
    selection["note"] = "The GX10's answers. The prompt was chosen on the Mac's (results/czech/dev/selection.json); this result chooses nothing."
    if not args.check:
        sel_path.write_text(dumped(selection), encoding="utf-8")

    rep = repeat_check(gx10, again)
    rates = {"gx10": rate(gx10_state), "mac": rate(mac_state)}
    shares = {k: rates["gx10"][k] / rates["mac"][k] for k in rates["gx10"] if k in rates["mac"]}
    ranks = ranking_check_from_records() if records else ranking_check()
    cond = {
        "1_the_repeat_gives_the_same_string_for_every_answer": all(x["same_string"] == x["answers"] > 0 for x in rep.values()),
        "2_at_most_a_fifth_of_the_macs_seconds_per_answer": len(shares) == 2 and all(s <= MAX_SHARE for s in shares.values()),
        "3_every_pilot_ranking_top_10_is_the_shipped_one": all(r["top_10_differs"] == 0 for by in ranks.values() for r in by.values()),
    }
    out = {"conditions": cond, "go": all(cond.values()), "repeat": rep, "seconds_per_answer": rates, "share_of_the_macs_time": shares, "rankings": ranks,
           "agreement_with_the_mac": agreement(gx10, read_jsonl_gz(dev.ROWS)), "rule_on_the_gx10_answers": {"first_rule": selection["first_rule"], "selection": selection["selection"]},
           "rule_on_the_macs_answers": json.loads(dev.OUT.read_text(encoding="utf-8"))["selection"]}
    if args.check:
        same = OUT.read_text(encoding="utf-8") == json.dumps(out, indent=1, sort_keys=True) + "\n" and sel_path.read_text(encoding="utf-8") == dumped(selection)
        print("czech_pilot_check:", "OK: the pilot recomputes from the shipped records" if same else "DIFFERENT")
        return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": cond, "go": out["go"], "share_of_the_macs_time": shares}, indent=1))
    return 0 if out["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
