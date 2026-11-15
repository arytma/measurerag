"""Every number in the public text, with the file or the script that recomputes it.

    .venv/bin/python scripts/number_audit.py [--out <file>]

Reads the templates, so it sees which token each number came from. A number written directly in a
template is traced by value, as scripts/check_docs.py does, or is not a result (a small count, a year,
a date, a version, a list number). Exit 1 if a number has no source, or if a token has no known route.

The figures' text is drawn as paths, so it is not read here; instead every value a figure draws is listed
in results/figures/figures.json with its file and JSON path (scripts/figures.py), and each one is read
again from that file and must be equal.

The routes are the release's rule:
    tree         recomputes from the tree alone, by the script named
    download     recomputes from the tree plus a file the reader downloads and checks by MD5
    rerun        English and CsFEVER: in a results file in the tree, recomputed by running the shipped
                 config against the original downloads, with corpus hashes rather than MD5
    setting      a setting of the method, in a config or in the code: not a measurement
    frozen       a dated record of what happened on one machine, labelled as such
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from check_docs import ANY_NUMBER, NUMBER, RANGE, czech_documents, exempt_number, from_czech_numbers, norm, prose_of  # noqa: E402
from fill import docs_config, token_name  # noqa: E402

TOKEN = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
# Numbers in words that are not results, and would otherwise match some unrelated token by value.
NOT_RESULTS = (
    (re.compile(r"\bSQAD\s+3\.[02]\b"), "a dataset's version"),
    (re.compile(r"\b95%"), "the level of the intervals, a setting of the method"),
    (re.compile(r"\bfirst 50 characters\b"), "the debias rule, `first` in src/measurerag/datasets/sqad32.py"),
    (re.compile(r"\bv0\.[23]\.0\b"), "a version of this release"),
)
RESCORE = "scripts/rescore_czech.py --check"
ARCHIVE = "scripts/sqad_copy_vs_archive.py --archive <SQAD 3.0 archive>"
CROSSOVER = "scripts/crossover.py, from the shipped configs and the downloads (docs/REPRODUCE.md section 5)"

PILOT = "scripts/czech_pilot_check.py --check"
BENCH = "scripts/czech_model_bench.py --check"
BENCH_M = "scripts/czech_model_bench_mistral.py --check"
QUEUES = "results/czech/complete/compute.json, results/czech/complete/compute_gx10.json"
# Checked before the prefixes: a token's kind can sit at its end. (pattern, route, file, how)
PATTERN_ROUTES: tuple[tuple[str, str, str, str], ...] = (
    (r"\.gpu_seconds$", "frozen", "results/czech/model_bench/", "the GX10's record of its own benchmark run"),
    (r"^cz\.[a-z0-9_]+\.(retriever|generator)_detects$", "tree", "results/czech/plan/plan.json", "scripts/czech_plan.py --check, byte for byte on macOS only"),
    (r"^cz\.table_power$", "tree", "results/czech/plan/plan.json and results/czech/analysis.json", "scripts/czech_plan.py --check (macOS only) and " + RESCORE),
)
# First match wins. (prefix, route, file, how)
ROUTES: tuple[tuple[str, str, str, str], ...] = (
    # Ministral 3, the family whose English sweeps were timed: its own record of the machine, its own sweeps, and the
    # same analysis as every English number, under its prefix (configs/crossover.yaml, headline_families)
    # the estimate without the double-counted mapped pages: the probe is a record of one machine; the estimates are the
    # same analysis on the rows' peaks less the probe's surplus, written beside the measured values
    ("en_m3.est.probe.", "frozen", "results/crossover/mmap_probe.jsonl", "the Mac's record of its own probe run (scripts/mmap_probe.py)"),
    ("en_m3.est.", "rerun", "results/crossover/mmap_estimate.json", CROSSOVER + ", with the probe's record results/crossover/mmap_probe.jsonl"),
    ("en_m3.hygiene.", "frozen", "results/crossover/calibration_ministral/", "a record of the machine during Ministral 3's timed sweeps"),
    ("en_m3.budget.", "rerun", "results/crossover/natural/*_ministral/results.json", "scripts/sweep_queue.py --phase ministral --no-hygiene"),
    ("en_m3.", "rerun", "results/crossover.json, block headline_families", CROSSOVER),
    ("cz.pilot.seconds_", "frozen", QUEUES, "each machine's record of its own dev-set run"),
    ("cz.pilot.share.", "frozen", QUEUES, "each machine's record of its own dev-set run"),
    ("cz.pilot.", "tree", "results/czech/complete/pilot_gx10.json", PILOT),
    ("cz.table_pilot_", "tree", "results/czech/complete/pilot_gx10.json", PILOT),
    ("cz.bench.ministral3", "tree", "results/czech/model_bench/mistral.json", BENCH_M),
    ("cz.table_model_bench_ministral", "tree", "results/czech/model_bench/mistral.json", BENCH_M + "; its seconds per answer are the GX10's record"),
    ("cz.table_model_bench", "tree", "results/czech/model_bench/bench.json", BENCH + "; its seconds per answer are the GX10's record"),
    ("cz.bench.", "tree", "results/czech/model_bench/bench.json, results/czech/model_bench/mistral.json", BENCH + " and " + BENCH_M),
    ("cz.dev.", "tree", "results/czech/dev/selection.json", "scripts/czech_dev_prompts.py --check"),
    ("cz.table_dev", "tree", "results/czech/dev/selection.json", "scripts/czech_dev_prompts.py --check"),
    ("cz.compute.", "frozen", QUEUES, "the queues' records of their own runs"),
    ("cz.quotes.", "download", "results/czech/quote_scan.json", "scripts/czech_quote_scan.py --out results/czech/quote_scan.json, with the slices built from the SQAD downloads (docs/REPRODUCE.md section 4)"),
    ("czr.", "tree", "results/czech/v040/retrievers.json", "scripts/czech_retrievers.py --check"),
    ("cz.", "tree", "results/czech/analysis.json", RESCORE),
    ("stefanik.", "setting", "src/measurerag/datasets/sqad32.py", "Štefánik et al. (2023)'s filter as their section 3 states it (STEFANIK_2023_FILTER): a citation, not a measurement"),
    ("bcm.", "setting", "src/measurerag/datasets/sqad32.py", "BenCzechMark's filter as its Appendix B.7 states it (BENCZECHMARK_FILTER): a citation, not a measurement"),
    ("czech_checks.archive_records_stated", "frozen", "results/czech_checks.json", "what LINDAT's page for the item states, read on 10 Nov 2026; beside the count the archive gives"),
    ("czech_checks.", "download", "results/czech_checks.json", ARCHIVE),
    ("czech.", "tree", "results/crossover.json, block czech", RESCORE),
    ("headline.czech.", "tree", "results/crossover.json, block czech", RESCORE),
    ("runs.cz_rag.max_tokens", "setting", "configs/cz_rag.yaml", "a setting of the method"),
    ("runs.cz_rag.k", "setting", "configs/cz_rag.yaml", "a setting of the method"),
    ("runs.cz_rag.", "tree", "results/cz_rag/results.json", RESCORE + " (bench_check)"),
    ("runs.cz_rag_32.", "tree", "results/cz_rag_32/results.json", RESCORE + " (bench_check)"),
    ("data.czech.", "tree", "results/crossover.json, block czech", RESCORE),
    ("mtcheck.czech.", "tree", "results/crossover/max_tokens_check_czech.json", RESCORE + " (token_cap_check on released_v020)"),
    ("runs.cz_small.", "rerun", "results/cz_small/results.json", "measurerag run --config configs/cz_small.yaml"),
    ("runs.cz_full.", "rerun", "results/cz_full/results.json", "measurerag run --config configs/cz_full.yaml"),
    ("runs.en_", "rerun", "results/en_*/results.json", "measurerag run --config configs/en_*.yaml"),
    ("runs.crossover/", "rerun", "results/crossover/natural/*/results.json", "scripts/sweep_queue.py"),
    ("runs.", "rerun", "results/crossover/natural/*/results.json", "scripts/sweep_queue.py"),
    ("hotpotqa.", "rerun", "results/crossover.json, block hotpotqa", CROSSOVER),
    ("nq.", "rerun", "results/crossover.json, block nq", CROSSOVER),
    ("headline.", "rerun", "results/crossover.json", CROSSOVER),
    ("agree", "rerun", "results/crossover.json, block agreement", CROSSOVER),
    ("table_agreement", "rerun", "results/crossover.json, block agreement", CROSSOVER),
    ("priced_step", "rerun", "results/crossover.json", CROSSOVER),
    ("mtcheck.", "rerun", "results/crossover/max_tokens_check.json", "scripts/max_tokens_check.py"),
    ("remeasure.", "rerun", "results/stats.json", CROSSOVER),
    ("repro.", "rerun", "results/crossover/repro_check.json", "scripts/repro_check.py"),
    ("run1.", "rerun", "results/en_rag/results.json", "measurerag run --config configs/en_rag.yaml"),
    ("data.", "rerun", "slice meta.json, results/crossover.json", "scripts/make_slice.py"),
    ("budget.", "rerun", "results/crossover.json", CROSSOVER),
    ("gens", "frozen", "results/stats.json", "counts the answers cached on the machine that ran them"),
    ("hygiene.", "frozen", "results/crossover/calibration/", "a record of the machine during the timed sweeps"),
    ("machine.", "frozen", "results/stats.json", "the machine the timings were taken on"),
    ("models.", "setting", "results/crossover/models_manifest.json", "scripts/fetch_verify_models.py"),
    ("design.", "setting", "configs/crossover.yaml", "a setting of the method"),
    ("max_tokens", "setting", "configs/crossover.yaml", "a setting of the method"),
    ("n_boot", "setting", "configs/crossover.yaml", "a setting of the method"),
)


FIGURES = "results/figures/figures.json"
# the files a figure may read, and how each recomputes
FIGURE_SOURCES = {"results/czech/analysis.json": ("tree", RESCORE), "results/crossover.json": ("rerun", CROSSOVER)}


def audit_figures(repo: Path) -> tuple[dict, list[str]]:
    """Each value results/figures/figures.json says a figure draws, read again from its source."""
    path = repo / FIGURES
    if not path.exists():
        return {}, [f"{FIGURES} is missing (scripts/figures.py)"]
    m = json.loads(path.read_text(encoding="utf-8"))
    cache: dict = {}
    out: dict = {}
    problems: list[str] = []
    for name, fig in sorted(m["figures"].items()):
        for f in fig["files"]:
            if not (path.parent / f).exists():
                problems.append(f"{FIGURES}: {name}: {f} is missing")
        traced = 0
        for v in fig["values"]:
            src = v["source"]
            if src not in FIGURE_SOURCES:
                problems.append(f"{FIGURES}: {name}: {src} is not a file a figure may read")
                continue
            if src not in cache:
                cache[src] = json.loads((repo / src).read_text(encoding="utf-8"))
            x = cache[src]
            try:
                for k in v["path"]:
                    x = x[k]
            except (KeyError, IndexError, TypeError):
                problems.append(f"{FIGURES}: {name}: {src} has no {v['path']}")
                continue
            if x != v["value"] or isinstance(x, bool):
                problems.append(f"{FIGURES}: {name}: {src} {v['path']} is {x!r}; the figure drew {v['value']!r}")
                continue
            traced += 1
        out[name] = {"files": fig["files"], "values_traced": traced,
                     "sources": {s: {"route": FIGURE_SOURCES[s][0], "how": FIGURE_SOURCES[s][1]} for s in fig["sources"] if s in FIGURE_SOURCES}}
    listed = {f for fig in m["figures"].values() for f in fig["files"]} | {path.name}
    for p in sorted(path.parent.iterdir()):
        if p.is_file() and p.name not in listed:
            problems.append(f"{FIGURES}: {p.name} is in {path.parent.relative_to(repo)}/ but no figure lists it")
    return out, problems


def route_of(token: str) -> tuple[str, str, str] | None:
    for pattern, route, file, how in PATTERN_ROUTES:
        if re.search(pattern, token):
            return route, file, how
    for prefix, route, file, how in ROUTES:
        if token.startswith(prefix):
            return route, file, how
    return None


def numbers_in(value: str) -> list[str]:
    """The numbers a rendered token holds. Identifiers (a hash, a revision) hold none worth listing."""
    return [] if re.fullmatch(r"[0-9a-f]{16,}", value) else ANY_NUMBER.findall(value)


def audit(repo: Path) -> dict:
    cfg = docs_config(repo)
    stats = json.loads((repo / cfg["stats"]).read_text(encoding="utf-8"))["tokens"]
    exempt = cfg.get("number_rule_exempt") or {}
    small = cfg.get("small_integer_max", 10)
    from_heading = cfg.get("number_rule_from_heading") or {}
    traced = {norm(n): k for k, v in sorted(stats.items()) for n in ANY_NUMBER.findall(str(v))}
    tokens_seen: dict[str, dict] = {}
    literal: list[dict] = []
    problems: list[str] = []
    documents: dict[str, dict] = {}
    for rel in cfg["documents"]:
        if rel in exempt:
            documents[rel] = {"status": "a dated record or attribution text: outside the rule", "why": " ".join(str(exempt[rel]).split())}
            continue
        path = repo / cfg["templates"] / rel
        if not path.exists():
            documents[rel] = {"status": "no template"}
            continue
        text = path.read_text(encoding="utf-8")
        start = from_heading.get(rel)
        frozen_part = 0
        if start:
            i = text.index(start)
            frozen_part = text[:i].count("\n")
            text = "\n" * frozen_part + text[i:]
        counts = Counter()
        for ln, line in enumerate(text.split("\n"), 1):
            for tok in map(token_name, TOKEN.findall(line)):     # {{a.b|cs}} is a.b, with Czech number marks
                r = route_of(tok)
                if tok not in stats:
                    problems.append(f"{rel}:{ln}: token {tok} is not in {cfg['stats']}")
                    continue
                if not numbers_in(str(stats[tok])):
                    counts["tokens without a number"] += 1
                    continue
                if r is None:
                    problems.append(f"{rel}:{ln}: token {tok} has no route")
                    continue
                e = tokens_seen.setdefault(tok, {"token": tok, "value": str(stats[tok])[:160], "route": r[0], "file": r[1], "how": r[2], "used_in": []})
                if rel not in e["used_in"]:
                    e["used_in"].append(rel)
                counts[r[0]] += 1
        prose = TOKEN.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        if rel in czech_documents(cfg):
            prose = from_czech_numbers(prose)
        for pattern, _why in NOT_RESULTS:
            prose = pattern.sub(lambda m: " " + "\n" * m.group(0).count("\n"), prose)
        prose = prose_of(prose)
        for ln, line in enumerate(prose.split("\n"), 1):
            for m in NUMBER.finditer(RANGE.sub(r"\1 to \2", line)):
                raw = m.group(1).rstrip(",").lstrip("+-−")
                if not raw or exempt_number(raw, small):
                    continue
                if norm(raw) in traced:
                    r = route_of(traced[norm(raw)])
                    literal.append({"document": rel, "line": ln, "number": raw, "traced_by_value_to": traced[norm(raw)], "route": r[0] if r else "none"})
                    counts["written directly, traced by value"] += 1
                else:
                    problems.append(f"{rel}:{ln}: the number {raw} has no source")
        documents[rel] = {"status": "under the rule", "numbers": dict(sorted(counts.items()))}
        if frozen_part:
            documents[rel]["frozen_lines_before_the_rule"] = frozen_part
    figures, figure_problems = audit_figures(repo)
    problems += figure_problems
    routes = Counter(e["route"] for e in tokens_seen.values())
    return {"figures": figures, "note": "Every number in the public text and how it recomputes. Written by scripts/number_audit.py.",
            "left_out_as_not_results": [why for _, why in NOT_RESULTS],
            "tokens_by_route": dict(sorted(routes.items())), "documents": documents,
            "tokens": sorted(tokens_seen.values(), key=lambda e: (e["route"], e["token"])), "written_directly": literal, "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = audit(REPO)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in report["problems"]:
        print("number_audit:", p)
    print("number_audit: tokens by route", report["tokens_by_route"], f"; {len(report['written_directly'])} numbers written directly and traced by value")
    figs = report["figures"]
    print(f"number_audit: figures: {sum(f['values_traced'] for f in figs.values())} values in {len(figs)} figures, each read again from its source")
    print("number_audit:", "FAILED" if report["problems"] else "OK: every number has a source")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
