"""Does any SQAD text sit in the tree? Counts only; no text is printed or written.

    .venv/bin/python scripts/sqad_text_scan.py [--out <file>]

For every tracked file, gzipped records opened and JSON strings decoded: is there a question of either
SQAD version, a gold answer of four words or more, or a run of WINDOW words from any passage? SQAD text may
appear only in a generated answer inside the shipped answer files, under answers/:
there a quoted passage and a repeated question are counted, and scripts/czech_quote_scan.py records how
much. Everywhere else, an `answer` field included, every count must be zero. Needs the data on disk, so it
is a check before a release, not a reader's step. Exit 1 on any hit outside the shipped answers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

WINDOW = 8
LEAD = 4
WORD = re.compile(r"\w+", re.U)
SKIP_SUFFIXES = (".png", ".lock", ".tar.gz", ".bundle")
CORPORA = ("results/crossover/slices/czech30_testall", "results/crossover/slices/czech32_testall")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def grams(ws: list[str], n: int) -> set[int]:
    return {hash(tuple(ws[i: i + n])) for i in range(len(ws) - n + 1)}


def strings_in(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [*strings_in(k), *strings_in(v)]]
    if isinstance(value, list):
        return [s for v in value for s in strings_in(v)]
    return []


def read_file(path: Path) -> tuple[list[str], list[str]]:
    """(text outside answers, generated answers)"""
    raw = path.read_bytes()
    if path.name.endswith(".jsonl.gz"):
        raw = gzip.decompress(raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [], []
    other: list[str] = []
    answers: list[str] = []
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                other.append(line)
                continue
            if isinstance(row, dict) and "answer" in row:
                answers.append(row.pop("answer"))
            other.extend(strings_in(row))
    elif path.suffix == ".json":
        try:
            other.extend(strings_in(json.loads(text)))
        except json.JSONDecodeError:
            other.append(text)
    else:
        other.append(text)
    return other, answers


def parts_to_check(rel: str, other: list[str], answers: list[str]) -> tuple[list[str], list[str]]:
    """(strings that must hold no SQAD text, generated answers counted apart). Only the shipped answer files keep
    their answers apart; an `answer` field anywhere else is checked like any other string."""
    if rel.startswith("answers/"):
        return other, answers
    return other + answers, []


def main() -> int:
    from measurerag.datasets.base import Dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    passage_windows: set[int] = set()
    questions: set[tuple[str, ...]] = set()
    golds: set[tuple[str, ...]] = set()
    for rel in CORPORA:
        ds = Dataset.load_slice(REPO / rel)
        for d in ds.docs:
            passage_windows |= grams(words(d.indexable()), WINDOW)
        questions |= {tuple(words(q)) for q in ds.queries.values() if len(words(q)) >= LEAD}
        golds |= {tuple(words(a)) for al in ds.answers.values() for a in al if len(words(a)) >= LEAD}
    print(f"checking {len(questions):,} questions, {len(golds):,} gold answers of {LEAD} words or more, {len(passage_windows):,} passage windows of {WINDOW} words", flush=True)
    control = positive_control(questions, golds, passage_windows, ds)
    print("positive control:", control, flush=True)
    if not all(control.values()):
        print("sqad_text_scan: FAILED: the scan does not find text planted in a string")
        return 1

    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    out = {"window_words": WINDOW, "questions_checked": len(questions), "gold_answers_checked": len(golds), "passage_windows_checked": len(passage_windows), "files_checked": 0,
           "outside_answers": {"questions": 0, "gold_answers": 0, "passage_windows": 0, "files": []},
           "in_generated_answers": {"answers": 0, "whole_questions": 0, "answers_with_a_passage_window": 0}}
    for rel in tracked:
        path = REPO / rel
        if rel.endswith(SKIP_SUFFIXES) or not path.is_file():
            continue
        other, answers = parts_to_check(rel, *read_file(path))
        out["files_checked"] += 1
        hit = 0
        for s in other:
            ws = words(s)
            if len(ws) < LEAD:
                continue
            if len(ws) >= WINDOW:
                n = len(grams(ws, WINDOW) & passage_windows)
                hit += n
                out["outside_answers"]["passage_windows"] += n
            joined = " " + " ".join(ws) + " "
            for name, pool in (("questions", questions), ("gold_answers", golds)):
                for q in pool_hits(ws, joined, pool):
                    hit += 1
                    out["outside_answers"][name] += 1
        if hit:
            out["outside_answers"]["files"].append(rel)
        for a in answers:
            ws = words(a)
            out["in_generated_answers"]["answers"] += 1
            if len(ws) >= WINDOW and grams(ws, WINDOW) & passage_windows:
                out["in_generated_answers"]["answers_with_a_passage_window"] += 1
            if len(ws) >= LEAD:
                out["in_generated_answers"]["whole_questions"] += sum(1 for _ in pool_hits(ws, " " + " ".join(ws) + " ", questions))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    o = out["outside_answers"]
    bad = o["questions"] or o["gold_answers"] or o["passage_windows"]
    print("sqad_text_scan:", "FAILED" if bad else "OK: no SQAD question, gold answer or passage run outside the shipped answers")
    return 1 if bad else 0


def positive_control(questions: set, golds: set, passage_windows: set, ds) -> dict[str, bool]:
    """Zero hits mean something only if a hit would be found: plant each kind of text in a string."""
    def found(item: tuple[str, ...], pool: set) -> bool:
        ws = ["před", *item, "po"]
        return any(True for _ in pool_hits(ws, " " + " ".join(ws) + " ", pool))

    question = next(iter(sorted(questions)))
    gold = next(iter(sorted(golds)))
    passage = next(words(d.indexable()) for d in ds.docs if len(words(d.indexable())) >= 3 * WINDOW)
    planted = ["před", *passage[WINDOW: 2 * WINDOW], "po"]
    return {"question": found(question, questions), "gold_answer": found(gold, golds), "passage_window": bool(grams(planted, WINDOW) & passage_windows),
            "no_false_hit": not grams(["slovo"] * (2 * WINDOW), WINDOW) & passage_windows}


_LEADS: dict[int, tuple[set, dict[tuple[str, ...], list[tuple[str, ...]]]]] = {}


def pool_hits(ws: list[str], joined: str, pool: set[tuple[str, ...]]):
    """Which items of the pool sit in this string, found through their first words."""
    entry = _LEADS.get(id(pool))
    if entry is None or entry[0] is not pool:      # holding the pool keeps its id from being reused
        table: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        for item in pool:
            table.setdefault(item[:LEAD], []).append(item)
        entry = _LEADS[id(pool)] = (pool, table)
    for lead in {tuple(ws[i: i + LEAD]) for i in range(len(ws) - LEAD + 1)}:
        for item in entry[1].get(lead, ()):
            if " " + " ".join(item) + " " in joined:
                yield item


if __name__ == "__main__":
    sys.exit(main())
