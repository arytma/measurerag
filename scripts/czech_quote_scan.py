"""How much SQAD text do the generated answers quote? Counts only; no text is printed or written.

    .venv/bin/python scripts/czech_quote_scan.py [--out <file>]

For every row of answers/*/answers_*.jsonl.gz: the answer's length, whether it holds a
whole question of its corpus, and the longest run of it that appears verbatim in a passage shown to
the generator. Needs the data (the slices on disk), so it is a check before a release, not a reader's
step. A row is `long` above LONG_CHARS of verbatim quotation, the line this release draws between a
short span and a passage. It also lists, by id only, each row holding a whole question and each
answer past that line, and writes results/czech/quote_scan.json, which the public text quotes.
"""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

LONG_CHARS = 200


def longest_quote(answer: str, passages: list[str]) -> int:
    best = 0
    for p in passages:
        if len(answer) <= best:
            break
        m = SequenceMatcher(None, answer, p, autojunk=False).find_longest_match(0, len(answer), 0, len(p))
        best = max(best, m.size)
    return best


def generator_of(config: str) -> str:
    return config.split("+", 1)[1]


def scan(rows: list[dict], docs: dict[str, str], questions: dict[str, str]) -> dict:
    lengths, quotes = [], []
    whole, long_ = [], {}
    texts = set(questions.values())
    seen: dict[tuple, int] = {}
    for r in rows:
        a = r["answer"]
        lengths.append(len(a))
        if len(a) >= 15 and any(q in a for q in texts):
            whole.append({"q": r["q"], "config": r["config"], "p": r.get("p"), "em": r["em"], "own_question": questions.get(r["q"], "\0") in a})
        key = (a, tuple(r["context"]))
        if key in seen:
            seen[key] += 1
            continue
        seen[key] = 1
        q = longest_quote(a, [docs[d] for d in r["context"] if d in docs]) if len(a) >= 20 else 0
        quotes.append(q)
        if q > LONG_CHARS:
            long_[key] = {"q": r["q"], "config": r["config"], "quote_chars": q, "em": r["em"]}
    for key, v in long_.items():
        v["rows"] = seen[key]
    lengths.sort()
    quotes.sort()
    return {"rows": len(rows), "distinct_answers_with_context": len(seen), "answer_chars_median": lengths[len(lengths) // 2], "answer_chars_p99": lengths[int(len(lengths) * 0.99)], "answer_chars_max": lengths[-1],
            "longest_quote_p99": quotes[int(len(quotes) * 0.99)], "longest_quote_max": quotes[-1], "rows_holding_a_whole_question": len(whole), f"distinct_answers_quoting_over_{LONG_CHARS}_chars": len(long_),
            "whole_question_rows": sorted(whole, key=lambda x: (x["q"], x["config"], x["p"] or 0)), "answers_over_the_line": sorted(long_.values(), key=lambda x: (x["q"], x["config"]))}


def summary(report: dict) -> dict:
    """What the public text quotes, over every shipped answers file."""
    whole = [dict(x, file=k) for k, v in report["sets"].items() for x in v["whole_question_rows"]]
    over = [dict(x, file=k) for k, v in report["sets"].items() for x in v["answers_over_the_line"]]
    return {"longest_quote": max(report["longest_quote_by_source"].values()),
            "rows_holding_a_whole_question": len(whole),
            "questions_repeated": len({x["q"] for x in whole}),
            "every_one_its_own_question": all(x["own_question"] for x in whole),
            "generators_repeating": sorted({generator_of(x["config"]) for x in whole}),
            "em_of_those_rows": sorted({x["em"] for x in whole}),
            "answers_over_the_line": len(over),
            "rows_over_the_line": sum(x["rows"] for x in over),
            "generators_over_the_line": sorted({generator_of(x["config"]) for x in over}),
            "em_over_the_line": sorted({x["em"] for x in over})}


def main() -> int:
    from czech_records import SETS
    from measurerag.czech_store import ANSWERS, read_jsonl_gz
    from measurerag.datasets.base import Dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report: dict = {"long_chars": LONG_CHARS, "sets": {}}
    questions: dict[str, set[str]] = {}
    docs: dict[str, dict[str, str]] = {}

    def corpus_of(source: str) -> str:
        """Every set of a source shares its corpus's passages. The whole test split names its questions."""
        corpus = "copy_test" if source == "sqad30_copy" else "sqad32_test"
        if corpus not in docs:
            one = next(s for s in SETS.values() if s["source"] == source)
            docs[corpus] = {d.doc_id: d.indexable()[:1200] for d in Dataset.load_slice(REPO / one["slice"]).docs}
            all_q = Dataset.load_slice(REPO / ("results/crossover/slices/czech30_testall" if corpus == "copy_test" else "results/crossover/slices/czech32_testall"))
            questions[corpus] = dict(all_q.queries)
        return corpus

    # every answers file that ships, each against the passages of its own source
    for path in sorted((REPO / ANSWERS).rglob("*.jsonl.gz")):
        source = path.relative_to(REPO / ANSWERS).parts[0]
        corpus = corpus_of(source)
        name = str(path.relative_to(REPO / ANSWERS))
        report["sets"][name] = scan(read_jsonl_gz(path), docs[corpus], questions[corpus])
        print(name, report["sets"][name], flush=True)
    longest = {source: max(v["longest_quote_max"] for k, v in report["sets"].items() if k.startswith(source + "/")) for source in {k.split("/")[0] for k in report["sets"]}}
    report["longest_quote_by_source"] = longest
    report["summary"] = summary(report)
    print("longest", longest, flush=True)
    print("summary", report["summary"], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
