"""Choose the Czech prompt on the dev set, by rules fixed before the dev set was read.

    .venv/bin/python scripts/czech_dev_prompts.py --select   # from the answers: writes the records and the choice
    .venv/bin/python scripts/czech_dev_prompts.py --check    # recompute the choice from the shipped records
    .venv/bin/python scripts/czech_dev_prompts.py --select --tag gx10   # the GX10's answers, beside the Mac's

The answers come from scripts/czech_complete_queue.py --phase dev: e5-small's top 3 on SQAD 3.2, each variant and
generator in one process from an empty cache of its own. Writes answers/sqad32/answers_dev.jsonl.gz (ids, scores and
generated answers) and results/czech/dev/selection.json. Prints no question and no passage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from measurerag.czech_sensitivity import YES_NO_CLASSES, choose_after_amendment, choose_variant, compliance, yes_no_class  # noqa: E402
from measurerag.czech_store import load_labels, read_jsonl_gz, write_jsonl_gz  # noqa: E402
from measurerag.metrics import normalize_answer  # noqa: E402

# The list in the V2 and V3 instruction. A whole answer that is one of these repeats the instruction and names nothing.
INSTRUCTION_WORDS = ("jménem", "číslem", "datem", "slovním spojením")
VARIANTS = ("v1", "v2", "v3", "v4")
FIRST_RULE_VARIANTS = ("v1", "v2", "v3")   # the first rule's variants; v4 has a rule of its own
GENERATORS = ("qwen2.5-1.5b", "qwen2.5-3b")
K, MT, SEED = 3, 64, 20260101
ROWS = REPO / "answers/sqad32/answers_dev.jsonl.gz"
OUT = REPO / "results/czech/dev/selection.json"
SLICE = REPO / "results/crossover/slices/complete_dev"
RANKS = REPO / "results/crossover/cache/v030_32"


def cache_of(variant: str) -> Path:
    return REPO / "results/crossover/cache/complete" / f"dev_{variant}"


def paths(tag: str | None) -> tuple[Path, Path]:
    """The records and the result. The Mac's answers chose the prompt; another machine's are reported beside them, and choose nothing."""
    if not tag:
        return ROWS, OUT
    return ROWS.with_name(f"answers_dev_{tag}.jsonl.gz"), OUT.with_name(f"selection_{tag}.json")


def export_rows(caches=cache_of, variants: tuple[str, ...] = VARIANTS) -> list[dict]:
    from czech_records import scores
    from measurerag.crossover.generate import GenCache, gen_key
    from measurerag.crossover.retrieval import load_run
    from measurerag.datasets.base import Dataset
    from measurerag.rag.generator import build_prompt

    ds = Dataset.load_slice(SLICE)
    by_id = {d.doc_id: d for d in ds.docs}
    run = load_run(RANKS, "e5-small")["run"]
    rows = []
    for variant in variants:
        for g in GENERATORS:
            cache = GenCache(caches(variant) / f"gen_{g}.jsonl")
            for q in sorted(ds.queries):
                ctx = list(run[q][:K])
                prompt = build_prompt(ds.queries[q], [by_id[d].indexable() for d in ctx if d in by_id], language="cs", variant=variant)
                got = cache.get(gen_key(g, MT, SEED, prompt))
                if got is None:
                    raise SystemExit(f"{variant} {g} {q}: no answer")
                rows.append({"set": "dev", "q": q, "variant": variant, "config": f"e5-small+{g}", "kind": "dev", "p": 0.0, "max_tokens": MT, "seed": SEED,
                             "context": ctx, "answer": got["text"], **scores(got["text"], ds.answers[q])})
    rows.sort(key=lambda r: (r["q"], r["variant"], r["config"]))
    return rows


def gold_split(rows: list[dict], labels: dict) -> dict[str, int]:
    """How many yes/no questions have the gold ano, and how many ne, read from any answer of ano or ne and its score."""
    gold: dict[str, str] = {}
    for r in rows:
        w = normalize_answer(r["answer"])
        if labels[r["q"]]["type"] == "yes/no" and w in ("ano", "ne"):
            gold[r["q"]] = w if r["em"] else ("ne" if w == "ano" else "ano")
    return {w: sum(g == w for g in gold.values()) for w in ("ano", "ne")}


def analyse(rows: list[dict], labels: dict) -> dict:
    out: dict = {"questions": len({r["q"] for r in rows}), "yes_no_questions": len({r["q"] for r in rows if labels[r["q"]]["type"] == "yes/no"}),
                 "gold_of_yes_no_questions": gold_split(rows, labels), "variants": {}}
    table: dict = {}
    for v in VARIANTS:
        out["variants"][v] = {}
        table[v] = {}
        for g in GENERATORS:
            mine = [r for r in rows if r["variant"] == v and r["config"] == f"e5-small+{g}"]
            yn = [r for r in mine if labels[r["q"]]["type"] == "yes/no"]
            span = [r for r in mine if labels[r["q"]]["type"] != "yes/no"]
            classes = {c: sum(yes_no_class(r["answer"]) == c for r in yn) for c in YES_NO_CLASSES}
            share = compliance(classes)
            table[v][g] = share
            words = [r["answer"].strip().strip(".,!").lower() for r in yn]
            out["variants"][v][g] = {
                "compliance": share, "strict_share_of_all_yes_no": classes["ano or ne"] / len(yn), "classes": classes,
                "answered_ano": sum(w == "ano" for w in words), "answered_ne": sum(w == "ne" for w in words),
                "em_on_yes_no": sum(r["em"] for r in yn) / len(yn), "em_on_yes_no_mapped": sum(r["em_yes_no_mapped"] for r in yn) / len(yn),
                "em_on_span": sum(r["em"] for r in span) / len(span), "em_mapped_on_span": sum(r["em_yes_no_mapped"] for r in span) / len(span),
                "f1_on_span": sum(r["f1"] for r in span) / len(span),
                "span_answers_declined": sum(yes_no_class(r["answer"]) == "unanswerable" for r in span),
                # Beyond the plan, counted after the answers were seen, and not part of the rule.
                "exploratory": {
                    "span_answered_ano": sum(normalize_answer(r["answer"]) == "ano" for r in span),
                    "span_answered_ano_or_ne": sum(yes_no_class(r["answer"]) == "ano or ne" for r in span),
                    "span_answered_with_the_instructions_list": sum(normalize_answer(r["answer"]) in INSTRUCTION_WORDS for r in span),
                    "yes_no_answered_with_the_instructions_list": sum(normalize_answer(r["answer"]) in INSTRUCTION_WORDS for r in yn),
                },
            }
    out["first_rule"] = choose_variant({v: table[v] for v in FIRST_RULE_VARIANTS})
    span = {v: {g: out["variants"][v][g]["em_on_span"] for g in GENERATORS} for v in ("v1", "v4")}
    out["selection"] = choose_after_amendment(table["v4"], span["v4"], span["v1"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tag", default=None, help="another machine's answers, written beside the Mac's")
    args = ap.parse_args()
    os.chdir(REPO)
    rows_path, out_path = paths(args.tag)
    if args.select:
        write_jsonl_gz(rows_path, export_rows())
    labels = load_labels(REPO / "results/czech/records", "sqad32_test")
    result = analyse(read_jsonl_gz(rows_path), labels)
    if args.tag:
        result["note"] = f"The {args.tag}'s answers. The prompt was chosen on the Mac's (results/czech/dev/selection.json); this result chooses nothing."
    text = json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if args.check:
        same = out_path.read_text(encoding="utf-8") == text
        print("czech_dev_prompts:", "OK" if same else "DIFFERENT")
        return 0 if same else 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
