"""The plan of the complete Czech run: the dev set, the eligible pools, the power sizes, the sets. No model runs.

    .venv/bin/python scripts/czech_plan.py --join     # needs the data: writes the join of the two SQAD versions
    .venv/bin/python scripts/czech_plan.py            # from the tree: writes results/czech/plan/
    .venv/bin/python scripts/czech_plan.py --check    # recompute, compare byte for byte, exit 1 on a difference
    .venv/bin/python scripts/czech_plan.py --slices   # needs the data: builds the git-ignored slices of the plan

The order is the point. The dev set is reserved first, from questions in none of the four reported sets, and
its questions leave every other pool, by question text across both SQAD versions. Only then are the sizes
computed and the sets drawn. Record ids are not stable between SQAD versions, so the join is on the text.

Writes results/czech/plan/: plan.json (counts, sizes, arithmetic) and questions.json (ids only).
results/czech/plan/question_join.jsonl.gz holds hashes of question texts and ids, never a question.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from math import ceil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from measurerag.crossover.stats import interval  # noqa: E402
from measurerag.czech_store import FIRST_CUT as FIRST_CUT_DIR, RECORDS as CURRENT_RECORDS, load_answers, load_first_cut_same_questions, load_labels, read_jsonl_gz, write_jsonl_gz  # noqa: E402
from measurerag.seeds import rng  # noqa: E402

PLAN = REPO / "results/czech/plan"
# the plan reads the first cut, wherever its description is by now: it moves when the complete run's records are written
RECORDS = REPO / FIRST_CUT_DIR if (REPO / FIRST_CUT_DIR / "sets.json").exists() else REPO / CURRENT_RECORDS
SL = REPO / "results/crossover/slices"
BASE, BIG = "qwen2.5-1.5b", "qwen2.5-3b"
DEV_YES_NO, DEV_SPAN = 100, 100
DELTA, ALPHA, POWER = 0.03, 0.05, 0.80
Z_ALPHA, Z_POWER = 1.959963984540054, 0.8416212335729143
SIM_REPS, SIM_BOOT = 10000, 2000
FIRST_CUT_SETS = {"sqad32_debiased": "sqad32_debiased", "sqad32_full": "sqad32_full", "released": "released", "copy_debiased": "copy_debiased"}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:20]


# --------------------------------------------------------------------------
# the join, from the data
# --------------------------------------------------------------------------
def write_join() -> None:
    import pandas as pd

    from measurerag.datasets.sqad32 import ARCHIVE, gold_answers, load_parsed
    from measurerag.metrics import normalize_answer

    os.chdir(REPO)
    rows: dict[str, dict] = {}
    for n in ("train", "validation", "test"):
        df = pd.read_parquet(glob.glob(str(REPO / f"data/hf/datasets--Pehy--cs_sqad-3.0/snapshots/*/data/{n}-*.parquet"))[0])
        for r in df.itertuples(index=False):
            texts = [str(t) for t in r.answers["text"]]
            if not texts:
                continue
            row = rows.setdefault(text_hash(str(r.question)), {"copy": [], "sqad32": [], "copy_gold": None, "sqad32_gold": None})
            row["copy"].append({"id": str(r.id), "split": n})
            if row["copy_gold"] is None:
                row["copy_gold"] = normalize_answer(texts[0])
    recs, arts = load_parsed(ARCHIVE.parent / "parsed")
    for r in recs:
        g = gold_answers(r)
        if not (r.get("question") and g and r.get("article")):
            continue
        row = rows.setdefault(text_hash(r["question"]), {"copy": [], "sqad32": [], "copy_gold": None, "sqad32_gold": None})
        row["sqad32"].append({"id": r["id"], "split": r.get("split")})
        if row["sqad32_gold"] is None:
            row["sqad32_gold"] = normalize_answer(g[0])
    out = []
    for h in sorted(rows):
        row = rows[h]
        both = bool(row["copy"] and row["sqad32"])
        out.append({"text": h, "copy": sorted(row["copy"], key=lambda x: x["id"]), "sqad32": sorted(row["sqad32"], key=lambda x: x["id"]),
                    "same_gold": (row["copy_gold"] == row["sqad32_gold"]) if both else None,
                    "yes_no": (row["copy_gold"] or row["sqad32_gold"]) in ("ano", "ne")})
    PLAN.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(PLAN / "question_join.jsonl.gz", out)
    print("join rows", len(out), "in both", sum(bool(r["copy"] and r["sqad32"]) for r in out))


# --------------------------------------------------------------------------
# power, as the statistics code tests: a paired bootstrap whose 95% interval excludes zero
# --------------------------------------------------------------------------
def paired_size(discordant: float, delta: float = DELTA) -> dict:
    """Questions for `delta` at the plan's alpha and power, when a share `discordant` of the pairs differ.

    One pair's difference is +1, -1 or 0, so its variance under the alternative is discordant - delta^2.
    n = (z_alpha + z_power)^2 * variance / delta^2, rounded up.
    """
    variance = max(discordant - delta**2, 0.0)
    n = ceil((Z_ALPHA + Z_POWER) ** 2 * variance / delta**2)
    return {"discordant": discordant, "variance": variance, "n": n}


def minimum_detectable(discordant: float, n: int) -> float:
    """The difference this many pairs detect at the plan's alpha and power."""
    return float((Z_ALPHA + Z_POWER) * np.sqrt(discordant / n))


def simulated_power(discordant: float, n: int, tag: str, delta: float = DELTA) -> float:
    """The statistics code's test on made-up pairs: resample questions, take the percentile interval.

    Differences are -1, 0 or +1, so a resample is three counts: the same distribution as stats.bootstrap_counts
    without its n-sized matrix. `interval` is the code that decides.
    """
    g = rng("czech-complete", "power", tag, n)
    p = [(discordant - delta) / 2, 1 - discordant, (discordant + delta) / 2]
    hits = 0
    for _ in range(SIM_REPS):
        k = g.multinomial(n, p)
        boot = g.multinomial(n, k / n, size=SIM_BOOT)
        lo, _hi = interval((boot[:, 2] - boot[:, 0]) / n)
        hits += lo > 0
    return hits / SIM_REPS


def size_for_power(discordant: float, eligible: int, tag: str) -> dict:
    """The formula's n, raised 1% at a time until the statistics code's own test reaches the plan's power.

    Past the pool the wanted size is the formula's, since no question is left to test it with.
    """
    start = paired_size(discordant)
    n = start["n"]
    while n <= eligible and simulated_power(discordant, n, tag) < POWER:
        n = ceil(n * 1.01)
    return {**start, "formula_n": start["n"], "n": n, "simulated_power_at_n": simulated_power(discordant, n, tag) if n <= eligible else None}


def natural_rows(rows: list[dict], config: str) -> dict[str, dict]:
    """One configuration at its natural top 3: dial p = 0, else natural, else the bench (rescore_czech's rule)."""
    qids = {r["q"] for r in rows}
    for kind in ("dial", "natural", "bench"):
        got = {r["q"]: r for r in rows if r["config"] == config and r["kind"] == kind and float(r["p"]) == 0.0}
        if len(got) == len(qids):
            return got
    raise SystemExit(f"{config}: no complete rows")


def discordance(rows: list[dict], a: str, b: str) -> dict:
    ra, rb = natural_rows(rows, a), natural_rows(rows, b)
    d = np.array([ra[q]["em"] - rb[q]["em"] for q in sorted(ra)], dtype=float)
    return {"questions": len(d), "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "discordant": float((d != 0).mean()), "observed_difference": float(d.mean())}


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------
def draw(ids: list[str], k: int, *tags: object) -> list[str]:
    ordered = sorted(ids)
    order = rng("czech-complete", *tags).permutation(len(ordered)).tolist()
    return sorted(ordered[i] for i in order[:k])


def rates(compute: dict) -> dict:
    """Seconds per generation, from the first v0.3.0 cut's stages."""
    def per(*names: str) -> float:
        st = [compute["stages"][n] for n in names]
        return sum(s["seconds"] for s in st) / sum(s["generations"] for s in st)

    return {
        "bench_1.5b": per("released_bench", "sqad32_full_bench"),
        "dial_1.5b": per("released_dial_1.5b", "sqad32_full_dial_1.5b", "sqad32_debiased_dial_1.5b", "copy_debiased_dial_1.5b"),
        "dial_3b": per("released_dial_3b", "sqad32_full_dial_3b", "sqad32_debiased_dial_3b", "copy_debiased_dial_3b"),
        "natural_1.5b": per("sqad32_debiased_bm25_1.5b", "sqad32_debiased_retromae_1.5b", "copy_debiased_bm25_1.5b", "copy_debiased_retromae_1.5b"),
    }


def distinct_dial_prompts(rows: list[dict], generator: str) -> float:
    """A dial's levels share prompts: distinct contexts per question, from the first cut."""
    seen: dict[str, set] = {}
    for r in rows:
        if r["kind"] == "dial" and r["config"] == f"e5-small+{generator}":
            seen.setdefault(r["q"], set()).add(tuple(r["context"]))
    return sum(len(v) for v in seen.values()) / len(seen)


def build(records: Path = RECORDS, compute_path: Path = REPO / "results/czech/compute.json") -> tuple[dict, dict]:
    meta = json.loads((records / "sets.json").read_text(encoding="utf-8"))["sets"]
    lab32, lab30 = load_labels(records, "sqad32_test"), load_labels(records, "copy_test")
    join = read_jsonl_gz(PLAN / "question_join.jsonl.gz")
    text32, text30 = {}, {}
    for row in join:
        for x in row["sqad32"]:
            text32[x["id"]] = row["text"]
        for x in row["copy"]:
            text30[x["id"]] = row["text"]
    reported = {k: list(meta[k]["questions"]) for k in FIRST_CUT_SETS}
    reported_text = {text32[q] for k in ("sqad32_full", "sqad32_debiased") for q in reported[k]} | {text30[q] for k in ("released", "copy_debiased") for q in reported[k]}
    copy_test_text = {text30[q] for q in lab30}

    # ---- 1. the dev set: first, from SQAD 3.2's test split, in none of the four reported sets ----
    # Not in the copy's test split either: that pool is small and every question of it is used.
    candidates = sorted(q for q in lab32 if text32[q] not in reported_text and text32[q] not in copy_test_text)
    dev_yes_no = draw([q for q in candidates if lab32[q]["type"] == "yes/no"], DEV_YES_NO, "dev", "yes-no")
    dev_span = draw([q for q in candidates if lab32[q]["type"] != "yes/no"], DEV_SPAN, "dev", "span")
    dev = sorted(dev_yes_no + dev_span)
    dev_text = {text32[q] for q in dev}

    # ---- 2. the pools, with the dev set out, by text ----------------------------------------------
    def out_32(q: str) -> bool:
        return text32[q] not in dev_text

    def out_30(q: str) -> bool:
        return text30[q] not in dev_text

    pool = {
        "sqad32_full": sorted(q for q in lab32 if out_32(q)),
        "sqad32_debiased": sorted(q for q in lab32 if out_32(q) and not lab32[q]["flagged_0138"]),
        "copy_debiased": sorted(q for q in lab30 if out_30(q) and not lab30[q]["flagged_0138"]),
        "extra010_sqad32": sorted(q for q in lab32 if out_32(q) and not lab32[q]["flagged_010"]),
        "extra010_copy": sorted(q for q in lab30 if out_30(q) and not lab30[q]["flagged_010"]),
    }
    in_both = [r for r in join if r["copy"] and r["sqad32"]]
    pairs = sorted((r["copy"][0]["id"], r["sqad32"][0]["id"]) for r in in_both if r["text"] not in dev_text)
    same_gold_by_pair = {(r["copy"][0]["id"], r["sqad32"][0]["id"]): bool(r["same_gold"]) for r in in_both}

    # ---- 3. the sizes -------------------------------------------------------------------------------
    answers = {k: load_answers(records, k) for k in FIRST_CUT_SETS}
    sized: dict = {}
    for key in ("sqad32_debiased", "sqad32_full", "copy_debiased"):
        ret = discordance(answers[key], f"e5-small+{BASE}", f"bm25+{BASE}")
        gen = discordance(answers[key], f"e5-small+{BIG}", f"e5-small+{BASE}")
        ns_ret, ns_gen = size_for_power(ret["discordant"], len(pool[key]), key + "-retriever"), size_for_power(gen["discordant"], len(pool[key]), key + "-generator")
        wanted = max(ns_ret["n"], ns_gen["n"])
        n = min(wanted, len(pool[key]))
        sized[key] = {"eligible": len(pool[key]), "retriever_step": {**ret, **ns_ret}, "generator_step": {**gen, **ns_gen}, "wanted": wanted, "n": n, "capped": n < wanted,
                      "at_this_size": {"retriever_step_detects": minimum_detectable(ret["discordant"], n), "generator_step_detects": minimum_detectable(gen["discordant"], n)},
                      "simulated_power_at_this_size": {"retriever_step": simulated_power(ret["discordant"], n, key + "-retriever"), "generator_step": simulated_power(gen["discordant"], n, key + "-generator")}}
    # the released 200 stays those 200: no sizing, and what it can detect is stated
    rel_ret = discordance(answers["released"], f"e5-small+{BASE}", f"bm25+{BASE}")
    rel_gen = discordance(answers["released"], f"e5-small+{BIG}", f"e5-small+{BASE}")
    fixed = {"n": len(reported["released"]), "why": "the continuity row against v0.2.0: exactly those questions",
             "retriever_step": rel_ret, "generator_step": rel_gen,
             "at_this_size": {"retriever_step_detects": minimum_detectable(rel_ret["discordant"], len(reported["released"])), "generator_step_detects": minimum_detectable(rel_gen["discordant"], len(reported["released"]))}}
    # the cross-version gap: paired on the same question, from the first cut's same-question run
    same = load_first_cut_same_questions()
    on_copy = {r["q"]: r for r in same if r["corpus"] == "copy"}
    on_32 = {r["q"]: r for r in same if r["corpus"] == "sqad32"}
    d = np.array([on_32[r["other_q"]]["em"] - r["em"] for _, r in sorted(on_copy.items())], dtype=float)
    gap = {"questions": len(d), "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "discordant": float((d != 0).mean()), "observed_difference": float(d.mean())}
    wanted_cross = size_for_power(gap["discordant"], len(pairs), "cross-version")
    n_cross = min(wanted_cross["n"], len(pairs))
    cross = {"eligible": len(pairs), "first_cut": {**gap, **wanted_cross}, "wanted": wanted_cross["n"], "n": n_cross, "capped": n_cross < wanted_cross["n"],
             "at_this_size_detects": minimum_detectable(gap["discordant"], n_cross), "simulated_power_at_this_size": simulated_power(gap["discordant"], n_cross, "cross-version")}

    # ---- 4. the sets: what stands stays, and a seeded draw from the rest of the pool tops it up ---------
    questions: dict[str, list] = {"dev": dev, "released": sorted(reported["released"])}
    for key in ("sqad32_debiased", "sqad32_full", "copy_debiased"):
        kept = sorted(reported[key])
        if not set(kept) <= set(pool[key]):
            raise SystemExit(f"{key}: a question of the set as it stands left the pool")
        rest = [q for q in pool[key] if q not in set(kept)]
        questions[key] = sorted(kept + draw(rest, sized[key]["n"] - len(kept), key, "topup"))
    for key, base_key, flags in (("extra010_sqad32", "sqad32_debiased", lab32), ("extra010_copy", "copy_debiased", lab30)):
        kept = sorted(q for q in questions[base_key] if not flags[q]["flagged_010"])
        rest = [q for q in pool[key] if q not in set(kept)]
        n = min(len(questions[base_key]), len(pool[key]))
        questions[key] = sorted(kept + draw(rest, n - len(kept), key, "topup"))
    cross_keys = [f"{a} {b}" for a, b in pairs]
    chosen = set(draw(cross_keys, n_cross, "cross-version"))
    questions["cross_version"] = [list(p) for p in pairs if f"{p[0]} {p[1]}" in chosen]

    # The gaps between two sets are different questions, so they are not paired: what those sizes can detect.
    def between(a: str, b: str) -> dict:
        out = {}
        for config in (f"e5-small+{BASE}", f"e5-small+{BIG}"):
            ra, rb = natural_rows(answers[a], config), natural_rows(answers[b], config)
            pa, pb = float(np.mean([r["em"] for r in ra.values()])), float(np.mean([r["em"] for r in rb.values()]))
            na, nb = len(questions[a]), len(questions[b])
            out[config] = {"em_first_cut": {a: pa, b: pb}, "questions": {a: na, b: nb},
                           "detects": float((Z_ALPHA + Z_POWER) * np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)),
                           "questions_each_for_the_plans_difference": ceil((Z_ALPHA + Z_POWER) ** 2 * (pa * (1 - pa) + pb * (1 - pb)) / DELTA**2)}
        return out

    gaps = {"sqad32_debiased minus copy_debiased": between("sqad32_debiased", "copy_debiased"), "sqad32_full minus released": between("sqad32_full", "released")}

    # ---- 5. what it costs ------------------------------------------------------------------------------
    compute = json.loads(Path(compute_path).read_text(encoding="utf-8"))
    r = rates(compute)
    per_q = {"dial_1.5b": float(np.mean([distinct_dial_prompts(answers[k], BASE) for k in FIRST_CUT_SETS])), "dial_3b": float(np.mean([distinct_dial_prompts(answers[k], BIG) for k in FIRST_CUT_SETS]))}

    def question_set(n: int) -> dict:
        return {"dial_1.5b": round(n * per_q["dial_1.5b"]), "dial_3b": round(n * per_q["dial_3b"]), "natural_1.5b": 2 * n}

    def whole_set(n: int) -> dict:
        return {"bench_1.5b": 3 * n, "dial_1.5b": round(n * per_q["dial_1.5b"]), "dial_3b": round(n * per_q["dial_3b"])}

    stages = {
        "dev": {"dial_1.5b": 3 * len(dev), "dial_3b": 3 * len(dev)},
        "released": whole_set(len(questions["released"])),
        "sqad32_debiased": question_set(len(questions["sqad32_debiased"])),
        "copy_debiased": question_set(len(questions["copy_debiased"])),
        "sqad32_full": whole_set(len(questions["sqad32_full"])),
        "cross_version": {"dial_1.5b": 2 * n_cross, "dial_3b": 2 * n_cross},
        "extra010_copy": question_set(len(questions["extra010_copy"])),
        "extra010_sqad32": question_set(len(questions["extra010_sqad32"])),
    }
    hours = {k: sum(v * r[kind] for kind, v in st.items()) / 3600 for k, st in stages.items()}

    plan = {
        "seed_tags": "measurerag.seeds.rng('czech-complete', ...)",
        "rule": {"delta_em": DELTA, "alpha": ALPHA, "power": POWER, "z_alpha": Z_ALPHA, "z_power": Z_POWER, "simulation": {"reps": SIM_REPS, "resamples": SIM_BOOT, "raised_until_power_reached": "1% at a time"}},
        "dev": {"questions": len(dev), "yes_no": len(dev_yes_no), "span": len(dev_span), "corpus": "sqad32_test", "candidates": len(candidates),
                "candidates_yes_no": sum(lab32[q]["type"] == "yes/no" for q in candidates),
                "also_in_another_split_of_the_copy": sum(text32[q] in text30.values() for q in dev)},
        "ids": {"shared_ids_all_splits": len(set(text30) & set(text32)), "same_question_on_a_shared_id_all_splits": sum(text30[i] == text32[i] for i in set(text30) & set(text32)),
                "shared_ids_test_splits": len(set(lab30) & set(lab32)), "same_question_on_a_shared_id_test_splits": sum(text30[i] == text32[i] for i in set(lab30) & set(lab32)),
                "questions_in_both_by_text": len(in_both), "questions_in_both_test_splits_by_text": sum(1 for q in lab30 if lab30[q].get("same_question"))},
        "pools": {k: {"eligible": len(v), "yes_no": sum((lab32 if k.endswith("sqad32") or k.startswith("sqad32") else lab30)[q]["type"] == "yes/no" for q in v)} for k, v in pool.items()},
        "sizes": {**sized, "released": fixed, "cross_version": cross},
        "between_set_gaps": gaps,
        "sets": {k: len(v) for k, v in questions.items()},
        "kept_from_the_first_cut": {k: len(set(reported[k]) & set(questions[k])) for k in FIRST_CUT_SETS},
        "extra010_kept_from_0138": {"extra010_sqad32": len(set(questions["extra010_sqad32"]) & set(questions["sqad32_debiased"])), "extra010_copy": len(set(questions["extra010_copy"]) & set(questions["copy_debiased"]))},
        "cross_version_same_gold": sum(same_gold_by_pair[tuple(p)] for p in questions["cross_version"]),
        "compute_estimate": {"seconds_per_generation": r, "dial_prompts_per_question": per_q, "generations": {k: sum(v.values()) for k, v in stages.items()}, "by_kind": stages, "hours": hours,
                             "total_generations": sum(sum(v.values()) for v in stages.values()), "total_hours": sum(hours.values())},
    }
    return plan, questions


def write(plan: dict, questions: dict) -> None:
    PLAN.mkdir(parents=True, exist_ok=True)
    (PLAN / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (PLAN / "questions.json").write_text(json.dumps(questions, indent=0, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# the slices: git-ignored, and they hold SQAD text
# --------------------------------------------------------------------------
def build_slices() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from sqad32_slices import restrict

    from measurerag.datasets.base import Dataset
    from measurerag.datasets.sqad import load as load_30
    from measurerag.datasets.sqad32 import load as load_32

    os.chdir(REPO)
    q = json.loads((PLAN / "questions.json").read_text(encoding="utf-8"))
    t32 = Dataset.load_slice(SL / "czech32_testall")
    t30 = Dataset.load_slice(SL / "czech30_testall")
    made = {"complete_dev": (t32, q["dev"], "sqad-3.2-dev"), "complete_sqad32_debiased": (t32, q["sqad32_debiased"], "sqad-3.2-test-debiased"), "complete_sqad32_full": (t32, q["sqad32_full"], "sqad-3.2-test"),
            "complete_copy_debiased": (t30, q["copy_debiased"], "sqad-3.0-test-debiased"), "complete_extra010_sqad32": (t32, q["extra010_sqad32"], "sqad-3.2-test-debiased-010"), "complete_extra010_copy": (t30, q["extra010_copy"], "sqad-3.0-test-debiased-010")}
    for name, (ds, ids, label) in made.items():
        restrict(ds, sorted(ids), label, "A set of the complete run; see results/czech/plan/.").save(SL / name)
        print(name, len(ids), flush=True)
    # the cross-version check reaches beyond the test splits, so it needs every split's questions
    full_32 = load_32(split="train", max_queries=None)
    pairs = q["cross_version"]
    a32 = {**{k: v for k, v in full_32.queries.items()}}
    ans32, rel32 = dict(full_32.answers), dict(full_32.qrels)
    for split in ("test", "eval"):
        other = load_32(split=split, max_queries=None)
        if other.corpus_hash() != full_32.corpus_hash():
            raise SystemExit("SQAD 3.2: the corpus changed between splits")
        a32.update(other.queries)
        ans32.update(other.answers)
        rel32.update(other.qrels)
    if full_32.corpus_hash() != t32.corpus_hash():
        raise SystemExit("SQAD 3.2: the corpus is not the first cut's")
    ids32 = sorted(b for _, b in pairs)
    Dataset(name="sqad-3.2-cross-version", language="cs", docs=full_32.docs, queries={b: a32[b] for b in ids32}, qrels={b: rel32[b] for b in ids32}, answers={b: ans32[b] for b in ids32},
            license=full_32.license, source=full_32.source, notes=[*full_32.notes, "Questions in both SQAD versions, joined on the text."]).save(SL / "complete_cross_32")
    full_30 = load_30(cache_dir=REPO / "data", split="train", max_queries=None)
    a30, ans30, rel30 = dict(full_30.queries), dict(full_30.answers), dict(full_30.qrels)
    for split in ("validation", "test"):
        other = load_30(cache_dir=REPO / "data", split=split, max_queries=None)
        a30.update(other.queries)
        ans30.update(other.answers)
        rel30.update(other.qrels)
    if full_30.corpus_hash() != t30.corpus_hash():
        raise SystemExit("the SQAD 3.0 copy: the corpus is not the first cut's")
    ids30 = sorted(a for a, _ in pairs)
    Dataset(name="sqad-3.0-cross-version", language="cs", docs=full_30.docs, queries={a: a30[a] for a in ids30}, qrels={a: rel30[a] for a in ids30}, answers={a: ans30[a] for a in ids30},
            license=full_30.license, source=full_30.source, notes=[*full_30.notes, "Questions in both SQAD versions, joined on the text."]).save(SL / "complete_cross_copy")
    bad = [(a, b) for a, b in pairs if a30[a].strip() != a32[b].strip()]
    if bad:
        raise SystemExit(f"{len(bad)} pairs are not the same question")
    print("cross", len(pairs), "pairs, every one the same question text", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--slices", action="store_true")
    ap.add_argument("--records", default=str(RECORDS))
    args = ap.parse_args()
    if args.join:
        write_join()
        return 0
    if args.slices:
        build_slices()
        return 0
    plan, questions = build(Path(args.records))
    if args.check:
        a = json.dumps(plan, indent=1, sort_keys=True) + "\n"
        b = json.dumps(questions, indent=0, sort_keys=True) + "\n"
        ok = (PLAN / "plan.json").read_text(encoding="utf-8") == a and (PLAN / "questions.json").read_text(encoding="utf-8") == b
        print("czech_plan:", "OK" if ok else "DIFFERENT")
        return 0 if ok else 1
    write(plan, questions)
    print(json.dumps(plan, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
