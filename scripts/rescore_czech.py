"""Recompute every Czech number from the records that ship. No model, no dataset, no cache.

    .venv/bin/python scripts/rescore_czech.py                 # write results/czech/analysis.json
    .venv/bin/python scripts/rescore_czech.py --check         # recompute, compare byte for byte, exit 1 on a difference
    .venv/bin/python scripts/rescore_czech.py --verify-scores # also recompute EM and F1 from the answers,
                                                              # against a copy of SQAD you download yourself

Reads answers/ (the generated answers, ids and scores, no SQAD text), results/czech/records/ (rankings and labels),
results/czech/question_join.json and results/cz_rag*/results.json. `--check` compares three things
with the tree: results/czech/analysis.json, the Czech block of results/crossover.json, and every
Czech token in results/stats.json that the public text is filled from. The first v0.3.0 cut is analysed the same way,
from results/czech/first_cut/, and results/czech/first_cut/analysis.json is compared too.

Gold answers do not ship, so EM and F1 are stored per row. `--verify-scores` recomputes them from each
generated answer and the gold of the pinned SQAD 3.0 copy and the SQAD 3.2 archive on LINDAT.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import crossover  # noqa: E402
from measurerag.crossover.stats import bootstrap_counts, established, interval  # noqa: E402
from measurerag.czech_labels import BANDS, TYPES, length_band  # noqa: E402
from measurerag.czech_sensitivity import yes_no_class  # noqa: E402
from measurerag.czech_store import FIRST_CUT, RECORDS, is_first_cut, load_answers, load_cross_version, load_first_cut_same_questions, load_labels, load_rankings, load_sets, read_jsonl_gz  # noqa: E402
from measurerag.metrics import gold_in_context, ndcg_at_k, normalize_answer, recall_at_k  # noqa: E402

BASE, BIG = "qwen2.5-1.5b", "qwen2.5-3b"
E2, E4 = "gemma-4-e2b", "gemma-4-e4b"
# Two model families answer every Czech set. Qwen2.5 is the family v0.2.0
# and the first cut ran, so its readings sit at the top level of a set and every comparison with those releases is its own.
# Every other family's readings sit under the set's "families", made the same way with its smaller model in 1.5B's place
# and its larger in 3B's. "family_difference" is the reading a second family adds: that family's model minus Qwen2.5's of its size.
M3, M8 = "ministral-3-3b", "ministral-3-8b"
Q2, Q4 = "qwen3.5-2b", "qwen3.5-4b"                  # v0.4.0's fourth family
N2, N4 = "gemma-3n-e2b", "gemma-3n-e4b"              # v0.4.2: Gemma 4's previous generation
FAMILIES = {"qwen2.5": (BASE, BIG), "gemma-4": (E2, E4), "ministral-3": (M3, M8), "qwen3.5": (Q2, Q4), "gemma-3n": (N2, N4)}
FIRST_FAMILY = "qwen2.5"
OTHER_FAMILIES = tuple(f for f in FAMILIES if f != FIRST_FAMILY)
# v0.3.1's added reading: a family against another besides Qwen2.5, as "first minus second"
BETWEEN = (("ministral-3", "gemma-4"), ("qwen3.5", "gemma-4"), ("qwen3.5", "ministral-3"),     # Qwen3.5 against each of the two
           ("gemma-4", "gemma-3n"))                                                              # one maker's two generations
GENERATORS = tuple(g for pair in FAMILIES.values() for g in pair)
SIZES = ("smaller", "larger")
N_BOOT = 10000


def rungs(small: str, big: str) -> tuple[str, ...]:
    """A family's configurations, in the order of every table: the retriever ladder with the smaller model, then e5-small with the larger."""
    return (f"bm25+{small}", f"e5-small+{small}", f"retromae-cs+{small}", f"e5-small+{big}")


GENERATOR_RUNGS = rungs(BASE, BIG)
RANK_RETRIEVERS = ("bm25", "bm25-nodiacritics", "e5-small", "retromae-cs")
MAIN_SETS = ("sqad32_debiased", "sqad32_full", "released", "copy_debiased")    # the order of every table
METRICS = ("em_yes_no_mapped", "em")                                           # the primary metric first
SENSITIVITY_SETS = ("extra010_sqad32", "extra010_copy")
POPULATION_LABELS = {"copy_test": "copy", "sqad32_test": "sqad32_test"}
LABEL_FILE = {"copy_test": "copy_test", "sqad32_test": "sqad32_test"}
DIAL = {"retriever": "e5-small", "k": 3, "levels": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
PASSAGE_BANDS = ("under 500", "500 to 799", "800 or more")
ORIGINS = ("in the copy", "dropped from the copy", "new in SQAD 3.2")
PLAN = REPO / "results/czech/plan"


def ci(point: float, boot: np.ndarray) -> dict:
    lo, hi = interval(boot)
    return {"point": float(point), "lo": lo, "hi": hi, "established": established(lo, hi)}


def boot_mean(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return (counts @ values) / counts.shape[1]


def mean_ci(values: np.ndarray, counts: np.ndarray) -> dict:
    return ci(float(values.mean()), boot_mean(values, counts))


def verdict_sign(base: dict, cur: dict) -> str:
    """HOLDS, WEAKENS or REVERSES for a signed effect against v0.2.0's value (the follow-up's rule)."""
    s = np.sign(base["point"]) or 1.0
    p = cur["point"]
    if cur["established"] and np.sign(p) == -s:
        return "REVERSES"
    if np.sign(p) == s and (cur["established"] or not base["established"]) and abs(p) >= 0.5 * abs(base["point"]):
        return "HOLDS"
    return "WEAKENS"


# --------------------------------------------------------------------------
# one set
# --------------------------------------------------------------------------
class SetData:
    def __init__(self, root: Path, key: str, meta: dict, qids: list[str] | None = None) -> None:
        self.key = key
        self.meta = meta
        self.qids = sorted(meta["questions"] if qids is None else qids)
        self.n = len(self.qids)
        self.index = {q: i for i, q in enumerate(self.qids)}
        pop = meta["ranking"]
        self.population = pop
        self.rank = load_rankings(root, pop)
        self.labels = load_labels(root, LABEL_FILE[pop])
        self.rows = [r for r in load_answers(root, meta.get("answers_from", key)) if r["q"] in self.index]
        self.is_yn = np.array([self.labels[q]["type"] == "yes/no" for q in self.qids])
        self.counts = bootstrap_counts(self.n, N_BOOT, "czech-v030", key, self.n).astype(float)
        self._yes_no_counts: np.ndarray | None = None

    def yes_no_counts(self) -> np.ndarray:
        """Resamples of the yes/no questions alone: compliance is read on those."""
        if self._yes_no_counts is None:
            self._yes_no_counts = bootstrap_counts(int(self.is_yn.sum()), N_BOOT, "czech-v030", self.key, "yes-no", self.n).astype(float)
        return self._yes_no_counts

    def config_rows(self, config: str) -> list[dict] | None:
        """The answers of one configuration at its natural top 3: dial p = 0 or natural, else the bench."""
        for kind in ("dial", "natural", "bench"):
            rows = {r["q"]: r for r in self.rows if r["config"] == config and r["kind"] == kind and float(r["p"]) == 0.0}
            if len(rows) == self.n:
                return [rows[q] for q in self.qids]
        return None

    def bench_rows(self, config: str) -> list[dict] | None:
        rows = {r["q"]: r for r in self.rows if r["config"] == config and r["kind"] == "bench"}
        return [rows[q] for q in self.qids] if len(rows) == self.n else None


def boot_ratio(top: np.ndarray, bottom: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n = counts @ bottom
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (counts @ top) / n, np.nan)


def compliance_parts(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """On yes/no questions: is the answer ano or ne, and did it answer at all?"""
    classes = [yes_no_class(r["answer"]) for r in rows]
    return np.array([c == "ano or ne" for c in classes], dtype=float), np.array([c != "unanswerable" for c in classes], dtype=float)


def compliance_block(rows: list[dict], counts: np.ndarray) -> dict | None:
    """`rows` are the answers to yes/no questions, and `counts` resamples them."""
    if not rows:
        return None
    czech, answered = compliance_parts(rows)
    return {"compliance": ci(float(czech.sum() / answered.sum()), boot_ratio(czech, answered, counts)) if answered.sum() else None,
            "strict_share": mean_ci(czech, counts), "declined": int(len(rows) - answered.sum()), "yes_no_questions": len(rows)}


def compliance_difference(a: list[dict], b: list[dict], counts: np.ndarray) -> dict | None:
    """Paired, on the same yes/no questions: a minus b."""
    if not a:
        return None
    ca, na = compliance_parts(a)
    cb, nb = compliance_parts(b)
    block: dict = {"strict_share": mean_ci(ca - cb, counts)}
    if na.sum() and nb.sum():
        block["compliance"] = ci(float(ca.sum() / na.sum() - cb.sum() / nb.sum()), boot_ratio(ca, na, counts) - boot_ratio(cb, nb, counts))
    return block


def yes_no_rows(sd: SetData, rows: list[dict]) -> list[dict]:
    return [rows[i] for i in np.flatnonzero(sd.is_yn)]


def answer_block(sd: SetData, rows: list[dict]) -> dict:
    em = np.array([r["em"] for r in rows], dtype=float)
    f1 = np.array([r["f1"] for r in rows], dtype=float)
    folded = np.array([r["f1_folded"] for r in rows], dtype=float)
    pref = np.array([r["prefix5"] for r in rows], dtype=float)
    mapped = np.array([r["em_yes_no_mapped"] for r in rows], dtype=float)
    tolerant = np.maximum.reduce([em, pref, mapped])
    words = [normalize_answer(r["answer"]) for r in rows]
    yn = sd.is_yn
    classes = {"ano or ne": 0, "yes or no in English": 0, "unanswerable": 0, "something else": 0}
    for i in np.flatnonzero(yn):
        classes[yes_no_class(rows[i]["answer"])] += 1
    out = {
        "source": rows[0]["kind"], "max_tokens": rows[0]["max_tokens"],
        "em": mean_ci(em, sd.counts), "f1": mean_ci(f1, sd.counts),
        "secondary_f1_folded": float(folded.mean()),
        "secondary_prefix5": mean_ci(pref, sd.counts),
        "secondary_em_yes_no_mapped": mean_ci(mapped, sd.counts),
        "tolerant": mean_ci(tolerant, sd.counts),
        "undercount_points": 100 * float(tolerant.mean() - em.mean()),
        "undercount_yes_no_points": 100 * float(mapped.mean() - em.mean()),
        "undercount_declension_points": 100 * float(pref.mean() - em.mean()),
        "said_unanswerable": int(sum(w == "unanswerable" for w in words)),
        "yes_no_answer_classes": classes,
    }
    if yn.any():
        out["compliance"] = compliance_block(yes_no_rows(sd, rows), sd.yes_no_counts())
    if yn.any():
        out["em_on_yes_no"] = float(em[yn].mean())
        out["em_on_yes_no_mapped"] = float(mapped[yn].mean())
    if (~yn).any():
        out["em_on_span"] = float(em[~yn].mean())
    other = [r.get("em_other_key") for r in rows]
    have = [i for i, v in enumerate(other) if v is not None]
    if have:
        out["key_swap"] = {"questions_with_a_record_in_the_other_version": len(have), "em_own_key": float(em[have].mean()), "em_other_key": float(np.mean([other[i] for i in have]))}
    return out


def retrieval_table(rank: dict, qids: list[str], counts: np.ndarray) -> dict:
    arrs: dict = {}
    for rid in RANK_RETRIEVERS:
        if rid not in rank[qids[0]]:
            continue
        rel = {q: {g: 1 for g in rank[q]["gold"]} for q in qids}
        arrs[rid] = {"ndcg10": np.array([ndcg_at_k(rank[q][rid], rel[q], 10) for q in qids]),
                     "recall10": np.array([recall_at_k(rank[q][rid], rel[q], 10) for q in qids]),
                     "g3": np.array([gold_in_context(rank[q][rid], rel[q], 3) for q in qids])}
    t: dict = {"means": {}, "deltas": {}}
    for rid, a in arrs.items():
        t["means"][rid] = {m: mean_ci(a[m], counts) for m in ("ndcg10", "g3")}
        t["means"][rid]["recall10"] = float(a["recall10"].mean())
    for hi, lo in (("e5-small", "bm25"), ("e5-small", "bm25-nodiacritics"), ("bm25-nodiacritics", "bm25"), ("e5-small", "retromae-cs")):
        if hi in arrs and lo in arrs:
            t["deltas"][f"{hi} - {lo}"] = {m: mean_ci(arrs[hi][m] - arrs[lo][m], counts) for m in ("ndcg10", "g3")}
    return t


def dial_summary(d: dict, small: str = BASE, big: str = BIG) -> dict:
    lv = d["dial"]["levels"]
    with_gold = [x for x in lv[1:] if x["ret_minus_gen"]]
    pts = [x["ret_minus_gen"]["point"] for x in with_gold]
    roots = d["dial"]["roots"]
    return {
        "n": d["n_questions"], "g_top": lv[0]["g_any"]["point"],
        "em_top_base": lv[0]["em"][small]["point"], "em_top_big": lv[0]["em"][big]["point"],
        "em_no_gold_base": lv[-1]["em"][small]["point"], "em_no_gold_big": lv[-1]["em"][big]["point"],
        "answer_in_nongold_no_gold": lv[-1]["answer_in_nongold"], "delta_gen_top": lv[0]["delta_gen"],
        "levels_compared": len(with_gold),
        "levels_generator_larger": int(sum(p < 0 for p in pts)), "levels_retriever_larger": int(sum(p > 0 for p in pts)), "levels_tied": int(sum(p == 0 for p in pts)),
        "levels_established_generator": int(sum(x["ret_minus_gen"]["established"] and x["ret_minus_gen"]["point"] < 0 for x in with_gold)),
        "levels_established_retriever": int(sum(x["ret_minus_gen"]["established"] and x["ret_minus_gen"]["point"] > 0 for x in with_gold)),
        "levels_established_with_gold": int(sum(x["ret_minus_gen"]["established"] and x["g_any"]["point"] > 0 for x in with_gold)),
        "root_isotonic": roots["raw"]["isotonic"]["point"], "root_linear": roots["raw"]["linear"]["point"],
        "break_even_raw": [{"p": b["p"], "g": b["g_any"], "point": b["raw"]["point"], "lo": b["raw"]["lo"], "hi": b["raw"]["hi"],
                            "n_censored": b["raw"]["n_censored"], "headroom": b["headroom"]} for b in roots["break_even_gold_gain"]],
    }


def dial_block(root: Path, key: str, cfg: dict, meta: dict, name: str, qids: list[str] | None = None, leak_field: str = "leak", em_field: str = "em",
               small: str = BASE, big: str = BIG) -> dict | None:
    dcfg = {"records": str(root.relative_to(REPO)), "set": meta.get("answers_from", key), "leak_field": leak_field, "em_field": em_field, "dial": DIAL, "ladder": ["bm25", "e5-small"], "harness": [], "dataset_name": name, "qids": qids}
    try:
        d = crossover.clean(crossover.analyse_dataset(meta["tag"], dcfg, {**cfg, "base_generator": small, "big_generator": big}, False))
    except KeyError:
        return None
    return d if "dial" in d else None


def steps_block(sd: SetData, metric_rows: dict[str, np.ndarray], small: str = BASE, big: str = BIG) -> dict:
    hi, base, bm = metric_rows[f"e5-small+{big}"], metric_rows[f"e5-small+{small}"], metric_rows[f"bm25+{small}"]
    return {"generator_step": mean_ci(hi - base, sd.counts), "retriever_step": mean_ci(base - bm, sd.counts),
            "generator_step_minus_retriever_step": mean_ci((hi - base) - (base - bm), sd.counts)}


def detectable(a: np.ndarray, b: np.ndarray) -> dict:
    """What a paired difference on these questions can detect at the plan's alpha and power, from its own discordance
. Read for a family whose sizes were not planned on it."""
    from czech_plan import minimum_detectable

    disc = float((a != b).mean())
    return {"questions": int(len(a)), "discordant": disc, "detects": minimum_detectable(disc, len(a)) if disc else 0.0}


def metric_arrays_for(rows_by_config: dict[str, list[dict]]) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict = {"em": {}, "mapped": {}, "tolerant": {}, "f1": {}}
    for config, rows in rows_by_config.items():
        arrays["em"][config] = np.array([r["em"] for r in rows], dtype=float)
        arrays["f1"][config] = np.array([r["f1"] for r in rows], dtype=float)
        arrays["mapped"][config] = np.array([r["em_yes_no_mapped"] for r in rows], dtype=float)
        arrays["tolerant"][config] = np.maximum.reduce([arrays["em"][config], np.array([r["prefix5"] for r in rows], dtype=float), arrays["mapped"][config]])
    return arrays


def family_block(root: Path, sd: SetData, key: str, cfg: dict, meta: dict, qids: list[str] | None, label: str, small: str, big: str) -> dict:
    """One family's readings on one set: its answers, the steps in each metric, compliance, the dial and the token cap."""
    out: dict = {"answers": {}}
    by_config: dict = {}
    for config in rungs(small, big):
        rows = sd.config_rows(config)
        if rows is not None:
            by_config[config] = rows
            out["answers"][config] = answer_block(sd, rows)
    arrays = metric_arrays_for(by_config)
    gen, base, bm, rm = f"e5-small+{big}", f"e5-small+{small}", f"bm25+{small}", f"retromae-cs+{small}"
    if all(c in by_config for c in (gen, base, bm)):
        out["steps"] = {"em": steps_block(sd, arrays["em"], small, big), "em_yes_no_mapped": steps_block(sd, arrays["mapped"], small, big),
                        "tolerant": steps_block(sd, arrays["tolerant"], small, big), "f1": steps_block(sd, arrays["f1"], small, big)}
        out["detectable"] = {field: {"generator_step": detectable(arrays[a][gen], arrays[a][base]), "retriever_step": detectable(arrays[a][base], arrays[a][bm])}
                             for field, a in (("em_yes_no_mapped", "mapped"), ("em", "em"))}
    if base in by_config and rm in by_config:
        out["e5small_minus_retromae_em"] = mean_ci(arrays["em"][base] - arrays["em"][rm], sd.counts)
        out["e5small_minus_retromae_mapped"] = mean_ci(arrays["mapped"][base] - arrays["mapped"][rm], sd.counts)
    if sd.is_yn.any() and gen in by_config and base in by_config:
        # H3: does the larger generator follow the language instruction more often?
        out["compliance_big_minus_small"] = compliance_difference(yes_no_rows(sd, by_config[gen]), yes_no_rows(sd, by_config[base]), sd.yes_no_counts())
    d = dial_block(root, key, cfg, meta, label, qids, small=small, big=big)
    if d:
        out["dial_summary"] = dial_summary(d, small, big)
        span = dial_block(root, key, cfg, meta, label, qids, leak_field="leak_span_only", small=small, big=big)
        out["dial_summary"]["answer_in_nongold_no_gold_span_only"] = span["dial"]["levels"][-1]["answer_in_nongold"]
        out["dial_levels"] = d["dial"]["levels"]
        # the same dial on the labelled secondary metric: does a level rest on the language of the answer?
        out["dial_summary_em_yes_no_mapped"] = dial_summary(dial_block(root, key, cfg, meta, label, qids, em_field="em_yes_no_mapped", small=small, big=big), small, big)
    # the token cap: the bench at 256 tokens against the dial at 64, on the same prompts
    a, b = sd.bench_rows(base), {r["q"]: r for r in sd.rows if r["config"] == base and r["kind"] == "dial" and float(r["p"]) == 0.0}
    if a and len(b) == sd.n:
        out["token_cap_check"] = {"questions": sd.n, "same_string": sum(r["answer"] == b[r["q"]]["answer"] for r in a), "same_em": sum(r["em"] == b[r["q"]]["em"] for r in a)}
    return out


def family_difference(sd: SetData, ours: tuple[str, str], theirs: tuple[str, str]) -> dict | None:
    """An added reading: another family's model minus Qwen2.5's of its size, both with e5-small's top 3, paired."""
    out: dict = {}
    for size, a, b in zip(SIZES, theirs, ours):
        ra, rb = sd.config_rows(f"e5-small+{a}"), sd.config_rows(f"e5-small+{b}")
        if ra is None or rb is None:
            continue
        block: dict = {"minuend": a, "subtrahend": b, "detectable": {}}
        for field in METRICS:
            x, y = np.array([r[field] for r in ra], dtype=float), np.array([r[field] for r in rb], dtype=float)
            block[field] = mean_ci(x - y, sd.counts)
            block["detectable"][field] = detectable(x, y)
        if sd.is_yn.any():
            block["compliance"] = compliance_difference(yes_no_rows(sd, ra), yes_no_rows(sd, rb), sd.yes_no_counts())
        out[size] = block
    return out or None


def family_view(set_out: dict, family: str) -> dict:
    """A family's readings on one set, wherever they sit."""
    return set_out if family == FIRST_FAMILY else set_out.get("families", {}).get(family, {})


def analyse_set(root: Path, key: str, cfg: dict, meta: dict, qids: list[str] | None = None, name: str | None = None) -> dict:
    sd = SetData(root, key, meta, qids)
    out: dict = {"label": name or meta["label"], "n": sd.n, "n_docs": meta["n_docs"], "corpus_sha256": meta["corpus_sha256"],
                 "yes_no_questions": int(sd.is_yn.sum()), "span_questions": int((~sd.is_yn).sum()),
                 "flagged_at_0138": sum(sd.labels[q]["flagged_0138"] for q in sd.qids), "flagged_at_010": sum(sd.labels[q]["flagged_010"] for q in sd.qids),
                 "answer_sentence_opens_article": sum(bool(sd.labels[q]["opens_article"]) for q in sd.qids),
                 "retrieval": retrieval_table(sd.rank, sd.qids, sd.counts)}
    out.update(family_block(root, sd, key, cfg, meta, qids, out["label"], *FAMILIES[FIRST_FAMILY]))
    for family in OTHER_FAMILIES:
        block = family_block(root, sd, key, cfg, meta, qids, out["label"], *FAMILIES[family])
        if not block["answers"]:
            continue
        out.setdefault("families", {})[family] = block
        diff = family_difference(sd, FAMILIES[FIRST_FAMILY], FAMILIES[family])
        if diff:
            out.setdefault("family_difference", {})[family] = diff
    for a, b in BETWEEN:
        if a in out.get("families", {}) and b in out.get("families", {}):
            diff = family_difference(sd, FAMILIES[b], FAMILIES[a])
            if diff:
                out.setdefault("family_difference_between", {})[f"{a} minus {b}"] = diff
    return out


# --------------------------------------------------------------------------
# the defect's size
# --------------------------------------------------------------------------
def defect(root: Path, sets: dict) -> dict:
    new, old = SetData(root, "released", sets["released"]), SetData(root, "released_v020", {**sets["released_v020"]})
    if new.qids != old.qids:
        raise SystemExit("the released set and v0.2.0's are different questions")
    counts = new.counts
    yn = new.is_yn
    c_y = bootstrap_counts(int(yn.sum()), N_BOOT, "czech-v030", "defect", "yes-no").astype(float)
    c_s = bootstrap_counts(int((~yn).sum()), N_BOOT, "czech-v030", "defect", "span").astype(float)
    out: dict = {"questions": new.n, "yes_no_questions": int(yn.sum()), "span_questions": int((~yn).sum()), "configurations": {}}
    for config in GENERATOR_RUNGS:
        a, b = new.config_rows(config), old.config_rows(config)
        if a is None or b is None:
            continue
        ea, eb = np.array([r["em"] for r in a], dtype=float), np.array([r["em"] for r in b], dtype=float)
        fa, fb = np.array([r["f1"] for r in a], dtype=float), np.array([r["f1"] for r in b], dtype=float)
        changed_span = sum(x["answer"] != y["answer"] for x, y, s in zip(a, b, ~yn) if s)
        out["configurations"][config] = {
            "em_v020": float(eb.mean()), "em_fixed": float(ea.mean()),
            "em_difference": mean_ci(ea - eb, counts),
            "em_difference_on_yes_no": ci(float((ea - eb)[yn].mean()), boot_mean((ea - eb)[yn], c_y)),
            "em_difference_on_span": ci(float((ea - eb)[~yn].mean()), boot_mean((ea - eb)[~yn], c_s)),
            "em_on_yes_no_v020": float(eb[yn].mean()), "em_on_yes_no_fixed": float(ea[yn].mean()),
            "em_on_span_v020": float(eb[~yn].mean()), "em_on_span_fixed": float(ea[~yn].mean()),
            "f1_v020": float(fb.mean()), "f1_fixed": float(fa.mean()), "f1_difference": mean_ci(fa - fb, counts),
            "span_answers_whose_string_changed": changed_span,
            "yes_no_classes_v020": answer_block(old, b)["yes_no_answer_classes"], "yes_no_classes_fixed": answer_block(new, a)["yes_no_answer_classes"],
            "secondary_em_yes_no_mapped_v020": float(np.mean([r["em_yes_no_mapped"] for r in b])), "secondary_em_yes_no_mapped_fixed": float(np.mean([r["em_yes_no_mapped"] for r in a])),
        }
    return out


# --------------------------------------------------------------------------
# diagnostics, from the labels
# --------------------------------------------------------------------------
def describe_population(labels: dict, qids: list[str]) -> dict:
    rows = [labels[q] for q in qids]
    n = len(rows)
    span = [r for r in rows if r["type"] != "yes/no"]
    pos = [r["answer_start"] for r in span if r["answer_start"] is not None]
    def med(v):
        return float(statistics.median(v)) if v else None
    def quart(v):
        return [float(x) for x in statistics.quantiles(v, n=4, method="inclusive")] if len(v) > 1 else None
    out = {
        "questions": n, "span_questions": len(span),
        "answer_type": {t: sum(r["type"] == t for r in rows) / n for t in TYPES},
        "sqad_answer_type": {t: sum(r["sqad_a_type"] == t for r in rows) / n for t in sorted({str(r["sqad_a_type"]) for r in rows})},
        "span_answer_tokens_median": med([r["answer_tokens"] for r in span]),
        "span_answer_tokens_mean": float(np.mean([r["answer_tokens"] for r in span])),
        "span_answer_length_band": {b: sum(length_band(r["answer_tokens"]) == b for r in span) / max(len(span), 1) for b in BANDS},
        "answer_start_quartiles": quart(pos), "answer_start_under_50_of_span": sum(p < 50 for p in pos) / max(len(span), 1),
        "answer_not_located_in_its_context": len(span) - len(pos),
        "overlap_quartiles": quart([r["overlap"] for r in rows]),
        "gold_passage_chars_quartiles": quart([r["gold_chars"] for r in rows]),
        "answer_sentence_opens_article": sum(bool(r["opens_article"]) for r in rows) / n,
        "flagged_by_the_rule_0138": sum(bool(r["flagged_0138"]) for r in rows) / n,
        "gold_with_spaced_punctuation": sum(r["spaced_punct"] for r in rows) / n,
        "answer_in_top3_verbatim": {}, "answer_in_top3_normalised": {},
    }
    for field, tab in (("answer_in_top3_verbatim", out["answer_in_top3_verbatim"]), ("answer_in_top3_normalised", out["answer_in_top3_normalised"])):
        for rid in RANK_RETRIEVERS:
            vals = [r[field][rid] for r in span if r[field].get(rid) is not None]
            tab[rid] = sum(vals) / len(vals) if vals else None
    return out


# --------------------------------------------------------------------------
# the gap between SQAD 3.2 and the copy
# --------------------------------------------------------------------------
def band3(tokens: int) -> str:
    return "1" if tokens <= 1 else "2" if tokens == 2 else "3+"


def passage_band(chars: int) -> str:
    return PASSAGE_BANDS[0] if chars < 500 else PASSAGE_BANDS[1] if chars < 800 else PASSAGE_BANDS[2]


def cell_keys(sd: SetData, parts: tuple[str, ...]) -> list[tuple]:
    out = []
    for q in sd.qids:
        lab = sd.labels[q]
        c = []
        for part in parts:
            if part == "type":
                c.append(lab["type"])
            elif part == "length":
                c.append("yes/no" if lab["type"] == "yes/no" else band3(lab["answer_tokens"]))
            elif part == "opens":
                c.append(bool(lab["opens_article"]))
            elif part == "passage":
                c.append(passage_band(lab["gold_chars"]))
            elif part == "flag":
                c.append(bool(lab["flagged_0138"]))
        out.append(tuple(c))
    return out


def reweighted_gap(copy: SetData, target: SetData, config: str, parts: tuple[str, ...], field: str = "em") -> dict | None:
    """Post-stratify the copy's EM to the target's mix of cells, and see what is left of the gap.

    The two sets are different questions, so they are resampled independently. A target cell the copy
    has no question in takes the copy's mean over its whole set in that resample.
    """
    ra, rb = copy.config_rows(config), target.config_rows(config)
    if ra is None or rb is None:
        return None
    ex, et = np.array([r[field] for r in ra], dtype=float), np.array([r[field] for r in rb], dtype=float)
    cx, ct = copy.counts, target.counts
    ka, kb = cell_keys(copy, parts), cell_keys(target, parts)
    cells = sorted(set(kb), key=str)
    em_x_all, em_t_all = boot_mean(ex, cx), boot_mean(et, ct)
    rw_boot = np.zeros(cx.shape[0])
    rw_point = 0.0
    uncovered = 0.0
    for c in cells:
        ix = np.array([k == c for k in ka], dtype=float)
        it = np.array([k == c for k in kb], dtype=float)
        share_pt = float(it.mean())
        if ix.sum():
            rw_point += share_pt * float((ex * ix).sum() / ix.sum())
        else:
            rw_point += share_pt * float(ex.mean())
            uncovered += share_pt
        nx = cx @ ix
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = np.where(nx > 0, (cx @ (ex * ix)) / nx, em_x_all)
        rw_boot += (ct @ it) / ct.shape[1] * mx
    raw_pt, after_pt = float(et.mean() - ex.mean()), float(et.mean()) - rw_point
    return {
        "em_copy": float(ex.mean()), "em_target": float(et.mean()), "em_copy_reweighted": rw_point,
        "gap": ci(raw_pt, em_t_all - em_x_all), "gap_after_reweighting": ci(after_pt, em_t_all - rw_boot),
        "share_of_the_gap_closed": (1 - after_pt / raw_pt) if raw_pt else None,
        "target_share_with_no_copy_question": uncovered, "cells": len(cells),
    }


def em_by(sd: SetData, config: str, key_fn, order: tuple) -> dict:
    rows = sd.config_rows(config)
    if rows is None:
        return {}
    out = {}
    for c in order:
        kept = [r for r, q in zip(rows, sd.qids) if key_fn(sd.labels[q]) == c]
        out[str(c)] = {"n": len(kept), "em": float(np.mean([r["em"] for r in kept])) if kept else None,
                       "em_yes_no_mapped": float(np.mean([r["em_yes_no_mapped"] for r in kept])) if kept else None}
    return out


def subset_gap(copy: SetData, target: SetData, config: str, keep_copy, keep_target, field: str = "em") -> dict | None:
    """The gap between part of one set and part of the other. Different questions, resampled independently."""
    ra, rb = copy.config_rows(config), target.config_rows(config)
    if ra is None or rb is None:
        return None
    ex, et = np.array([r[field] for r in ra], dtype=float), np.array([r[field] for r in rb], dtype=float)
    ix = np.array([bool(keep_copy(copy.labels[q])) for q in copy.qids], dtype=float)
    it = np.array([bool(keep_target(target.labels[q])) for q in target.qids], dtype=float)
    if not ix.sum() or not it.sum():
        return None

    def boot(e: np.ndarray, i: np.ndarray, c: np.ndarray) -> np.ndarray:
        n = c @ i
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (c @ (e * i)) / n, (e * i).sum() / i.sum())

    px, pt = float((ex * ix).sum() / ix.sum()), float((et * it).sum() / it.sum())
    return {"n_copy": int(ix.sum()), "n_target": int(it.sum()), "em_copy": px, "em_target": pt, "gap": ci(pt - px, boot(et, it, target.counts) - boot(ex, ix, copy.counts))}


def gap_by_dial_level(copy: SetData, target: SetData, config: str) -> dict:
    out = {}
    for p in DIAL["levels"]:
        a = {r["q"]: r["em"] for r in copy.rows if r["config"] == config and r["kind"] == "dial" and float(r["p"]) == p}
        b = {r["q"]: r["em"] for r in target.rows if r["config"] == config and r["kind"] == "dial" and float(r["p"]) == p}
        if len(a) == copy.n and len(b) == target.n:
            ex, et = np.array([a[q] for q in copy.qids], dtype=float), np.array([b[q] for q in target.qids], dtype=float)
            out[str(p)] = {"em_copy": float(ex.mean()), "em_target": float(et.mean()), "gap": ci(float(et.mean() - ex.mean()), boot_mean(et, target.counts) - boot_mean(ex, copy.counts))}
    return out


def same_questions_retrieval(root: Path) -> dict | None:
    """Every question in both test splits: does the larger corpus put more gold in the top 3? No generation."""
    la, lb = load_labels(root, "copy_test"), load_labels(root, "sqad32_test")
    pairs = sorted((q, row["same_question"]) for q, row in la.items() if row.get("same_question"))
    if not pairs:
        return None
    ra, rb = load_rankings(root, "copy_test"), load_rankings(root, "sqad32_test")
    counts = bootstrap_counts(len(pairs), N_BOOT, "czech-v030", "same_questions", "retrieval").astype(float)
    out: dict = {"questions": len(pairs)}
    for rid in RANK_RETRIEVERS:
        ga = np.array([gold_in_context(ra[a][rid], {g: 1 for g in ra[a]["gold"]}, 3) for a, b in pairs])
        gb = np.array([gold_in_context(rb[b][rid], {g: 1 for g in rb[b]["gold"]}, 3) for a, b in pairs])
        out[rid] = {"g3_copy": float(ga.mean()), "g3_sqad32": float(gb.mean()), "difference": ci(float(gb.mean() - ga.mean()), boot_mean(gb - ga, counts))}
    for name, fn in (("gold_chars_median", lambda r: r["gold_chars"]), ("answer_start_median", lambda r: r["answer_start"])):
        out[name] = {side: float(statistics.median(v)) for side, v in (("copy", [fn(la[a]) for a, b in pairs if fn(la[a]) is not None]), ("sqad32", [fn(lb[b]) for a, b in pairs if fn(lb[b]) is not None]))}
    out["answer_sentence_opens_article"] = {"copy": sum(bool(la[a]["opens_article"]) for a, b in pairs), "sqad32": sum(bool(lb[b]["opens_article"]) for a, b in pairs)}
    return out


def same_questions_answers(root: Path) -> dict | None:
    """One question, one model, two corpora: only the passages differ. Paired. The first cut's small run."""
    rows = load_first_cut_same_questions() if is_first_cut(root) else []
    if not rows:
        return None
    a = {r["q"]: r for r in rows if r["corpus"] == "copy"}
    b = {r["q"]: r for r in rows if r["corpus"] == "sqad32"}
    pairs = sorted((q, r["other_q"]) for q, r in a.items())
    if {b[y]["other_q"] for _, y in pairs} != set(a):
        raise SystemExit("same_questions: the pairs do not match")
    counts = bootstrap_counts(len(pairs), N_BOOT, "czech-v030", "same_questions", "answers").astype(float)

    def col(side: dict, ids: list[str], field: str) -> np.ndarray:
        return np.array([side[q][field] for q in ids], dtype=float)

    ids_a, ids_b = [x for x, _ in pairs], [y for _, y in pairs]
    ea, eb = col(a, ids_a, "em"), col(b, ids_b, "em")

    def diff(x: np.ndarray, y: np.ndarray) -> dict:
        return ci(float(y.mean() - x.mean()), boot_mean(y - x, counts))

    return {
        "questions": len(pairs), "config": rows[0]["config"], "max_tokens": rows[0]["max_tokens"], "seed": rows[0]["seed"],
        "from_the_released_200": sum(a[x]["from"] == "released" for x in ids_a), "from_sqad32_full": sum(b[y]["from"] == "sqad32_full" for y in ids_b),
        "em_on_copy_passages": float(ea.mean()), "em_on_sqad32_passages": float(eb.mean()), "sqad32_passages_minus_copy_passages": diff(ea, eb),
        "em_with_the_copys_gold_on_both": diff(ea, col(b, ids_b, "em_other_key")), "em_with_sqad32s_gold_on_both": diff(col(a, ids_a, "em_other_key"), eb),
        "f1": diff(col(a, ids_a, "f1"), col(b, ids_b, "f1")),
        "em_yes_no_mapped": diff(col(a, ids_a, "em_yes_no_mapped"), col(b, ids_b, "em_yes_no_mapped")),
        "same_answer_string": sum(normalize_answer(a[x]["answer"]) == normalize_answer(b[y]["answer"]) for x, y in pairs),
        "right_only_on_copy_passages": int(((ea == 1) & (eb == 0)).sum()), "right_only_on_sqad32_passages": int(((ea == 0) & (eb == 1)).sum()),
    }


def gap_analysis(root: Path, sets: dict) -> dict:
    out: dict = {"em_by_answer_type": {}, "em_by_length_band": {}, "em_by_opening_sentence": {}, "em_by_gold_passage_length": {}, "em_by_answer_in_top3": {}, "em_by_origin": {}, "em_by_the_debias_rule": {},
                 "reweighting": {}, "on_questions_the_copy_also_holds": {}, "on_questions_the_rule_passes": {}, "gap_by_dial_level": {},
                 "same_questions_retrieval": same_questions_retrieval(root), "same_questions_answers": same_questions_answers(root)}
    loaded = {k: SetData(root, k, sets[k]) for k in MAIN_SETS if k in sets}
    for key, sd in loaded.items():
        out["em_by_answer_type"][key] = {c: em_by(sd, c, lambda l: l["type"], TYPES) for c in GENERATOR_RUNGS}
        out["em_by_length_band"][key] = {c: em_by(sd, c, lambda l: "yes/no" if l["type"] == "yes/no" else length_band(l["answer_tokens"]), (*BANDS, "yes/no")) for c in GENERATOR_RUNGS}
        out["em_by_opening_sentence"][key] = {c: em_by(sd, c, lambda l: bool(l["opens_article"]), (True, False)) for c in GENERATOR_RUNGS}
        out["em_by_gold_passage_length"][key] = {c: em_by(sd, c, lambda l: passage_band(l["gold_chars"]), PASSAGE_BANDS) for c in GENERATOR_RUNGS}
        out["em_by_the_debias_rule"][key] = {c: em_by(sd, c, lambda l: bool(l["flagged_0138"]), (True, False)) for c in GENERATOR_RUNGS}
        out["em_by_answer_in_top3"][key] = {c: em_by(sd, c, lambda l, rid=c.split("+")[0]: "yes/no" if l["type"] == "yes/no" else "answer in the top 3" if l["answer_in_top3_normalised"].get(rid) else "answer not in the top 3",
                                                     ("answer in the top 3", "answer not in the top 3", "yes/no")) for c in GENERATOR_RUNGS}
        if sd.population == "sqad32_test":
            out["em_by_origin"][key] = {c: em_by(sd, c, lambda l: l["origin"], ORIGINS) for c in GENERATOR_RUNGS}
    pairs = (("released", "sqad32_full"), ("copy_debiased", "sqad32_debiased"))
    variants = (("answer type and length", ("type", "length")), ("the opening sentence", ("opens",)), ("answer type, length and the opening sentence", ("type", "length", "opens")),
                ("the gold passage's length", ("passage",)), ("the debias rule", ("flag",)))
    for a, b in pairs:
        if a not in loaded or b not in loaded:
            continue
        tab: dict = {}
        for config in (f"e5-small+{BASE}", f"e5-small+{BIG}", f"bm25+{BASE}"):
            tab[config] = {name: reweighted_gap(loaded[a], loaded[b], config, parts) for name, parts in variants}
        out["reweighting"][f"{b} minus {a}"] = tab
        out["on_questions_the_copy_also_holds"][f"{b} minus {a}"] = {
            config: {"all of the copy's set": subset_gap(loaded[a], loaded[b], config, lambda l: True, lambda l: l["origin"] == "in the copy"),
                     "answer not in an opening sentence": subset_gap(loaded[a], loaded[b], config, lambda l: not l["opens_article"], lambda l: l["origin"] == "in the copy" and not l["opens_article"])}
            for config in GENERATOR_RUNGS}
        out["gap_by_dial_level"][f"{b} minus {a}"] = {config: gap_by_dial_level(loaded[a], loaded[b], config) for config in (f"e5-small+{BASE}", f"e5-small+{BIG}")}
        out["on_questions_the_rule_passes"][f"{b} minus {a}"] = {config: subset_gap(loaded[a], loaded[b], config, lambda l: not l["flagged_0138"], lambda l: not l["flagged_0138"]) for config in GENERATOR_RUNGS}
    return out


# --------------------------------------------------------------------------
# the claims, against v0.2.0
# --------------------------------------------------------------------------
def dial_verdict(cs: dict) -> str:
    crossed = cs["root_isotonic"] is not None or cs["root_linear"] is not None
    if cs["levels_established_retriever"] > cs["levels_compared"] / 2:
        return "REVERSES"
    if crossed or cs["levels_established_retriever"] > 0 or cs["levels_retriever_larger"] > cs["levels_generator_larger"]:
        return "WEAKENS"
    return "HOLDS"


DIAL_CLAIM_FIELDS = ("levels_generator_larger", "levels_retriever_larger", "levels_tied", "levels_established_generator", "levels_established_retriever", "levels_compared", "root_isotonic", "root_linear")


def claims(sets_out: dict, large: dict, base_of) -> dict:
    """Every Czech claim, against a comparator: `base_of(set key)` is v0.2.0's block or the first cut's on that set.
    The field named v020 holds the comparator's value."""
    out: dict = {}
    for key in MAIN_SETS:
        cur = sets_out.get(key)
        base = base_of(key)
        if not cur or not base:
            out[key] = {"status": "not run"}
            continue
        bd = base["dial_summary"]
        base_steps = base["steps"]["em"]
        v: dict = {"label": cur["label"], "n": cur["n"]}
        if "steps" in cur:
            st = cur["steps"]["em"]
            v["generator_step_is_established"] = {"value": st["generator_step"], "v020": base_steps["generator_step"], "verdict": verdict_sign(base_steps["generator_step"], st["generator_step"])}
            r = st["retriever_step"]
            v["retriever_step_is_not_established"] = {"value": r, "v020": base_steps["retriever_step"], "verdict": "HOLDS" if not r["established"] else ("REVERSES" if r["point"] > 0 else "WEAKENS")}
            v["generator_step_exceeds_retriever_step"] = {"value": st["generator_step_minus_retriever_step"], "v020": base_steps["generator_step_minus_retriever_step"],
                                                          "verdict": verdict_sign(base_steps["generator_step_minus_retriever_step"], st["generator_step_minus_retriever_step"])}
            # the same step once yes counts as ano: what is left when the language of the answer is set aside
            sm, bm = cur["steps"]["em_yes_no_mapped"], base["steps"]["em_yes_no_mapped"]
            m, mb = sm["generator_step"], bm["generator_step"]
            v["generator_step_with_yes_and_no_mapped"] = {"value": m, "v020": mb, "verdict": verdict_sign(mb, m)}
            rm = sm["retriever_step"]
            v["retriever_step_is_not_established_with_yes_and_no_mapped"] = {"value": rm, "v020": bm["retriever_step"], "verdict": "HOLDS" if not rm["established"] else ("REVERSES" if rm["point"] > 0 else "WEAKENS")}
            v["generator_step_exceeds_retriever_step_with_yes_and_no_mapped"] = {"value": sm["generator_step_minus_retriever_step"], "v020": bm["generator_step_minus_retriever_step"],
                                                                                "verdict": verdict_sign(bm["generator_step_minus_retriever_step"], sm["generator_step_minus_retriever_step"])}
            # "counting both kinds of miss as correct barely moves the steps": moved by 0.02 or less
            tol = cur["steps"]["tolerant"]
            moved = {name: abs(tol[name]["point"] - st[name]["point"]) for name in ("generator_step", "retriever_step")}
            base_tol = base["steps"]["tolerant"]
            v["a_tolerant_metric_barely_moves_the_steps"] = {"moved": moved, "tolerant_generator_step": tol["generator_step"], "tolerant_retriever_step": tol["retriever_step"],
                                                             "v020_moved": {name: abs(base_tol[name]["point"] - base_steps[name]["point"]) for name in moved},
                                                             "verdict": "HOLDS" if max(moved.values()) <= 0.02 else "WEAKENS"}
        if "dial_summary" in cur:
            cs = cur["dial_summary"]
            v["dial_has_no_crossing_and_leans_to_the_generator"] = {k: cs[k] for k in DIAL_CLAIM_FIELDS} | {"verdict": dial_verdict(cs)}
            mp = cur["dial_summary_em_yes_no_mapped"]
            v["dial_has_no_crossing_and_leans_to_the_generator_with_yes_and_no_mapped"] = {k: mp[k] for k in DIAL_CLAIM_FIELDS} | {"verdict": dial_verdict(mp)}
            est = cs["levels_established_generator"] + cs["levels_established_retriever"]
            v["the_dial_cannot_separate_the_two_moves"] = {"established": est, "established_with_yes_and_no_mapped": mp["levels_established_generator"] + mp["levels_established_retriever"], "levels_compared": cs["levels_compared"],
                                                           "v020_established": bd["levels_established_generator"] + bd["levels_established_retriever"],
                                                           "v020_established_with_yes_and_no_mapped": base["dial_summary_em_yes_no_mapped"]["levels_established_generator"] + base["dial_summary_em_yes_no_mapped"]["levels_established_retriever"],
                                                           "verdict": "HOLDS" if est == 0 else ("REVERSES" if est > cs["levels_compared"] / 2 else "WEAKENS")}
            rb = {b["p"]: b for b in bd["break_even_raw"]}
            inside = located = 0
            for b in cs["break_even_raw"]:
                r_ = rb[b["p"]]
                if b["point"] is not None and r_["point"] is not None:
                    located += 1
                    inside += r_["lo"] <= b["point"] <= r_["hi"]
            v["break_even_values_sit_inside_v020_intervals"] = {"v020_located": sum(b["point"] is not None for b in bd["break_even_raw"]), "located_in_both": located, "inside": inside,
                                                                "points": {str(b["p"]): b["point"] for b in cs["break_even_raw"] if b["point"] is not None},
                                                                "v020_points": {str(b["p"]): b["point"] for b in bd["break_even_raw"] if b["point"] is not None},
                                                                "verdict": "WEAKENS" if located == 0 else ("HOLDS" if inside == located else ("REVERSES" if inside == 0 else "WEAKENS"))}
            v["non_gold_passages_leak_answers"] = {"answer_in_nongold": cs["answer_in_nongold_no_gold"], "v020": bd["answer_in_nongold_no_gold"], "em_no_gold_base": cs["em_no_gold_base"], "em_no_gold_big": cs["em_no_gold_big"],
                                                   "v020_em_no_gold_base": bd["em_no_gold_base"], "v020_em_no_gold_big": bd["em_no_gold_big"],
                                                   "verdict": "HOLDS" if cs["answer_in_nongold_no_gold"] >= 0.5 * bd["answer_in_nongold_no_gold"] and cs["em_no_gold_base"] >= 0.5 * bd["em_no_gold_base"] else "WEAKENS"}
        if "e5small_minus_retromae_em" in cur:
            v["retromae_answers_worst"] = {"value": cur["e5small_minus_retromae_em"], "v020": base["e5small_minus_retromae_em"], "verdict": verdict_sign(base["e5small_minus_retromae_em"], cur["e5small_minus_retromae_em"])}
            v["retromae_answers_worst_with_yes_and_no_mapped"] = {"value": cur["e5small_minus_retromae_mapped"], "v020": base["e5small_minus_retromae_mapped"], "verdict": verdict_sign(base["e5small_minus_retromae_mapped"], cur["e5small_minus_retromae_mapped"])}
        g = cur["retrieval"]["deltas"]["e5-small - bm25"]["g3"]
        v["e5small_puts_more_gold_in_front_of_the_generator"] = {"value": g, "v020": base["retrieval"]["deltas"]["e5-small - bm25"]["g3"], "verdict": verdict_sign(base["retrieval"]["deltas"]["e5-small - bm25"]["g3"], g)}
        x = cur["retrieval"]["deltas"]["e5-small - bm25"]["ndcg10"]
        v["e5small_beats_bm25_on_ndcg_in_this_sample"] = {"value": x, "v020": base["retrieval"]["deltas"]["e5-small - bm25"]["ndcg10"], "verdict": verdict_sign(base["retrieval"]["deltas"]["e5-small - bm25"]["ndcg10"], x)}
        yn = [a.get("em_on_yes_no") for a in cur["answers"].values() if a.get("em_on_yes_no") is not None]
        v["every_yes_no_question_scores_0"] = {"em_on_yes_no_by_configuration": {c: a.get("em_on_yes_no") for c, a in cur["answers"].items()}, "v020": {c: a.get("em_on_yes_no") for c, a in base["answers"].items()}, "verdict": "REVERSES" if any(yn) else "HOLDS"}
        v["exact_match_under_counts"] = {"points_by_configuration": {c: {"all": a["undercount_points"], "yes_no": a["undercount_yes_no_points"], "declension": a["undercount_declension_points"]} for c, a in cur["answers"].items()},
                                         "v020_by_configuration": {c: {"all": a["undercount_points"], "yes_no": a["undercount_yes_no_points"], "declension": a["undercount_declension_points"]} for c, a in base["answers"].items()}}
        v["exact_match_under_counts"]["verdict"] = "HOLDS" if any(a["undercount_points"] > 0 for a in cur["answers"].values()) else "REVERSES"
        # the fix was meant to remove the yes/no part: gone means under one point in every configuration
        yn_pts = {c: a["undercount_yes_no_points"] for c, a in cur["answers"].items()}
        v["the_fixed_prompt_removes_the_yes_no_under_count"] = {"points_by_configuration": yn_pts, "v020_by_configuration": {c: a["undercount_yes_no_points"] for c, a in base["answers"].items()},
                                                                "verdict": "HOLDS" if max(yn_pts.values()) < 1.0 else "WEAKENS"}
        de_pts = {c: a["undercount_declension_points"] for c, a in cur["answers"].items()}
        v["declension_still_under_counts"] = {"points_by_configuration": de_pts, "v020_by_configuration": {c: a["undercount_declension_points"] for c, a in base["answers"].items()},
                                              "verdict": "HOLDS" if max(de_pts.values()) > 0 else "WEAKENS"}
        if "token_cap_check" in cur and "token_cap_check" in base:
            tc = cur["token_cap_check"]
            v["the_token_cap_does_not_change_an_answer"] = {**tc, "v020": base["token_cap_check"], "verdict": "HOLDS" if tc["same_string"] == tc["questions"] else "WEAKENS"}
        out[key] = v
    # the large samples: retrieval, so the prompt cannot touch them
    out["large_samples"] = {}
    for pop, row in large.items():
        d = row["all"]["deltas"]
        lead = d["e5-small - bm25-nodiacritics"]["ndcg10"]
        out["large_samples"][pop] = {"n": row["n"], "e5small_minus_bm25_ndcg": d["e5-small - bm25"]["ndcg10"], "e5small_minus_bm25_nodiacritics_ndcg": lead,
                                     "verdict": "HOLDS" if lead["established"] and lead["point"] > 0 else "WEAKENS"}
    return out


# --------------------------------------------------------------------------
# the bench's numbers, recomputed
# --------------------------------------------------------------------------
def bench_check(root: Path, sets: dict) -> dict:
    out = {}
    files = (("released", f"{FIRST_CUT}/results_cz_rag.json"), ("sqad32_full", f"{FIRST_CUT}/results_cz_rag_32.json")) if is_first_cut(root) else (("released", "results/cz_rag/results.json"), ("sqad32_full", "results/cz_rag_32/results.json"))
    for key, rel in files:
        path = REPO / rel
        if key not in sets or not path.exists():
            continue
        sd = SetData(root, key, sets[key])
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks = {}
        for r in payload["results"]:
            cfg = r["config_id"]
            rows = sd.bench_rows(cfg)
            if rows is None:
                checks[cfg] = {"records": False}
                continue
            rid = r["retriever_id"]
            rel_ = {q: {g: 1 for g in sd.rank[q]["gold"]} for q in sd.qids}
            checks[cfg] = {
                "records": True,
                "em": abs(np.mean([x["em"] for x in rows]) - r["answers"]["em"]) < 1e-9,
                "f1": abs(np.mean([x["f1"] for x in rows]) - r["answers"]["f1"]) < 1e-9,
                "ndcg10": abs(np.mean([ndcg_at_k(sd.rank[q][rid], rel_[q], 10) for q in sd.qids]) - r["metrics"]["ndcg@10"]) < 1e-9,
                "g3": abs(np.mean([gold_in_context(sd.rank[q][rid], rel_[q], 3) for q in sd.qids]) - r["metrics"]["gold_in_context@3"]) < 1e-9,
            }
        out[key] = {"file": rel, "corpus_matches": payload["datasets"][0]["corpus_sha256"] == sets[key]["corpus_sha256"], "configurations": checks,
                    "all_match": all(all(v.values()) for v in checks.values())}
    return out


# --------------------------------------------------------------------------
# what the complete run adds
# --------------------------------------------------------------------------
GAP_CONFIGS = tuple(c for small, big in FAMILIES.values() for c in (f"e5-small+{small}", f"e5-small+{big}", f"bm25+{small}"))
MIX = ("type", "length")


def metric_arrays(sd: SetData, field: str, configs: tuple[str, ...] = GENERATOR_RUNGS) -> dict[str, np.ndarray]:
    out = {}
    for config in configs:
        rows = sd.config_rows(config)
        if rows is not None:
            out[config] = np.array([r[field] for r in rows], dtype=float)
    return out


def conjunction(vals: dict[str, dict]) -> dict:
    established_on = [k for k, v in vals.items() if v["established"]]
    signs = {float(np.sign(v["point"])) for v in vals.values() if v["established"]}
    return {"established_on": established_on, "holds_on_all_four": len(established_on) == len(MAIN_SETS) and len(signs) == 1, "sign": signs.pop() if len(signs) == 1 else None}


def holds_on_all_four(sets_out: dict) -> dict:
    """Section 6: a claim across sets is a conjunction, established with the same sign on each of the four. Per family, and for
    each family difference against Qwen2.5."""
    out: dict = {}
    for family in FAMILIES:
        if not any("steps" in family_view(sets_out.get(k, {}), family) for k in MAIN_SETS):
            continue                                   # a family with no answers yet adds nothing
        out[family] = {}
        for field in METRICS:
            out[family][field] = {}
            for step in ("generator_step", "retriever_step", "generator_step_minus_retriever_step"):
                views = {k: family_view(sets_out.get(k, {}), family) for k in MAIN_SETS}
                vals = {k: v["steps"][field][step] for k, v in views.items() if "steps" in v}
                if len(vals) == len(MAIN_SETS):
                    out[family][field][step] = conjunction(vals)
    for family in OTHER_FAMILIES:
        if family not in out:
            continue
        name = f"{family} minus {FIRST_FAMILY}"
        out[name] = {}
        for field in METRICS:
            out[name][field] = {}
            for size in SIZES:
                vals = {k: sets_out[k]["family_difference"][family][size][field] for k in MAIN_SETS if size in sets_out.get(k, {}).get("family_difference", {}).get(family, {})}
                if len(vals) == len(MAIN_SETS):
                    out[name][field][size] = conjunction(vals)
    for a, b in BETWEEN:
        name = f"{a} minus {b}"
        if a not in out or b not in out:
            continue
        out[name] = {}
        for field in METRICS:
            out[name][field] = {}
            for size in SIZES:
                vals = {k: sets_out[k]["family_difference_between"][name][size][field] for k in MAIN_SETS if size in sets_out.get(k, {}).get("family_difference_between", {}).get(name, {})}
                if len(vals) == len(MAIN_SETS):
                    out[name][field][size] = conjunction(vals)
    return out


def continuity(root: Path, first_root: Path, sets: dict, first_sets: dict) -> dict:
    """The released 200: this run against the first cut's answers and against v0.2.0's, paired (H3)."""
    new = SetData(root, "released", sets["released"])
    others = {"first_cut": SetData(first_root, "released", first_sets["released"]), "v020": SetData(root, "released_v020", sets["released_v020"])}
    out: dict = {}
    for name, old in others.items():
        if new.qids != old.qids:
            raise SystemExit(f"the released set and {name}'s are different questions")
        tab: dict = {}
        for config in GENERATOR_RUNGS:
            a, b = new.config_rows(config), old.config_rows(config)
            if a is None or b is None:
                continue
            block: dict = {"same_string": sum(x["answer"] == y["answer"] for x, y in zip(a, b)), "questions": new.n}
            for field in METRICS:
                ea, eb = np.array([r[field] for r in a], dtype=float), np.array([r[field] for r in b], dtype=float)
                block[field] = {"this_run": float(ea.mean()), "other": float(eb.mean()), "difference": mean_ci(ea - eb, new.counts)}
            block["compliance_this_run"] = compliance_block(yes_no_rows(new, a), new.yes_no_counts())
            block["compliance_other"] = compliance_block(yes_no_rows(old, b), new.yes_no_counts())
            block["compliance_difference"] = compliance_difference(yes_no_rows(new, a), yes_no_rows(old, b), new.yes_no_counts())
            tab[config] = block
        out[name] = tab
    return out


def cross_data() -> dict | None:
    """The cross-version check as arrays, one question per position: the copy's passages, then SQAD 3.2's."""
    rows = load_cross_version()
    if not rows:
        return None
    pairs = [tuple(p) for p in json.loads((PLAN / "questions.json").read_text(encoding="utf-8"))["cross_version"]]
    same_gold = {(r["copy"][0]["id"], r["sqad32"][0]["id"]): bool(r["same_gold"]) for r in read_jsonl_gz(PLAN / "question_join.jsonl.gz") if r["copy"] and r["sqad32"]}
    side = {s: {(r["q"], r["config"]): r for r in rows if r["corpus"] == s} for s in ("copy", "sqad32")}
    out: dict = {"pairs": pairs, "same_gold": np.array([same_gold[p] for p in pairs]), "generators": {}}
    for g in GENERATORS:
        config = f"e5-small+{g}"
        if all((a, config) in side["copy"] and (b, config) in side["sqad32"] for a, b in pairs):
            out["generators"][g] = {"copy": [side["copy"][(a, config)] for a, _ in pairs], "sqad32": [side["sqad32"][(b, config)] for _, b in pairs],
                                    "counts": bootstrap_counts(len(pairs), N_BOOT, "czech-v030", "cross_version", g).astype(float)}
    return out


def cross_column(rows: list[dict], field: str) -> np.ndarray:
    return np.array([r[field] for r in rows], dtype=float)


def cross_version(cd: dict) -> dict:
    """One question, one generator, each version's passages. Paired. Own gold unless a row says otherwise."""
    same = cd["same_gold"]
    out: dict = {"pairs": len(cd["pairs"]), "same_gold_pairs": int(same.sum()), "generators": {}}
    for g, d in cd["generators"].items():
        a, b, counts = d["copy"], d["sqad32"], d["counts"]
        same_counts = bootstrap_counts(int(same.sum()), N_BOOT, "czech-v030", "cross_version", g, "same-gold").astype(float)
        block: dict = {}

        def diff(x: np.ndarray, y: np.ndarray, c: np.ndarray = counts) -> dict:
            return {"copy": float(x.mean()), "sqad32": float(y.mean()), "difference": mean_ci(y - x, c)}

        for field in (*METRICS, "f1"):
            x, y = cross_column(a, field), cross_column(b, field)
            block[field] = diff(x, y)
            block[f"{field}_same_gold"] = diff(x[same], y[same], same_counts)
        block["em_with_the_copys_gold_on_both"] = diff(cross_column(a, "em"), cross_column(b, "em_other_key"))
        block["em_with_sqad32s_gold_on_both"] = diff(cross_column(a, "em_other_key"), cross_column(b, "em"))
        block["gold_in_top3"] = diff(cross_column(a, "gold_in_top3"), cross_column(b, "gold_in_top3"))
        ea, eb = cross_column(a, "em"), cross_column(b, "em")
        block["same_answer_string"] = sum(normalize_answer(x["answer"]) == normalize_answer(y["answer"]) for x, y in zip(a, b))
        block["right_only_on_copy_passages"] = int(((ea == 1) & (eb == 0)).sum())
        block["right_only_on_sqad32_passages"] = int(((ea == 0) & (eb == 1)).sum())
        yes_no = [i for i, r in enumerate(a) if r["type"] == "yes/no"]
        if yes_no:
            c_y = bootstrap_counts(len(yes_no), N_BOOT, "czech-v030", "cross_version", g, "yes-no").astype(float)
            block["compliance"] = {"copy": compliance_block([a[i] for i in yes_no], c_y), "sqad32": compliance_block([b[i] for i in yes_no], c_y)}
        out["generators"][g] = block
    # the generator step on each version's passages: a second reading of H1 on the copy, whose test split is capped.
    # Every family's step and every family difference read the same resamples of the same questions.
    gens = cd["generators"]
    counts = {s: bootstrap_counts(len(cd["pairs"]), N_BOOT, "czech-v030", "cross_version", "generator-step", s).astype(float) for s in ("copy", "sqad32")}

    def minus(a: str, b: str, s: str) -> dict:
        return {field: mean_ci(cross_column(gens[a][s], field) - cross_column(gens[b][s], field), counts[s]) for field in METRICS}

    out["generator_step"] = {family: {s: minus(big, small, s) for s in counts} for family, (small, big) in FAMILIES.items() if small in gens and big in gens}
    out["family_difference"] = {}
    for family in OTHER_FAMILIES:
        block = {size: {s: minus(a, b, s) for s in counts} for size, a, b in zip(SIZES, FAMILIES[family], FAMILIES[FIRST_FAMILY]) if a in gens and b in gens}
        if block:
            out["family_difference"][family] = block
    for a_fam, b_fam in BETWEEN:
        block = {size: {s: minus(a, b, s) for s in counts} for size, a, b in zip(SIZES, FAMILIES[a_fam], FAMILIES[b_fam]) if a in gens and b in gens}
        if block:
            out.setdefault("family_difference_between", {})[f"{a_fam} minus {b_fam}"] = block
    return out


def planned_gap(loaded: dict, cd: dict | None) -> dict:
    """Section 7, in order: the gaps, the mix, the rule, the passages, what is left. Different questions are resampled independently."""
    out: dict = {}
    for field in METRICS:
        out[field] = {}
        for config in GAP_CONFIGS:
            block: dict = {}
            for name, tkey, ckey in (("G1", "sqad32_full", "released"), ("G2", "sqad32_debiased", "copy_debiased")):
                if tkey not in loaded or ckey not in loaded:
                    continue
                t, c = loaded[tkey], loaded[ckey]
                whole = subset_gap(c, t, config, lambda _: True, lambda _: True, field)
                if whole is None:
                    continue
                block[name] = {"gap": whole, "after_matching_answer_type_and_length": reweighted_gap(c, t, config, MIX, field)}
            if "G1" in block:
                rule = subset_gap(loaded["released"], loaded["sqad32_full"], config, lambda l: not l["flagged_0138"], lambda l: not l["flagged_0138"], field)
                g1 = block["G1"]["gap"]["gap"]["point"]
                block["G1"]["on_questions_the_rule_passes"] = rule
                block["G1"]["share_of_the_gap_the_rule_accounts_for"] = (1 - rule["gap"]["point"] / g1) if (rule and g1) else None
            generator = config.split("+", 1)[1]
            if cd and config.startswith("e5-small") and generator in cd["generators"]:
                d = cd["generators"][generator]
                x, y = cross_column(d["copy"], field), cross_column(d["sqad32"], field)
                passages = mean_ci(y - x, d["counts"])
                same_counts = bootstrap_counts(int(cd["same_gold"].sum()), N_BOOT, "czech-v030", "cross_version", generator, "same-gold").astype(float)
                block["passages"] = {"own_gold": passages, "same_gold": mean_ci((y - x)[cd["same_gold"]], same_counts)}
                if "G2" in block:
                    t, c = loaded["sqad32_debiased"], loaded["copy_debiased"]
                    et, ec = np.array([r[field] for r in t.config_rows(config)], dtype=float), np.array([r[field] for r in c.config_rows(config)], dtype=float)
                    g2_boot = boot_mean(et, t.counts) - boot_mean(ec, c.counts)
                    block["what_is_left"] = ci(float(et.mean() - ec.mean()) - passages["point"], g2_boot - boot_mean(y - x, d["counts"]))
            if block:                                  # a configuration with no answers yet adds nothing
                out[field][config] = block
    return out


def added_questions(root: Path, sets: dict, first_sets: dict) -> dict:
    """Section 8: H1, H2 and the gap's steps 1 to 3 on the questions this run added, which had no part in forming the
    hypotheses. The released 200 has none, so G1's other side stays the released 200."""
    out: dict = {"sets": {}}
    loaded: dict = {}
    for key in MAIN_SETS:
        if key not in sets or key not in first_sets:
            continue
        added = sorted(set(sets[key]["questions"]) - set(first_sets[key]["questions"]))
        if not added:
            continue
        sd = SetData(root, key, sets[key], added)
        loaded[key] = sd
        block: dict = {"questions": sd.n, "steps": {}}
        for family, (small, big) in FAMILIES.items():
            steps: dict = {}
            for field in METRICS:
                arrays = metric_arrays(sd, field, rungs(small, big))
                if all(c in arrays for c in (f"e5-small+{big}", f"e5-small+{small}", f"bm25+{small}")):
                    steps[field] = steps_block(sd, arrays, small, big)
            if family == FIRST_FAMILY:
                block["steps"] = steps
            elif steps:
                block.setdefault("families", {})[family] = {"steps": steps}
        out["sets"][key] = block
    if "released" in sets:
        loaded["released"] = SetData(root, "released", sets["released"])
    out["gap"] = planned_gap(loaded, None)
    return out


def order_check(root: Path, sets: dict) -> dict | None:
    """Section 4: the copy's 0.10 row holds the same questions as part of its 0.138 set, with the same prompts. How many
    answers differ measures the order effect at this size."""
    if "copy_debiased" not in sets or "extra010_copy" not in sets:
        return None
    a = {(r["q"], r["config"], r["kind"], float(r["p"])): r for r in load_answers(root, "copy_debiased")}
    b = {(r["q"], r["config"], r["kind"], float(r["p"])): r for r in load_answers(root, "extra010_copy")}
    out: dict = {}
    for g in GENERATORS:
        both = [k for k in b if k in a and k[1].endswith("+" + g)]
        if not both:
            continue
        out[g] = {"answers_compared": len(both), "same_string": sum(a[k]["answer"] == b[k]["answer"] for k in both), "same_em": sum(a[k]["em"] == b[k]["em"] for k in both),
                  "same_em_yes_no_mapped": sum(a[k]["em_yes_no_mapped"] == b[k]["em_yes_no_mapped"] for k in both)}
    return out


THINKING_FILES = tuple(f"answers/sqad32/answers_{n}.jsonl.gz" for n in ("sqad32_debiased", "sqad32_full", "extra010_sqad32", "cross_version")) + \
    tuple(f"answers/sqad30_copy/answers_{n}.jsonl.gz" for n in ("released", "copy_debiased", "extra010_copy", "cross_version"))


def thinking_tags() -> dict:
    """Every test-set answer of Qwen3.5 that holds a thinking tag is counted and reported; none is removed."""
    out: dict = {"files": list(THINKING_FILES), "tags": ["<think", "</think"]}
    for g in FAMILIES["qwen3.5"]:
        out[g] = {"answer_rows": 0, "with_a_tag": 0}
    for rel in THINKING_FILES:
        p = REPO / rel
        if not p.exists():
            continue
        for r in read_jsonl_gz(p):
            g = r["config"].split("+")[-1]
            if g in FAMILIES["qwen3.5"]:
                out[g]["answer_rows"] += 1
                out[g]["with_a_tag"] += int(any(tag in r["answer"] for tag in out["tags"]))
    return out


def complete_compute() -> dict | None:
    path = REPO / "results/czech/complete/compute.json"
    if not path.exists():
        return None
    c = json.loads(path.read_text(encoding="utf-8"))

    def total(stages: list[dict]) -> dict:
        return {"stages": len(stages), "generations": sum(int(s.get("generations", 0)) for s in stages), "seconds": round(sum(float(s.get("seconds", 0.0)) for s in stages), 1)}

    dev = [s for n, s in c["stages"].items() if n.startswith("dev_")]
    main_ = [s for n, s in c["stages"].items() if not n.startswith("dev_")]
    out = {"dev": total(dev), "main": total(main_), "superseded_v3": total(list(c.get("superseded_v3", {}).values())), "superseded_mac": total(list(c.get("superseded_mac", {}).values())),
           "restarts": len(c.get("restarts", [])), "warnings": list(c.get("warnings", []))}
    # the GX10: its pilot, and then every generation of the main run
    gx = REPO / "results/czech/complete/compute_gx10.json"
    if gx.exists():
        g = json.loads(gx.read_text(encoding="utf-8"))
        pilot = [s for n, s in g["stages"].items() if n.startswith(("dev_", "pilot_"))]
        on_gx10 = [s for n, s in g["stages"].items() if not n.startswith(("dev_", "pilot_"))]
        out["main"] = total(main_ + on_gx10)
        out["gx10"] = {"pilot": total(pilot), "main": total(on_gx10), "restarts": len(g.get("restarts", [])), "warnings": list(g.get("warnings", []))}
    # v0.3.1: Ministral 3, alone, in its own record. Not part of v0.3.0's main run.
    gm = REPO / "results/czech/complete/compute_gx10_ministral.json"
    if gm.exists():
        g = json.loads(gm.read_text(encoding="utf-8"))
        out["gx10_ministral"] = {**total(list(g["stages"].values())), "restarts": len(g.get("restarts", [])), "warnings": list(g.get("warnings", []))}
    # v0.4.0: Qwen3.5, alone, in its own record
    gq = REPO / "results/czech/complete/compute_gx10_qwen35.json"
    if gq.exists():
        g = json.loads(gq.read_text(encoding="utf-8"))
        out["gx10_qwen35"] = {**total(list(g["stages"].values())), "restarts": len(g.get("restarts", [])), "warnings": list(g.get("warnings", []))}
    # v0.4.2: Gemma 3n, alone, in its own record
    gn = REPO / "results/czech/complete/compute_gx10_gemma3n.json"
    if gn.exists():
        g = json.loads(gn.read_text(encoding="utf-8"))
        out["gx10_gemma3n"] = {**total(list(g["stages"].values())), "restarts": len(g.get("restarts", [])), "warnings": list(g.get("warnings", []))}
    return out


# --------------------------------------------------------------------------
def analyse(root: Path, cfg: dict, first: dict | None = None) -> tuple[dict, dict | None]:
    """`first` is the first cut's analysis, given when `root` is the complete run's records."""
    sets = load_sets(root)["sets"]
    out: dict = {"note": "Recomputed by scripts/rescore_czech.py from results/czech/records/. Uncertainty is the paired bootstrap of scripts/crossover.py, 95%.",
                 "sets": {}, "sensitivity_010": {}, "large_retrieval": {}, "diagnostics": {}}
    for key in (*MAIN_SETS, "released_v020"):
        if key in sets and sets[key].get("complete"):
            out["sets"][key] = analyse_set(root, key, cfg, sets[key])
    # the 0.10 threshold: first from the same outputs, where the questions overlap
    for key, parent in (("sqad32_debiased_passing_010", "sqad32_debiased"), ("copy_debiased_passing_010", "copy_debiased")):
        if parent in out["sets"]:
            labels = load_labels(root, LABEL_FILE[sets[parent]["ranking"]])
            qids = [q for q in sets[parent]["questions"] if not labels[q]["flagged_010"]]
            meta = {**sets[parent], "answers_from": parent}
            out["sensitivity_010"][key] = analyse_set(root, key, cfg, meta, qids, f"{sets[parent]['label']}, the questions that also pass 0.10, no top-up")
    # then the extra questions, if the cap left room
    for key in SENSITIVITY_SETS:
        if key in sets and sets[key].get("complete"):
            out["sensitivity_010"][key] = analyse_set(root, key, cfg, sets[key])

    for pop, label in (("copy_test", "SQAD 3.0 copy, whole test split"), ("sqad32_test", "SQAD 3.2, whole test split")):
        rank = load_rankings(root, pop)
        labels = load_labels(root, LABEL_FILE[pop])
        qids = sorted(rank)
        row: dict = {"label": label, "n": len(qids), "all": retrieval_table(rank, qids, bootstrap_counts(len(qids), N_BOOT, "czech-v030", pop, "all").astype(float))}
        for name, fld in (("passes_0138", "flagged_0138"), ("passes_010", "flagged_010")):
            use = [q for q in qids if not labels[q][fld]]
            row[name] = {"n": len(use), **retrieval_table(rank, use, bootstrap_counts(len(use), N_BOOT, "czech-v030", pop, name).astype(float))}
        out["large_retrieval"][pop] = row

    # ---- diagnostics: no generations ---------------------------------------------------------
    pops = {"copy_test_split": ("copy_test", None), "sqad32_test_split": ("sqad32_test", None)}
    for k in MAIN_SETS:
        if k in sets:
            pops[k] = (sets[k]["ranking"], sets[k]["questions"])
    for name, (pop, qids) in pops.items():
        labels = load_labels(root, LABEL_FILE[pop])
        out["diagnostics"][name] = describe_population(labels, sorted(qids or labels))
    join = REPO / "results/czech/question_join.json"
    if join.exists():
        out["question_join"] = json.loads(join.read_text(encoding="utf-8"))
    compute = {name: json.loads((REPO / "results/czech" / file).read_text(encoding="utf-8")) for name, file in (("queue", "compute.json"), ("same_questions", "same_questions_compute.json")) if (REPO / "results/czech" / file).exists()}
    if compute:
        out["compute"] = compute

    block = None
    if "released" in out["sets"]:
        out["defect"] = defect(root, sets) if "released_v020" in sets else None
        block = dial_block(root, "released", cfg, sets["released"], "sqad-3.0-test")
    if all(k in out["sets"] for k in ("released", "released_v020")):
        out["claims"] = claims(out["sets"], out["large_retrieval"], lambda _: out["sets"]["released_v020"])
    out["gap"] = gap_analysis(root, {k: sets[k] for k in MAIN_SETS if k in sets and sets[k].get("complete")})
    out["bench_check"] = bench_check(root, {k: sets[k] for k in MAIN_SETS if k in sets and sets[k].get("complete")})
    if first is not None:
        first_root = REPO / FIRST_CUT
        first_sets = load_sets(first_root)["sets"]
        complete = {k: sets[k] for k in MAIN_SETS if k in sets and sets[k].get("complete")}
        loaded = {k: SetData(root, k, meta) for k, meta in complete.items()}
        cd = cross_data()
        out["claims_against_first_cut"] = claims(out["sets"], out["large_retrieval"], lambda k: first["sets"].get(k))
        # the prompt's effect alone: the first cut ran the same instruction on the Mac, where v0.2.0 ran
        out["defect_same_machine"] = first.get("defect")
        out["holds_on_all_four"] = holds_on_all_four(out["sets"])
        out["cross_version"] = cross_version(cd) if cd else None
        out["planned_gap"] = planned_gap(loaded, cd)
        out["added_questions"] = added_questions(root, complete, first_sets)
        out["continuity_released"] = continuity(root, first_root, sets, first_sets) if "released" in complete and "released_v020" in sets else None
        out["order_check"] = order_check(root, sets)
        out["compute_complete"] = complete_compute()
        out["thinking_tags"] = thinking_tags()
        out["dev_selection"] = json.loads((REPO / "results/czech/dev/selection.json").read_text(encoding="utf-8"))
        out["planned_sizes"] = json.loads((PLAN / "plan.json").read_text(encoding="utf-8"))["sizes"]
    if Path(root).resolve() == (REPO / RECORDS).resolve():
        # what the public text quotes from the move to the GX10 and the model benchmark, each recomputed by its own script's --check.
        # Dev-set records, so they do not wait for the test sets; the first cut's own analysis predates them.
        for name, rel in (("gx10_pilot", "results/czech/complete/pilot_gx10.json"), ("dev_selection_gx10", "results/czech/dev/selection_gx10.json"),
                          ("model_bench", "results/czech/model_bench/bench.json"), ("model_bench_ministral", "results/czech/model_bench/mistral.json")):
            if (REPO / rel).exists():
                out[name] = json.loads((REPO / rel).read_text(encoding="utf-8"))
    return crossover.clean(out), block


def verify_rows(rows: list[dict], gold: dict[str, list[str]]) -> dict:
    """Recompute each row's scores from its answer and the reader's own gold. Returns the mismatches."""
    from czech_records import scores

    bad, seen, no_gold = 0, 0, 0
    for r in rows:
        if r["q"] not in gold:
            no_gold += 1
            continue
        seen += 1
        s = scores(r["answer"], gold[r["q"]])
        if any(abs(s[k] - r[k]) > 1e-9 for k in s):
            bad += 1
    return {"rows": seen, "mismatches": bad, "rows_with_no_gold": no_gold}


def verify_scores(root: Path) -> int:
    """EM and F1 from each answer and the gold of the data itself."""
    from measurerag.datasets.base import Dataset  # noqa: F401
    from measurerag.datasets.sqad import load as load_copy
    from measurerag.datasets.sqad32 import load as load_32

    copy = load_copy(cache_dir=REPO / "data", max_queries=None)
    s32 = load_32(max_queries=None)
    rc = 0

    def check(name: str, rows: list[dict], gold: dict) -> None:
        nonlocal rc
        res = verify_rows(rows, gold)
        print(name, res)
        rc = rc or int(res["mismatches"] > 0 or res["rows"] == 0 or res["rows_with_no_gold"] > 0)

    roots = [Path(root)] if is_first_cut(root) else [Path(root), REPO / FIRST_CUT]
    for r in roots:
        for key, meta in load_sets(r)["sets"].items():
            ds = copy if meta["ranking"] == "copy_test" else s32
            if ds.corpus_hash() != meta["corpus_sha256"]:
                print(f"{key}: the corpus hash differs from the data you downloaded")
                rc = 1
                continue
            check(f"{r.name} {key}", load_answers(r, key), ds.answers)
    same = load_first_cut_same_questions()
    for side, ds in (("copy", copy), ("sqad32", s32)):
        check(f"same_questions {side}", [r for r in same if r["corpus"] == side], ds.answers)
    # the dev set's answers: the prompt's selection on each machine, the GX10's repeat, and both benchmarks
    for name in ("dev", "dev_gx10", "dev_gx10_repeat", "model_bench", "model_bench_mistral"):
        check(name, read_jsonl_gz(REPO / f"answers/sqad32/answers_{name}.jsonl.gz"), s32.answers)
    # the cross-version check reaches past the test splits
    cross = load_cross_version()
    if cross:
        gold_copy, gold_32 = dict(copy.answers), dict(s32.answers)
        for split in ("train", "validation"):          # the copy's names for its other two splits
            gold_copy.update(load_copy(cache_dir=REPO / "data", split=split, max_queries=None).answers)
        for split in ("train", "eval"):                # SQAD 3.2's
            gold_32.update(load_32(split=split, max_queries=None).answers)
        for side, gold in (("copy", gold_copy), ("sqad32", gold_32)):
            check(f"cross_version {side}", [r for r in cross if r["corpus"] == side], gold)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=RECORDS)
    ap.add_argument("--out", default="results/czech/analysis.json")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify-scores", action="store_true")
    args = ap.parse_args()
    root = REPO / args.records
    cfg = yaml.safe_load((REPO / "configs/crossover.yaml").read_text(encoding="utf-8"))
    if args.verify_scores:
        return verify_scores(root)
    def dumped(analysis: dict) -> str:
        return json.dumps(analysis, indent=1, sort_keys=True, ensure_ascii=False) + "\n"

    # until the complete run's records are written, the records describe the first cut, and there is one analysis
    first = None if is_first_cut(root) else analyse(REPO / FIRST_CUT, cfg)[0]
    first_path = REPO / FIRST_CUT / "analysis.json"
    out, block = analyse(root, cfg, first)
    text = dumped(out)
    path = REPO / args.out
    if not args.check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
        if first is not None:
            first_path.write_text(dumped(first), encoding="utf-8")
            print(f"wrote {FIRST_CUT}/analysis.json")
        return 0
    from czech_tokens import check_tree

    problems = check_tree(REPO, text, block, cfg)
    if first is not None and (not first_path.exists() or first_path.read_text(encoding="utf-8") != dumped(first)):
        problems.append(f"{FIRST_CUT}/analysis.json differs from what the first cut's records give")
    for p in problems:
        print("rescore_czech:", p)
    print("rescore_czech:", "FAILED" if problems else "OK: every Czech table and number recomputes byte for byte")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
