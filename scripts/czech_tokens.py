"""The Czech numbers the public text quotes, as tokens, from results/czech/analysis.json.

scripts/stats_tokens.py merges these into results/stats.json. `check_tree` is what
`scripts/rescore_czech.py --check` runs: it recomputes the analysis, the Czech block of
results/crossover.json and every Czech token from the shipped records, and compares them with the tree.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE, BIG = "qwen2.5-1.5b", "qwen2.5-3b"
E2, E4 = "gemma-4-e2b", "gemma-4-e4b"
M3, M8 = "ministral-3-3b", "ministral-3-8b"      # v0.3.1's third family
Q2, Q4 = "qwen3.5-2b", "qwen3.5-4b"              # v0.4.0's fourth family
N2, N4 = "gemma-3n-e2b", "gemma-3n-e4b"          # v0.4.2: Gemma 4's previous generation
SETS = (("released", "SQAD 3.0 copy, the released 200"), ("sqad32_full", "SQAD 3.2, 200 unfiltered"),
        ("sqad32_debiased", "SQAD 3.2, 200 debiased at 0.138"), ("copy_debiased", "SQAD 3.0 copy, 200 debiased at 0.138"))
CONFIG_LABEL = {f"bm25+{BASE}": "BM25 + 1.5B", f"e5-small+{BASE}": "e5-small + 1.5B", f"retromae-cs+{BASE}": "retromae-cs + 1.5B", f"e5-small+{BIG}": "e5-small + 3B",
                f"bm25+{E2}": "BM25 + E2B", f"e5-small+{E2}": "e5-small + E2B", f"retromae-cs+{E2}": "retromae-cs + E2B", f"e5-small+{E4}": "e5-small + E4B",
                f"bm25+{M3}": "BM25 + Ministral 3B", f"e5-small+{M3}": "e5-small + Ministral 3B", f"retromae-cs+{M3}": "retromae-cs + Ministral 3B", f"e5-small+{M8}": "e5-small + Ministral 8B",
                f"bm25+{Q2}": "BM25 + Qwen3.5 2B", f"e5-small+{Q2}": "e5-small + Qwen3.5 2B", f"retromae-cs+{Q2}": "retromae-cs + Qwen3.5 2B", f"e5-small+{Q4}": "e5-small + Qwen3.5 4B",
                f"bm25+{N2}": "BM25 + Gemma 3n E2B", f"e5-small+{N2}": "e5-small + Gemma 3n E2B", f"retromae-cs+{N2}": "retromae-cs + Gemma 3n E2B", f"e5-small+{N4}": "e5-small + Gemma 3n E4B"}
CONFIGS = (f"bm25+{BASE}", f"e5-small+{BASE}", f"retromae-cs+{BASE}", f"e5-small+{BIG}")
# The families of the complete run (scripts/rescore_czech.py, FAMILIES). Qwen2.5's tokens keep the names v0.2.0's text
# used; another family's carry its prefix, so cz.<set>.g4.generator_step is Gemma 4's generator step.
FAMILY_LABEL = {"gemma-3n": "Gemma 3n", "gemma-4": "Gemma 4", "ministral-3": "Ministral 3", "qwen2.5": "Qwen2.5", "qwen3.5": "Qwen3.5"}
FAMILY_MODELS = {"gemma-3n": (N2, N4), "gemma-4": (E2, E4), "ministral-3": (M3, M8), "qwen2.5": (BASE, BIG), "qwen3.5": (Q2, Q4)}   # US, France, China
FAMILY_CONFIGS = {"qwen2.5": CONFIGS, "gemma-4": (f"bm25+{E2}", f"e5-small+{E2}", f"retromae-cs+{E2}", f"e5-small+{E4}"),
                  "ministral-3": (f"bm25+{M3}", f"e5-small+{M3}", f"retromae-cs+{M3}", f"e5-small+{M8}"),
                  "qwen3.5": (f"bm25+{Q2}", f"e5-small+{Q2}", f"retromae-cs+{Q2}", f"e5-small+{Q4}"),
                  "gemma-3n": (f"bm25+{N2}", f"e5-small+{N2}", f"retromae-cs+{N2}", f"e5-small+{N4}")}
FAMILY_SHORTS = {"qwen2.5": ("bm25", "e5", "retromae", "e5_3b"), "gemma-4": ("bm25", "e5", "retromae", "e5_e4b"), "ministral-3": ("bm25", "e5", "retromae", "e5_8b"),
                 "qwen3.5": ("bm25", "e5", "retromae", "e5_4b"), "gemma-3n": ("bm25", "e5", "retromae", "e5_e4b")}
PREFIX = {"qwen2.5": "", "gemma-4": "g4.", "ministral-3": "m3.", "qwen3.5": "q35.", "gemma-3n": "g3n."}
MODEL_LABEL = {BASE: "1.5B", BIG: "3B", E2: "E2B", E4: "E4B", M3: "Ministral 3B", M8: "Ministral 8B", Q2: "Qwen3.5 2B", Q4: "Qwen3.5 4B", N2: "Gemma 3n E2B", N4: "Gemma 3n E4B"}
MODEL_SHORT = {BASE: "1_5b", BIG: "3b", E2: "e2b", E4: "e4b", M3: "m3b", M8: "m8b", Q2: "q2b", Q4: "q4b", N2: "n2b", N4: "n4b"}
FULL_LABEL = {BASE: "Qwen2.5 1.5B", BIG: "Qwen2.5 3B", E2: "Gemma 4 E2B", E4: "Gemma 4 E4B", M3: "Ministral 3 3B", M8: "Ministral 3 8B", Q2: "Qwen3.5 2B", Q4: "Qwen3.5 4B", N2: "Gemma 3n E2B", N4: "Gemma 3n E4B"}
SIZE_NAME = {BASE: "1.5B", BIG: "3B", E2: "E2B", E4: "E4B", M3: "3B", M8: "8B", Q2: "2B", Q4: "4B", N2: "E2B", N4: "E4B"}       # a model's name within its family
SIZE_LABEL = {"smaller": "E2B minus 1.5B", "larger": "E4B minus 3B"}
# the dev-set benchmark's models (results/czech/model_bench/), named as their makers name them
# in the order of every benchmark table: the release's family, the Western families, then Qwen3.5
BENCH_LABEL = {"qwen2.5-1.5b": "Qwen2.5 1.5B", "qwen2.5-3b": "Qwen2.5 3B", "gemma-4-e2b": "Gemma 4 E2B", "gemma-4-e4b": "Gemma 4 E4B", "granite-4.0-1b": "Granite 4.0 1B",
               "granite-4.0-micro": "Granite 4.0 Micro", "ministral-3-3b": "Ministral 3 3B, Mistral AI's GGUF", "ministral-3-8b": "Ministral 3 8B, Mistral AI's GGUF",
               "qwen3.5-2b": "Qwen3.5 2B", "qwen3.5-4b": "Qwen3.5 4B",
               "ministral-3-3b-unsloth": "Ministral 3 3B, Unsloth's GGUF", "ministral-3-8b-unsloth": "Ministral 3 8B, Unsloth's GGUF", "ministral-3-14b-unsloth": "Ministral 3 14B, Unsloth's GGUF"}
BENCH_FAMILY_LABEL = {"qwen2.5": "Qwen2.5", "gemma-4": "Gemma 4", "granite-4.0": "Granite 4.0", "ministral-3": "Ministral 3", "qwen3.5": "Qwen3.5"}


def in_order(d: dict, order) -> list:
    """The keys of `d` in a table's order, any others after them. JSON objects come back sorted."""
    return [k for k in order if k in d] + sorted(k for k in d if k not in order)


def view(s: dict, family: str) -> dict:
    """A family's readings on one set: Qwen2.5's at the top level, another's under families."""
    return s if family == "qwen2.5" else s.get("families", {}).get(family, {})


def families_in(s: dict) -> list[str]:
    return [fam for fam in FAMILY_MODELS if view(s, fam).get("answers")]
POPULATIONS = (("copy_test_split", "SQAD 3.0 copy, whole test split"), ("released", "SQAD 3.0 copy, the released 200"), ("copy_debiased", "SQAD 3.0 copy, 200 debiased at 0.138"),
               ("sqad32_test_split", "SQAD 3.2, whole test split"), ("sqad32_full", "SQAD 3.2, 200 unfiltered"), ("sqad32_debiased", "SQAD 3.2, 200 debiased at 0.138"))
# the complete run: no set is 200 by default, and every table reads in this order
COMPLETE_SETS = (("sqad32_debiased", "SQAD 3.2, debiased at 0.138"), ("sqad32_full", "SQAD 3.2, full"), ("released", "SQAD 3.0 copy, the released 200"), ("copy_debiased", "SQAD 3.0 copy, debiased at 0.138"))
COMPLETE_POPULATIONS = (("sqad32_test_split", "SQAD 3.2, whole test split"), ("sqad32_debiased", "SQAD 3.2, debiased at 0.138"), ("sqad32_full", "SQAD 3.2, full"), ("copy_test_split", "SQAD 3.0 copy, whole test split"),
                        ("released", "SQAD 3.0 copy, the released 200"), ("copy_debiased", "SQAD 3.0 copy, debiased at 0.138"))
RELABEL = (("SQAD 3.2, 200 unfiltered", "SQAD 3.2, full"), ("SQAD 3.2, 200 debiased at 0.138", "SQAD 3.2, debiased at 0.138"), ("SQAD 3.0 copy, 200 debiased at 0.138", "SQAD 3.0 copy, debiased at 0.138"),
           ("both 200 debiased at 0.138", "both debiased at 0.138"), ("fixed prompt", "v0.3.0's prompt"), ("in this sample of 200", "in this sample"))


def f(v, nd=3, signed=False) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def pct(v, nd=0) -> str:
    if v is None:
        return "n/a"
    s = f"{100 * v:.{nd}f}"
    return ("0" + s[2:] if s.startswith("-0") and not s.strip("-0.") else s) + "%"   # no "-0%"


def pct_range(xs: list[float]) -> str:
    return f"{pct(min(xs))} to {pct(max(xs))}"


def of(a: int, b: int) -> str:
    return f"{a:,} of {b:,}"


def with_ci(c: dict | None, nd=3) -> str:
    if not c:
        return "n/a"
    return f"{f(c['point'], nd, True)} [{f(c['lo'], nd, True)}, {f(c['hi'], nd, True)}]" + ("" if c["established"] else " (not established)")


import re as _re

FLAG = {"US": "\U0001F1FA\U0001F1F8", "FR": "\U0001F1EB\U0001F1F7", "CN": "\U0001F1E8\U0001F1F3", "CZ": "\U0001F1E8\U0001F1FF"}
FAMILY_COUNTRY = {"gemma-3n": "US", "gemma-4": "US", "ministral-3": "FR", "qwen2.5": "CN", "qwen3.5": "CN"}
MODEL_NAMES = {**{FULL_LABEL[g]: FAMILY_COUNTRY[f] for f, pair in FAMILY_MODELS.items() for g in pair},
               **{FAMILY_LABEL[f]: c for f, c in FAMILY_COUNTRY.items()},
               "Qwen2.5-1.5B-Instruct": "CN", "Qwen2.5-3B-Instruct": "CN", "Qwen2.5-1.5B": "CN", "Qwen2.5-3B": "CN", "Ministral 3B": "FR", "Ministral 8B": "FR",
               "Ministral 3 14B": "FR", "Granite 4.0 1B": "US", "Granite 4.0 Micro": "US", "Granite 4.0": "US"}   # the dev-set benchmark's others (IBM's Granite)
# v0.3.0's configuration labels name Qwen2.5's and Gemma 4's models by size alone ("BM25 + 1.5B", "e5-small + E2B")
CONFIG_SIZE_COUNTRY = {"1.5B": "CN", "3B": "CN", "E2B": "US", "E4B": "US"}
RETRIEVER_NAMES = {"e5-small": "US", "e5-base": "US", "granite-107m": "US", "granite-278m": "US", "EmbeddingGemma": "US",
                   "simcse-cs-en": "CZ", "retromae-cs": "CZ"}
def _name_re(names) -> "_re.Pattern":
    # the longest name first, with any flag already after it, so that a flagged name is left as it is and a shorter name
    # (Gemma 4) is never matched inside a longer one (Gemma 4 E4B)
    alt = "|".join(_re.escape(n) for n in sorted(names, key=len, reverse=True))
    # nor a family's name before a size it does not list (Ministral 3 14B is not "Ministral 3" and "14B")
    # and the flag after a possessive: "Qwen2.5's 🇨🇳", not "Qwen2.5 🇨🇳's"
    return _re.compile(rf"(?<![\w.+/`-])({alt})(?![\w+/`-])(?!\s(?:E?\d[\d.]*B|Micro)\b)((?:'|\u2019)s)?(\s?[\U0001F1E6-\U0001F1FF]{{2}})?")


_MODEL_RE, _RETR_RE = _name_re(MODEL_NAMES), _name_re(RETRIEVER_NAMES)
_CONFIG_SIZE_RE = _re.compile(r"\b((?:BM25|e5-small|retromae-cs) \+ )(1\.5B|3B|E2B|E4B)(?![\w.+-])(\s?[\U0001F1E6-\U0001F1FF]{2})?")


def _flag(m, names: dict) -> str:
    return m.group(0) if m.group(3) else f"{m.group(1)}{m.group(2) or ''} {FLAG[names[m.group(1)]]}"


def flagged(text: str, retrievers: bool = False) -> str:
    """A flag after each model name, and after each retriever name in a retriever table; flagging twice changes nothing."""
    out = _MODEL_RE.sub(lambda m: _flag(m, MODEL_NAMES), text)
    out = _CONFIG_SIZE_RE.sub(lambda m: m.group(0) if m.group(3) else f"{m.group(1)}{m.group(2)} {FLAG[CONFIG_SIZE_COUNTRY[m.group(2)]]}", out)
    return _RETR_RE.sub(lambda m: _flag(m, RETRIEVER_NAMES), out) if retrievers else out


def with_flags(t: dict[str, str]) -> dict[str, str]:
    return {k: flagged(v, retrievers=k.startswith("czr.table")) if isinstance(v, str) else v for k, v in t.items()}


_ANY_FLAG = _re.compile(r" ?[\U0001F1E6-\U0001F1FF]{2}")


def unflagged(text: str) -> str:
    """The text without its flags, as v0.4.0 and earlier wrote it; the tests that pin earlier releases' numbers read this."""
    return _ANY_FLAG.sub("", text)


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def points(v: float) -> str:
    return f"{v:.1f}"


def span_of(values: list[float], nd=1) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f} to {hi:.{nd}f}"


def _ci_cell(e: dict) -> str:
    return f"{e['verdict']}: {with_ci(e['value'])}"


def _ci_base(e: dict) -> str:
    return with_ci(e["v020"])


def _span_points(d: dict) -> str:
    return span_of(list(d.values()))


def is_complete(a: dict) -> bool:
    """Does this analysis hold the complete run, or only the first cut?"""
    return "holds_on_all_four" in a


# in the complete run EM with yes and no mapped is primary, so its rows lead
MAPPED_CLAIMS = (
    ("generator_step_with_yes_and_no_mapped", "The generator step, primary: EM with yes and no mapped", lambda e: _ci_cell(e), lambda e: _ci_base(e)),
    ("retriever_step_is_not_established_with_yes_and_no_mapped", "The retriever step is not established, EM with yes and no mapped", lambda e: _ci_cell(e), lambda e: _ci_base(e)),
    ("generator_step_exceeds_retriever_step_with_yes_and_no_mapped", "The generator step exceeds the retriever step, EM with yes and no mapped", lambda e: _ci_cell(e), lambda e: _ci_base(e)),
    ("dial_has_no_crossing_and_leans_to_the_generator_with_yes_and_no_mapped", "The dial has no crossing and leans to the generator, EM with yes and no mapped",
     lambda e: f"{e['verdict']}: generator larger at {e['levels_generator_larger']} of {e['levels_compared']} levels, retriever at {e['levels_retriever_larger']}, tied at {e['levels_tied']}", lambda e: "no crossing"),
    ("retromae_answers_worst_with_yes_and_no_mapped", "retromae-cs answers worse than e5-small, EM with yes and no mapped, 1.5B", lambda e: _ci_cell(e), lambda e: _ci_base(e)),
)

CLAIMS = (
    ("generator_step_is_established", "The generator step (e5-small + 3B over e5-small + 1.5B) is established, EM", _ci_cell, _ci_base),
    ("generator_step_with_yes_and_no_mapped", "The same step, secondary: EM with yes and no mapped", _ci_cell, _ci_base),
    ("retriever_step_is_not_established", "The retriever step (e5-small over BM25, 1.5B) is not established: e5-small answers no better", _ci_cell, _ci_base),
    ("generator_step_exceeds_retriever_step", "The generator step exceeds the retriever step", _ci_cell, _ci_base),
    ("dial_has_no_crossing_and_leans_to_the_generator", "The dial has no crossing, and its levels lean to the generator",
     lambda e: f"{e['verdict']}: generator larger at {e['levels_generator_larger']} of {e['levels_compared']} levels, retriever at {e['levels_retriever_larger']}, tied at {e['levels_tied']}", lambda e: "no crossing"),
    ("the_dial_cannot_separate_the_two_moves", "The dial cannot separate the two moves: no level established",
     lambda e: f"{e['verdict']}: {e['established']} of {e['levels_compared']} levels established; {e['established_with_yes_and_no_mapped']} of {e['levels_compared']} with yes and no mapped",
     lambda e: f"{e['v020_established']} of {e['levels_compared']}; {e['v020_established_with_yes_and_no_mapped']} of {e['levels_compared']} with yes and no mapped"),
    ("break_even_values_sit_inside_v020_intervals", "The break-even gains sit inside v0.2.0's intervals",
     lambda e: f"{e['verdict']}: {e['inside']} of {e['located_in_both']} located in both; gains {span_of(list(e['points'].values()), 2)}", lambda e: f"{e['v020_located']} levels located; gains {span_of(list(e['v020_points'].values()), 2)}"),
    ("non_gold_passages_leak_answers", "Non-gold passages leak answers: with no gold, EM stays above zero",
     lambda e: f"{e['verdict']}: answer in a non-gold passage {pct(e['answer_in_nongold'])}, EM {f(e['em_no_gold_base'])} (1.5B) and {f(e['em_no_gold_big'])} (3B)",
     lambda e: f"{pct(e['v020'])}, EM {f(e['v020_em_no_gold_base'])} and {f(e['v020_em_no_gold_big'])}"),
    ("retromae_answers_worst", "retromae-cs answers worse than e5-small (EM, 1.5B)", _ci_cell, _ci_base),
    ("e5small_puts_more_gold_in_front_of_the_generator", "e5-small puts more gold in the top 3 than BM25 (g @3)", _ci_cell, _ci_base),
    ("e5small_beats_bm25_on_ndcg_in_this_sample", "e5-small leads BM25 on nDCG@10 in this sample of 200", _ci_cell, _ci_base),
    ("every_yes_no_question_scores_0", "Every yes/no question scores 0",
     lambda e: f"{e['verdict']}: EM on yes/no {span_of([v for v in e['em_on_yes_no_by_configuration'].values() if v is not None], 3)}", lambda e: "0.000"),
    ("exact_match_under_counts", "Exact match under-counts correct answers (points of EM)",
     lambda e: f"{e['verdict']}: {span_of([v['all'] for v in e['points_by_configuration'].values()])}", lambda e: span_of([v["all"] for v in e["v020_by_configuration"].values()])),
    ("the_fixed_prompt_removes_the_yes_no_under_count", "The new prompt removes the yes/no under-count: under one point in every configuration",
     lambda e: f"{e['verdict']}: {_span_points(e['points_by_configuration'])} points", lambda e: f"{_span_points(e['v020_by_configuration'])} points"),
    ("declension_still_under_counts", "Declension under-counts too", lambda e: f"{e['verdict']}: {_span_points(e['points_by_configuration'])} points", lambda e: f"{_span_points(e['v020_by_configuration'])} points"),
    ("a_tolerant_metric_barely_moves_the_steps", "Counting both kinds of miss as correct barely moves the steps: by 0.02 or less",
     lambda e: f"{e['verdict']}: generator step moves {f(e['moved']['generator_step'])}, retriever step {f(e['moved']['retriever_step'])}",
     lambda e: f"{f(e['v020_moved']['generator_step'])} and {f(e['v020_moved']['retriever_step'])}"),
    ("the_token_cap_does_not_change_an_answer", "The token cap of 64 changes no answer: the bench and the dial agree",
     lambda e: f"{e['verdict']}: {of(e['same_string'], e['questions'])} the same string, EM the same on {e['same_em']:,}", lambda e: of(e["v020"]["same_string"], e["v020"]["questions"])),
)


def claims_table(a: dict, field: str = "claims", base_header: str = "v0.2.0, the released 200") -> str | None:
    """One plain table: each Czech claim and its verdict, per set, against the comparator in `field`."""
    cl = a.get(field)
    if not cl:
        return None
    complete = is_complete(a)
    have = [(k, lab) for k, lab in (COMPLETE_SETS if complete else SETS) if "status" not in cl.get(k, {"status": 1})]
    if not have:
        return None
    rows = []
    for key, label, cell, base in ((*MAPPED_CLAIMS, *CLAIMS[:1], *CLAIMS[2:]) if complete else CLAIMS):
        entries = [cl[k].get(key) for k, _ in have]
        first = next((e for e in entries if e), None)
        if first is None:
            continue
        rows.append([label, base(first), *[cell(e) if e else "n/a" for e in entries]])
    return table(["claim", base_header, *[lab for _, lab in have]], rows)


def tokens(a: dict) -> dict[str, str]:
    """Every Czech token, a flag after each model's name."""
    return with_flags(_tokens(a))


def _tokens(a: dict) -> dict[str, str]:
    t: dict[str, str] = {}
    sets = a["sets"]
    complete = is_complete(a)
    have = [(k, lab) for k, lab in (COMPLETE_SETS if complete else SETS) if k in sets]

    # ---- one row per set ------------------------------------------------------------------
    rows = []
    for k, lab in have:
        s = sets[k]
        m = s["retrieval"]["means"]
        ans = s["answers"]
        rows.append([lab, f"{s['n']:,}", f"{s['yes_no_questions']:,}", f"{s['answer_sentence_opens_article']:,}", f(m["e5-small"]["ndcg10"]["point"]), f(m["e5-small"]["g3"]["point"]),
                     *[f(ans[c]["em"]["point"]) if c in ans else "n/a" for c in CONFIGS], f(ans[f"e5-small+{BASE}"]["f1"]["point"]) if f"e5-small+{BASE}" in ans else "n/a"])
        t[f"cz.{k}.n"] = f"{s['n']:,}"
        t[f"cz.{k}.yes_no"] = f"{s['yes_no_questions']:,}"
        t[f"cz.{k}.opens"] = f"{s['answer_sentence_opens_article']:,}"
        t[f"cz.{k}.flagged"] = str(s["flagged_at_0138"])
        if s.get("token_cap_check"):
            t[f"cz.{k}.bench_dial_same"] = of(s["token_cap_check"]["same_string"], s["token_cap_check"]["questions"])
        for fam in families_in(s):
            v, p = view(s, fam), PREFIX[fam]
            for c, short in zip(FAMILY_CONFIGS[fam], FAMILY_SHORTS[fam]):
                if c in v["answers"]:
                    t[f"cz.{k}.{p}{short}.em"] = f(v["answers"][c]["em"]["point"])
                    t[f"cz.{k}.{p}{short}.f1"] = f(v["answers"][c]["f1"]["point"])
                    t[f"cz.{k}.{p}{short}.mapped"] = f(v["answers"][c]["secondary_em_yes_no_mapped"]["point"])
            if p and v.get("token_cap_check"):
                t[f"cz.{k}.{p}bench_dial_same"] = of(v["token_cap_check"]["same_string"], v["token_cap_check"]["questions"])
    t["cz.table_sets"] = table(["set", "questions", "yes/no questions", "answer sentence opens its article", "nDCG@10, e5-small", "g @3, e5-small", *[f"EM, {CONFIG_LABEL[c]}" for c in CONFIGS], "F1, e5-small + 1.5B"], rows)
    # v0.4.2: the same table read with Gemma 4, the family the documents name when they name one
    g4c = FAMILY_CONFIGS["gemma-4"]
    g4_rows = []
    for k, lab in have:
        s, v = sets[k], view(sets[k], "gemma-4")
        if not v.get("answers"):
            continue
        m, ans = s["retrieval"]["means"], v["answers"]
        g4_rows.append([lab, f"{s['n']:,}", f"{s['yes_no_questions']:,}", f"{s['answer_sentence_opens_article']:,}", f(m["e5-small"]["ndcg10"]["point"]),
                        f(m["e5-small"]["g3"]["point"]), *[f(ans[c]["secondary_em_yes_no_mapped"]["point"]) if c in ans else "n/a" for c in g4c],
                        *[f(ans[c]["em"]["point"]) if c in ans else "n/a" for c in g4c[1:2]],
                        f(ans[g4c[1]]["f1"]["point"]) if g4c[1] in ans else "n/a"])
    g4_label = {g4c[0]: "BM25 + Gemma 4 E2B", g4c[1]: "e5-small + Gemma 4 E2B", g4c[2]: "retromae-cs + Gemma 4 E2B", g4c[3]: "e5-small + Gemma 4 E4B"}
    t["cz.table_sets_g4"] = table(["set", "questions", "yes/no questions", "answer sentence opens its article", "nDCG@10, e5-small", "g @3, e5-small",
                                   *[f"EM mapped, {g4_label[c]}" for c in g4c], f"raw EM, {g4_label[g4c[1]]}", f"F1, {g4_label[g4c[1]]}"], g4_rows) if g4_rows else "_not measured._"

    # ---- the steps and the dial ---------------------------------------------------------------
    step_rows = []
    for k, lab in have:
        s = sets[k]
        if "steps" not in s or "dial_summary" not in s:
            continue
        st, d = s["steps"]["em"], s["dial_summary"]
        step_rows.append([lab, with_ci(st["generator_step"]), with_ci(st["retriever_step"]), with_ci(st["generator_step_minus_retriever_step"]),
                          f"{d['levels_generator_larger']} of {d['levels_compared']}", f"{d['levels_retriever_larger']} of {d['levels_compared']}", f"{d['levels_established_with_gold']} of {d['levels_compared']}",
                          f(d["em_no_gold_base"]), f(d["em_no_gold_big"]), pct(d["answer_in_nongold_no_gold"])])
        t[f"cz.{k}.generator_step"] = with_ci(st["generator_step"])
        t[f"cz.{k}.retriever_step"] = with_ci(st["retriever_step"])
        t[f"cz.{k}.gap_steps"] = with_ci(st["generator_step_minus_retriever_step"])
    rows = []
    for k, lab in (("released_v020", "SQAD 3.0 copy, the released 200, v0.2.0's prompt"), *have):
        s = sets.get(k)
        if not s or "steps" not in s:
            continue
        st, sm = s["steps"]["em"], s["steps"]["em_yes_no_mapped"]
        d, dm = s.get("dial_summary"), s.get("dial_summary_em_yes_no_mapped")
        t[f"cz.{k}.generator_step"] = with_ci(st["generator_step"])
        t[f"cz.{k}.generator_step_mapped"] = with_ci(sm["generator_step"])
        t[f"cz.{k}.retriever_step_mapped"] = with_ci(sm["retriever_step"])
        est = lambda x: x["levels_established_generator"] + x["levels_established_retriever"]  # noqa: E731
        if d and dm:
            t[f"cz.{k}.levels_established"] = f"{est(d)} of {d['levels_compared']}"
            t[f"cz.{k}.levels_established_mapped"] = f"{est(dm)} of {dm['levels_compared']}"
        rows.append([lab, with_ci(st["generator_step"]), with_ci(sm["generator_step"]), f"{est(d)} of {d['levels_compared']}" if d else "n/a", f"{est(dm)} of {dm['levels_compared']}" if dm else "n/a"])
    t["cz.table_steps_mapped"] = table(["set", "generator step, EM [95%]", "generator step, secondary: EM with yes and no mapped [95%]", "dial levels established, EM", "dial levels established, EM with yes and no mapped"], rows)
    tc = sets.get("released_v020", {}).get("token_cap_check")
    if tc:
        t["cz.released_v020.bench_dial_same"] = f"{tc['same_string']} of {tc['questions']}"
    t["cz.table_steps"] = table(["set", "generator step [95%]", "retriever step [95%]", "generator step minus retriever step [95%]", "dial levels: generator larger", "dial levels: retriever larger",
                                 "dial levels established, with gold", "EM, no gold, 1.5B", "EM, no gold, 3B", "answer in a non-gold passage, no gold"], step_rows)

    # ---- labelled secondary metrics --------------------------------------------------------------
    rows = []
    under: dict = {fam: ([], [], []) for fam in FAMILY_MODELS}
    for fam in FAMILY_MODELS:
        for k, lab in have:
            for c in FAMILY_CONFIGS[fam]:
                x = view(sets[k], fam).get("answers", {}).get(c)
                if not x:
                    continue
                rows.append([lab, CONFIG_LABEL[c], f(x["em"]["point"]), f(x["secondary_em_yes_no_mapped"]["point"]), f(x["secondary_prefix5"]["point"]), f(x["f1"]["point"]), f(x["secondary_f1_folded"]),
                             f(x.get("em_on_yes_no")), f(x.get("em_on_span"))])
                for bucket, field in zip(under[fam], ("undercount_points", "undercount_yes_no_points", "undercount_declension_points")):
                    bucket.append(x[field])
    t["cz.table_secondary"] = table(["set", "configuration", "EM", "EM with yes and no mapped (primary)", "secondary: five-character match", "F1", "secondary: F1 without diacritics", "EM on yes/no questions", "EM on span questions"], rows)
    for fam, (all_, yn, dec) in under.items():
        if all_:
            p = PREFIX[fam]
            t[f"cz.{p}undercount_all"] = span_of(all_)
            t[f"cz.{p}undercount_yes_no"] = span_of(yn)
            t[f"cz.{p}undercount_declension"] = span_of(dec)

    # ---- the defect ---------------------------------------------------------------------------------
    d = a.get("defect")
    if d:
        rows = []
        for c in CONFIGS:
            x = d["configurations"].get(c)
            if not x:
                continue
            rows.append([CONFIG_LABEL[c], f(x["em_v020"]), f(x["em_fixed"]), with_ci(x["em_difference"]), f(x["em_on_yes_no_v020"]), f(x["em_on_yes_no_fixed"]), with_ci(x["em_difference_on_yes_no"]), f(x["em_on_span_v020"]), f(x["em_on_span_fixed"]), with_ci(x["em_difference_on_span"])])
        t["cz.table_defect"] = table(["configuration", "EM, v0.2.0's prompt", "EM, fixed prompt", "difference [95%]", "EM on yes/no, v0.2.0's prompt", "EM on yes/no, fixed prompt", "difference on yes/no [95%]", "EM on span, v0.2.0's prompt", "EM on span, fixed prompt", "difference on span [95%]"], rows)
        rows = []
        for c in CONFIGS:
            x = d["configurations"].get(c)
            if not x:
                continue
            for tag, cl in (("v0.2.0's prompt", x["yes_no_classes_v020"]), ("fixed prompt", x["yes_no_classes_fixed"])):
                rows.append([CONFIG_LABEL[c], tag, cl["ano or ne"], cl["yes or no in English"], cl["unanswerable"], cl["something else"]])
        t["cz.table_yes_no_answers"] = table(["configuration", "prompt", "answered ano or ne", "answered yes or no in English", "answered unanswerable", "something else"], rows)
        t["cz.defect.questions"] = str(d["questions"])
        t["cz.defect.yes_no"] = str(d["yes_no_questions"])
        t["cz.defect.span"] = str(d["span_questions"])
        for c, short in zip(CONFIGS, ("bm25", "e5", "retromae", "e5_3b")):
            x = d["configurations"].get(c)
            if x:
                t[f"cz.defect.{short}.diff"] = with_ci(x["em_difference"])
                t[f"cz.defect.{short}.diff_yes_no"] = with_ci(x["em_difference_on_yes_no"])
                t[f"cz.defect.{short}.diff_span"] = with_ci(x["em_difference_on_span"])
                t[f"cz.defect.{short}.em_v020"] = f(x["em_v020"])
                t[f"cz.defect.{short}.em_fixed"] = f(x["em_fixed"])
                t[f"cz.defect.{short}.english"] = str(x["yes_no_classes_fixed"]["yes or no in English"])
                t[f"cz.defect.{short}.em_yes_no_fixed"] = f(x["em_on_yes_no_fixed"])
                t[f"cz.defect.{short}.em_yes_no_v020"] = f(x["em_on_yes_no_v020"])
                t[f"cz.defect.{short}.english_v020"] = str(x["yes_no_classes_v020"]["yes or no in English"])
                t[f"cz.defect.{short}.mapped_v020"] = f(x["secondary_em_yes_no_mapped_v020"])
                t[f"cz.defect.{short}.mapped_fixed"] = f(x["secondary_em_yes_no_mapped_fixed"])

    # ---- retrieval at large samples ---------------------------------------------------------------------
    rows = []
    for pop, row in a.get("large_retrieval", {}).items():
        for tag, label in (("all", ""), ("passes_0138", ", passes 0.138"), ("passes_010", ", passes 0.10")):
            b = row[tag]
            m, dl = b["means"], b["deltas"]
            n = b.get("n", row["n"])
            rows.append([row["label"] + label, f"{n:,}", f(m["bm25"]["ndcg10"]["point"]), f(m["bm25-nodiacritics"]["ndcg10"]["point"]), f(m["e5-small"]["ndcg10"]["point"]), f(m["retromae-cs"]["ndcg10"]["point"]),
                         with_ci(dl["e5-small - bm25"]["ndcg10"]), with_ci(dl["e5-small - bm25-nodiacritics"]["ndcg10"])])
        t[f"cz.large.{pop}.n"] = f"{row['n']:,}"
        t[f"cz.large.{pop}.lead_nd"] = with_ci(row["all"]["deltas"]["e5-small - bm25-nodiacritics"]["ndcg10"])
        t[f"cz.large.{pop}.lead_bm25"] = with_ci(row["all"]["deltas"]["e5-small - bm25"]["ndcg10"])
    if rows:
        t["cz.table_large"] = table(["data", "questions", "nDCG@10, BM25", "nDCG@10, BM25 without diacritics", "nDCG@10, e5-small", "nDCG@10, retromae-cs", "e5-small minus BM25 [95%]", "e5-small minus BM25 without diacritics [95%]"], rows)

    # ---- the 0.10 threshold ------------------------------------------------------------------------------
    rows = []
    for k, s in a.get("sensitivity_010", {}).items():
        if "steps" not in s:
            continue
        st = s["steps"]["em"]
        rows.append([s["label"], f"{s['n']:,}", f(s["answers"][f"e5-small+{BASE}"]["em"]["point"]), f(s["answers"][f"e5-small+{BIG}"]["em"]["point"]), with_ci(st["generator_step"]), with_ci(st["retriever_step"]), with_ci(st["generator_step_minus_retriever_step"])])
    if rows:
        t["cz.table_sensitivity"] = table(["sensitivity row", "questions", "EM, e5-small + 1.5B", "EM, e5-small + 3B", "generator step [95%]", "retriever step [95%]", "generator step minus retriever step [95%]"], rows)

    # ---- diagnostics ----------------------------------------------------------------------------------------
    di = a.get("diagnostics", {})
    pops = [(k, lab) for k, lab in (COMPLETE_POPULATIONS if complete else POPULATIONS) if k in di]
    if pops:
        def row(name, fn):
            return [name, *[fn(di[k]) for k, _ in pops]]
        rows = [row("questions", lambda x: f"{x['questions']:,}"),
                row("answer type: yes/no", lambda x: pct(x["answer_type"]["yes/no"])),
                row("answer type: number", lambda x: pct(x["answer_type"]["number"])),
                row("answer type: date", lambda x: pct(x["answer_type"]["date"])),
                row("answer type: single word", lambda x: pct(x["answer_type"]["single word"])),
                row("answer type: multi-word name", lambda x: pct(x["answer_type"]["multi-word name"])),
                row("answer type: other", lambda x: pct(x["answer_type"]["other"])),
                row("span answers, mean tokens", lambda x: f(x["span_answer_tokens_mean"], 2)),
                row("span answers of one token", lambda x: pct(x["span_answer_length_band"]["1"])),
                row("answer start in its context, median characters", lambda x: f(x["answer_start_quartiles"][1], 0)),
                row("answer starts in the first 50 characters", lambda x: pct(x["answer_start_under_50_of_span"])),
                row("answer sentence opens its article", lambda x: pct(x["answer_sentence_opens_article"])),
                row("flagged by the debias rule at 0.138", lambda x: pct(x["flagged_by_the_rule_0138"], 1)),
                row("question-context overlap, median", lambda x: f(x["overlap_quartiles"][1])),
                row("gold passage, median characters", lambda x: f(x["gold_passage_chars_quartiles"][1], 0)),
                row("answer verbatim in the top 3, BM25", lambda x: pct(x["answer_in_top3_verbatim"]["bm25"])),
                row("answer verbatim in the top 3, e5-small", lambda x: pct(x["answer_in_top3_verbatim"]["e5-small"])),
                row("answer verbatim in the top 3, retromae-cs", lambda x: pct(x["answer_in_top3_verbatim"]["retromae-cs"]))]
        t["cz.table_diagnostics"] = table(["", *[lab for _, lab in pops]], rows)

    # ---- the gap ----------------------------------------------------------------------------------------------
    g = a.get("gap", {}).get("reweighting", {})
    rows = []
    names = {"sqad32_full minus released": "SQAD 3.2, 200 unfiltered, minus the SQAD 3.0 copy, the released 200", "sqad32_debiased minus copy_debiased": "SQAD 3.2 minus the SQAD 3.0 copy, both 200 debiased at 0.138"}
    for pair in names:
        for c, variants in g.get(pair, {}).items():
            first = next(iter(variants.values()))
            if not first:
                continue
            rows.append([names.get(pair, pair), CONFIG_LABEL[c], with_ci(first["gap"]),
                         *[with_ci(variants[v]["gap_after_reweighting"]) for v in ("answer type and length", "the opening sentence", "answer type, length and the opening sentence", "the gold passage's length", "the debias rule")]])
    if rows:
        t["cz.table_gap"] = table(["gap in EM", "configuration", "gap [95%]", "left after matching answer type and length [95%]", "left after matching the opening sentence [95%]", "left after matching all three [95%]", "left after matching the gold passage's length [95%]", "left after matching the debias rule [95%]"], rows)
    t.update(gap_tokens(a, names, have))
    comp = a.get("compute", {})
    if comp.get("queue") and not complete:
        q = comp["queue"]
        made = q["generations"] + comp.get("same_questions", {}).get("generations", 0)
        t["cz.compute.generations"] = f"{made:,}"
        t["cz.compute.cap"] = f"{q['max_generations']:,}"
        t["cz.compute.hours"] = f"{q['max_seconds'] / 3600:.0f}"
    changes = changes_from_v020(a)
    if changes:
        t["cz.table_changes"] = table(["number", "v0.2.0", "v0.3.0"], changes)
        t["cz.changes_count"] = str(len(changes))
    table_claims = claims_table(a)
    if table_claims:
        t["cz.table_claims"] = table_claims
    qj = a.get("question_join")
    if qj:
        t["cz.join.joined"] = f"{qj['joined_on_question_text']:,}"
        t["cz.join.same"] = f"{qj['gold_same_after_normalising']:,}"
        t["cz.join.same_pct"] = pct(qj["gold_same_after_normalising"] / qj["joined_on_question_text"], 1)
    t.update(gx10_tokens(a))
    t.update(bench_tokens(a))
    t.update(defect_same_machine_tokens(a))
    t.update(quote_tokens())
    if complete:
        t.update(complete_tokens(a))
        for old, new in RELABEL:
            t = {k: v.replace(old, new) for k, v in t.items()}
    return t


def level(c: dict | None, nd: int = 3) -> str:
    """A level, not a difference: no sign, and no word on whether it is established."""
    return "n/a" if not c else f"{f(c['point'], nd)} [{f(c['lo'], nd)}, {f(c['hi'], nd)}]"


def rules_out(c: dict) -> str:
    """What an interval that holds zero says: an effect outside it is ruled out, at 95%."""
    return f"below {f(c['lo'], 3, True)} or above {f(c['hi'], 3, True)}"


def diff_label(fam: str, size: str, other: str = "qwen2.5") -> str:
    """'Gemma 4 E2B minus Qwen2.5 1.5B': a family difference against Qwen2.5, or against another family."""
    i = 0 if size == "smaller" else 1
    return f"{FULL_LABEL[FAMILY_MODELS[fam][i]]} minus {FULL_LABEL[FAMILY_MODELS[other][i]]}"


def fam_tok(fam: str) -> str:
    """A family difference's token name: Gemma 4's keeps v0.3.0's, cz.<set>.family; another's carries its prefix."""
    return "family" if fam == "gemma-4" else f"{PREFIX[fam]}family"


def pair_tok(name: str) -> str:
    """'ministral-3 minus gemma-4' -> 'm3_minus_g4'."""
    a, b = name.split(" minus ")
    return f"{PREFIX[a].rstrip('.')}_minus_{PREFIX[b].rstrip('.')}"


SET_PROSE = {"sqad32_debiased": "SQAD 3.2 debiased", "sqad32_full": "SQAD 3.2 full", "released": "the released 200", "copy_debiased": "the copy debiased"}


def and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def holds_text(h: dict) -> str:
    return "established on all four, with one sign" if h["holds_on_all_four"] else f"established on {len(h['established_on'])} of 4"


def signed_span(values: list[float], nd=3) -> str:
    lo, hi = f(min(values), nd, True), f(max(values), nd, True)
    return lo if lo == hi else f"{lo} to {hi}"


def range_tokens(a: dict, have: list, fams: list) -> dict[str, str]:
    """Ranges of point estimates, for the prose of README and CROSSOVER: across the four sets, and across the three
    besides the released 200. A range says nothing on whether each value is established; the text says that."""
    t: dict[str, str] = {}
    sets = a["sets"]
    M, E = "em_yes_no_mapped", "em"
    three = [(k, lab) for k, lab in have if k != "released"]
    for fam in fams:
        p = PREFIX[fam]
        for step in ("generator_step", "retriever_step", "generator_step_minus_retriever_step"):
            for field, short in ((M, "mapped"), (E, "em")):
                for name, among in (("range", have), ("range3", three)):
                    vals = [view(sets[k], fam)["steps"][field][step]["point"] for k, _ in among if "steps" in view(sets[k], fam)]
                    if vals:
                        t[f"cz.{p}{name}.{step}.{short}"] = signed_span(vals)
        comp = [view(sets[k], fam)["compliance_big_minus_small"]["compliance"]["point"] for k, _ in have if view(sets[k], fam).get("compliance_big_minus_small")]
        if comp:
            t[f"cz.{p}range.compliance_step"] = signed_span(comp)
    retro = [view(sets[k], fam)["e5small_minus_retromae_mapped"]["point"] for k, _ in have for fam in fams if "e5small_minus_retromae_mapped" in view(sets[k], fam)]
    if retro:
        t["cz.range.e5_minus_retromae_mapped"] = signed_span(retro)
    for fam in (x for x in fams if x != "qwen2.5"):          # every family against Qwen2.5, whatever the order
        for size in ("smaller", "larger"):
            for field, short in ((M, "mapped"), (E, "em")):
                vals = [sets[k]["family_difference"][fam][size][field]["point"] for k, _ in have if sets[k].get("family_difference", {}).get(fam, {}).get(size)]
                if vals:
                    t[f"cz.{fam_tok(fam)}.{size}.{short}_range"] = signed_span(vals)
    names = sorted({n for k, _ in have for n in sets[k].get("family_difference_between", {})})
    for name in names:
        for size in ("smaller", "larger"):
            for field, short in ((M, "mapped"), (E, "em")):
                vals = [sets[k]["family_difference_between"][name][size][field]["point"] for k, _ in have if sets[k].get("family_difference_between", {}).get(name, {}).get(size)]
                if vals:
                    t[f"cz.{pair_tok(name)}.{size}.{short}_range"] = signed_span(vals)
    cross = a.get("cross_version")
    if cross:
        for field, short in ((M, "mapped"), (E, "em")):
            t[f"cz.cross.{short}_range"] = signed_span([x[field]["difference"]["point"] for x in cross["generators"].values()])
            t[f"cz.cross.{short}_established"] = f"{sum(x[field]['difference']['established'] for x in cross['generators'].values())} of {len(cross['generators'])}"
        for fam, fd in (cross.get("family_difference") or {}).items():
            vals = [fd[size][s][M]["point"] for size in fd for s in fd[size]]
            if vals:
                t[f"cz.cross.{fam_tok(fam)}.mapped_range"] = signed_span(vals)
        for name, fd in (cross.get("family_difference_between") or {}).items():
            vals = [fd[size][s][M]["point"] for size in fd for s in fd[size]]
            if vals:
                t[f"cz.cross.{pair_tok(name)}.mapped_range"] = signed_span(vals)
    for field, short in ((M, "mapped"), (E, "em")):
        pg = [b for b in a.get("planned_gap", {}).get(field, {}).values() if b.get("G1") and b.get("G2")]
        if not pg:
            continue
        g1 = [b["G1"]["gap"]["gap"] for b in pg]
        g2 = [b["G2"]["gap"]["gap"] for b in pg]
        shares = [b["G1"]["share_of_the_gap_the_rule_accounts_for"] for b in pg if b["G1"].get("share_of_the_gap_the_rule_accounts_for") is not None]
        rule = [b["G1"]["on_questions_the_rule_passes"]["gap"] for b in pg if b["G1"].get("on_questions_the_rule_passes")]
        left = [b["what_is_left"] for b in pg if b.get("what_is_left")]
        t[f"cz.planned.{short}.configurations"] = str(len(pg))
        t[f"cz.planned.{short}.g1_established"] = f"{sum(x['established'] for x in g1)} of {len(g1)}"
        t[f"cz.planned.{short}.g1_range"] = signed_span([x["point"] for x in g1])
        t[f"cz.planned.{short}.g1_rule_established"] = f"{sum(x['established'] for x in rule)} of {len(rule)}"
        t[f"cz.planned.{short}.g2_established"] = f"{sum(x['established'] for x in g2)} of {len(g2)}"
        t[f"cz.planned.{short}.g2_range"] = signed_span([x["point"] for x in g2])
        t[f"cz.planned.{short}.g1_established_sign"] = ", ".join(sorted({"SQAD 3.2 higher" if x["point"] > 0 else "SQAD 3.2 lower" for x in g1 if x["established"]})) or "none"
        t[f"cz.planned.{short}.g2_established_sign"] = ", ".join(sorted({"SQAD 3.2 higher" if x["point"] > 0 else "SQAD 3.2 lower" for x in g2 if x["established"]})) or "none"
        if shares:
            t[f"cz.planned.{short}.rule_share"] = pct_range(shares)
        # a share of G1 is read beside the planned range only where G1 itself is established; where G1 is near zero
        # the ratio of two small differences can take any value
        firm = [b["G1"]["share_of_the_gap_the_rule_accounts_for"] for b in pg
                if b["G1"].get("share_of_the_gap_the_rule_accounts_for") is not None and b["G1"]["gap"]["gap"]["established"]]
        if firm:
            t[f"cz.planned.{short}.rule_share_where_g1_established"] = pct_range(firm)
            t[f"cz.planned.{short}.rule_share_g1_established_n"] = f"{len(firm)} of {len(pg)}"
        if left:
            t[f"cz.planned.{short}.left_established"] = f"{sum(x['established'] for x in left)} of {len(left)}"
    return t


def family_tokens(a: dict, have: list) -> dict[str, str]:
    """The comparison of every family: each model with e5-small's top 3, every family
    difference, and each family's steps. Only once every family has answers."""
    sets = a["sets"]
    fams = tuple(FAMILY_MODELS)
    if not all(view(sets[k], fam).get("answers") for k, _ in have for fam in fams):
        return {}
    t: dict[str, str] = {}
    M, E = "em_yes_no_mapped", "em"
    order = [FAMILY_MODELS[f][0] for f in fams] + [FAMILY_MODELS[f][1] for f in fams]
    fam_of = {g: f for f, pair in FAMILY_MODELS.items() for g in pair}
    for field, key, cell in ((M, "mapped", lambda x: level(x["secondary_em_yes_no_mapped"])), (E, "em", lambda x: level(x["em"])),
                             (None, "compliance", lambda x: level((x.get("compliance") or {}).get("compliance")))):
        rows = []
        for k, lab in have:
            rows.append([lab, f"{sets[k]['n']:,}", *[cell(view(sets[k], fam_of[g])["answers"][f"e5-small+{g}"]) for g in order]])
        t[f"cz.table_families_levels_{key}"] = table(["set", "questions", *[FULL_LABEL[g] for g in order]], rows)
    # every family difference, paired, with e5-small's top 3: each newer family against Qwen2.5, then the newer ones against each other
    pairs = [("gemma-4", "qwen2.5"), ("ministral-3", "qwen2.5"), ("qwen3.5", "qwen2.5"), ("gemma-3n", "qwen2.5"), ("ministral-3", "gemma-4"),
             ("qwen3.5", "gemma-4"), ("qwen3.5", "ministral-3"), ("gemma-4", "gemma-3n")]

    def diff_of(s: dict, x: str, y: str) -> dict | None:
        return s.get("family_difference", {}).get(x) if y == "qwen2.5" else s.get("family_difference_between", {}).get(f"{x} minus {y}")

    for field, key in ((M, "mapped"), (E, "em")):
        rows = []
        for k, lab in have:
            for x, y in pairs:
                cells = []
                for i, size in enumerate(("smaller", "larger")):
                    d = (diff_of(sets[k], x, y) or {}).get(size)
                    cells.append(f"{SIZE_NAME[FAMILY_MODELS[x][i]]} minus {SIZE_NAME[FAMILY_MODELS[y][i]]}: {with_ci(d[field])}" if d else "n/a")
                rows.append([lab, f"{FAMILY_LABEL[x]} minus {FAMILY_LABEL[y]}", *cells])
        t[f"cz.table_families_differences_{key}"] = table(["set", "families", "smaller models [95%]", "larger models [95%]"], rows)
    # each family's steps, with yes and no mapped
    rows = []
    for k, lab in have:
        for fam in fams:
            st = view(sets[k], fam).get("steps", {}).get(M)
            if not st:
                continue
            small, big = FAMILY_MODELS[fam]
            rows.append([lab, FAMILY_LABEL[fam], f"{SIZE_NAME[big]} minus {SIZE_NAME[small]}", with_ci(st["generator_step"]), with_ci(st["retriever_step"]), with_ci(st["generator_step_minus_retriever_step"])])
    t["cz.table_families_steps"] = table(["set", "family", "models", "generator step [95%]", "retriever step, e5-small minus BM25 [95%]", "generator step minus retriever step [95%]"], rows)
    return t


def complete_tokens(a: dict) -> dict[str, str]:
    """The complete run's tables. Every table reads in the order of COMPLETE_SETS, and
    within a set Qwen2.5 before Gemma 4."""
    t: dict[str, str] = {}
    sets = a["sets"]
    have = [(k, lab) for k, lab in COMPLETE_SETS if k in sets]
    fams = [fam for fam in FAMILY_MODELS if any(view(sets[k], fam).get("answers") for k, _ in have)]
    M, E = "em_yes_no_mapped", "em"
    passages = {"copy": "the SQAD 3.0 copy", "sqad32": "SQAD 3.2"}
    est = lambda x: x["levels_established_generator"] + x["levels_established_retriever"]  # noqa: E731

    # ---- the answers: both metrics side by side, and compliance ---------------------------------------
    rows, classes, gaps = [], [], []
    compliance: dict = {fam: [] for fam in fams}
    english: dict = {fam: 0 for fam in fams}          # yes/no answers given in English, every configuration on the four sets
    same: dict = {fam: True for fam in fams}          # EM and EM with yes and no mapped equal in every configuration
    for k, lab in have:
        for fam in fams:
            v, p = view(sets[k], fam), PREFIX[fam]
            for c, short in zip(FAMILY_CONFIGS[fam], FAMILY_SHORTS[fam]):
                x = v.get("answers", {}).get(c)
                if not x:
                    continue
                english[fam] += x["yes_no_answer_classes"]["yes or no in English"]
                same[fam] = same[fam] and x["em"]["point"] == x["secondary_em_yes_no_mapped"]["point"]
                comp = x.get("compliance") or {}
                rows.append([lab, CONFIG_LABEL[c], level(x["secondary_em_yes_no_mapped"]), level(x["em"]), level(comp.get("compliance")), f(comp["strict_share"]["point"]) if comp else "n/a"])
                cl = x["yes_no_answer_classes"]
                classes.append([lab, CONFIG_LABEL[c], comp.get("yes_no_questions", "n/a"), cl["ano or ne"], cl["yes or no in English"], cl["unanswerable"], cl["something else"]])
                if comp.get("compliance"):
                    t[f"cz.{k}.{p}{short}.compliance"] = level(comp["compliance"])
                    compliance[fam].append(comp["compliance"]["point"])
            d = v.get("compliance_big_minus_small")
            if d:
                small, big = FAMILY_MODELS[fam]
                gaps.append([lab, FAMILY_LABEL[fam], f"{MODEL_LABEL[big]} minus {MODEL_LABEL[small]}", with_ci(d.get("compliance")), with_ci(d["strict_share"])])
                t[f"cz.{k}.{p}compliance_big_minus_small"] = with_ci(d.get("compliance"))
    t["cz.table_answers"] = table(["set", "configuration", "EM with yes and no mapped [95%]", "EM [95%]", "compliance [95%]", "Czech yes or no, share of all yes/no questions"], rows)
    t["cz.table_compliance_classes"] = table(["set", "configuration", "yes/no questions", "answered ano or ne", "answered yes or no in English", "declined", "something else"], classes)
    t["cz.table_compliance_generators"] = table(["set", "family", "models", "compliance, larger minus smaller, with e5-small [95%]", "Czech share of all yes/no questions, larger minus smaller [95%]"], gaps)
    for fam, vals in compliance.items():
        if vals:
            t[f"cz.{PREFIX[fam]}compliance_range"] = span_of(vals, 3)
            t[f"cz.{PREFIX[fam]}yes_no_in_english"] = f"{english[fam]:,}"
            t[f"cz.{PREFIX[fam]}metrics_agree"] = "the same in every configuration" if same[fam] else "not the same in every configuration"

    # ---- the steps, and what each set could detect -----------------------------------------------------
    rows, power, observed = [], [], []
    sizes = a.get("planned_sizes", {})
    for k, lab in have:
        s = sets[k]
        for fam in fams:
            v, p = view(s, fam), PREFIX[fam]
            if "steps" not in v:
                continue
            m, e = v["steps"][M], v["steps"][E]
            rows.append([lab, FAMILY_LABEL[fam], f"{s['n']:,}", with_ci(m["generator_step"]), with_ci(e["generator_step"]), with_ci(m["retriever_step"]), with_ci(e["retriever_step"]),
                         with_ci(m["generator_step_minus_retriever_step"]), with_ci(e["generator_step_minus_retriever_step"])])
            t[f"cz.{k}.{p}gap_steps_mapped"] = with_ci(m["generator_step_minus_retriever_step"])
            if p:
                t[f"cz.{k}.{p}generator_step"] = with_ci(e["generator_step"])
                t[f"cz.{k}.{p}generator_step_mapped"] = with_ci(m["generator_step"])
                t[f"cz.{k}.{p}retriever_step"] = with_ci(e["retriever_step"])
                t[f"cz.{k}.{p}retriever_step_mapped"] = with_ci(m["retriever_step"])
                t[f"cz.{k}.{p}gap_steps"] = with_ci(e["generator_step_minus_retriever_step"])
            r = m["retriever_step"]
            ruled = "established" if r["established"] else rules_out(r)
            t[f"cz.{k}.{p}retriever_rules_out"] = ruled
            if not p:
                # the sizes the plan set, from the first cut's Qwen2.5 discordance
                detect = sizes.get(k, {}).get("at_this_size", {})
                power.append([lab, f"{s['n']:,}", f(detect.get("retriever_step_detects")), with_ci(r), ruled, f(detect.get("generator_step_detects")), with_ci(m["generator_step"])])
                t[f"cz.{k}.retriever_detects"] = f(detect.get("retriever_step_detects"))
                t[f"cz.{k}.generator_detects"] = f(detect.get("generator_step_detects"))
            own = v.get("detectable", {}).get(M)
            if own:
                observed.append([lab, FAMILY_LABEL[fam], f"{s['n']:,}", pct(own["generator_step"]["discordant"], 1), f(own["generator_step"]["detects"]),
                                 pct(own["retriever_step"]["discordant"], 1), f(own["retriever_step"]["detects"]), with_ci(r), ruled])
                t[f"cz.{k}.{p}generator_detects_observed"] = f(own["generator_step"]["detects"])
                t[f"cz.{k}.{p}retriever_detects_observed"] = f(own["retriever_step"]["detects"])
    t["cz.table_steps_both"] = table(["set", "family", "questions", "generator step, EM with yes and no mapped [95%]", "generator step, EM [95%]", "retriever step, EM with yes and no mapped [95%]", "retriever step, EM [95%]",
                                      "generator minus retriever, EM with yes and no mapped [95%]", "generator minus retriever, EM [95%]"], rows)
    t["cz.table_power"] = table(["set", "questions", "retriever step this set could detect at 80% power", "retriever step, EM with yes and no mapped [95%]", "a retriever step ruled out", "generator step this set could detect at 80% power", "generator step, EM with yes and no mapped [95%]"], power)
    t["cz.table_power_observed"] = table(["set", "family", "questions", "generator step: questions answered differently", "generator step this set could detect at 80% power", "retriever step: questions answered differently",
                                          "retriever step this set could detect at 80% power", "retriever step [95%]", "a retriever step ruled out"], observed)

    # ---- README's two short tables: the primary metric only, one row per set -----------------------------------
    levels, short = [], []
    for k, lab in have:
        s = sets[k]

        def lv(fam: str, config: str) -> str:
            x = view(s, fam).get("answers", {}).get(config)
            return level(x["secondary_em_yes_no_mapped"]) if x else "n/a"

        levels.append([lab, f"{s['n']:,}", lv("qwen2.5", f"bm25+{BASE}"), lv("qwen2.5", f"e5-small+{BASE}"), lv("qwen2.5", f"e5-small+{BIG}"),
                       lv("gemma-4", f"bm25+{E2}"), lv("gemma-4", f"e5-small+{E2}"), lv("gemma-4", f"e5-small+{E4}")])
        two = ("qwen2.5", "gemma-4")                    # v0.3.0's table; the three families have their own, below
        steps = {fam: view(s, fam).get("steps", {}).get(M) for fam in two}
        fd = s.get("family_difference", {}).get("gemma-4", {})
        short.append([lab, *[with_ci(steps[fam]["generator_step"]) if steps[fam] else "n/a" for fam in two],
                      *[with_ci(steps[fam]["retriever_step"]) if steps[fam] else "n/a" for fam in two],
                      *[with_ci(fd[size][M]) if fd.get(size) else "n/a" for size in ("smaller", "larger")]])
    t["cz.table_levels_mapped"] = table(["set", "questions", "BM25 + 1.5B", "e5-small + 1.5B", "e5-small + 3B", "BM25 + E2B", "e5-small + E2B", "e5-small + E4B"], levels)
    t["cz.table_steps_families"] = table(["set", "generator step, Qwen2.5: 3B minus 1.5B [95%]", "generator step, Gemma 4: E4B minus E2B [95%]", "retriever step, Qwen2.5: e5-small minus BM25 [95%]",
                                          "retriever step, Gemma 4 [95%]", "Gemma 4 E2B minus Qwen2.5 1.5B [95%]", "Gemma 4 E4B minus Qwen2.5 3B [95%]"], short)

    # ---- the two families, model for model ----------------------------------
    rows = []
    for k, lab in have:
        fdiff = sets[k].get("family_difference", {})
        for fam in in_order(fdiff, FAMILY_LABEL):
            diff = fdiff[fam]
            for size in ("smaller", "larger"):
                x = diff.get(size)
                if not x:
                    continue
                comp = x.get("compliance") or {}
                rows.append([lab, diff_label(fam, size), with_ci(x[M]), with_ci(x[E]), with_ci(comp.get("compliance")), f(x["detectable"][M]["detects"])])
                n = fam_tok(fam)
                t[f"cz.{k}.{n}.{size}.mapped"] = with_ci(x[M])
                t[f"cz.{k}.{n}.{size}.em"] = with_ci(x[E])
                t[f"cz.{k}.{n}.{size}.compliance"] = with_ci(comp.get("compliance"))
                t[f"cz.{k}.{n}.{size}.detects"] = f(x["detectable"][M]["detects"])
        for name, diff in sorted(sets[k].get("family_difference_between", {}).items()):
            a_fam, b_fam = name.split(" minus ")
            for size in ("smaller", "larger"):
                x = diff.get(size)
                if not x:
                    continue
                comp = x.get("compliance") or {}
                rows.append([lab, diff_label(a_fam, size, b_fam), with_ci(x[M]), with_ci(x[E]), with_ci(comp.get("compliance")), f(x["detectable"][M]["detects"])])
                t[f"cz.{k}.{pair_tok(name)}.{size}.mapped"] = with_ci(x[M])
                t[f"cz.{k}.{pair_tok(name)}.{size}.em"] = with_ci(x[E])
    if rows:
        t["cz.table_family"] = table(["set", "difference, both with e5-small's top 3", "EM with yes and no mapped [95%]", "EM [95%]", "compliance [95%]", "this set could detect, EM with yes and no mapped"], rows)

    # ---- the rest of the released protocol, per family: the dial, retromae-cs, the token cap ---------------
    rows, protocol = [], []
    for k, lab in have:
        for fam in fams:
            v, p = view(sets[k], fam), PREFIX[fam]
            d, dm = v.get("dial_summary"), v.get("dial_summary_em_yes_no_mapped")
            if d and dm:
                crosses = lambda x: "no" if x["root_isotonic"] is None and x["root_linear"] is None else "yes"  # noqa: E731
                n = dm["levels_compared"]
                rows.append([lab, FAMILY_LABEL[fam], f"{dm['levels_generator_larger']} of {n}", f"{dm['levels_retriever_larger']} of {n}", f"{est(dm)} of {n}", f"{est(d)} of {d['levels_compared']}",
                             crosses(dm), crosses(d), f"{f(dm['em_no_gold_base'])} and {f(dm['em_no_gold_big'])}", pct(d["answer_in_nongold_no_gold"])])
                if p:
                    t[f"cz.{k}.{p}levels_established"] = f"{est(d)} of {d['levels_compared']}"
                    t[f"cz.{k}.{p}levels_established_mapped"] = f"{est(dm)} of {n}"
                t[f"cz.{k}.{p}dial_crosses_mapped"] = crosses(dm)
            # v0.4.2: Gemma 4's dial level by level, raw EM, the family the documents name when they name one
            if fam == "gemma-4" and v.get("dial_levels"):
                a_, b_ = FAMILY_MODELS[fam]
                ci = lambda x: f"{f(x['point'], signed=True)} [{f(x['lo'], signed=True)}, {f(x['hi'], signed=True)}]"  # noqa: E731
                lv = sorted(v["dial_levels"], key=lambda x: -x["p"])
                t[f"cz.{k}.{p}table_dial_levels"] = table(
                    ["p", "g any", f"EM {SIZE_NAME[a_]}", f"EM {SIZE_NAME[b_]}", "Δ_gen [95%]", "Δ_ret [95%]"],
                    [[f(x["p"], 1), f(x["g_any"]["point"]), f(x["em"][a_]["point"]), f(x["em"][b_]["point"]), ci(x["delta_gen"]),
                      "n/a" if x.get("delta_ret") is None else ci(x["delta_ret"])] for x in lv])
                gs = [x["delta_gen"] for x in v["dial_levels"]]
                t[f"cz.{k}.{p}dial_gen_step_range"] = f"{f(min(x['point'] for x in gs), signed=True)} to {f(max(x['point'] for x in gs), signed=True)}"
                t[f"cz.{k}.{p}dial_gen_step_established_above"] = f"{sum(x['established'] and x['point'] > 0 for x in gs)} of {len(gs)}"
                if d:
                    t[f"cz.{k}.{p}leak_no_gold"] = pct(d["answer_in_nongold_no_gold"])
            if "e5small_minus_retromae_mapped" in v:
                tc = v.get("token_cap_check")
                protocol.append([lab, FAMILY_LABEL[fam], with_ci(v["e5small_minus_retromae_mapped"]), with_ci(v["e5small_minus_retromae_em"]), of(tc["same_string"], tc["questions"]) if tc else "n/a"])
                t[f"cz.{k}.{p}e5_minus_retromae_mapped"] = with_ci(v["e5small_minus_retromae_mapped"])
    if rows:
        t["cz.table_dial"] = table(["set", "family", "levels where the generator step is larger, EM with yes and no mapped", "levels where the retriever step is larger, the same", "levels established, the same", "levels established, EM",
                                    "the dial crosses, EM with yes and no mapped", "the dial crosses, EM", "EM with yes and no mapped with no gold: smaller and larger model", "answer in a non-gold passage, no gold"], rows)
    if protocol:
        t["cz.table_protocol"] = table(["set", "family", "e5-small minus retromae-cs, smaller model, EM with yes and no mapped [95%]", "the same, EM [95%]", "the token cap: the same answer at 64 and at 256 tokens"], protocol)

    # ---- what holds on all four ------------------------------------------------------------------------
    names = {"generator_step": "generator step", "retriever_step": "retriever step", "generator_step_minus_retriever_step": "generator step minus retriever step"}
    rows = []
    holds = a["holds_on_all_four"]
    for fam in fams:
        for step, name in names.items():
            for field, tag in ((M, "EM with yes and no mapped"), (E, "EM")):
                h = holds.get(fam, {}).get(field, {}).get(step)
                if not h:
                    continue
                rows.append([FAMILY_LABEL[fam], name, tag, holds_text(h)])
                t[f"cz.holds.{PREFIX[fam]}{step}.{'mapped' if field == M else 'em'}"] = holds_text(h)
                t[f"cz.holds.{PREFIX[fam]}{step}.{'mapped' if field == M else 'em'}_on"] = and_list([SET_PROSE[k] for k in h["established_on"]]) or "none of the four"
    for fam in (x for x in fams if x != "qwen2.5"):          # every family against Qwen2.5, whatever the order
        for size in ("smaller", "larger"):
            for field, tag in ((M, "EM with yes and no mapped"), (E, "EM")):
                h = holds.get(f"{fam} minus qwen2.5", {}).get(field, {}).get(size)
                if not h:
                    continue
                text = holds_text(h) + ("" if h["sign"] is None or not h["holds_on_all_four"] else f", {FAMILY_LABEL[fam] if h['sign'] > 0 else 'Qwen2.5'} higher")
                rows.append([f"{FAMILY_LABEL[fam]} and Qwen2.5", diff_label(fam, size), tag, text])
                t[f"cz.holds.{fam_tok(fam)}.{size}.{'mapped' if field == M else 'em'}"] = text
    for name in sorted(n for n in holds if " minus " in n and not n.endswith(" minus qwen2.5")):
        a_fam, b_fam = name.split(" minus ")
        for size in ("smaller", "larger"):
            for field, tag in ((M, "EM with yes and no mapped"), (E, "EM")):
                h = holds.get(name, {}).get(field, {}).get(size)
                if not h:
                    continue
                higher = None if h["sign"] is None else (FAMILY_LABEL[a_fam] if h["sign"] > 0 else FAMILY_LABEL[b_fam])
                if h["holds_on_all_four"]:
                    text = holds_text(h) + (f", {higher} higher" if higher else "")
                else:                     # name the sets, and say whether the established ones agree in sign
                    where = and_list([SET_PROSE[k] for k in h["established_on"]])
                    text = holds_text(h) + (f" ({where}), {higher} higher on each" if higher else f" ({where})" if where else "")
                rows.append([f"{FAMILY_LABEL[a_fam]} and {FAMILY_LABEL[b_fam]}", diff_label(a_fam, size, b_fam), tag, text])
                short = "mapped" if field == M else "em"
                t[f"cz.holds.{pair_tok(name)}.{size}.{short}"] = text
                t[f"cz.holds.{pair_tok(name)}.{size}.{short}_on"] = and_list([SET_PROSE[k] for k in h["established_on"]]) or "none of the four"
                pts = [sets[k]["family_difference_between"][name][size][field]["point"] for k, _ in have]
                t[f"cz.{pair_tok(name)}.{size}.{short}_points"] = ("below zero on all four" if all(x < 0 for x in pts) else
                                                                  "above zero on all four" if all(x > 0 for x in pts) else "of both signs across the four")
                # v0.4.2: a difference of exactly zero is no sign change
                zeros = sum(x == 0 for x in pts)
                t[f"cz.{pair_tok(name)}.{size}.{short}_sign"] = (
                    "below zero on all four" if all(x < 0 for x in pts) else "above zero on all four" if all(x > 0 for x in pts) else
                    f"at or below zero on all four (zero on {zeros})" if all(x <= 0 for x in pts) else
                    f"at or above zero on all four (zero on {zeros})" if all(x >= 0 for x in pts) else "of both signs across the four")
    t["cz.table_holds"] = table(["family", "reading", "metric", "across the four sets"], rows)

    # ---- the questions this run added ---------------------------------------------------------------------
    added = a.get("added_questions", {}).get("sets", {})
    rows = []
    for k, lab in have:
        x = added.get(k)
        if not x:
            continue
        for fam in fams:
            steps = x["steps"] if fam == "qwen2.5" else x.get("families", {}).get(fam, {}).get("steps")
            if not steps:
                continue
            m, e = steps[M], steps[E]
            rows.append([lab, FAMILY_LABEL[fam], f"{x['questions']:,}", with_ci(m["generator_step"]), with_ci(e["generator_step"]), with_ci(m["retriever_step"]), with_ci(e["retriever_step"])])
    if rows:
        t["cz.table_added"] = table(["set, added questions only", "family", "questions", "generator step, EM with yes and no mapped [95%]", "generator step, EM [95%]", "retriever step, EM with yes and no mapped [95%]", "retriever step, EM [95%]"], rows)

    # ---- the released 200: this run, the first cut, v0.2.0. Qwen2.5 only: neither ran Gemma 4 -----------------
    cont = a.get("continuity_released") or {}
    rows = []
    for c in CONFIGS:
        x, y = cont.get("first_cut", {}).get(c), cont.get("v020", {}).get(c)
        if not (x and y):
            continue
        comp = lambda b: level((b or {}).get("compliance"))  # noqa: E731
        rows.append([CONFIG_LABEL[c], f"{x['same_string']} of {x['questions']}", f"{y['same_string']} of {y['questions']}", f(x[M]["this_run"]), f(y[M]["other"]), with_ci(y[M]["difference"]), f(x[E]["this_run"]), f(y[E]["other"]), with_ci(y[E]["difference"]),
                     comp(x["compliance_this_run"]), comp(y["compliance_other"])])
        t[f"cz.continuity.{c.split('+')[0]}.{'3b' if c.endswith('3b') else '1_5b'}.first_cut_same"] = f"{x['same_string']} of {x['questions']}"
    if rows:
        t["cz.table_continuity"] = table(["configuration", "same answer as the first cut", "same answer as v0.2.0", "EM with yes and no mapped", "the same, v0.2.0", "difference from v0.2.0 [95%]", "EM", "EM, v0.2.0", "difference from v0.2.0 [95%]", "compliance [95%]", "compliance, v0.2.0 [95%]"], rows)

    # ---- the cross-version check ---------------------------------------------------------------------------------
    cross = a.get("cross_version")
    if cross:
        t["cz.cross.n"] = f"{cross['pairs']:,}"
        t["cz.cross.same_gold"] = f"{cross['same_gold_pairs']:,}"
        rows = []
        for g in in_order(cross["generators"], MODEL_LABEL):
            x = cross["generators"][g]
            short = MODEL_SHORT.get(g, g)
            rows.append([MODEL_LABEL.get(g, g), f(x[M]["copy"]), f(x[M]["sqad32"]), with_ci(x[M]["difference"]), with_ci(x[f"{M}_same_gold"]["difference"]), f(x[E]["copy"]), f(x[E]["sqad32"]), with_ci(x[E]["difference"]), with_ci(x[f"{E}_same_gold"]["difference"]),
                         with_ci(x["gold_in_top3"]["difference"]), f"{x['same_answer_string']:,}"])
            t[f"cz.cross.{short}.mapped"] = with_ci(x[M]["difference"])
            t[f"cz.cross.{short}.em"] = with_ci(x[E]["difference"])
            t[f"cz.cross.{short}.mapped_same_gold"] = with_ci(x[f"{M}_same_gold"]["difference"])
        t["cz.table_cross"] = table(["model", "EM with yes and no mapped, the copy's passages", "the same, SQAD 3.2's passages", "SQAD 3.2 minus the copy [95%]", "the same, same gold in both [95%]", "EM, the copy's passages", "EM, SQAD 3.2's passages",
                                     "SQAD 3.2 minus the copy [95%]", "the same, same gold in both [95%]", "gold in the top 3, SQAD 3.2 minus the copy [95%]", "same answer on both"], rows)
        step = cross.get("generator_step")
        if step:
            rows = []
            for fam in in_order(step, FAMILY_LABEL):
                for s in in_order(step[fam], passages):
                    v = step[fam][s]
                    rows.append([FAMILY_LABEL.get(fam, fam), passages[s], with_ci(v[M]), with_ci(v[E])])
                    t[f"cz.cross.{PREFIX.get(fam, fam + '.')}step.{s}.mapped"] = with_ci(v[M])
            t["cz.table_cross_steps"] = table(["family", "passages", "generator step, EM with yes and no mapped [95%]", "generator step, EM [95%]"], rows)
        rows = []
        fd = cross.get("family_difference") or {}
        for fam in in_order(fd, FAMILY_LABEL):
            for size in in_order(fd[fam], SIZE_LABEL):
                for s in in_order(fd[fam][size], passages):
                    v = fd[fam][size][s]
                    rows.append([diff_label(fam, size), passages[s], with_ci(v[M]), with_ci(v[E])])
                    t[f"cz.cross.{fam_tok(fam)}.{size}.{s}.mapped"] = with_ci(v[M])
                    t[f"cz.cross.{fam_tok(fam)}.{size}.{s}.em"] = with_ci(v[E])
        for name, block in sorted((cross.get("family_difference_between") or {}).items()):
            a_fam, b_fam = name.split(" minus ")
            for size in in_order(block, SIZE_LABEL):
                for s in in_order(block[size], passages):
                    v = block[size][s]
                    rows.append([diff_label(a_fam, size, b_fam), passages[s], with_ci(v[M]), with_ci(v[E])])
                    t[f"cz.cross.{pair_tok(name)}.{size}.{s}.mapped"] = with_ci(v[M])
        if rows:
            t["cz.table_cross_family"] = table(["difference, both with e5-small's top 3", "passages", "EM with yes and no mapped [95%]", "EM [95%]"], rows)

    # ---- the planned gap ---------------------------------------------------------------------------------------------
    for field, short in ((M, "mapped"), (E, "em")):
        rows = []
        for c, b in a.get("planned_gap", {}).get(field, {}).items():
            g1, g2 = b.get("G1"), b.get("G2")
            if not (g1 and g2):
                continue
            rule = g1.get("on_questions_the_rule_passes") or {}
            share = g1.get("share_of_the_gap_the_rule_accounts_for")
            rows.append([CONFIG_LABEL[c], with_ci(g1["gap"]["gap"]), with_ci(g1["after_matching_answer_type_and_length"]["gap_after_reweighting"]), with_ci(rule.get("gap")), "n/a" if share is None else pct(share),
                         with_ci(g2["gap"]["gap"]), with_ci(g2["after_matching_answer_type_and_length"]["gap_after_reweighting"]), with_ci(b.get("passages", {}).get("own_gold")), with_ci(b.get("passages", {}).get("same_gold")), with_ci(b.get("what_is_left"))])
        if rows:
            t[f"cz.table_planned_gap_{short}"] = table(["configuration", "G1: SQAD 3.2 full minus the released 200 [95%]", "G1 after matching answer type and length [95%]", "G1 on questions the rule passes [95%]", "share of G1 the rule accounts for",
                                                        "G2: SQAD 3.2 debiased minus the copy debiased [95%]", "G2 after matching answer type and length [95%]", "the passages, same question [95%]", "the passages, same gold [95%]", "what is left of G2 [95%]"], rows)

    # ---- the order check, the 0.10 rows ---------------------------------------------------------------------------------
    order = a.get("order_check")
    if order:
        t["cz.table_order"] = table(["model", "answers compared", "same string", "same EM", "same EM with yes and no mapped"],
                                    [[MODEL_LABEL.get(g, g), f"{order[g]['answers_compared']:,}", f"{order[g]['same_string']:,}", f"{order[g]['same_em']:,}", f"{order[g]['same_em_yes_no_mapped']:,}"] for g in in_order(order, MODEL_LABEL)])
    rows = []
    for k, s in a.get("sensitivity_010", {}).items():
        for fam in FAMILY_MODELS:
            v = view(s, fam)
            if "steps" not in v:
                continue
            m, e = v["steps"][M], v["steps"][E]
            rows.append([s["label"], FAMILY_LABEL[fam], f"{s['n']:,}", with_ci(m["generator_step"]), with_ci(e["generator_step"]), with_ci(m["retriever_step"]), with_ci(e["retriever_step"])])
    if rows:
        t["cz.table_sensitivity_both"] = table(["sensitivity row", "family", "questions", "generator step, EM with yes and no mapped [95%]", "generator step, EM [95%]", "retriever step, EM with yes and no mapped [95%]", "retriever step, EM [95%]"], rows)

    # ---- the prompt, on the dev set --------------------------------------------------------------------------------------
    dev = a.get("dev_selection")
    if dev:
        rows = []
        for v, by_g in dev["variants"].items():
            for g, x in by_g.items():
                rows.append([v.upper(), "1.5B" if g == BASE else "3B", f(x["compliance"]), f(x["strict_share_of_all_yes_no"]), f(x["em_on_yes_no"], 2), f(x["em_on_yes_no_mapped"], 2), f(x["em_on_span"], 2), x["span_answers_declined"], x["exploratory"]["span_answered_ano_or_ne"]])
        t["cz.table_dev"] = table(["prompt", "generator", "compliance", "Czech share of all yes/no", "EM, yes/no", "EM with yes and no mapped, yes/no", "EM, span", "span answers declined", "span answers of ano or ne"], rows)
        sel = dev["selection"]
        t["cz.dev.chosen"] = sel["chosen"].upper()
        t["cz.dev.questions"] = str(dev["questions"])
        t["cz.dev.yes_no"] = str(dev["yes_no_questions"])
        for g, short in ((BASE, "1_5b"), (BIG, "3b")):
            t[f"cz.dev.v4_below_v1.{short}"] = f(sel["span_em_below_v1"][g], 2)
            t[f"cz.dev.v1_compliance.{short}"] = f(dev["variants"]["v1"][g]["compliance"])
            t[f"cz.dev.v4_compliance.{short}"] = f(dev["variants"]["v4"][g]["compliance"])

    # ---- compute ---------------------------------------------------------------------------------------------------------------
    run = a.get("compute_complete")
    if run:
        none = {"generations": 0, "seconds": 0.0}
        gx = run.get("gx10", {})
        set_aside = (run["superseded_v3"], run.get("superseded_mac", none))
        parts = (run["dev"], run["main"], *set_aside, gx.get("pilot", none))
        t["cz.compute.generations"] = f"{sum(x['generations'] for x in parts):,}"
        t["cz.compute.hours"] = f"{sum(x['seconds'] for x in parts) / 3600:.1f}"
        t["cz.compute.main_generations"] = f"{run['main']['generations']:,}"
        t["cz.compute.main_hours"] = f"{run['main']['seconds'] / 3600:.1f}"
        t["cz.compute.dev_generations"] = f"{run['dev']['generations']:,}"
        t["cz.compute.set_aside"] = f"{sum(x['generations'] for x in set_aside):,}"
        t["cz.compute.set_aside_v3"] = f"{run['superseded_v3']['generations']:,}"
        t["cz.compute.set_aside_mac"] = f"{run.get('superseded_mac', none)['generations']:,}"
        if gx:
            t["cz.compute.pilot_generations"] = f"{gx['pilot']['generations']:,}"
            t["cz.compute.gx10_main_hours"] = f"{gx['main']['seconds'] / 3600:.1f}"
        gm = run.get("gx10_ministral")
        if gm:
            t["cz.compute.ministral_generations"] = f"{gm['generations']:,}"
            t["cz.compute.ministral_hours"] = f"{gm['seconds'] / 3600:.1f}"
            t["cz.compute.ministral_stages"] = str(gm["stages"])
        gq = run.get("gx10_qwen35")
        if gq:
            t["cz.compute.qwen35_generations"] = f"{gq['generations']:,}"
            t["cz.compute.qwen35_hours"] = f"{gq['seconds'] / 3600:.1f}"
            t["cz.compute.qwen35_stages"] = str(gq["stages"])
        gn = run.get("gx10_gemma3n")
        if gn:
            t["cz.compute.gemma3n_generations"] = f"{gn['generations']:,}"
            t["cz.compute.gemma3n_hours"] = f"{gn['seconds'] / 3600:.1f}"
            t["cz.compute.gemma3n_stages"] = str(gn["stages"])
    tags = a.get("thinking_tags")
    if tags:
        t["cz.q35.thinking.answer_rows"] = f"{sum(tags[g]['answer_rows'] for g in (Q2, Q4)):,}"
        t["cz.q35.thinking.with_a_tag"] = f"{sum(tags[g]['with_a_tag'] for g in (Q2, Q4)):,}"
    t.update(range_tokens(a, have, fams))
    t.update(family_tokens(a, have))
    table_first = claims_table(a, "claims_against_first_cut", "the first v0.3.0 cut, on the same set")
    if table_first:
        t["cz.table_claims_first_cut"] = table_first
    return t


QUOTE_SCAN = Path(__file__).resolve().parents[1] / "results/czech/quote_scan.json"
FAMILY_OF_MODEL = FULL_LABEL       # every family's models by their full names (Ministral 3 and Qwen3.5 included)


def quote_tokens() -> dict[str, str]:
    """What the shipped answers quote, from scripts/czech_quote_scan.py's record, which needs the data."""
    if not QUOTE_SCAN.exists():
        return {}
    q = json.loads(QUOTE_SCAN.read_text(encoding="utf-8"))
    s = q["summary"]
    names = lambda gs: " and ".join(FAMILY_OF_MODEL.get(g, g) for g in gs)  # noqa: E731
    scoring = lambda ems: "all scoring 0" if ems == [0] else "some scoring 1"  # noqa: E731
    return {"cz.quotes.longest": str(s["longest_quote"]), "cz.quotes.line": str(q["long_chars"]),
            "cz.quotes.repeated_rows": str(s["rows_holding_a_whole_question"]), "cz.quotes.repeated_questions": str(s["questions_repeated"]),
            "cz.quotes.repeated_by": names(s["generators_repeating"]), "cz.quotes.repeated_scoring": scoring(s["em_of_those_rows"]),
            "cz.quotes.repeated_whose": "its own question" if s["every_one_its_own_question"] else "a question",
            "cz.quotes.over_line": str(s["answers_over_the_line"]), "cz.quotes.over_line_rows": str(s["rows_over_the_line"]),
            "cz.quotes.over_line_by": names(s["generators_over_the_line"]), "cz.quotes.over_line_scoring": scoring(s["em_over_the_line"])}


def defect_same_machine_tokens(a: dict) -> dict[str, str]:
    """The prompt's defect read on one machine: the first cut against v0.2.0, both on the Mac."""
    t: dict[str, str] = {}
    d = a.get("defect_same_machine")
    if not d:
        return t
    rows = []
    for c, short in zip(CONFIGS, ("bm25", "e5", "retromae", "e5_3b")):
        x = d["configurations"].get(c)
        if not x:
            continue
        rows.append([CONFIG_LABEL[c], f(x["em_v020"]), f(x["em_fixed"]), with_ci(x["em_difference"]), f(x["em_on_yes_no_v020"]), f(x["em_on_yes_no_fixed"]), with_ci(x["em_difference_on_yes_no"]),
                     f(x["em_on_span_v020"]), f(x["em_on_span_fixed"]), with_ci(x["em_difference_on_span"])])
        t[f"cz.defect_mac.{short}.diff"] = with_ci(x["em_difference"])
        t[f"cz.defect_mac.{short}.diff_yes_no"] = with_ci(x["em_difference_on_yes_no"])
        t[f"cz.defect_mac.{short}.em_yes_no_v020"] = f(x["em_on_yes_no_v020"])
        t[f"cz.defect_mac.{short}.em_yes_no_fixed"] = f(x["em_on_yes_no_fixed"])
        t[f"cz.defect_mac.{short}.english"] = str(x["yes_no_classes_fixed"]["yes or no in English"])
        t[f"cz.defect_mac.{short}.english_v020"] = str(x["yes_no_classes_v020"]["yes or no in English"])
    t["cz.table_defect_same_machine"] = table(["configuration", "EM, v0.2.0's prompt", "EM, v0.3.0's prompt", "difference [95%]", "EM on yes/no, v0.2.0's prompt", "EM on yes/no, v0.3.0's prompt",
                                                "difference on yes/no [95%]", "EM on span, v0.2.0's prompt", "EM on span, v0.3.0's prompt", "difference on span [95%]"], rows)
    # v0.4.2: the same table, its columns named by the instruction rather than by a release
    t["cz.table_prompt_same_machine"] = table(["configuration", "EM, yes-or-no instruction", "EM, ano-or-ne instruction", "difference [95%]",
                                               "EM on yes/no, yes-or-no instruction", "EM on yes/no, ano-or-ne instruction", "difference on yes/no [95%]",
                                               "EM on span, yes-or-no instruction", "EM on span, ano-or-ne instruction", "difference on span [95%]"], rows)
    t["cz.defect_mac.questions"] = str(d["questions"])
    t["cz.defect_mac.yes_no"] = str(d["yes_no_questions"])
    return t


def gx10_tokens(a: dict) -> dict[str, str]:
    """The pilot that moved the Czech generations to the GX10."""
    t: dict[str, str] = {}
    pilot = a.get("gx10_pilot")
    if not pilot:
        return t
    rep = pilot["repeat"]
    t["cz.pilot.repeat_same"] = f"{sum(x['same_string'] for x in rep.values())} of {sum(x['answers'] for x in rep.values())}"
    sec, share = pilot["seconds_per_answer"], pilot["share_of_the_macs_time"]
    for kind, short in (("dial_1.5b", "1_5b"), ("dial_3b", "3b")):
        t[f"cz.pilot.seconds_gx10.{short}"] = f(sec["gx10"][kind], 2)
        t[f"cz.pilot.seconds_mac.{short}"] = f(sec["mac"][kind], 1)
        t[f"cz.pilot.share.{short}"] = pct(share[kind], 1)
    for i, (name, ok) in enumerate(sorted(pilot["conditions"].items()), 1):
        t[f"cz.pilot.condition_{i}"] = "met" if ok else "not met"
    rows = []
    differs = total = 0
    for key, by_r in pilot["rankings"].items():
        for rid, x in by_r.items():
            rows.append([{"released": "SQAD 3.0 copy, the released 200", "sqad32_full": "SQAD 3.2, full"}.get(key, key), rid, f"{x['questions']:,}", x["top_10_differs"], x["top_3_differs"]])
            differs += x["top_10_differs"]
            total += x["questions"]
    t["cz.table_pilot_rankings"] = table(["set", "retriever", "questions", "top 10 differs from the shipped ranking", "top 3 differs"], rows)
    t["cz.pilot.top10_differs"] = f"{differs:,} of {total:,}"
    t["cz.pilot.top10_differs_n"] = f"{differs:,}"
    t["cz.pilot.rankings_checked"] = f"{total:,}"
    rows, shares = [], []
    for name, x in pilot["agreement_with_the_mac"].items():
        v, g = name.split(" ", 1)
        rows.append([v.upper(), MODEL_LABEL.get(g, g), f"{x['same_string']} of {x['answers']}", f(x["em_yes_no_mapped"]["mac"]), f(x["em_yes_no_mapped"]["gx10"]), with_ci(x["em_yes_no_mapped"]["gx10_minus_mac"]),
                     f(x["em"]["mac"]), f(x["em"]["gx10"]), with_ci(x["em"]["gx10_minus_mac"])])
        short = MODEL_SHORT.get(g, g)
        t[f"cz.pilot.same.{v}.{short}"] = f"{x['same_string']} of {x['answers']}"
        t[f"cz.pilot.mapped_diff.{v}.{short}"] = with_ci(x["em_yes_no_mapped"]["gx10_minus_mac"])
        shares.append(x["same_string"] / x["answers"])
    t["cz.pilot.same_range"] = f"{pct(min(shares))} to {pct(max(shares))}"
    t["cz.pilot.questions"] = str(next(iter(pilot["agreement_with_the_mac"].values()))["answers"])
    t["cz.table_pilot_agreement"] = table(["prompt", "generator", "same answer on both machines", "EM with yes and no mapped, the Mac", "the same, the GX10", "GX10 minus the Mac [95%]", "EM, the Mac", "EM, the GX10", "GX10 minus the Mac [95%]"], rows)
    t["cz.pilot.rule_on_gx10_answers"] = pilot["rule_on_the_gx10_answers"]["selection"]["chosen"].upper()
    return t


def bench_tokens(a: dict) -> dict[str, str]:
    """The dev-set benchmark of model families."""
    t: dict[str, str] = {}
    short = {"qwen2.5": "qwen25", "gemma-4": "gemma4", "granite-4.0": "granite4", "ministral-3": "ministral3", "qwen3.5": "qwen35"}

    def model_rows(models: dict) -> list[list]:
        rows = []
        for m in in_order(models, BENCH_LABEL):
            x = models[m]
            if not x.get("answered"):
                rows.append([BENCH_LABEL.get(m, m), "did not load", *["n/a"] * 6])
                continue
            rows.append([BENCH_LABEL.get(m, m), f"{x['gguf_gb']:.1f} GB", f(x["em_yes_no_mapped"]), f(x["em"]), f(x["compliance"]), f(x["em_on_span"]), f(x["gpu_seconds_per_answer"], 2), f(x["cpu_seconds_per_answer"], 2)])
        return rows

    header = ["model", "file at Q4_K_M", "EM with yes and no mapped", "EM", "compliance", "EM on span questions", "seconds per answer, GPU", "seconds per answer, CPU"]
    mb = a.get("model_bench")
    if mb:
        t["cz.bench.questions"] = str(mb["questions"])
        t["cz.bench.yes_no"] = str(mb["yes_no_questions"])
        t["cz.table_model_bench"] = table(header, model_rows(mb["models"]))
        rows = []
        for fam in in_order(mb["families"], BENCH_FAMILY_LABEL):
            x = mb["families"][fam]
            s = short.get(fam, fam)
            if not x.get("eligible"):
                rows.append([BENCH_FAMILY_LABEL.get(fam, fam), "did not load", "n/a", "n/a"])
                continue
            rows.append([BENCH_FAMILY_LABEL.get(fam, fam), level(x["score"]), with_ci(x.get("minus_qwen")), with_ci(x.get("minus_qwen3_5"))])
            t[f"cz.bench.{s}.score"] = level(x["score"])
            for k in ("minus_qwen", "minus_qwen3_5"):
                if x.get(k):
                    t[f"cz.bench.{s}.{k}"] = with_ci(x[k])
        t["cz.table_model_bench_families"] = table(["family", "score: EM with yes and no mapped, mean of its two models [95%]", "minus Qwen2.5 [95%]", "minus Qwen3.5 [95%]"], rows)
        t["cz.bench.joins"] = BENCH_FAMILY_LABEL.get(mb["rule"]["joins_beside_qwen"], mb["rule"]["joins_beside_qwen"]) if mb["rule"].get("joins_beside_qwen") else "none"
        for m, x in mb["models"].items():
            if x.get("answered"):
                t[f"cz.bench.{m}.mapped"] = f(x["em_yes_no_mapped"])
                t[f"cz.bench.{m}.compliance"] = f(x["compliance"])
                t[f"cz.bench.{m}.gpu_seconds"] = f(x["gpu_seconds_per_answer"], 2)
                t[f"cz.bench.{m}.gb"] = f"{x['gguf_gb']:.1f}"
    mi = a.get("model_bench_ministral")
    if mi:
        t["cz.table_model_bench_ministral"] = table(header, model_rows(mi["models"]))
        fb = mi.get("family_3b_and_8b")
        if fb:
            t["cz.bench.ministral3.score"] = level(fb["score"])
            for fam, c in fb["minus"].items():
                t[f"cz.bench.ministral3.minus_{short.get(fam, fam)}"] = with_ci(c)
        for m, c in (mi.get("14b_minus") or {}).items():
            t[f"cz.bench.ministral3_14b.minus_{m}"] = with_ci(c)
        for m, x in mi["models"].items():
            if x.get("answered"):
                t[f"cz.bench.{m}.mapped"] = f(x["em_yes_no_mapped"])
                t[f"cz.bench.{m}.compliance"] = f(x["compliance"])
                t[f"cz.bench.{m}.gpu_seconds"] = f(x["gpu_seconds_per_answer"], 2)
                t[f"cz.bench.{m}.gb"] = f"{x['gguf_gb']:.1f}"
    return t


def changes_from_v020(a: dict) -> list[list[str]]:
    """Every Czech number of v0.2.0's text that moved: the released 200, old prompt and fixed prompt.

    Both columns come from the records. The old column is what v0.2.0 printed: the same analysis over
    the answers it reported (results/czech/records/answers_released_v020.jsonl.gz).
    """
    old, new = a["sets"].get("released_v020"), a["sets"].get("released")
    if not (old and new):
        return []
    rows: list[list[str]] = []

    def add(name: str, x: str, y: str) -> None:
        if x != y:
            rows.append([name, x, y])

    for c in CONFIGS:
        label = CONFIG_LABEL[c]
        for metric, title in (("em", "EM"), ("f1", "F1")):
            add(f"{label}, {title}", f(old["answers"][c][metric]["point"]), f(new["answers"][c][metric]["point"]))
    by_p = {x["p"]: x for x in new["dial_levels"]}
    for x in old["dial_levels"]:
        y = by_p[x["p"]]
        at = f"Dial, p = {x['p']:.1f}"
        add(f"{at}, EM 1.5B", f(x["em"][BASE]["point"]), f(y["em"][BASE]["point"]))
        add(f"{at}, EM 3B", f(x["em"][BIG]["point"]), f(y["em"][BIG]["point"]))
        add(f"{at}, generator step [95%]", with_ci(x["delta_gen"]), with_ci(y["delta_gen"]))
        add(f"{at}, retriever step [95%]", with_ci(x["delta_ret"]), with_ci(y["delta_ret"]))
        add(f"{at}, retriever step minus generator step [95%]", with_ci(x["ret_minus_gen"]), with_ci(y["ret_minus_gen"]))
        add(f"{at}, answer in a non-gold passage", pct(x["answer_in_nongold"], 1), pct(y["answer_in_nongold"], 1))
    newer = {x["p"]: x for x in new["dial_summary"]["break_even_raw"]}

    def gain(x: dict) -> str:
        return "not located" if x["point"] is None else f"{f(x['point'])} [{f(x['lo'])}, {f(x['hi'])}]"

    for x in old["dial_summary"]["break_even_raw"]:
        add(f"Break-even gold gain, p = {x['p']:.1f} [95%]", gain(x), gain(newer[x["p"]]))
    add("EM with no gold, 1.5B", f(old["dial_summary"]["em_no_gold_base"]), f(new["dial_summary"]["em_no_gold_base"]))
    add("EM with no gold, 3B", f(old["dial_summary"]["em_no_gold_big"]), f(new["dial_summary"]["em_no_gold_big"]))
    add("Dial levels where the generator step is the larger", f"{old['dial_summary']['levels_generator_larger']} of {old['dial_summary']['levels_compared']}", f"{new['dial_summary']['levels_generator_larger']} of {new['dial_summary']['levels_compared']}")
    add("Dial levels with the gap established", f"{old['dial_summary']['levels_established_with_gold']} of {old['dial_summary']['levels_compared']}", f"{new['dial_summary']['levels_established_with_gold']} of {new['dial_summary']['levels_compared']}")
    return rows


def _em_cell(x: dict | None) -> str:
    return f"{f(x['em'])} ({x['n']})" if x and x.get("n") else "none"


def gap_tokens(a: dict, names: dict, have: list) -> dict[str, str]:
    """How the gap between the versions breaks down. Exploratory, and labelled so where it is quoted."""
    t: dict[str, str] = {}
    gap = a.get("gap", {})
    both = (f"e5-small+{BASE}", f"e5-small+{BIG}")
    for name, field, header in (("cz.table_em_by_type", "em_by_answer_type", "answer type"), ("cz.table_em_by_length", "em_by_length_band", "gold answer, tokens"),
                                ("cz.table_em_by_opening", "em_by_opening_sentence", "answer sentence opens its article"), ("cz.table_em_by_passage", "em_by_gold_passage_length", "gold passage, characters"),
                                ("cz.table_em_by_rule", "em_by_the_debias_rule", "flagged by the debias rule at 0.138"),
                                ("cz.table_em_by_answer_in_top3", "em_by_answer_in_top3", "the answer's string")):
        tab = gap.get(field, {})
        sets = [(k, lab) for k, lab in have if k in tab]
        if not sets:
            continue
        rows = []
        for c in both:
            for cell in tab[sets[0][0]][c]:
                rows.append([{"True": "yes", "False": "no"}.get(cell, cell), CONFIG_LABEL[c], *[_em_cell(tab[k][c].get(cell)) for k, _ in sets]])
        t[name] = table([header, "configuration", *[f"{lab}: EM (questions)" for _, lab in sets]], rows)
    tab = gap.get("em_by_origin", {})
    shared = gap.get("on_questions_the_copy_also_holds", {})
    rows = []
    for pair in names:
        key = pair.split(" minus ")[0]
        for c, parts in shared.get(pair, {}).items():
            x, later = parts.get("all of the copy's set"), parts.get("answer not in an opening sentence")
            o = tab.get(key, {}).get(c)
            if not (x and o):
                continue
            rows.append([names.get(pair, pair), CONFIG_LABEL[c], f(x["em_copy"]), _em_cell(o["in the copy"]), _em_cell(o["dropped from the copy"]), _em_cell(o["new in SQAD 3.2"]), with_ci(x["gap"]), with_ci(later["gap"]) if later else "n/a"])
    if rows:
        t["cz.table_gap_origin"] = table(["gap in EM", "configuration", "EM, the copy's set", "SQAD 3.2: EM on questions the copy also holds (questions)", "SQAD 3.2: EM on questions dropped from the copy (questions)", "SQAD 3.2: EM on questions new in 3.2 (questions)",
                                          "gap on questions the copy also holds [95%]", "the same, answer not in an opening sentence [95%]"], rows)
    full = shared.get("sqad32_full minus released", {})
    for c, short in zip(CONFIGS, ("bm25", "e5", "retromae", "e5_3b")):
        if full.get(c) and full[c].get("all of the copy's set"):
            t[f"cz.shared.{short}.gap"] = with_ci(full[c]["all of the copy's set"]["gap"])
    levels = gap.get("gap_by_dial_level", {})
    rows = []
    for pair in names:
        for c, by_p in levels.get(pair, {}).items():
            for p, x in by_p.items():
                rows.append([names.get(pair, pair), CONFIG_LABEL[c], p, f(x["em_copy"]), f(x["em_target"]), with_ci(x["gap"])])
    if rows:
        t["cz.table_gap_levels"] = table(["gap in EM", "configuration", "dial level p", "EM, SQAD 3.0 copy", "EM, SQAD 3.2", "gap [95%]"], rows)
    for c, short in zip(both, ("e5", "e5_3b")):
        x = levels.get("sqad32_full minus released", {}).get(c, {}).get("1.0")
        if x:
            t[f"cz.gap_no_gold.{short}"] = with_ci(x["gap"])
    passes = gap.get("on_questions_the_rule_passes", {})
    full_passes, debiased = passes.get("sqad32_full minus released", {}), gap.get("reweighting", {}).get("sqad32_debiased minus copy_debiased", {})
    rows = []
    closed = []
    for c, short in zip(CONFIGS, ("bm25", "e5", "retromae", "e5_3b")):
        x = full_passes.get(c)
        if x:
            t[f"cz.passes.{short}.gap"] = with_ci(x["gap"])
            t[f"cz.passes.{short}.n"] = f"{x['n_copy']} and {x['n_target']}"
            whole = gap.get("reweighting", {}).get("sqad32_full minus released", {}).get(c, {}).get("answer type and length")
            if whole:
                closed.append(1 - x["gap"]["point"] / whole["gap"]["point"])
            else:
                st = a["sets"]
                closed.append(1 - x["gap"]["point"] / (st["sqad32_full"]["answers"][c]["em"]["point"] - st["released"]["answers"][c]["em"]["point"]))
        v = debiased.get(c, {}).get("answer type and length")
        if v:
            t[f"cz.gap_debiased.{short}"] = with_ci(v["gap"])
        r_full = gap.get("reweighting", {}).get("sqad32_full minus released", {}).get(c, {}).get("answer type and length")
        if r_full and x and v:
            rows.append([CONFIG_LABEL[c], with_ci(r_full["gap"]), f"{x['n_copy']} and {x['n_target']}", with_ci(x["gap"]), with_ci(v["gap"])])
    if rows:
        t["cz.table_gap_rule"] = table(["configuration", "SQAD 3.2, 200 unfiltered, minus the SQAD 3.0 copy, the released 200 [95%]", "questions the rule passes, in each of those sets", "the same two sets, only questions the rule passes [95%]", "SQAD 3.2 minus the SQAD 3.0 copy, both 200 debiased at 0.138 [95%]"], rows)
    if closed:
        t["cz.rule.closed"] = pct_range(closed)
    di = a.get("diagnostics", {})
    if di.get("copy_test_split") and di.get("sqad32_test_split"):
        t["cz.rule.share_copy"] = pct(di["copy_test_split"]["flagged_by_the_rule_0138"], 1)
        t["cz.rule.share_32"] = pct(di["sqad32_test_split"]["flagged_by_the_rule_0138"], 1)
    by_rule = gap.get("em_by_the_debias_rule", {}).get("sqad32_full", {}).get(f"e5-small+{BASE}")
    if by_rule:
        t["cz.rule.em_flagged_32"] = f(by_rule["True"]["em"])
        t["cz.rule.em_passing_32"] = f(by_rule["False"]["em"])
    rw = gap.get("reweighting", {}).get("sqad32_full minus released", {})
    for name, short in (("answer type and length", "type_length"), ("the opening sentence", "opening"), ("answer type, length and the opening sentence", "all"), ("the gold passage's length", "passage")):
        shares = [v[name]["share_of_the_gap_closed"] for v in rw.values() if v.get(name)]
        if shares:
            t[f"cz.gap_closed.{short}"] = pct_range(shares)
    for c, short in zip((f"bm25+{BASE}", *both), ("bm25", "e5", "e5_3b")):
        v = rw.get(c, {}).get("answer type and length")
        if v:
            t[f"cz.gap.{short}"] = with_ci(v["gap"])
    t.pop("cz.gap_debiased.retromae", None)
    r = gap.get("same_questions_retrieval")
    if r:
        t["cz.same_retrieval.n"] = str(r["questions"])
        t["cz.same_retrieval.e5_copy"] = f(r["e5-small"]["g3_copy"])
        t["cz.same_retrieval.e5_32"] = f(r["e5-small"]["g3_sqad32"])
        t["cz.same_retrieval.e5_diff"] = with_ci(r["e5-small"]["difference"])
        t["cz.same_retrieval.bm25_diff"] = with_ci(r["bm25"]["difference"])
        t["cz.same_retrieval.chars_copy"] = f(r["gold_chars_median"]["copy"], 0)
        t["cz.same_retrieval.chars_32"] = f(r["gold_chars_median"]["sqad32"], 0)
        t["cz.same_retrieval.start_copy"] = f(r["answer_start_median"]["copy"], 0)
        t["cz.same_retrieval.start_32"] = f(r["answer_start_median"]["sqad32"], 0)
    s = gap.get("same_questions_answers")
    if s:
        t["cz.same.n"] = str(s["questions"])
        t["cz.same.em_copy"] = f(s["em_on_copy_passages"])
        t["cz.same.em_32"] = f(s["em_on_sqad32_passages"])
        t["cz.same.diff"] = with_ci(s["sqad32_passages_minus_copy_passages"])
        t["cz.same.diff_copy_gold"] = with_ci(s["em_with_the_copys_gold_on_both"])
        t["cz.same.diff_32_gold"] = with_ci(s["em_with_sqad32s_gold_on_both"])
        t["cz.same.diff_f1"] = with_ci(s["f1"])
        t["cz.same.diff_mapped"] = with_ci(s["em_yes_no_mapped"])
        t["cz.same.same_string"] = str(s["same_answer_string"])
        t["cz.same.right_copy"] = str(s["right_only_on_copy_passages"])
        t["cz.same.right_32"] = str(s["right_only_on_sqad32_passages"])
    return t


def czech_only(tokens_: dict) -> dict:
    return {k: v for k, v in tokens_.items() if k.startswith(("cz.", "czech.", "headline.czech."))}


def check_tree(repo: Path, analysis_text: str, block: dict | None, cfg: dict) -> list[str]:
    """Compare what was just recomputed with the tree. Returns the differences."""
    import crossover
    from stats_tokens import headline

    problems: list[str] = []
    path = repo / "results/czech/analysis.json"
    if not path.exists():
        return [f"{path.relative_to(repo)} does not exist"]
    if path.read_text(encoding="utf-8") != analysis_text:
        problems.append("results/czech/analysis.json differs from what the records give")
    analysis = json.loads(analysis_text)
    for key, check in analysis.get("bench_check", {}).items():
        if not check["all_match"] or not check["corpus_matches"]:
            problems.append(f"{check['file']}: EM, F1, nDCG@10 or g @3 differ from the records ({key})")

    released = json.loads((repo / "results/crossover.json").read_text(encoding="utf-8"))
    shipped = released["datasets"]["czech"]
    if json.dumps(shipped, sort_keys=True) != json.dumps(json.loads(json.dumps(block)), sort_keys=True):
        problems.append("the Czech block of results/crossover.json differs from what the records give")

    want: dict[str, str] = dict(tokens(analysis))
    built = crossover.build_tokens({"config": cfg, "datasets": {"czech": block}}, cfg)
    want.update({k: v for k, v in built.items() if k.startswith("czech.")})
    want.update(headline({"config": cfg, "datasets": {"czech": block}}, built))
    stats = json.loads((repo / "results/stats.json").read_text(encoding="utf-8"))["tokens"]
    for k, v in sorted(czech_only(want).items()):
        if stats.get(k) != v:
            problems.append(f"results/stats.json: token {k} is {stats.get(k)!r}; the records give {v!r}")
    for k in sorted(czech_only(stats)):
        if k not in want:
            problems.append(f"results/stats.json: token {k} does not come from the records")
    return problems


# ---- v0.4.0: the retrievers on SQAD's two test splits ----------------------------------

RETRIEVER_LABEL = {"bm25": "BM25", "bm25-nodiacritics": "BM25 without diacritics", "bm25-lemma": "BM25 over lemmas", "e5-small": "e5-small",
                   "e5-base": "e5-base", "granite-107m": "granite-107m", "granite-278m": "granite-278m", "embeddinggemma": "EmbeddingGemma",
                   "simcse-cs-en": "simcse-cs-en", "retromae-cs": "retromae-cs"}
RETRIEVER_POPS = (("sqad32_test", "SQAD 3.2, whole test split"), ("copy_test", "SQAD 3.0 copy, whole test split"))


# A difference read after the rankings, outside the analysis fixed before them (scripts/czech_retrievers.py, PAIRS)
ADDED_AFTER_READINGS = {("bm25-lemma", "bm25")}


def retriever_tokens(r: dict) -> dict[str, str]:
    """czr.*, a flag after each model's and retriever's name."""
    return with_flags(_retriever_tokens(r))


def _retriever_tokens(r: dict) -> dict[str, str]:
    """czr.*: the ten retrievers' readings, from results/czech/v040/retrievers.json (scripts/czech_retrievers.py --check)."""
    t: dict[str, str] = {}
    pops = r.get("populations", {})
    for pop, lab in RETRIEVER_POPS:
        row = pops.get(pop)
        if not row:
            continue
        t[f"czr.{pop}.n"] = f"{row['n']:,}"
        rows = []
        for rid in sorted(row["means"], key=lambda x: -row["means"][x]["ndcg10"]["point"]):
            m = row["means"][rid]
            rows.append([RETRIEVER_LABEL.get(rid, rid), with_ci(m["ndcg10"]), with_ci(m["recall10"]), with_ci(m["g3"])])
            for k in ("ndcg10", "recall10", "g3"):
                t[f"czr.{pop}.{rid}.{k}"] = f(m[k]["point"])
        t[f"czr.table_{pop}"] = table(["retriever", "nDCG@10 [95%]", "Recall@10 [95%]", "gold in the top 3 [95%]"], rows)
        drows = []
        for name, d in row["differences"].items():
            hi, lo = name.split(" - ")
            key = f"{hi}_minus_{lo}".replace("-", "_")
            for k in ("ndcg10", "g3"):
                t[f"czr.{pop}.{key}.{k}"] = with_ci(d[k])
            added = " (added after the rankings were read)" if (hi, lo) in ADDED_AFTER_READINGS else ""
            drows.append([f"{RETRIEVER_LABEL.get(hi, hi)} minus {RETRIEVER_LABEL.get(lo, lo)}{added}", with_ci(d["ndcg10"]), with_ci(d["g3"])])
        t[f"czr.table_{pop}_differences"] = table(["difference", "nDCG@10 [95%]", "gold in the top 3 [95%]"], drows)
    rule = r.get("phase_b_rule") or {}
    if rule.get("complete"):
        g = rule["largest_gain"]
        t["czr.rule.retriever"] = RETRIEVER_LABEL.get(g["retriever"], g["retriever"])
        t["czr.rule.gain"] = with_ci(g)
        t["czr.rule.anchor"] = with_ci(rule["anchor"])
        t["czr.rule.verdict"] = "runs" if rule["runs"] else "does not run"
    t.update(retriever_cost_tokens(r))
    return t


def retriever_cost_tokens(r: dict) -> dict[str, str]:
    """czr.cost.*: the timed rows on SQAD 3.2's corpus, once every retriever has an accepted row."""
    c = r.get("cost") or {}
    if not c.get("complete"):
        return {}
    t: dict[str, str] = {}
    rows_by = c["retrievers"]
    quality = (r.get("populations", {}).get("sqad32_test") or {}).get("means", {})
    order = sorted(rows_by, key=lambda x: -quality[x]["ndcg10"]["point"]) if all(x in quality for x in rows_by) else list(rows_by)
    rows = []
    for rid in order:
        m, key = rows_by[rid], rid.replace("-", "_")
        t[f"czr.cost.{key}.gb"] = f(m["peak_rss_gb"], 2)
        t[f"czr.cost.{key}.p50_ms"] = f(m["retrieval_ms_p50"], 1)
        t[f"czr.cost.{key}.p95_ms"] = f(m["retrieval_ms_p95"], 1)
        t[f"czr.cost.{key}.model_mb"] = f"{m['model_bytes'] / 1e6:,.0f}"
        t[f"czr.cost.{key}.index_mb"] = f"{m['index_bytes'] / 1e6:,.1f}"
        rows.append([RETRIEVER_LABEL.get(rid, rid), f(quality[rid]["ndcg10"]["point"]) if rid in quality else "n/a",
                     f(m["peak_rss_gb"], 2), f"{m['model_bytes'] / 1e6:,.0f}", f"{m['index_bytes'] / 1e6:,.1f}",
                     f(m["retrieval_ms_p50"], 1), f(m["retrieval_ms_p95"], 1)])
    t["czr.cost.table"] = table(["retriever", "nDCG@10, SQAD 3.2 test", "peak memory (GB)", "model (MB)", "index (MB)",
                                 "latency p50 (ms)", "latency p95 (ms)"], rows)
    for name, d in c["differences"].items():
        hi, lo = name.split(" - ")
        key = f"{hi}_minus_{lo}".replace("-", "_")
        t[f"czr.cost.{key}.gb"] = f(d["peak_rss_gb"], 2, True)
        t[f"czr.cost.{key}.p95_ms"] = f(d["retrieval_ms_p95"], 1, True)
    one = next(iter(rows_by.values()))
    t["czr.cost.n_queries"] = f"{one['n_queries']:,}"
    t["czr.cost.chip"] = one["chip"]
    t["czr.cost.macos"] = and_list(sorted({m["macos"] for m in rows_by.values()}))
    t["czr.cost.baseline_macos"] = c["probe_baseline_macos"]
    drifts = [m["pre_probe_drift"] for m in rows_by.values()]
    t["czr.cost.drift_range"] = f"{100 * min(drifts):+.1f}% to {100 * max(drifts):+.1f}%"
    t["czr.cost.unsampled"] = and_list([RETRIEVER_LABEL.get(x, x) for x in order if rows_by[x]["samples"] == 0]) or "none"
    return t
