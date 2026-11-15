"""The complete Czech run: the plan's sets, the prompt rule, a resumed stage's order, the queue's guards.
No model, no SQAD text: made-up questions, and the ids and hashes that ship."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import czech_complete_queue as cq  # noqa: E402
import dial_generate as dg  # noqa: E402
from measurerag.czech_sensitivity import choose_after_amendment, choose_variant, compliance, yes_no_class  # noqa: E402
from measurerag.czech_store import read_jsonl_gz  # noqa: E402
from measurerag.datasets.base import Dataset, Doc  # noqa: E402
from measurerag.rag.generator import PROMPT_PATH, build_prompt, load_prompt_template  # noqa: E402

PLAN = REPO / "results/czech/plan"


def plan_files():
    return json.loads((PLAN / "plan.json").read_text(encoding="utf-8")), json.loads((PLAN / "questions.json").read_text(encoding="utf-8")), read_jsonl_gz(PLAN / "question_join.jsonl.gz")


# ---- the plan ------------------------------------------------------------------------------------
def test_the_dev_set_is_in_no_pool_by_text_in_either_version():
    _, q, join = plan_files()
    text = {}
    for row in join:
        for side in ("copy", "sqad32"):
            for x in row[side]:
                text[(side, x["id"])] = row["text"]
    dev = {text[("sqad32", d)] for d in q["dev"]}
    assert len(dev) == len(q["dev"]) >= 150
    for key in ("sqad32_debiased", "sqad32_full", "extra010_sqad32"):
        assert not dev & {text[("sqad32", x)] for x in q[key]}, key
    for key in ("released", "copy_debiased", "extra010_copy"):
        assert not dev & {text[("copy", x)] for x in q[key]}, key
    assert not dev & {text[("copy", a)] for a, _ in q["cross_version"]}
    assert not dev & {text[("sqad32", b)] for _, b in q["cross_version"]}


def test_the_dev_set_has_its_yes_no_questions():
    p, _, _ = plan_files()
    assert p["dev"]["yes_no"] >= 60 and p["dev"]["questions"] >= 150


def test_the_cross_version_pairs_are_the_same_question():
    _, q, join = plan_files()
    a_text = {x["id"]: r["text"] for r in join for x in r["copy"]}
    b_text = {x["id"]: r["text"] for r in join for x in r["sqad32"]}
    assert all(a_text[a] == b_text[b] for a, b in q["cross_version"])


def test_sizes_are_power_sizes_capped_and_the_released_200_is_not_sized():
    p, q, _ = plan_files()
    assert p["sets"]["released"] == 200 == len(q["released"]) and "wanted" not in p["sizes"]["released"]
    for key in ("sqad32_debiased", "sqad32_full", "copy_debiased"):
        s = p["sizes"][key]
        assert s["n"] == min(s["wanted"], s["eligible"]) == len(q[key])
        assert s["wanted"] == max(s["retriever_step"]["n"], s["generator_step"]["n"])
        if not s["capped"]:
            assert s["simulated_power_at_this_size"]["generator_step"] >= 0.80
    assert p["sizes"]["copy_debiased"]["capped"]


def test_every_set_keeps_the_first_cut_questions():
    p, _, _ = plan_files()
    assert p["kept_from_the_first_cut"] == {"copy_debiased": 200, "released": 200, "sqad32_debiased": 200, "sqad32_full": 200}


# ---- the prompt -----------------------------------------------------------------------------------
def test_variants_leave_english_alone():
    assert PROMPT_PATH.read_text(encoding="utf-8") == load_prompt_template("en")
    with pytest.raises(ValueError):
        load_prompt_template("en", "v2")


def test_v4_is_v2_without_the_list_and_with_two_worked_examples():
    v2, v4 = load_prompt_template("cs", "v2"), load_prompt_template("cs", "v4")
    words = ": jménem, číslem, datem nebo slovním spojením"
    assert words in v2 and not any(w in v4 for w in ("jménem", "číslem", "datem", "spojením"))
    instruction, rest = v4.split("\n\n", 1)
    assert instruction == v2.split("\n\n", 1)[0].replace(words, "")
    assert v4.endswith(v2.split("\n\n", 1)[1]) and v4.count("Příklad:") == 2
    answers = [line for line in rest.splitlines() if line.startswith("Odpověď: ")]
    assert answers == ["Odpověď: ne", "Odpověď: Josef Novák"]    # one yes/no answer, then one span answer


def test_v4s_example_questions_are_not_questions_of_sqad():
    """The join holds the hash of every question, in every split of both versions, and never a question."""
    import hashlib

    from measurerag.czech_store import read_jsonl_gz

    known = {r["text"] for r in read_jsonl_gz(REPO / "results/czech/plan/question_join.jsonl.gz")}
    asked = [line.removeprefix("Otázka: ") for line in load_prompt_template("cs", "v4").splitlines() if line.startswith("Otázka: ") and "{question}" not in line]
    assert len(asked) == 2
    assert not {hashlib.sha256(q.strip().encode("utf-8")).hexdigest()[:20] for q in asked} & known


@pytest.mark.parametrize("c4,s4,chosen,metric", [
    ({"a": 0.95, "b": 1.0}, {"a": 0.34, "b": 0.45}, "v4", "em"),                 # exactly on both limits: passes
    ({"a": 0.94, "b": 1.0}, {"a": 0.39, "b": 0.50}, "v1", "em_yes_no_mapped"),   # compliance fails on one generator
    ({"a": 1.0, "b": 1.0}, {"a": 0.33, "b": 0.50}, "v1", "em_yes_no_mapped"),    # span EM fails on one generator
    ({"a": 1.0, "b": 1.0}, {"a": 0.39, "b": 0.44}, "v1", "em_yes_no_mapped"),
    ({"a": 1.0, "b": None}, {"a": 0.39, "b": 0.50}, "v1", "em_yes_no_mapped"),   # every answer declined: not a pass
])
def test_the_amended_prompt_rule(c4, s4, chosen, metric):
    got = choose_after_amendment(c4, s4, {"a": 0.39, "b": 0.50})
    assert (got["chosen"], got["primary_metric"], got["passed"]) == (chosen, metric, chosen == "v4")


def test_v2_and_v3_are_in_czech_with_the_rule():
    for v in ("v2", "v3", "v4"):
        t = load_prompt_template("cs", v)
        assert "odpověz pouze slovem ano, nebo ne" in t and "Odpověď:" in t and "Answer" not in t and "unanswerable" not in t
        assert "{context}" in t and "{question}" in t
    assert "Příklad" in load_prompt_template("cs", "v3") and "Příklad" not in load_prompt_template("cs", "v2")
    assert "exactly ano or ne" in load_prompt_template("cs", "v1")
    p = build_prompt("Kdy?", ["první"], language="cs", variant="v3")
    assert p.endswith("Otázka: Kdy?\nOdpověď:\n") and "[1] první" in p and "[1] Vltava" in p


@pytest.mark.parametrize("answer,cls", [("Ano.", "ano or ne"), ("ne", "ano or ne"), ("Yes", "yes or no in English"), ("no.", "yes or no in English"), ("unanswerable", "unanswerable"),
                                        ("Nelze odpovědět.", "unanswerable"), ("Ano, byl", "something else"), ("1923", "something else")])
def test_the_class_of_a_yes_no_answer(answer, cls):
    assert yes_no_class(answer) == cls


def test_compliance_leaves_declined_answers_out_and_counts_other_against():
    assert compliance({"ano or ne": 19, "yes or no in English": 0, "unanswerable": 30, "something else": 1}) == 0.95
    assert compliance({"unanswerable": 3}) is None


@pytest.mark.parametrize("table,chosen,passed", [
    ({"v1": {"a": 0.2, "b": 0.3}, "v2": {"a": 0.95, "b": 0.99}, "v3": {"a": 1.0, "b": 1.0}}, "v2", True),      # first that passes, not the best
    ({"v1": {"a": 0.2, "b": 0.3}, "v2": {"a": 0.94, "b": 0.99}, "v3": {"a": 0.95, "b": 0.95}}, "v3", True),
    ({"v1": {"a": 0.2, "b": 0.3}, "v2": {"a": 0.90, "b": 0.92}, "v3": {"a": 0.93, "b": 0.91}}, "v3", False),   # the higher of the lower values
    ({"v1": {"a": 0.2, "b": 0.3}, "v2": {"a": 0.90, "b": 0.94}, "v3": {"a": 0.90, "b": 0.92}}, "v2", False),   # a tie: the higher mean
    ({"v1": {"a": 0.2, "b": 0.3}, "v2": {"a": 0.90, "b": 0.90}, "v3": {"a": 0.90, "b": 0.90}}, "v2", False),   # still tied: the order
    ({"v1": {"a": 0.8, "b": 0.8}, "v2": {"a": 0.5, "b": 0.9}, "v3": {"a": 0.6, "b": None}}, "v1", False),      # all three compete when none passes
])
def test_the_prompt_rule(table, chosen, passed):
    got = choose_variant(table)
    assert (got["chosen"], got["passed"]) == (chosen, passed)
    assert passed or got["shortfall"] > 0


# ---- a resumed stage keeps its order ------------------------------------------------------------------
class FakeGenerator:
    calls: list[str] = []

    def __init__(self, *a, **k):
        self.llm = None

    def generate(self, prompt):
        FakeGenerator.calls.append(prompt)
        return f"answer {len(prompt)}", {"seconds": 0.0, "completion_tokens": 1, "prompt_tokens": 1}

    def close(self):
        pass


def small_slice(tmp_path: Path) -> tuple[Path, Path]:
    docs = [Doc(doc_id=f"d{i}", text=f"Text číslo {i}.", title="") for i in range(12)]
    queries = {f"q{i}": f"Otázka {i}?" for i in range(6)}
    ds = Dataset(name="t", language="cs", docs=docs, queries=queries, qrels={q: {f"d{i}": 1} for i, q in enumerate(queries)}, answers={q: ["ano"] for q in queries}, license="x", source="y", notes=[])
    ds.save(tmp_path / "slice")
    rank = tmp_path / "rank"
    rank.mkdir()
    run = {q: [f"d{i}", f"d{i + 1}", f"d{i + 2}"] for i, q in enumerate(queries)}
    (rank / "retrieval_e5-small.json").write_text(json.dumps({"run": run}), encoding="utf-8")
    return tmp_path / "slice", rank


def test_a_resumed_stage_regenerates_its_last_answer_first(tmp_path, monkeypatch):
    import measurerag.openmp
    import measurerag.rag.generator as gen_mod

    monkeypatch.setattr(gen_mod, "LlamaGenerator", FakeGenerator)
    monkeypatch.setattr(measurerag.openmp, "preflight", lambda: None)
    slice_dir, rank = small_slice(tmp_path)
    base = ["dial_generate", "--dataset", "t", "--slice", str(slice_dir), "--cache", str(tmp_path / "c"), "--rank-cache", str(rank), "--retriever", "e5-small", "--k", "3",
            "--levels", "natural", "--generator", "qwen2.5-1.5b", "--stage", "s1"]
    FakeGenerator.calls = []
    monkeypatch.setattr(sys, "argv", [*base, "--limit", "3"])
    assert dg.main() == 0
    first = list(FakeGenerator.calls)
    assert len(first) == 3
    FakeGenerator.calls = []
    monkeypatch.setattr(sys, "argv", base)
    assert dg.main() == 0
    assert FakeGenerator.calls[0] == first[-1] and len(FakeGenerator.calls) == 4     # the last answer again, then the three left
    uninterrupted = list(first) + FakeGenerator.calls[1:]
    FakeGenerator.calls = []
    monkeypatch.setattr(sys, "argv", ["dial_generate", *base[1:6], str(tmp_path / "c2"), *base[7:]])
    assert dg.main() == 0
    assert FakeGenerator.calls == uninterrupted                                      # the same prompts in the same order


def test_a_stage_with_nothing_left_generates_nothing(tmp_path, monkeypatch):
    import measurerag.openmp
    import measurerag.rag.generator as gen_mod

    monkeypatch.setattr(gen_mod, "LlamaGenerator", FakeGenerator)
    monkeypatch.setattr(measurerag.openmp, "preflight", lambda: None)
    slice_dir, rank = small_slice(tmp_path)
    argv = ["dial_generate", "--dataset", "t", "--slice", str(slice_dir), "--cache", str(tmp_path / "c"), "--rank-cache", str(rank), "--retriever", "e5-small", "--k", "3", "--levels", "natural", "--generator", "qwen2.5-1.5b", "--stage", "s1"]
    monkeypatch.setattr(sys, "argv", argv)
    assert dg.main() == 0
    FakeGenerator.calls = []
    assert dg.main() == 0
    assert FakeGenerator.calls == []


# ---- the queue's guards ------------------------------------------------------------------------------
def make_queue(tmp_path: Path, stages: list[cq.Stage], **kw) -> cq.Queue:
    kw.setdefault("poll_s", 0.02)
    kw.setdefault("no_progress_s", 3600)
    kw.setdefault("min_free_gb", 0)
    return cq.Queue(stages, tmp_path / "report", {"dial_1.5b": 1.0}, "test", root=tmp_path, env={}, export=False, **kw)


def script(tmp_path: Path, name: str, body: str) -> list[str]:
    p = tmp_path / f"{name}.sh"
    p.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8")
    p.chmod(0o755)
    return [str(p)]


def test_every_stage_runs_once_and_a_finished_stage_is_not_run_again(tmp_path):
    log = tmp_path / "ran.txt"
    stages = [cq.Stage("a", script(tmp_path, "a", f"echo a >> {log}")), cq.Stage("b", script(tmp_path, "b", f"echo b >> {log}"))]
    assert make_queue(tmp_path, stages).run() == 0
    assert make_queue(tmp_path, stages).run() == 0
    assert log.read_text().split() == ["a", "b"]
    progress = (tmp_path / "report/PROGRESS.md").read_text(encoding="utf-8")
    assert "2 done, 0 left" in progress and "Projected finish" in progress and "Generations in this phase" in progress


def test_the_stop_switch_stops_before_the_next_configuration(tmp_path):
    log = tmp_path / "ran.txt"
    report = tmp_path / "report"
    report.mkdir()
    a = cq.Stage("a", script(tmp_path, "a", f"echo a >> {log}; touch {report / 'STOP'}"))
    b = cq.Stage("b", script(tmp_path, "b", f"echo b >> {log}"))
    assert make_queue(tmp_path, [a, b]).run() == 3
    assert log.read_text().split() == ["a"]
    assert "the stop switch" in (report / "PROGRESS.md").read_text(encoding="utf-8")


def test_no_progress_restarts_once_and_then_stops(tmp_path):
    log = tmp_path / "ran.txt"
    s = cq.Stage("slow", script(tmp_path, "s", f"echo x >> {log}; sleep 100"))
    q = make_queue(tmp_path, [s], no_progress_s=0.3)
    assert q.run() == 1
    assert log.read_text().split() == ["x", "x"]
    assert "made no progress" in q.st["stopped"]["why"] and len(q.st["restarts"]) == 1


def test_a_stage_that_keeps_writing_is_not_stopped(tmp_path):
    s = cq.Stage("busy", script(tmp_path, "b", "for i in 1 2 3 4 5 6; do echo $i; sleep 0.15; done"))
    q = make_queue(tmp_path, [s], no_progress_s=0.5)
    assert q.run() == 0 and not q.st["restarts"]


def test_low_disk_stops_the_stage_and_says_so(tmp_path):
    s = cq.Stage("any", script(tmp_path, "s", "sleep 100"))
    q = make_queue(tmp_path, [s], min_free_gb=1e12)
    assert q.run() == 1 and "free disk" in q.st["stopped"]["why"]


def test_a_failed_check_is_reported_and_stops_nothing_and_a_failed_stage_stops(tmp_path):
    log = tmp_path / "ran.txt"
    check = cq.Stage("check", script(tmp_path, "c", "exit 2"), fatal=False)
    next_ = cq.Stage("next", script(tmp_path, "n", f"echo n >> {log}"))
    q = make_queue(tmp_path, [check, next_])
    assert q.run() == 0 and log.read_text().split() == ["n"] and "check exited 2" in q.st["warnings"][0]
    bad = cq.Stage("bad", script(tmp_path, "bad", "exit 5"))
    after = cq.Stage("after", script(tmp_path, "after", f"echo after >> {log}"))
    q2 = make_queue(tmp_path, [bad, after])
    assert q2.run() == 1 and log.read_text().split() == ["n"]


def test_the_stage_list_is_the_plans_and_each_set_has_its_own_cache():
    plan, _, _ = plan_files()
    stages = cq.main_stages(plan)
    names = [s.name for s in stages]
    assert len(names) == len(set(names))
    assert names[0].startswith("released_bench") and names.index("sqad32_debiased_dial_1.5b") < names.index("sqad32_full_bench_bm25")
    assert all(n in names for n in ("sqad32_full_agree", "extra010_sqad32_retromae_1.5b", "extra010_copy_bm25_1.5b", "cross_copy_3b", "cross_32_1.5b"))
    for s in stages:
        if s.count_stage:
            cache = s.cmd[s.cmd.index("--cache") + 1]
            assert cache.startswith(cq.COMPLETE + "/")
            key = s.name.rsplit("_", 2)[0] if "dial" in s.name or "bm25" in s.name or "retromae" in s.name else s.name.rsplit("_", 1)[0]
            assert cache == f"{cq.COMPLETE}/{key}", s.name


# ---- the planned analyses, on made-up answers ------------------------------------------------------------
def _resamples(n: int, seed: int = 1, size: int = 2000):
    import numpy as np

    return np.random.default_rng(seed).multinomial(n, [1 / n] * n, size=size).astype(float)


def _row(em: int, answer: str = "x", kind: str = "yes/no", mapped: int | None = None, other: int | None = None, top3: int = 1) -> dict:
    return {"em": em, "em_yes_no_mapped": em if mapped is None else mapped, "f1": float(em), "em_other_key": em if other is None else other, "gold_in_top3": top3, "answer": answer, "type": kind}


def test_compliance_counts_czech_answers_among_those_that_did_not_decline():
    import rescore_czech as rc

    rows = [_row(1, "ano"), _row(0, "ne"), _row(0, "yes"), _row(0, "unanswerable"), _row(0, "Praha")]
    got = rc.compliance_block(rows, _resamples(5))
    assert got["compliance"]["point"] == 0.5 and got["strict_share"]["point"] == 0.4
    assert (got["declined"], got["yes_no_questions"]) == (1, 5) and got["compliance"]["lo"] <= 0.5 <= got["compliance"]["hi"]
    assert rc.compliance_block([], _resamples(5)) is None
    assert rc.compliance_block([_row(0, "unanswerable")], _resamples(1))["compliance"] is None   # everything declined: no value, not zero
    diff = rc.compliance_difference([_row(1, "ano"), _row(1, "ne")], [_row(0, "yes"), _row(1, "ne")], _resamples(2))
    assert diff["compliance"]["point"] == 0.5 and diff["strict_share"]["point"] == 0.5


def _steps(gen: float, ret: float, established: bool = True) -> dict:
    def c(point: float) -> dict:
        return {"point": point, "established": established}

    return {"generator_step": c(gen), "retriever_step": c(ret), "generator_step_minus_retriever_step": c(gen - ret)}


def test_a_claim_holds_on_all_four_only_with_one_sign_on_each():
    import rescore_czech as rc

    def sets(a, b, c, d, field="em"):
        return {k: {"steps": {"em": v, "em_yes_no_mapped": v}} for k, v in zip(rc.MAIN_SETS, (a, b, c, d))}

    same = sets(_steps(0.1, 0.0), _steps(0.2, 0.0), _steps(0.1, 0.0), _steps(0.1, 0.0))
    got = rc.holds_on_all_four(same)["qwen2.5"]["em"]["generator_step"]
    assert got["holds_on_all_four"] and got["sign"] == 1.0 and got["established_on"] == list(rc.MAIN_SETS)
    assert not rc.holds_on_all_four(sets(_steps(0.1, 0), _steps(-0.1, 0), _steps(0.1, 0), _steps(0.1, 0)))["qwen2.5"]["em"]["generator_step"]["holds_on_all_four"]
    one_short = sets(_steps(0.1, 0), _steps(0.1, 0), _steps(0.1, 0), _steps(0.1, 0, established=False))
    got = rc.holds_on_all_four(one_short)["qwen2.5"]["em"]["generator_step"]
    assert not got["holds_on_all_four"] and len(got["established_on"]) == 3
    assert rc.holds_on_all_four({rc.MAIN_SETS[0]: same[rc.MAIN_SETS[0]]})["qwen2.5"]["em"] == {}  # fewer than four: no claim
    assert "gemma-4" not in rc.holds_on_all_four(same) and "ministral-3 minus gemma-4" not in rc.holds_on_all_four(same)  # a family with no answers claims nothing


def test_each_family_and_each_family_difference_is_its_own_conjunction():
    import rescore_czech as rc

    def c(point, established=True):
        return {"point": point, "established": established}

    sets = {k: {"steps": {"em": _steps(0.1, 0.0), "em_yes_no_mapped": _steps(0.1, 0.0)},
                "families": {"gemma-4": {"steps": {"em": _steps(0.1, 0.0, established=k != "released"), "em_yes_no_mapped": _steps(0.1, 0.0)}}},
                "family_difference": {"gemma-4": {"smaller": {"em": c(0.2), "em_yes_no_mapped": c(0.2)}, "larger": {"em": c(0.05, k != "copy_debiased"), "em_yes_no_mapped": c(-0.1)}}}}
            for k in rc.MAIN_SETS}
    got = rc.holds_on_all_four(sets)
    assert got["qwen2.5"]["em"]["generator_step"]["holds_on_all_four"]
    g = got["gemma-4"]["em"]["generator_step"]
    assert not g["holds_on_all_four"] and "released" not in g["established_on"] and got["gemma-4"]["em_yes_no_mapped"]["generator_step"]["holds_on_all_four"]
    d = got["gemma-4 minus qwen2.5"]
    assert d["em"]["smaller"]["holds_on_all_four"] and d["em"]["smaller"]["sign"] == 1.0
    assert not d["em"]["larger"]["holds_on_all_four"] and len(d["em"]["larger"]["established_on"]) == 3
    assert d["em_yes_no_mapped"]["larger"]["holds_on_all_four"] and d["em_yes_no_mapped"]["larger"]["sign"] == -1.0


def _cross(em_copy, em_32, same_gold, types=None) -> dict:
    """Both generators answer alike here, so the steps are zero and the passages' difference is plain."""
    import numpy as np

    from rescore_czech import BASE, BIG

    types = types or ["span"] * len(em_copy)
    a = [_row(e, "ano" if e else "ne", kind=t) for e, t in zip(em_copy, types)]
    b = [_row(e, "ano" if e else "ne", kind=t) for e, t in zip(em_32, types)]
    return {"pairs": [(f"a{i}", f"b{i}") for i in range(len(a))], "same_gold": np.array(same_gold),
            "generators": {g: {"copy": a, "sqad32": b, "counts": _resamples(len(a), seed=3)} for g in (BASE, BIG)}}


def test_the_cross_version_check_is_paired_and_reads_the_same_gold_questions_apart():
    import rescore_czech as rc

    out = rc.cross_version(_cross([1, 0, 0, 1], [1, 1, 1, 0], [True, True, True, False], types=["yes/no", "span", "span", "yes/no"]))
    g = out["generators"][rc.BASE]
    assert (out["pairs"], out["same_gold_pairs"]) == (4, 3)
    assert (g["em"]["copy"], g["em"]["sqad32"], g["em"]["difference"]["point"]) == (0.5, 0.75, 0.25)
    assert g["em_same_gold"]["difference"]["point"] == pytest.approx(1 - 1 / 3)
    assert (g["right_only_on_copy_passages"], g["right_only_on_sqad32_passages"], g["same_answer_string"]) == (1, 2, 1)
    assert g["compliance"]["copy"]["yes_no_questions"] == 2
    assert out["generator_step"]["qwen2.5"]["copy"]["em"]["point"] == 0.0
    assert "gemma-4" not in out["generator_step"] and out["family_difference"] == {}               # Gemma 4 did not answer here


def test_the_cross_version_check_reads_each_family_and_the_family_difference_on_the_same_questions():
    import rescore_czech as rc

    cd = _cross([1, 0, 0, 1], [1, 1, 1, 0], [True] * 4)
    a = [_row(e, kind="span") for e in (1, 1, 0, 1)]
    b = [_row(e, kind="span") for e in (1, 1, 1, 1)]
    for g in (rc.E2, rc.E4):
        cd["generators"][g] = {"copy": a, "sqad32": b, "counts": _resamples(4, seed=3)}
    m = [_row(e, kind="span") for e in (1, 0, 0, 0)]
    for g in (rc.M3, rc.M8):
        cd["generators"][g] = {"copy": m, "sqad32": m, "counts": _resamples(4, seed=3)}
    q = [_row(e, kind="span") for e in (1, 1, 1, 0)]
    for g in (rc.Q2, rc.Q4):
        cd["generators"][g] = {"copy": q, "sqad32": q, "counts": _resamples(4, seed=3)}
    n = [_row(e, kind="span") for e in (1, 1, 1, 1)]
    for g in (rc.N2, rc.N4):
        cd["generators"][g] = {"copy": n, "sqad32": n, "counts": _resamples(4, seed=3)}
    out = rc.cross_version(cd)
    assert set(out["generators"]) == set(rc.GENERATORS)
    assert out["generator_step"]["gemma-4"]["copy"]["em"]["point"] == 0.0                          # E4B and E2B answer alike here
    fd = out["family_difference"]["gemma-4"]
    assert fd["smaller"]["copy"]["em"]["point"] == 0.25 and fd["larger"]["sqad32"]["em"]["point"] == 0.25   # 3 of 4 against 2 of 4; 4 of 4 against 3 of 4
    # Ministral 3 against Qwen2.5, and against Gemma 4, on the same questions
    assert out["family_difference"]["ministral-3"]["smaller"]["copy"]["em"]["point"] == -0.25       # 1 of 4 against 2 of 4
    between = out["family_difference_between"]["ministral-3 minus gemma-4"]
    assert between["smaller"]["copy"]["em"]["point"] == -0.5 and between["larger"]["sqad32"]["em"]["point"] == -0.75   # 1 of 4 against 3 of 4; against 4 of 4
    # Qwen3.5 against Qwen2.5 and against the two other families, on the same questions
    assert out["family_difference"]["qwen3.5"]["smaller"]["copy"]["em"]["point"] == 0.25            # 3 of 4 against 2 of 4
    q_g = out["family_difference_between"]["qwen3.5 minus gemma-4"]
    assert q_g["smaller"]["copy"]["em"]["point"] == 0.0 and q_g["larger"]["sqad32"]["em"]["point"] == -0.25   # 3 of 4 against 3 of 4; against 4 of 4
    assert out["family_difference_between"]["qwen3.5 minus ministral-3"]["smaller"]["copy"]["em"]["point"] == 0.5   # 3 of 4 against 1 of 4
    # Gemma 3n against Qwen2.5, and Gemma 4 against its previous generation, on the same questions
    assert out["family_difference"]["gemma-3n"]["smaller"]["copy"]["em"]["point"] == 0.5             # 4 of 4 against 2 of 4
    g4_g3n = out["family_difference_between"]["gemma-4 minus gemma-3n"]
    assert g4_g3n["smaller"]["copy"]["em"]["point"] == -0.25 and g4_g3n["larger"]["sqad32"]["em"]["point"] == 0.0   # 3 of 4 against 4 of 4; 4 of 4 against 4 of 4


def test_the_planned_gap_runs_its_steps_in_order_and_leaves_what_the_passages_do_not_explain():
    from types import SimpleNamespace

    import rescore_czech as rc

    def sd(ems, flagged, seed):
        qids = [f"q{i}" for i in range(len(ems))]
        rows = [_row(e) for e in ems]
        labels = {q: {"type": "number", "answer_tokens": 1, "flagged_0138": f} for q, f in zip(qids, flagged)}
        return SimpleNamespace(config_rows=lambda c: rows, qids=qids, labels=labels, counts=_resamples(len(ems), seed=seed))

    loaded = {"sqad32_full": sd([1, 1, 0, 0], [True, False, False, False], 4), "released": sd([1, 0, 0, 0], [False, False, False, True], 5),
              "sqad32_debiased": sd([1, 1, 1, 0], [False] * 4, 6), "copy_debiased": sd([1, 0, 0, 0], [False] * 4, 7)}
    got = rc.planned_gap(loaded, _cross([0, 0, 1, 1], [1, 0, 1, 1], [True, True, True, True]))
    e5 = got["em_yes_no_mapped"][f"e5-small+{rc.BASE}"]
    assert e5["G1"]["gap"]["gap"]["point"] == 0.25 and e5["G2"]["gap"]["gap"]["point"] == 0.5
    rule = e5["G1"]["on_questions_the_rule_passes"]
    assert (rule["n_copy"], rule["n_target"]) == (3, 3)
    assert abs(rule["gap"]["point"] - (1 / 3 - 1 / 3)) < 1e-9 and e5["G1"]["share_of_the_gap_the_rule_accounts_for"] == 1.0
    assert e5["passages"]["own_gold"]["point"] == 0.25 and e5["what_is_left"]["point"] == 0.25    # 0.5 of G2, less 0.25 from the passages
    bm25 = got["em"][f"bm25+{rc.BASE}"]
    assert "passages" not in bm25 and "what_is_left" not in bm25                                  # no BM25 runs on the cross-version questions
    alone = rc.planned_gap(loaded, None)["em"][f"e5-small+{rc.BIG}"]
    assert "passages" not in alone and "G2" in alone


def test_the_order_check_compares_the_same_answers_only(monkeypatch):
    import rescore_czech as rc

    def row(q, answer, em=0):
        return {"q": q, "config": f"e5-small+{rc.BASE}", "kind": "dial", "p": 0.0, "answer": answer, "em": em, "em_yes_no_mapped": em}

    files = {"copy_debiased": [row("1", "a"), row("2", "b", 1), row("not in the row", "c")], "extra010_copy": [row("1", "a"), row("2", "x", 1)]}
    monkeypatch.setattr(rc, "load_answers", lambda root, key: files[key])
    got = rc.order_check("root", {"copy_debiased": {}, "extra010_copy": {}})[rc.BASE]
    assert got == {"answers_compared": 2, "same_string": 1, "same_em": 2, "same_em_yes_no_mapped": 2}
    assert rc.order_check("root", {"copy_debiased": {}}) is None


def test_compute_counts_the_dev_set_the_main_run_and_what_was_set_aside(tmp_path, monkeypatch):
    import rescore_czech as rc

    (tmp_path / "results/czech/complete").mkdir(parents=True)
    (tmp_path / "results/czech/complete/compute.json").write_text(json.dumps({
        "stages": {"dev_v1_1.5b": {"generations": 200, "seconds": 10.0}, "released_dial_3b": {"generations": 5, "seconds": 2.5}, "released_agree": {"seconds": 1.0}},
        "superseded_v3": {"released_bench_bm25": {"generations": 200, "seconds": 1770.7}}, "restarts": [{"stage": "x"}], "warnings": ["w"]}), encoding="utf-8")
    monkeypatch.setattr(rc, "REPO", tmp_path)
    got = rc.complete_compute()
    assert got["dev"] == {"stages": 1, "generations": 200, "seconds": 10.0} and got["main"] == {"stages": 2, "generations": 5, "seconds": 3.5}
    assert got["superseded_v3"]["generations"] == 200 and got["restarts"] == 1 and got["warnings"] == ["w"] and "gx10" not in got
    (tmp_path / "results/czech/complete/compute_gx10.json").write_text(json.dumps({
        "stages": {"dev_v1_1.5b": {"generations": 200, "seconds": 30.0}, "pilot_rank_released_bm25": {"seconds": 5.0}, "released_dial_gemma-4-e2b": {"generations": 367, "seconds": 75.0}},
        "restarts": [], "warnings": []}), encoding="utf-8")
    got = rc.complete_compute()
    assert got["gx10"]["pilot"] == {"stages": 2, "generations": 200, "seconds": 35.0} and got["gx10"]["main"] == {"stages": 1, "generations": 367, "seconds": 75.0}
    assert got["main"] == {"stages": 3, "generations": 372, "seconds": 78.5} and got["dev"]["generations"] == 200      # the Mac's dev set stays the Mac's


def test_the_family_difference_is_paired_on_the_same_questions_with_e5_small():
    from types import SimpleNamespace

    import numpy as np

    import rescore_czech as rc

    answers = {f"e5-small+{rc.BASE}": [_row(0, "yes"), _row(1, "ano"), _row(0, "Praha", kind="span"), _row(1, "Brno", kind="span")],
               f"e5-small+{rc.E2}": [_row(1, "ano"), _row(1, "ano"), _row(1, "Praha", kind="span"), _row(1, "Brno", kind="span")],
               f"e5-small+{rc.BIG}": [_row(1, "ano"), _row(1, "ne"), _row(1, "Praha", kind="span"), _row(0, "x", kind="span")]}
    sd = SimpleNamespace(config_rows=answers.get, is_yn=np.array([True, True, False, False]), counts=_resamples(4), yes_no_counts=lambda: _resamples(2, seed=2))
    got = rc.family_difference(sd, (rc.BASE, rc.BIG), (rc.E2, rc.E4))
    small = got["smaller"]
    assert (small["minuend"], small["subtrahend"]) == (rc.E2, rc.BASE) and small["em"]["point"] == 0.5
    assert small["compliance"]["compliance"]["point"] == 0.5                                           # 2 of 2 in Czech against 1 of 2
    assert small["detectable"]["em"]["discordant"] == 0.5 and small["detectable"]["em"]["detects"] > 0
    assert "larger" not in got                                                                          # E4B did not answer: no reading
    assert rc.family_difference(SimpleNamespace(config_rows=lambda c: None), (rc.BASE, rc.BIG), (rc.E2, rc.E4)) is None


def test_detectable_is_the_plans_formula_on_a_familys_own_discordance():
    import numpy as np

    import rescore_czech as rc
    from czech_plan import Z_ALPHA, Z_POWER

    a, b = np.array([1, 0, 1, 1] * 50, dtype=float), np.array([1, 1, 0, 1] * 50, dtype=float)
    got = rc.detectable(a, b)
    assert got["questions"] == 200 and got["discordant"] == 0.5 and got["detects"] == pytest.approx((Z_ALPHA + Z_POWER) * (0.5 / 200) ** 0.5)
    assert rc.detectable(a, a)["detects"] == 0.0


def test_the_family_tokens_keep_qwens_names_and_prefix_the_other_family():
    import czech_tokens as ct

    s = {"answers": {"x": 1}, "families": {"gemma-4": {"answers": {"y": 1}}}}
    # v0.4.2 reads the families by maker's country, the US's first, then France's, then China's
    assert ct.view(s, "qwen2.5") is s and ct.view(s, "gemma-4") == {"answers": {"y": 1}} and ct.families_in(s) == ["gemma-4", "qwen2.5"]
    assert ct.families_in({"answers": {"x": 1}}) == ["qwen2.5"]
    assert ct.PREFIX == {"qwen2.5": "", "gemma-4": "g4.", "ministral-3": "m3.", "qwen3.5": "q35.", "gemma-3n": "g3n."}
    assert list(ct.FAMILY_MODELS) == ["gemma-3n", "gemma-4", "ministral-3", "qwen2.5", "qwen3.5"]
    assert ct.diff_label("gemma-4", "smaller", "gemma-3n") == "Gemma 4 E2B minus Gemma 3n E2B" and ct.pair_tok("gemma-4 minus gemma-3n") == "g4_minus_g3n"
    assert ct.diff_label("gemma-4", "smaller") == "Gemma 4 E2B minus Qwen2.5 1.5B" and ct.diff_label("gemma-4", "larger") == "Gemma 4 E4B minus Qwen2.5 3B"
    # Ministral 3's names never read as Qwen2.5's 3B, and its tokens never overwrite Gemma 4's
    assert ct.diff_label("ministral-3", "larger") == "Ministral 3 8B minus Qwen2.5 3B" and ct.diff_label("ministral-3", "smaller", "gemma-4") == "Ministral 3 3B minus Gemma 4 E2B"
    assert ct.fam_tok("gemma-4") == "family" and ct.fam_tok("ministral-3") == "m3.family" and ct.pair_tok("ministral-3 minus gemma-4") == "m3_minus_g4"
    assert ct.in_order({"qwen2.5": 1, "gemma-4": 2, "zz": 3}, ct.FAMILY_LABEL) == ["gemma-4", "qwen2.5", "zz"]
    assert all(c in ct.CONFIG_LABEL for cs in ct.FAMILY_CONFIGS.values() for c in cs) and len(ct.CONFIGS) == 4


US, FR, CN, CZ = "\U0001F1FA\U0001F1F8", "\U0001F1EB\U0001F1F7", "\U0001F1E8\U0001F1F3", "\U0001F1E8\U0001F1FF"


@pytest.mark.parametrize("text, flagged", [
    ("Gemma 4 E4B minus Qwen2.5 3B", f"Gemma 4 E4B {US} minus Qwen2.5 3B {CN}"),        # the longer name, never Gemma 4 inside it
    ("Gemma 4 and Qwen2.5", f"Gemma 4 {US} and Qwen2.5 {CN}"),
    ("e5-small + Ministral 8B", f"e5-small + Ministral 8B {FR}"),
    ("BM25 + Gemma 3n E2B", f"BM25 + Gemma 3n E2B {US}"),
    ("| e5-small + 3B | 0.5 |", f"| e5-small + 3B {CN} | 0.5 |"),                     # v0.3.0's labels name a model by size alone
    ("BM25 + E2B", f"BM25 + E2B {US}"),
    ("Ministral 3 14B, Unsloth's GGUF", f"Ministral 3 14B {FR}, Unsloth's GGUF"),      # not "Ministral 3" and "14B"
    ("Granite 4.0 Micro", f"Granite 4.0 Micro {US}"),
    ("Gemma 4 E2B.", f"Gemma 4 E2B {US}."),
    ("Gemma 4 12B", "Gemma 4 12B"),                                                     # a size it does not know: no flag at all
    ("the 3B model", "the 3B model"),
    ("`Qwen/Qwen2.5-3B-Instruct`", "`Qwen/Qwen2.5-3B-Instruct`"),                      # code and repository ids stay as they are
    ("`Gemma 4 E2B`", "`Gemma 4 E2B`"),
    ("gemma-4-e2b", "gemma-4-e2b"),
])
def test_a_flag_follows_each_model_name_once(text, flagged):
    import czech_tokens as ct

    assert ct.flagged(text) == flagged and ct.flagged(flagged) == flagged
    assert ct.unflagged(flagged) == text


def test_a_retriever_is_flagged_only_in_the_retriever_tables():
    import czech_tokens as ct

    t = ct.with_flags({"czr.table_x": "| e5-base | retromae-cs |", "cz.y": "e5-small + Gemma 4 E2B"})
    assert t == {"czr.table_x": f"| e5-base {US} | retromae-cs {CZ} |", "cz.y": f"e5-small + Gemma 4 E2B {US}"}


def test_the_complete_runs_tables_wait_for_the_complete_run_and_read_in_the_asked_order():
    import czech_tokens as ct

    assert not ct.is_complete({"sets": {}}) and ct.is_complete({"sets": {}, "holds_on_all_four": {}})
    assert [k for k, _ in ct.COMPLETE_SETS] == ["sqad32_debiased", "sqad32_full", "released", "copy_debiased"]
    assert not any("200 debiased" in lab or "unfiltered" in lab for _, lab in ct.COMPLETE_SETS)
    c = {"point": -0.01, "lo": -0.045, "hi": 0.06, "established": False}
    assert ct.rules_out(c) == "below -0.045 or above +0.060" and ct.level({**c, "point": 0.5}) == "0.500 [-0.045, 0.060]" and ct.level(None) == "n/a"
    # the mapped claims lead, and the mapped generator row is not listed twice
    assert ct.MAPPED_CLAIMS[0][0] == "generator_step_with_yes_and_no_mapped" and ct.CLAIMS[1][0] == "generator_step_with_yes_and_no_mapped"


# ---- the move to the GX10 -----------------------------------------------------
def test_a_gx10_run_keeps_its_own_record_and_asks_for_the_gpu(tmp_path, monkeypatch):
    stages = cq.main_stages(json.loads((REPO / "results/czech/plan/plan.json").read_text(encoding="utf-8")))
    assert not any("--device" in s.cmd for s in stages)                      # the Mac's stages are unchanged
    monkeypatch.setattr(cq, "GEN_DEVICE", "cuda")
    pilot = cq.pilot_stages()
    dial = [s for s in pilot if "scripts/dial_generate.py" in s.cmd]
    assert len(dial) == 10 and all(s.cmd[-2:] == ["--device", "cuda"] for s in dial)
    assert {s.cmd[s.cmd.index("--cache") + 1] for s in dial if s.name.endswith("_repeat")} == {f"{cq.COMPLETE}/dev_v1_repeat"}
    assert pilot[-1].name == "pilot_check" and not pilot[-1].fatal
    q = cq.Queue([], tmp_path / "report", {}, "pilot", root=tmp_path, env={}, export=False, **{k: v for k, v in cq.MACHINES["gx10"].items() if k != "device"})
    assert q.state_path == tmp_path / cq.MACHINES["gx10"]["state"] and q.state_path != tmp_path / cq.STATE


def test_the_pilot_reads_its_conditions_from_its_own_numbers():
    import czech_pilot_check as pc

    def row(q, v, g, answer, em):
        return {"q": q, "variant": v, "config": f"e5-small+{g}", "answer": answer, "em": em, "em_yes_no_mapped": em}

    first = [row("1", "v1", g, "ano", 1) for g in pc.dev.GENERATORS] + [row("2", "v1", g, "Praha", 0) for g in pc.dev.GENERATORS] + [row("1", "v2", pc.dev.GENERATORS[0], "x", 0)]
    again = [row("1", "v1", g, "ano", 1) for g in pc.dev.GENERATORS] + [row("2", "v1", pc.dev.GENERATORS[0], "Praha", 0), row("2", "v1", pc.dev.GENERATORS[1], "Brno", 0)]
    got = pc.repeat_check(first, again)
    assert got[pc.dev.GENERATORS[0]] == {"answers": 2, "same_string": 2} and got[pc.dev.GENERATORS[1]] == {"answers": 2, "same_string": 1}
    state = {"stages": {"dev_v1_1.5b": {"kind": "dial_1.5b", "seconds": 100.0, "generations": 200}, "dev_v1_1.5b_repeat": {"kind": "dial_1.5b", "seconds": 60.0, "generations": 200},
                        "released_bench_bm25": {"kind": "bench_1.5b", "seconds": 9.0, "generations": 1}}}
    assert pc.rate(state) == {"dial_1.5b": 0.4}
    agree = pc.agreement(first, [row("1", "v1", g, "yes", 0) for g in pc.dev.GENERATORS] + [row("2", "v1", g, "Praha", 0) for g in pc.dev.GENERATORS])
    one = agree[f"v1 {pc.dev.GENERATORS[0]}"]
    assert one["same_string"] == 1 and one["em"]["gx10_minus_mac"]["point"] == 0.5 and "v2 " + pc.dev.GENERATORS[0] not in agree


def test_gemma_4_answers_every_czech_set_beside_qwen():
    """Every Czech run, both sizes; the smaller also runs the natural top 3 and the bench."""
    stages = {s.name: s for s in cq.main_stages(json.loads((REPO / "results/czech/plan/plan.json").read_text(encoding="utf-8")))}
    for key in ("sqad32_debiased", "copy_debiased", "extra010_copy", "extra010_sqad32"):
        for name in (f"{key}_dial_gemma-4-e2b", f"{key}_dial_gemma-4-e4b", f"{key}_bm25_gemma-4-e2b", f"{key}_retromae_gemma-4-e2b"):
            assert name in stages and stages[name].cmd[stages[name].cmd.index("--cache") + 1] == f"{cq.COMPLETE}/{key}"
    for key in ("released", "sqad32_full"):
        for rid in ("bm25", "e5-small", "retromae-cs"):
            assert stages[f"{key}_bench_{rid}_gemma-4-e2b"].cmd[-1] == f"{rid}+gemma-4-e2b"
        assert stages[f"{key}_agree_gemma-4-e2b"].cmd[stages[f"{key}_agree_gemma-4-e2b"].cmd.index("--generator") + 1] == "gemma-4-e2b"
    assert {f"cross_{side}_{g}" for side in ("copy", "32") for g in ("1.5b", "3b", "gemma-4-e2b", "gemma-4-e4b")} <= set(stages)
    assert [s.prepare for s in stages.values() if s.prepare] == ["results/cz_rag", "results/cz_rag_32"]    # the first cut moves aside once per bench


def test_ministral_3_runs_alone_every_stage_gemma_4_ran():
    """Ministral 3 in Gemma 4's place, nothing else generates, the benches keep their outputs."""
    plan = json.loads((REPO / "results/czech/plan/plan.json").read_text(encoding="utf-8"))
    main = [s.name for s in cq.main_stages(plan)]
    gemma = [n for n in main if "gemma-4" in n]
    ministral = cq.ministral_stages(plan)
    names = [s.name for s in ministral]
    expected = [n.replace("gemma-4-e2b", "ministral-3-3b").replace("gemma-4-e4b", "ministral-3-8b") for n in gemma]
    finishes = [f"{k}_bench_finish_ministral-3-3b" for k in ("released", "sqad32_full")]
    assert [n for n in names if n not in finishes] == expected
    for s in ministral:
        if "--generator" in s.cmd:
            assert s.cmd[s.cmd.index("--generator") + 1] in ("ministral-3-3b", "ministral-3-8b"), s.name
        if "--only" in s.cmd:
            assert s.cmd[-1].endswith("+ministral-3-3b"), s.name
    assert not any(s.prepare for s in ministral)                     # v0.3.0's bench outputs stay where they are
    assert not any("retrieval_cache.py" in " ".join(s.cmd) for s in ministral)   # the cross-version rankings exist


# ---- how a later release's tokens are read against an earlier release's -------------------------------------------------
# v0.4.2 puts a flag after each model's name and reads the families by maker's country (US, France, China), so a table's rows
# and its model columns come in another order. A pin compares tokens without their flags, and a table as the set of its rows,
# each row as its cells by column name: the same numbers under the same names, whatever the order.
def plain(v):
    import czech_tokens as ct

    return ct.unflagged(v) if isinstance(v, str) else v


def canon(table: str, drop=()) -> list[str]:
    """A markdown table as its sorted rows, each row its cells sorted by column name; rows and columns naming any of `drop`
    (lower case) are left out."""
    lines = plain(table).split("\n")
    cells = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]  # noqa: E731
    head = cells(lines[0])
    keep = [i for i, h in enumerate(head) if not any(d in h.lower() for d in drop)]
    rows = [cells(x) for x in lines[2:] if not any(d in x.lower() for d in drop)]
    return sorted(" | ".join(sorted(f"{head[i]}={r[i]}" for i in keep)) for r in rows)


GEMMA3N_ROWS = ("gemma 3n", "gemma-3n")          # a row or column of Gemma 3n's, by its name or its configuration's


def digest(d: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tokens_now() -> dict:
    stats = json.loads((REPO / "results/stats.json").read_text(encoding="utf-8"))
    return stats.get("tokens", stats)


# ---- v0.3.1 changes no Qwen2.5 or Gemma 4 record or number ----------------------------------
# SHA-256 of each shipped answers file as v0.3.0 shipped it, decompressed: v0.3.1's file without Ministral 3's rows must equal it.
V030_ANSWERS = {
    "answers/sqad30_copy/answers_copy_debiased.jsonl.gz": "952239f40b23aa127297061a0054157659ddba812e06f3fbd13f46687b2da11b",
    "answers/sqad30_copy/answers_cross_version.jsonl.gz": "5b9235ec533ff38f2edba88cae00417d1a21be8d134839a37357901a1aa13ae7",
    "answers/sqad30_copy/answers_extra010_copy.jsonl.gz": "57d0d9bf095cd3883b3dc7211065a4ae8491bb2a828c7aca1075a6b19413d72a",
    "answers/sqad30_copy/answers_released.jsonl.gz": "05342066d09937077a95671b685ad2d60f3eab7219f5c2c1d21a0ddd4d653b97",
    "answers/sqad30_copy/answers_released_v020.jsonl.gz": "db24af7b3e898de7242240e9b4591b1b72427b4fb8d11a4db38b289a0b3c5d89",
    "answers/sqad32/answers_cross_version.jsonl.gz": "0ddab3a431196d8314404a03cdea1cd4e6bbdad8ba8526f6d4a341b59a29a084",
    "answers/sqad32/answers_dev.jsonl.gz": "c441e5a62f9ddb78ca469ae539beaef3a925fcc5344b19e373e6f25c5bfe4425",
    "answers/sqad32/answers_dev_gx10.jsonl.gz": "db6502950c481d7bc7237ea43307c4b888c6afcf87984ca0469bc0aa7d608cda",
    "answers/sqad32/answers_dev_gx10_repeat.jsonl.gz": "98ec3fd91d8bc31644480271d90a128f8cae75a379a6e88ad6c3fdeaa46fd296",
    "answers/sqad32/answers_extra010_sqad32.jsonl.gz": "5c90e01de5d3f1850c220d5a4a228d44385de6a7f47c5e61c6e1eb7d99e0000e",
    "answers/sqad32/answers_model_bench.jsonl.gz": "d113a05a24dbc2ea14b61317beeb5d85818a0dd6f087cea4bfa7a3491756ced6",
    "answers/sqad32/answers_model_bench_mistral.jsonl.gz": "6bd2b9f839f92d41510a5c5c3f8e7edca6520cfe82cc223a0b14dc4a53f53a4a",
    "answers/sqad32/answers_sqad32_debiased.jsonl.gz": "dd0c3c1a6d7c6c6e2060c656d4467c1bbe48fb80b52c4ee26f0383b262172c6d",
    "answers/sqad32/answers_sqad32_full.jsonl.gz": "8a4de207f3ecaae0474e6a93093ecb5ee64b6e4eaa02eacaadc681862309a303",
}
MINISTRAL = ("ministral-3-3b", "ministral-3-8b")
# The tokens v0.3.1 adds, by name; every other token is one of v0.3.0's 1,329.
NEW_TOKEN = r"(^cz\.(?:[a-z0-9_]+\.)?m3[._]|^cz\.cross\.m[38]b\.|\+ministral-3-[38]b\.|^cz\.table_three_|^cz\.compute\.ministral_|_on$|\.(?:mapped|em)_points$|yes_no_in_english$|metrics_agree$)"
# Ranges and counts over every configuration, which now include Ministral 3's.
SPANS_EVERY_FAMILY = {f"cz.cross.{m}_{x}" for m in ("em", "mapped") for x in ("established", "range")} | {
    f"cz.planned.{m}.{x}" for m in ("em", "mapped") for x in ("configurations", "g1_established", "g1_range", "g1_rule_established", "g2_established", "left_established", "rule_share")}
# The tables that gained Ministral 3's rows.
GAINED_ROWS = ("cz.table_added", "cz.table_answers", "cz.table_compliance_classes", "cz.table_compliance_generators", "cz.table_cross",
               "cz.table_cross_family", "cz.table_cross_steps", "cz.table_dial", "cz.table_family", "cz.table_holds", "cz.table_order",
               "cz.table_planned_gap_em", "cz.table_planned_gap_mapped", "cz.table_power_observed", "cz.table_protocol", "cz.table_secondary",
               "cz.table_sensitivity_both", "cz.table_steps_both", "runs.cz_rag.table")
# v0.4.0 leaves the copy's G2 range out too, and v0.4.2 the quote check's readings (SPANS_V042); recomputed from
# release/v0.3.0's own stats.json, 1,280 tokens
# recomputed on 15 Nov 2026, when the hygiene table's label (hygiene.table_batches) lost a reference tag that pointed to an
# unpublished note; no number changed, and tests/test_english_frozen.py pins that token. The same holds for V031 and V040.
V030_UNCHANGED = (1280, "bbaa71e1ebf49627c72dea60faf3e8e7b78f31ecc538660fa3d5350fb00a6f0e")   # hygiene.heavily_contended without the process names
V030_TABLES = (19, "100251fee7c40ac63ad011c4903e89eafc1947329b1c6d8dd434d3e83688becc")   # release/v0.3.0's 19 tables, each as canon() reads it

# ---- v0.4.0 changes no Qwen2.5, Gemma 4 or Ministral 3 record or number ----------------------
QWEN35 = ("qwen3.5-2b", "qwen3.5-4b")
# The tokens v0.4.0 adds, by name: the retrievers (czr.*), Qwen3.5's, the four-family tables and the added reading.
NEW_TOKEN_V040 = (r"(^czr\.|^cz\.(?:[a-z0-9_]+\.)?q35[._]|^cz\.cross\.q[24]b\.|\+qwen3\.5-[24]b\.|^cz\.table_families_|^cz\.compute\.qwen35_|"
                  r"rule_share_where_g1_established$|rule_share_g1_established_n$|g1_established_sign$)")
# Ranges and counts over every configuration, which now include Qwen3.5's as well; the copy's G2 range among them.
SPANS_V040 = SPANS_EVERY_FAMILY | {f"cz.planned.{m}.g2_range" for m in ("em", "mapped")}
# SHA-256 of each shipped answers file as v0.3.1 shipped it, decompressed: v0.4.0's file without Qwen3.5's test-set rows must equal it.
V031_ANSWERS = {
    "answers/sqad30_copy/answers_copy_debiased.jsonl.gz": "3a32e5d0b7b475d8748391ff1aa74a7b0ebfac773c05286b2120ba535687307f",
    "answers/sqad30_copy/answers_cross_version.jsonl.gz": "922244104aa8ec76aa9bfdd492523e4426872b9bfa89db8665c9204ab050d9b8",
    "answers/sqad30_copy/answers_extra010_copy.jsonl.gz": "8fa323a676a5f89bd3559dde84cd1f3673e1b1f4076d8cd718d798c2b5f6abaf",
    "answers/sqad30_copy/answers_released.jsonl.gz": "3b8c9f315f9e5aabc1c037494506048ce08cee2582f10ce442e53b55db402076",
    "answers/sqad30_copy/answers_released_v020.jsonl.gz": "db24af7b3e898de7242240e9b4591b1b72427b4fb8d11a4db38b289a0b3c5d89",
    "answers/sqad32/answers_cross_version.jsonl.gz": "ae8539fecbe12294af21504e720558837dc0eb25ac107a57c15b8eab4ba89a6d",
    "answers/sqad32/answers_dev.jsonl.gz": "c441e5a62f9ddb78ca469ae539beaef3a925fcc5344b19e373e6f25c5bfe4425",
    "answers/sqad32/answers_dev_gx10.jsonl.gz": "db6502950c481d7bc7237ea43307c4b888c6afcf87984ca0469bc0aa7d608cda",
    "answers/sqad32/answers_dev_gx10_repeat.jsonl.gz": "98ec3fd91d8bc31644480271d90a128f8cae75a379a6e88ad6c3fdeaa46fd296",
    "answers/sqad32/answers_extra010_sqad32.jsonl.gz": "8cd00f5bf444ef91b132420e336c876b1041f58d58c0302c01f94e5bb1b645de",
    "answers/sqad32/answers_model_bench.jsonl.gz": "d113a05a24dbc2ea14b61317beeb5d85818a0dd6f087cea4bfa7a3491756ced6",
    "answers/sqad32/answers_model_bench_mistral.jsonl.gz": "6bd2b9f839f92d41510a5c5c3f8e7edca6520cfe82cc223a0b14dc4a53f53a4a",
    "answers/sqad32/answers_sqad32_debiased.jsonl.gz": "1f0a174c1056d1db6a8302db9d51391d2916d2215dab29bae9516c601f387ed0",
    "answers/sqad32/answers_sqad32_full.jsonl.gz": "dd1dd735cf526a48168e06bdb241a693c3f5da4ba21b4dafcf9dc23b0a6df59d",
}
# v0.3.1's 1,610 tokens: six were its three-family tables, renamed cz.table_families_*; of the rest, every token but the spans and
# the tables that gained Qwen3.5's rows is unchanged, and each such table, without those rows, is v0.3.1's.
V031_UNCHANGED = (1555, "26e09d3536988316f462c067f2f70afac502548d08e6dbbe963cf5a0e85623cf")   # recomputed from release/v0.3.1, SPANS_V042 left out; heavily_contended without the process names
V031_TABLES = (19, "61e74c9a5d21c15959eb34b503b76b3822f90ccf0440cffdbbfca0cc86585071")   # release/v0.3.1's 19 tables, each as canon() reads it


def test_v031_keeps_every_qwen_and_gemma_answer_byte_for_byte():
    import gzip
    import hashlib

    for rel, digest in V030_ANSWERS.items():
        with gzip.open(REPO / rel, "rb") as fh:
            lines = list(fh)
        dev = "_dev" in rel or "model_bench" in rel                     # Qwen3.5's dev-set rows were there in v0.3.0
        keep = b"".join(x for x in lines if json.loads(x)["config"].split("+")[-1] not in MINISTRAL + GEMMA3N
                        and (dev or json.loads(x)["config"].split("+")[-1] not in QWEN35))
        assert hashlib.sha256(keep).hexdigest() == digest, rel
        if "_dev" not in rel and "model_bench" not in rel and "v020" not in rel:     # every reported set has both Ministral models
            assert {json.loads(x)["config"].split("+")[-1] for x in lines} >= set(MINISTRAL), rel


def test_v031_keeps_every_v030_number_but_the_spans_over_every_family():
    import re

    t = tokens_now()
    old = sorted(k for k in t if not any(re.search(n, k) for n in (NEW_TOKEN, NEW_TOKEN_V040, NEW_TOKEN_V042, NEW_TOKEN_V043)))
    assert len(old) == 1329
    tables = [k for k in old if k in GAINED_ROWS]
    same = {k: plain(t[k]) for k in old if k not in SPANS_V042 and k not in GAINED_ROWS}
    assert (len(same), digest(same)) == V030_UNCHANGED
    # a table that gained Ministral 3's rows holds every row of v0.3.0's, unchanged but the holds table's "both", now named
    def without_later(k: str) -> list[str]:
        v = plain(t[k]).replace("| Gemma 4 and Qwen2.5 |", "| both |") if k == "cz.table_holds" else t[k]
        return canon(v, ("ministral", "qwen3.5") + GEMMA3N_ROWS)
    assert (len(tables), digest({k: without_later(k) for k in tables})) == V030_TABLES


def test_v040_keeps_every_v031_answer_byte_for_byte():
    import gzip
    import hashlib

    for rel, digest in V031_ANSWERS.items():
        with gzip.open(REPO / rel, "rb") as fh:
            lines = list(fh)
        dev = "_dev" in rel or "model_bench" in rel                     # Qwen3.5's dev-set rows were there before
        keep = b"".join(x for x in lines if json.loads(x)["config"].split("+")[-1] not in GEMMA3N
                        and (dev or json.loads(x)["config"].split("+")[-1] not in QWEN35))
        assert hashlib.sha256(keep).hexdigest() == digest, rel
        if not dev and "v020" not in rel:                                # every reported set has both Qwen3.5 models
            assert {json.loads(x)["config"].split("+")[-1] for x in lines} >= set(QWEN35), rel


def test_v040_keeps_every_v031_number_but_the_spans_over_every_family():
    import re

    t = tokens_now()
    old = sorted(k for k in t if not any(re.search(n, k) for n in (NEW_TOKEN_V040, NEW_TOKEN_V042, NEW_TOKEN_V043)))
    assert len(old) == 1604                                              # v0.3.1's 1,610 less its six three-family tables
    assert not any(k.startswith("cz.table_three_") for k in t)
    tables = [k for k in old if k in GAINED_ROWS]
    same = {k: plain(t[k]) for k in old if k not in SPANS_V042 and k not in GAINED_ROWS}
    assert (len(same), digest(same)) == V031_UNCHANGED
    assert (len(tables), digest({k: canon(t[k], ("qwen3.5",) + GEMMA3N_ROWS) for k in tables})) == V031_TABLES


# ---- v0.4.2 changes no v0.4.0 record or number -------------------------------------
GEMMA3N = ("gemma-3n-e2b", "gemma-3n-e4b")
# The tokens v0.4.2 adds, by name: Gemma 3n's, Gemma 4 against its previous generation, and the English dial of each family.
NEW_TOKEN_V042 = (r"(\.dial_family\.|^cz\.(?:[a-z0-9_]+\.)?g3n[._]|^cz\.cross\.n[24]b\.|\+gemma-3n-e[24]b\.|^cz\.compute\.gemma3n_|"
                  r"g4_minus_g3n|^cz\.table_sets_g4$|^cz\.table_prompt_same_machine$|^en_m3\.|^cz\.[a-z0-9_]+\.g4\.(?:table_dial_levels|dial_gen_step_range|dial_gen_step_established_above|leak_no_gold)$|^cz\.[a-z0-9]+_minus_[a-z0-9]+\.(?:smaller|larger)\.(?:em|mapped)_sign$)")
# Ranges and counts over every configuration, which now include Gemma 3n's as well; and the quote check's readings, which
# read every answer that ships and so move with each family added (v0.4.0's Qwen3.5 moved them first: THIRD_PARTY.md).
QUOTE_SPANS = {f"cz.quotes.{x}" for x in ("longest", "over_line", "over_line_by", "over_line_rows", "over_line_scoring",
                                          "repeated_by", "repeated_questions", "repeated_rows", "repeated_scoring", "repeated_whose")}
SPANS_V042 = SPANS_V040 | QUOTE_SPANS | {f"cz.planned.{m}.rule_share_g1_established_n" for m in ("em", "mapped")}
# The tables that gain Gemma 3n's rows (and, in the levels tables, its columns): v0.3.1's and v0.4.0's five-family tables.
GAINED_ROWS_V042 = GAINED_ROWS + ("cz.table_families_differences_em", "cz.table_families_differences_mapped", "cz.table_families_levels_compliance",
                                  "cz.table_families_levels_em", "cz.table_families_levels_mapped", "cz.table_families_steps")
# SHA-256 of each shipped answers file as v0.4.0 has it (afa9ccd), decompressed: v0.4.2's file without Gemma 3n's rows must equal it.
V040_ANSWERS = {
    "answers/sqad30_copy/answers_copy_debiased.jsonl.gz": "fd3840e4b45e95f39e279c25b7fabe34f999c9de8e7bedb8bbdbc5eb35655625",
    "answers/sqad30_copy/answers_cross_version.jsonl.gz": "5045c0f873c76dd72038388d7b42c751260a7035e2204c776b79b70759d60358",
    "answers/sqad30_copy/answers_extra010_copy.jsonl.gz": "398eb2405008a242c96e91e5950dba5a83ad69461c2bbe3b60cc042d3b866019",
    "answers/sqad30_copy/answers_released.jsonl.gz": "525a68da801ac94592659f338d8669671dac7496affbb2722e4f1301234868ec",
    "answers/sqad30_copy/answers_released_v020.jsonl.gz": "db24af7b3e898de7242240e9b4591b1b72427b4fb8d11a4db38b289a0b3c5d89",
    "answers/sqad32/answers_cross_version.jsonl.gz": "9bae46eee6e265a8ae0b2439b8e3cc59a3f534c2df8712cd7cfd3a15f59e9344",
    "answers/sqad32/answers_dev.jsonl.gz": "c441e5a62f9ddb78ca469ae539beaef3a925fcc5344b19e373e6f25c5bfe4425",
    "answers/sqad32/answers_dev_gx10.jsonl.gz": "db6502950c481d7bc7237ea43307c4b888c6afcf87984ca0469bc0aa7d608cda",
    "answers/sqad32/answers_dev_gx10_repeat.jsonl.gz": "98ec3fd91d8bc31644480271d90a128f8cae75a379a6e88ad6c3fdeaa46fd296",
    "answers/sqad32/answers_extra010_sqad32.jsonl.gz": "ead9a79a5516f0c7732b7b487351db2a775f3b338af9f3cf9736fc55bac2547a",
    "answers/sqad32/answers_model_bench.jsonl.gz": "d113a05a24dbc2ea14b61317beeb5d85818a0dd6f087cea4bfa7a3491756ced6",
    "answers/sqad32/answers_model_bench_mistral.jsonl.gz": "6bd2b9f839f92d41510a5c5c3f8e7edca6520cfe82cc223a0b14dc4a53f53a4a",
    "answers/sqad32/answers_sqad32_debiased.jsonl.gz": "ecbdae7f88b98297e32ead4b5f5fde5d45a068f46958e3085d494e3a9fa026aa",
    "answers/sqad32/answers_sqad32_full.jsonl.gz": "19b7f52fee622633b26c4036d7de125e3b31ab8296dc80d93128e164dbff9616",
}
# v0.4.0's 2,131 tokens: every one but the spans and the tables that gain Gemma 3n is unchanged without its flags, and each
# such table, without Gemma 3n's rows and columns, holds v0.4.0's rows. Recomputed from afa9ccd's own stats.json.
V040_UNCHANGED = (2074, "af8922b384b1dd7f6b3318d8300c468e197b3f9054dfec0a08f88fa0b509fc5d")   # hygiene.heavily_contended without the process names
V040_TABLES = (25, "05297d44e2cb433217f7fe7d5e62cc7f4eef34b7b0e7ae8d4a94a658948d0519")


def test_v042_keeps_every_v040_answer_byte_for_byte():
    import gzip
    import hashlib

    for rel, digest_ in V040_ANSWERS.items():
        with gzip.open(REPO / rel, "rb") as fh:
            lines = list(fh)
        keep = b"".join(x for x in lines if json.loads(x)["config"].split("+")[-1] not in GEMMA3N)
        assert hashlib.sha256(keep).hexdigest() == digest_, rel
        if "_dev" not in rel and "model_bench" not in rel and "v020" not in rel:     # every reported set has both Gemma 3n models
            assert {json.loads(x)["config"].split("+")[-1] for x in lines} >= set(GEMMA3N), rel


# ---- v0.4.3 changes no number ---------------------------------------------------------------------
# The tokens v0.4.3 adds, by name: the budget grid as prose, each retriever step's move in g, and BM25 over lemmas minus plain
# BM25; and the two retriever tables that gain that row.
NEW_TOKEN_V043 = r"(^design\.budget_[a-z0-9_]+_text$|\.step\.[a-z0-9.-]+->[a-z0-9.-]+\.k[0-9]+\.g(?:_change)?$|^czr\.[a-z0-9_]+\.bm25_lemma_minus_bm25\.)"
GAINED_ROWS_V043 = {"czr.table_copy_test_differences": ("bm25 over lemmas minus bm25 (added after the rankings were read) |",),
                    "czr.table_sqad32_test_differences": ("bm25 over lemmas minus bm25 (added after the rankings were read) |",)}


def test_v043_adds_tokens_and_rows_and_changes_no_number():
    import re

    t = tokens_now()
    new = sorted(k for k in t if re.search(NEW_TOKEN_V043, k))
    assert len(new) == 57
    for k, drop in GAINED_ROWS_V043.items():
        assert len(canon(t[k])) == len(canon(t[k], drop)) + 1, k


def test_v042_keeps_every_v040_number_but_the_spans_over_every_family():
    import re

    t = tokens_now()
    old = sorted(k for k in t if not re.search(NEW_TOKEN_V042, k) and not re.search(NEW_TOKEN_V043, k))
    assert len(old) == 2131
    tables = [k for k in old if k in GAINED_ROWS_V042]
    same = {k: plain(t[k]) for k in old if k not in SPANS_V042 and k not in GAINED_ROWS_V042}
    for k, drop in GAINED_ROWS_V043.items():                         # as v0.4.0 had them, without v0.4.3's row
        same[k] = "\n".join(plain(t[k]).split("\n")[:2] + [x for x in plain(t[k]).split("\n")[2:] if not any(d in x.lower() for d in drop)])
    assert (len(same), digest(same)) == V040_UNCHANGED
    assert (len(tables), digest({k: canon(t[k], GEMMA3N_ROWS) for k in tables})) == V040_TABLES


def test_gemma_3n_runs_alone_every_stage_ministral_3_ran():
    """Gemma 3n in Ministral 3's place, nothing else generates, the benches keep their outputs."""
    plan = json.loads((REPO / "results/czech/plan/plan.json").read_text(encoding="utf-8"))
    ministral = [s.name for s in cq.ministral_stages(plan)]
    gemma3n = cq.gemma3n_stages(plan)
    assert [s.name for s in gemma3n] == [n.replace("ministral-3-3b", "gemma-3n-e2b").replace("ministral-3-8b", "gemma-3n-e4b") for n in ministral]
    assert len(gemma3n) == 34
    for s in gemma3n:
        if "--generator" in s.cmd:
            assert s.cmd[s.cmd.index("--generator") + 1] in GEMMA3N, s.name
        if "--only" in s.cmd:
            assert s.cmd[-1].endswith("+gemma-3n-e2b"), s.name
    assert not any(s.prepare for s in gemma3n)
    assert not any("retrieval_cache.py" in " ".join(s.cmd) for s in gemma3n)


def test_qwen35_runs_alone_every_stage_ministral_3_ran():
    """Qwen3.5 in Ministral 3's place, nothing else generates, the benches keep their outputs."""
    plan = json.loads((REPO / "results/czech/plan/plan.json").read_text(encoding="utf-8"))
    ministral = [s.name for s in cq.ministral_stages(plan)]
    qwen35 = cq.qwen35_stages(plan)
    assert [s.name for s in qwen35] == [n.replace("ministral-3-3b", "qwen3.5-2b").replace("ministral-3-8b", "qwen3.5-4b") for n in ministral]
    for s in qwen35:
        if "--generator" in s.cmd:
            assert s.cmd[s.cmd.index("--generator") + 1] in ("qwen3.5-2b", "qwen3.5-4b"), s.name
        if "--only" in s.cmd:
            assert s.cmd[-1].endswith("+qwen3.5-2b"), s.name
    assert not any(s.prepare for s in qwen35)
    assert not any("retrieval_cache.py" in " ".join(s.cmd) for s in qwen35)


def test_a_pinned_generator_is_fetched_at_its_commit(monkeypatch, tmp_path):
    import huggingface_hub

    from measurerag.rag import generator as gm

    seen = {}

    def fake(repo_id, filename, revision=None, cache_dir=None):
        seen.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    gm.resolve_model_path("gemma-4-e4b")
    assert seen == {"repo_id": "unsloth/gemma-4-E4B-it-GGUF", "filename": "gemma-4-E4B-it-Q4_K_M.gguf", "revision": gm.GENERATORS["gemma-4-e4b"].revision}
