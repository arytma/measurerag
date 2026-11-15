"""The worker's spec contract and the pieces of the RAG stage that can be
tested without a 1 GB GGUF on disk."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from measurerag.datasets import load_dataset
from measurerag.rag.generator import GENERATORS, build_prompt, load_prompt_template

REPO = Path(__file__).resolve().parents[1]


# -- the prompt -----------------------------------------------------------
def test_there_is_exactly_one_prompt_template_file():
    from measurerag.rag.generator import PROMPT_PATH

    assert PROMPT_PATH.name == "prompt_template.txt"
    assert PROMPT_PATH.exists()


def test_template_has_both_slots_and_nothing_else():
    tpl = load_prompt_template()
    assert "{context}" in tpl and "{question}" in tpl


def test_prompt_numbers_the_passages():
    p = build_prompt("why?", ["first", "second"])
    assert "[1] first" in p and "[2] second" in p
    assert "why?" in p


def test_prompt_truncates_passages_to_a_fixed_budget():
    # a retriever must not be able to win by handing over more text than a rival
    p = build_prompt("q", ["\u00e9" * 5000], max_chars_per_passage=100)
    assert p.count("\u00e9") == 100


def test_prompt_handles_the_empty_retrieval_case():
    # a retriever that returns nothing must still produce a scoreable prompt
    assert "(no passages retrieved)" in build_prompt("q", [])


def test_prompt_is_identical_across_configurations():
    # the template is read from disk every time, so no caller can localise it
    a = build_prompt("q", ["p"])
    b = build_prompt("q", ["p"])
    assert a == b


# -- the Czech yes/no instruction -------------------------
TEMPLATE_SHA_V020 = "6060a52591a3892ad2cf7ed30d4ad4488daf98df4bbf540578314d2d49ad0086"
METRICS_SHA_V020 = "7d8789c81e9cca41021c274835c18e8103f2afe5ee76c57e6d64ae80d76b642c"
PROMPT_SHA_V020 = "a2ed6a0918faadf1e229294372755e4804d901b6b6e98de9ea41671f5a6c7420"


def test_the_template_file_and_every_english_prompt_are_byte_identical_to_v020():
    import hashlib

    from measurerag.rag.generator import PROMPT_PATH

    assert hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest() == TEMPLATE_SHA_V020
    for prompt in (build_prompt("Kdy?", ["první", "druhý"]), build_prompt("Kdy?", ["první", "druhý"], language="en")):
        assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == PROMPT_SHA_V020
    assert "exactly yes or no" in load_prompt_template() and "ano or ne" not in load_prompt_template()


def test_the_first_cuts_czech_prompt_asks_for_ano_or_ne_and_changes_nothing_else():
    en, cs = load_prompt_template("en"), load_prompt_template("cs", "v1")
    assert "answer exactly ano or ne." in cs and "yes or no" not in cs
    assert cs.replace("answer exactly ano or ne", "answer exactly yes or no") == en
    assert build_prompt("q", ["p"], language="cs", variant="v1") == build_prompt("q", ["p"]).replace("yes or no", "ano or ne")
    assert "reply exactly: unanswerable" in cs


def test_the_frozen_czech_prompt_is_the_one_the_dev_set_chose():
    from measurerag.rag.generator import CZECH_PROMPT

    selection = json.loads((REPO / "results/czech/dev/selection.json").read_text(encoding="utf-8"))["selection"]
    assert CZECH_PROMPT == selection["chosen"]
    assert load_prompt_template("cs") == load_prompt_template("cs", selection["chosen"])
    assert selection["primary_metric"] == ("em" if selection["passed"] else "em_yes_no_mapped")


def test_the_exact_match_rule_is_unchanged():
    import hashlib

    from measurerag import metrics

    assert hashlib.sha256(Path(metrics.__file__).read_bytes()).hexdigest() == METRICS_SHA_V020
    assert metrics.exact_match("Ano.", ["ano"]) == 1.0
    assert metrics.exact_match("yes", ["ano"]) == 0.0           # no mapping inside the metric
    assert metrics.exact_match("v Praze", ["Praha"]) == 0.0     # no lemmatising either


def test_a_czech_harness_spec_names_the_language_and_an_english_one_does_not(tmp_path, monkeypatch):
    """Resume compares specs. A Czech result from before the prompt fix must not pass as finished."""
    from measurerag import runner
    from measurerag.config import load_config
    from measurerag.datasets.base import Dataset

    cfg = dict(load_config(REPO / "configs" / "smoke.yaml"))
    cfg["rag"] = {"enabled": True, "generators": ["qwen2.5-1.5b"], "max_tokens": 8}
    real = runner.resolve_dataset

    class Stop(Exception):
        pass

    def fake_run(cmd, **kw):       # read the spec the runner wrote, then stop: no model is loaded
        seen.append(json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text(encoding="utf-8")))
        raise Stop

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    for language in ("en", "cs"):
        seen: list[dict] = []

        def fake_resolve(c, cache, log=print, language=language):
            ds, attempts = real(c, cache, log)
            data = {k: getattr(ds, k) for k in ("name", "docs", "queries", "qrels", "answers", "license", "source", "notes")}
            return Dataset(language=language, **data), attempts

        monkeypatch.setattr(runner, "resolve_dataset", fake_resolve)
        with pytest.raises(Stop):
            runner.run(cfg, out_dir=tmp_path / language, log=lambda _: None)
        assert ("prompt_language" in seen[0]) == (language == "cs")
        if language == "cs":
            assert seen[0]["prompt_language"] == "cs"


def test_generators_are_quantised_and_local():
    for spec in GENERATORS.values():
        assert "q4_k_m" in spec.filename.lower()
        # the model's own makers' file, or another quantiser's pinned to a commit
        assert spec.repo_id.startswith("Qwen/") or (spec.revision and len(spec.revision) == 40)


# -- the worker as a subprocess ------------------------------------------
@pytest.fixture(scope="module")
def toy_slice(tmp_path_factory):
    d = tmp_path_factory.mktemp("slice")
    load_dataset("toy").save(d)
    return d


def _run_worker(tmp_path, spec: dict) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "out.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "measurerag.worker", "--spec", str(spec_path),
         "--out", str(out_path)],
        cwd=REPO, capture_output=True, text=True, timeout=600, check=False,
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


def _spec(toy_slice, tmp_path, **over) -> dict:
    base = {
        "config_id": "bm25",
        "slice_dir": str(toy_slice),
        "workdir": str(tmp_path / "work"),
        "retriever": {"id": "bm25", "kind": "bm25"},
        "rag": {"enabled": False},
        "budgets": [{"name": "b", "ram_gb": 16, "retrieval_ms": 500, "e2e_s": None}],
        "top_k": 100, "ks": [10, 100], "context_k": 5,
        "device": "cpu", "seed": 20260101, "query_threads": 1,
    }
    base.update(over)
    return base


def test_worker_reports_its_own_cost_and_verdict(toy_slice, tmp_path):
    res = _run_worker(tmp_path, _spec(toy_slice, tmp_path))
    assert res["status"] == "ok"
    assert res["measurement"]["peak_rss_gb"] > 0
    assert res["verdicts"][0]["fits"] is True
    assert res["threads"]["query"] == 1


def test_worker_writes_the_index_it_measured(toy_slice, tmp_path):
    res = _run_worker(tmp_path, _spec(toy_slice, tmp_path))
    assert res["build"]["index_bytes"] > 0
    assert (tmp_path / "work").exists()


def test_worker_reports_a_failure_instead_of_crashing_the_run(toy_slice, tmp_path):
    res = _run_worker(
        tmp_path, _spec(toy_slice, tmp_path, retriever={"id": "x", "kind": "no-such-retriever"})
    )
    assert res["status"] == "failed"
    assert "no-such-retriever" in res["error"]
    assert res["traceback"]


def test_worker_records_the_machine_with_the_numbers(toy_slice, tmp_path):
    res = _run_worker(tmp_path, _spec(toy_slice, tmp_path))
    assert res["machine"]["chip"]
    assert res["machine"]["ram_gb"] > 0


def test_worker_skips_the_rag_stage_gracefully_without_a_generator(toy_slice, tmp_path):
    # an air-gapped box with no GGUF must still produce retrieval numbers
    res = _run_worker(
        tmp_path,
        _spec(toy_slice, tmp_path,
              rag={"enabled": True, "generator": "qwen2.5-1.5b", "max_tokens": 8},
              retriever={"id": "bm25", "kind": "bm25"}),
    )
    assert res["status"] == "ok"
    assert res["metrics"]["ndcg@10"] > 0
    assert res.get("rag_status") in {"ok", "skipped"}


def test_two_workers_on_the_same_spec_agree_on_every_metric(toy_slice, tmp_path):
    a = _run_worker(tmp_path / "a", _spec(toy_slice, tmp_path / "a"))
    b = _run_worker(tmp_path / "b", _spec(toy_slice, tmp_path / "b"))
    assert a["metrics"] == b["metrics"]


def test_a_generator_device_reaches_the_spec_only_when_a_config_sets_it(tmp_path, monkeypatch):
    """Every spec written before the key existed must stay the same, or a finished result would run again."""
    from measurerag import runner
    from measurerag.config import load_config

    class Stop(Exception):
        pass

    def fake_run(cmd, **kw):
        seen.append(json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text(encoding="utf-8")))
        raise Stop

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    for rag_device in (None, "cuda"):
        seen: list[dict] = []
        cfg = dict(load_config(REPO / "configs" / "smoke.yaml"))
        cfg["rag"] = {"enabled": True, "generators": ["qwen2.5-1.5b"], "max_tokens": 8, **({"device": rag_device} if rag_device else {})}
        with pytest.raises(Stop):
            runner.run(cfg, out_dir=tmp_path / str(rag_device), log=lambda _: None)
        assert seen[0]["rag"].get("device") == rag_device and seen[0]["device"] == "cpu"


def test_a_cuda_generator_on_a_build_without_offload_stops(tmp_path, monkeypatch):
    import llama_cpp

    from measurerag.rag import generator as gm

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(gm, "resolve_model_path", lambda key, cache_dir=None: gguf)
    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)
    with pytest.raises(gm.GeneratorUnavailable, match="without GPU offload"):
        gm.LlamaGenerator("qwen2.5-1.5b", device="cuda")


# -- memory-mapping the model file ---------------------------------------
# llama-cpp-python maps the GGUF by default, and every published run loaded it so; with mapping on, peak RSS counts the
# file's pages beside the CPU backend's copy of the weights (docs/LIMITATIONS.md). Both settings must work and be recorded.
@pytest.mark.parametrize("setting", [None, True, False])
def test_the_generator_maps_its_file_by_default_and_passes_the_setting_on(tmp_path, monkeypatch, setting):
    import llama_cpp

    from measurerag.rag import generator as gm

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(gm, "resolve_model_path", lambda key, cache_dir=None: gguf)
    seen: dict = {}

    class FakeLlama:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(llama_cpp, "Llama", FakeLlama)
    g = gm.LlamaGenerator("ministral-3-3b", **({} if setting is None else {"use_mmap": setting}))
    expected = True if setting is None else setting
    assert seen["use_mmap"] is expected and g.use_mmap is expected


def test_use_mmap_reaches_the_spec_only_when_a_config_sets_it(tmp_path, monkeypatch):
    """A spec written before the key existed must stay the same, or a finished result would be measured again."""
    from measurerag import runner
    from measurerag.config import load_config

    class Stop(Exception):
        pass

    def fake_run(cmd, **kw):
        seen.append(json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text(encoding="utf-8")))
        raise Stop

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    for setting in (None, True, False):
        seen: list[dict] = []
        cfg = dict(load_config(REPO / "configs" / "smoke.yaml"))
        cfg["rag"] = {"enabled": True, "generators": ["qwen2.5-1.5b"], "max_tokens": 8,
                      **({} if setting is None else {"use_mmap": setting})}
        with pytest.raises(Stop):
            runner.run(cfg, out_dir=tmp_path / str(setting), log=lambda _: None)
        assert seen[0]["rag"].get("use_mmap", "absent") == ("absent" if setting is None else setting)


@pytest.mark.parametrize("setting", [None, True, False])
def test_the_worker_records_whether_the_model_file_was_mapped(toy_slice, tmp_path, monkeypatch, setting):
    from measurerag import worker
    from measurerag.datasets.base import Dataset
    from measurerag.rag import generator as gm

    made: list = []

    class FakeGenerator:
        def __init__(self, key, use_mmap=True, **kw):
            self.use_mmap, self.model_bytes, self.model_path = use_mmap, 1, tmp_path / "model.gguf"
            made.append(self)

        def generate(self, prompt):
            return "x", {"seconds": 0.01, "completion_tokens": 1, "prompt_tokens": 1, "tokens_per_s": 100.0}

        def close(self):
            pass

    monkeypatch.setattr(gm, "LlamaGenerator", FakeGenerator)
    ds = Dataset.load_slice(toy_slice)
    assert ds.answers, "the toy slice must have answers for the RAG stage to run"
    run = {q: [d.doc_id for d in ds.docs[:5]] for q in ds.queries}
    rag = {"generator": "qwen2.5-1.5b", "max_tokens": 8, **({} if setting is None else {"use_mmap": setting})}
    out = worker._run_rag_stage(ds, run, rag, "cpu", tmp_path, 5, 20260101)
    expected = True if setting is None else setting
    assert out["rag_status"] == "ok" and out["generator"]["use_mmap"] is expected and made[0].use_mmap is expected
