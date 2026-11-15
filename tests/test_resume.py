"""Resume: a queue that loses the machine mid-run must not redo the hours behind it.

The first unattended build segfaulted in the Czech stage after two retrievers had
already finished, and the restart would have re-measured them from scratch. These
tests run the real runner twice over the bundled fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    from measurerag.config import load_config

    return load_config(REPO / "configs" / "smoke.yaml")


ONLY = ["bm25"]   # resume is per-configuration; one is enough and keeps this fast


def _run(cfg, out, only=ONLY, **kw):
    from measurerag.runner import run

    return run(
        dict(cfg), config_path="configs/smoke.yaml", out_dir=out,
        only=only, log=lambda _: None, **kw,
    )


@pytest.fixture(scope="module")
def first(cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("resume")
    return out, _run(cfg, out)


def test_second_run_reuses_every_finished_result(cfg, first):
    out, _ = first
    again = _run(cfg, out, resume=True)
    assert again["results"], "nothing ran"
    assert all(r.get("resumed") for r in again["results"])


def test_reused_metrics_are_the_ones_already_on_disk(cfg, first):
    out, original = first
    again = _run(cfg, out, resume=True)
    before = {r["config_id"]: r["metrics"] for r in original["results"]}
    after = {r["config_id"]: r["metrics"] for r in again["results"]}
    assert before == after


def test_resume_off_measures_again(cfg, first):
    out, _ = first
    again = _run(cfg, out, resume=False)
    assert not any(r.get("resumed") for r in again["results"])


def test_a_changed_spec_is_not_reused(cfg, first):
    """The guard is the spec, not the filename: change the run and it re-measures."""
    out, _ = first
    changed = dict(cfg)
    changed["retrieval"] = dict(cfg["retrieval"]) | {"top_k": cfg["retrieval"]["top_k"] + 7}
    again = _run(changed, out, resume=True)
    assert not any(r.get("resumed") for r in again["results"])


def test_reuse_only_reuses_and_never_measures(cfg, first, monkeypatch):
    """The queue rebuilds a sweep's files with MEASURERAG_REUSE_ONLY set.

    A spec that changed since a batch was accepted must stop the rebuild, not
    measure the configuration again: the release's timings are the accepted ones.
    """
    out, _ = first
    _run(cfg, out, resume=True)   # the module's earlier tests leave a changed spec behind
    monkeypatch.setenv("MEASURERAG_REUSE_ONLY", "1")
    again = _run(cfg, out, resume=True)
    assert again["results"] and all(r.get("resumed") for r in again["results"])
    before = {f.name: f.read_bytes() for f in (out / "specs").iterdir()}
    changed = dict(cfg)
    changed["retrieval"] = dict(cfg["retrieval"]) | {"top_k": cfg["retrieval"]["top_k"] + 7}
    with pytest.raises(RuntimeError, match="MEASURERAG_REUSE_ONLY"):
        _run(changed, out, resume=True)
    assert {f.name: f.read_bytes() for f in (out / "specs").iterdir()} == before


def test_a_failed_result_is_retried_not_reused(cfg, first):
    out, _ = first
    victim = sorted((out / "specs").glob("*.result.json"))[0]
    saved = json.loads(victim.read_text(encoding="utf-8"))
    victim.write_text(
        json.dumps({**saved, "status": "failed", "error": "pretend segfault"}), encoding="utf-8"
    )
    again = _run(cfg, out, resume=True)
    retried = [r for r in again["results"] if r["config_id"] == saved["config_id"]]
    assert retried and not retried[0].get("resumed")
    assert retried[0]["status"] == "ok"


def test_a_truncated_result_is_retried_not_fatal(cfg, first):
    """A process killed while writing its JSON leaves half a file behind."""
    out, _ = first
    victim = sorted((out / "specs").glob("*.result.json"))[0]
    config_id = json.loads(victim.read_text(encoding="utf-8"))["config_id"]
    victim.write_text('{"config_id": "trunc', encoding="utf-8")
    again = _run(cfg, out, resume=True)
    retried = [r for r in again["results"] if r["config_id"] == config_id]
    assert retried and retried[0]["status"] == "ok"


def test_repro_never_resumes(cfg):
    """The reproducibility claim is two independent runs; reuse would fake it."""
    import inspect

    from measurerag import cli

    src = inspect.getsource(cli.repro)
    assert src.count("resume=False") == 2


def test_a_changed_model_declaration_invalidates_a_dense_result(tmp_path):
    """The gap that the pooling fix exposed.

    Correcting retromae-cs from mean to CLS pooling changed what it measures
    without changing a single line of its config, so the spec stayed identical
    and resume would have served the stale number back. The spec carries the
    resolved ModelSpec for exactly this reason.
    """
    from dataclasses import replace

    from measurerag.retrievers import describe_retriever
    from measurerag.retrievers.dense import MODELS

    before = describe_retriever("dense", model="retromae-cs")
    patched = dict(MODELS)
    patched["retromae-cs"] = replace(MODELS["retromae-cs"], pooling="mean")

    import measurerag.retrievers.dense as dense

    original = dense.MODELS
    try:
        dense.MODELS = patched
        after = describe_retriever("dense", model="retromae-cs")
    finally:
        dense.MODELS = original

    assert before != after, "a pooling change must change the spec"


def test_an_unpinned_model_declares_no_revision():
    """Pinning one model must not change the spec of every other dense model:
    the key appears only where a revision is pinned."""
    from measurerag.retrievers import describe_retriever
    from measurerag.retrievers.dense import MODELS

    for key, m in MODELS.items():
        decl = describe_retriever("dense", model=key)["model_spec"]
        assert ("revision" in decl) == (m.revision is not None), key


def test_a_lexical_retriever_has_no_model_declaration():
    from measurerag.retrievers import describe_retriever

    assert describe_retriever("bm25") == {}
    assert describe_retriever("tfidf-svd") == {}


def test_only_selects_a_retriever_or_a_single_configuration():
    from measurerag.runner import wanted

    assert wanted("bm25", "qwen2.5-3b", None)
    assert wanted("bm25", "qwen2.5-3b", ["bm25"])
    assert wanted("bm25", None, ["bm25+qwen2.5-1.5b"])
    assert wanted("bm25", "qwen2.5-1.5b", ["bm25+qwen2.5-1.5b"])
    assert not wanted("bm25", "qwen2.5-3b", ["bm25+qwen2.5-1.5b"])
    assert not wanted("e5-base", None, ["bm25+qwen2.5-1.5b"])
    # a prefix of another id is not that id
    assert not wanted("e5", None, ["e5-base+qwen2.5-1.5b"])
