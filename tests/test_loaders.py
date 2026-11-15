"""Loaders, capping and slice round-tripping. All offline: no network, no
weights, nothing but the bundled fixture."""

from __future__ import annotations

import pytest

from measurerag.datasets import available, get_loader, load_dataset
from measurerag.datasets.base import Dataset, DatasetBlocked, Doc, apply_caps


# -- the bundled fixture --------------------------------------------------
def test_toy_fixture_is_the_advertised_size():
    ds = load_dataset("toy")
    assert len(ds.docs) == 200
    assert len(ds.queries) == 20


def test_toy_every_query_has_gold_and_an_answer():
    ds = load_dataset("toy")
    for qid in ds.queries:
        assert ds.qrels.get(qid), f"{qid} has no qrels"
        assert ds.answers.get(qid), f"{qid} has no gold answer"


def test_toy_gold_documents_exist_in_the_corpus():
    ds = load_dataset("toy")
    ids = set(ds.doc_ids)
    for rels in ds.qrels.values():
        assert set(rels).issubset(ids)


def test_toy_language_filter_splits_cleanly():
    cs = load_dataset("toy", language="cs")
    en = load_dataset("toy", language="en")
    assert len(cs.docs) + len(en.docs) == 200
    assert len(cs.queries) == len(en.queries) == 10


def test_toy_is_not_trivially_solvable():
    # every entity's five documents share the entity name and differ only in
    # which fact they carry; if that ever stops being true the smoke table goes
    # back to a wall of 1.0 and teaches nothing
    ds = load_dataset("toy")
    by_title: dict[str, int] = {}
    for d in ds.docs:
        by_title[d.title] = by_title.get(d.title, 0) + 1
    assert set(by_title.values()) == {5}


def test_toy_fingerprint_is_stable_across_loads():
    assert load_dataset("toy").fingerprint() == load_dataset("toy").fingerprint()


# -- capping --------------------------------------------------------------
def _synthetic(n_docs: int = 50, n_queries: int = 10) -> Dataset:
    docs = [Doc(doc_id=f"d{i:03d}", text=f"document {i}") for i in range(n_docs)]
    queries = {f"q{i}": f"query {i}" for i in range(n_queries)}
    qrels = {f"q{i}": {f"d{i:03d}": 1} for i in range(n_queries)}
    return Dataset(name="synthetic", language="en", docs=docs, queries=queries, qrels=qrels)


def test_cap_reduces_and_records_itself():
    ds = apply_caps(_synthetic(), max_docs=20, max_queries=5)
    assert len(ds.queries) == 5
    assert len(ds.docs) == 20
    assert ds.caps_applied["max_docs"] == 20
    assert ds.caps_applied["max_queries"] == 5


def test_cap_keeps_every_gold_document():
    # this is the whole point: a plain random cap would bound recall by the
    # sampling rate rather than by the retriever
    ds = apply_caps(_synthetic(1000, 20), max_docs=50, max_queries=20)
    ids = set(ds.doc_ids)
    for rels in ds.qrels.values():
        assert set(rels).issubset(ids)
    assert ds.caps_applied["gold_docs_forced"] == 20


def test_cap_drops_qrels_pointing_outside_the_slice():
    ds = _synthetic(100, 10)
    ds.qrels["q0"]["d099"] = 1  # a second gold we will not force in
    ds = apply_caps(ds, max_docs=12, max_queries=2)
    ids = set(ds.doc_ids)
    for rels in ds.qrels.values():
        assert set(rels).issubset(ids)


def test_cap_is_deterministic():
    a = apply_caps(_synthetic(500, 50), max_docs=40, max_queries=10)
    b = apply_caps(_synthetic(500, 50), max_docs=40, max_queries=10)
    assert a.corpus_hash() == b.corpus_hash()
    assert a.query_hash() == b.query_hash()


def test_cap_above_the_size_is_a_no_op():
    ds = apply_caps(_synthetic(10, 3), max_docs=999, max_queries=999)
    assert len(ds.docs) == 10 and ds.caps_applied == {}


def test_cap_never_sacrifices_gold_to_hit_the_number():
    ds = apply_caps(_synthetic(100, 30), max_docs=5, max_queries=30)
    assert ds.caps_applied.get("cap_overridden_by_gold") is True
    assert len(ds.docs) == 30  # the gold, all of it


# -- slices ---------------------------------------------------------------
def test_slice_round_trips_byte_for_byte(tmp_path):
    ds = load_dataset("toy")
    ds.save(tmp_path / "slice")
    back = Dataset.load_slice(tmp_path / "slice")
    assert back.corpus_hash() == ds.corpus_hash()
    assert back.query_hash() == ds.query_hash()
    assert back.qrels_hash() == ds.qrels_hash()
    assert back.answers == ds.answers


def test_saved_slice_carries_the_licence_and_the_notes(tmp_path):
    ds = load_dataset("toy")
    ds.save(tmp_path / "s")
    back = Dataset.load_slice(tmp_path / "s")
    assert back.license == ds.license
    assert back.notes == ds.notes


# -- registry -------------------------------------------------------------
def test_every_registered_dataset_is_importable():
    for name in available():
        assert callable(get_loader(name))


def test_unknown_dataset_names_itself_in_the_error():
    with pytest.raises(KeyError, match="toy"):
        get_loader("no-such-dataset")


def test_gated_datasets_raise_blocked_not_a_generic_error():
    # BLOCKED is what makes the runner fall back instead of dying, and what
    # gets recorded as "we did not work around the gate"
    for name in ("cwrczech", "dareczech"):
        with pytest.raises(DatasetBlocked):
            get_loader(name)(cache_dir=None)


def test_cwrczech_blocked_message_says_how_to_unblock():
    from measurerag.datasets.cwrczech import BLOCKED_MESSAGE

    assert "disclaimer.md" in BLOCKED_MESSAGE and "request access" in BLOCKED_MESSAGE
    assert "data_dir" in BLOCKED_MESSAGE


# -- HotpotQA column addressing (the bug that stopped the RAG stage) ------
class TestHotpotQaPairs:
    """BEIR names the query id column `_id`, which pandas cannot expose as an
    attribute. `itertuples` renames it to a position - `_1` with the index,
    `_0` without - and the loader asked for `_1` while passing `index=False`,
    so the English RAG stage failed the first time anything reached it.
    """

    @staticmethod
    def _df(rows):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(rows)

    def test_reads_a_column_python_cannot_name(self):
        from measurerag.datasets.hotpotqa import _pairs

        df = self._df([{"_id": "q1", "text": "why?"}, {"_id": "q2", "text": "how?"}])
        assert _pairs(df, "_id", "text") == {"q1": "why?", "q2": "how?"}

    def test_wrap_puts_each_answer_in_a_list(self):
        from measurerag.datasets.hotpotqa import _pairs

        df = self._df([{"id": "q1", "answer": "Prague"}])
        assert _pairs(df, "id", "answer", wrap=True) == {"q1": ["Prague"]}

    def test_blank_answers_are_dropped_not_kept_as_empty(self):
        from measurerag.datasets.hotpotqa import _pairs

        df = self._df([{"id": "a", "answer": "yes"}, {"id": "b", "answer": ""}])
        assert _pairs(df, "id", "answer", wrap=True) == {"a": ["yes"]}

    def test_a_renamed_upstream_column_fails_loudly(self):
        """If BEIR renames `_id`, say so rather than return an empty dataset."""
        from measurerag.datasets.base import DatasetUnavailable
        from measurerag.datasets.hotpotqa import _pairs

        df = self._df([{"query_id": "q1", "text": "why?"}])
        with pytest.raises(DatasetUnavailable, match="expected columns"):
            _pairs(df, "_id", "text")

    def test_itertuples_would_have_renamed_the_column(self):
        """Pins the pandas behaviour the original code guessed wrong."""
        df = self._df([{"_id": "q1", "text": "why?"}])
        assert next(iter(df.itertuples(index=False)))._fields[0] == "_0"
        assert next(iter(df.itertuples()))._fields[1] == "_1"
