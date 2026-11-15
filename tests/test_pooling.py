"""Pooling is declared per model, not inferred.

Neither Seznam model ships a `modules.json`. sentence-transformers responds by
building a mean-pooled model without warning, while both model cards specify the
CLS token. That produced 0.2778 nDCG@10 for retromae-cs on CsFEVER - a number
about the loader rather than the model. These tests keep the declaration honest
and are all offline.
"""

from __future__ import annotations

import pytest

from measurerag.retrievers.dense import MODELS, ModelSpec, _pooling_mode

VALID = {"cls", "max", "mean", "mean_sqrt_len_tokens", "weightedmean", "lasttoken"}


def test_every_model_declares_a_pooling():
    for key, spec in MODELS.items():
        assert spec.pooling in VALID, f"{key} declares {spec.pooling!r}"


def test_pooling_has_no_default_so_a_new_model_cannot_forget_it():
    with pytest.raises(TypeError):
        ModelSpec("x", "org/x")  # type: ignore[call-arg]


@pytest.mark.parametrize("key", ["retromae-cs", "simcse-cs-en"])
def test_the_seznam_models_are_cls_pooled(key):
    """Regression guard. Their cards say last_hidden_state[:, 0]; ST would mean-pool."""
    assert MODELS[key].pooling == "cls"


@pytest.mark.parametrize("key", ["granite-107m", "granite-278m"])
def test_the_granite_models_are_cls_pooled_with_no_prefix(key):
    """Their repos ship 1_Pooling with cls=true and their cards pool the CLS token."""
    spec = MODELS[key]
    assert spec.pooling == "cls" and spec.query_prefix == "" and spec.passage_prefix == "" and spec.max_seq_length == 512


def test_embeddinggemma_carries_its_repository_prompts():
    spec = MODELS["embeddinggemma"]
    assert spec.pooling == "mean" and spec.max_seq_length == 512
    assert spec.query_prefix == "task: search result | query: " and spec.passage_prefix == "title: none | text: "


@pytest.mark.parametrize("key", ["e5-small", "e5-base"])
def test_the_e5_models_stay_mean_pooled(key):
    """Their repos ship 1_Pooling with mean=true; changing this invalidates a curve."""
    assert MODELS[key].pooling == "mean"


def test_bge_m3_is_cls_pooled():
    assert MODELS["bge-m3"].pooling == "cls"


# -- the helper that reads the pooling back off a loaded model ------------
class _FakeModule:
    def __init__(self, mode=None):
        self.pooling_mode = mode


class _FakeEncoder:
    def __init__(self, modules):
        self._m = modules

    def __iter__(self):
        return iter(self._m)


def test_pooling_mode_reads_the_pooling_module():
    enc = _FakeEncoder([_FakeModule(None), _FakeModule("cls")])
    assert _pooling_mode(enc) == "cls"


def test_pooling_mode_unwraps_a_single_element_sequence():
    assert _pooling_mode(_FakeEncoder([_FakeModule(["mean"])])) == "mean"


def test_pooling_mode_is_none_when_nothing_pools():
    assert _pooling_mode(_FakeEncoder([_FakeModule(None)])) is None


def test_a_mismatch_is_detectable_not_silent():
    """The condition the loader branches on: declared != actual."""
    spec = MODELS["retromae-cs"]
    actual_if_left_to_sentence_transformers = "mean"
    assert spec.pooling != actual_if_left_to_sentence_transformers


# -- the deliverable carries the provenance -------------------------------
def test_the_csv_row_records_which_pooling_bought_the_score():
    from measurerag.report import flatten

    row = flatten(
        {
            "config_id": "retromae-cs",
            "build": {"pooling": "cls", "pooling_source": "declared ('mean' would have been used)"},
            "metrics": {"ndcg@10": 0.5},
            "measurement": {},
        }
    )
    assert row["pooling"] == "cls"
    assert "mean" in row["pooling_source"]


def test_a_lexical_row_has_no_pooling_columns():
    from measurerag.report import flatten

    row = flatten({"config_id": "bm25", "build": {}, "metrics": {}, "measurement": {}})
    assert "pooling" not in row and "pooling_source" not in row
