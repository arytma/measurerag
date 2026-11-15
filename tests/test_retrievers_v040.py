"""v0.4.0's retrievers on SQAD. All offline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from measurerag.retrievers.bm25 import BM25Retriever, tokenize  # noqa: E402

TEXT = "Hlavním městem České republiky je Praha; v Praze žije přes milion lidí."


def test_tokens_without_lemmas_are_what_they_always_were():
    """bm25 and bm25-nodiacritics rank the shipped records; their tokens must not move."""
    assert tokenize(TEXT) == ["hlavním", "městem", "české", "republiky", "je", "praha", "v", "praze", "žije", "přes", "milion", "lidí"]
    assert tokenize(TEXT, strip_diacritics=True) == ["hlavnim", "mestem", "ceske", "republiky", "je", "praha", "v", "praze", "zije", "pres", "milion", "lidi"]


def test_lemmas_join_the_inflected_forms():
    pytest.importorskip("simplemma")
    toks = tokenize(TEXT, lemmatize="cs")
    assert len(toks) == len(tokenize(TEXT))
    assert all(t == t.lower() for t in toks)
    assert toks[5] == toks[7], toks        # Praha and v Praze: one lemma
    assert tokenize("městem", lemmatize="cs") == tokenize("město", lemmatize="cs")


def test_diacritics_are_stripped_after_the_lemma_not_before():
    pytest.importorskip("simplemma")
    from measurerag.retrievers.bm25 import _strip

    with_marks = tokenize(TEXT, lemmatize="cs")
    assert tokenize(TEXT, lemmatize="cs", strip_diacritics=True) == [_strip(t) for t in with_marks]
    assert tokenize("městem", lemmatize="cs", strip_diacritics=True) == [_strip(tokenize("město", lemmatize="cs")[0])]


def test_the_lemma_retriever_is_declared_by_its_id():
    from retrieval_cache import parse

    assert parse("bm25-lemma:bm25") == {"id": "bm25-lemma", "kind": "bm25", "lemmatize": "cs"}
    assert "lemmatize" not in parse("bm25:bm25") and "lemmatize" not in parse("bm25-nodiacritics:bm25")
    assert BM25Retriever(lemmatize="cs").lemmatize == "cs"


def _row(gains: dict, anchor: float = 0.054) -> dict:
    import czech_retrievers as cr

    def ci(p, est=True):
        return {"g3": {"point": p, "lo": p - 0.01 if est else -0.01, "hi": p + 0.01, "established": est}}
    diffs = {"e5-small - bm25": ci(anchor)}
    for n in cr.NEW:
        if n in gains:
            p, est = gains[n]
            diffs[f"{n} - e5-small"] = ci(p, est)
    return {"differences": diffs}


def test_phase_b_runs_only_for_an_established_gain_at_least_the_anchor_once_every_row_is_in():
    import czech_retrievers as cr

    every = {n: (0.0, False) for n in cr.NEW}
    assert cr.phase_b_rule(_row({**every, "granite-278m": (0.060, True)}))["runs"] is True
    assert cr.phase_b_rule(_row({**every, "granite-278m": (0.050, True)}))["runs"] is False    # below the anchor
    assert cr.phase_b_rule(_row({**every, "granite-278m": (0.060, False)}))["runs"] is False   # not established
    assert cr.phase_b_rule(_row({"granite-278m": (0.060, True)}))["runs"] is False              # rows still missing
    assert cr.phase_b_rule(_row({**every, "granite-278m": (0.060, True)}))["largest_gain"]["retriever"] == "granite-278m"


def test_the_readings_use_the_shipped_resamples():
    """e5-small minus BM25 here is the shipped large-sample row, to the last digit."""
    import czech_retrievers as cr

    shipped = json.loads((REPO / "results/czech/analysis.json").read_text(encoding="utf-8"))["large_retrieval"]
    mine = cr.readings(REPO)["populations"]
    for pop in cr.POPULATIONS:
        for m in ("ndcg10", "g3"):
            assert mine[pop]["differences"]["e5-small - bm25"][m] == shipped[pop]["all"]["deltas"]["e5-small - bm25"][m]
            assert mine[pop]["means"]["e5-small"][m] == shipped[pop]["all"]["means"]["e5-small"][m]


def test_model_bytes_count_the_weights_once_and_no_onnx_export(tmp_path, monkeypatch):
    """granite ships model.safetensors, pytorch_model.bin and model.onnx: one copy is what the bench loads."""
    import huggingface_hub

    from measurerag.retrievers import dense

    (tmp_path / "model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "pytorch_model.bin").write_bytes(b"x" * 100)
    (tmp_path / "model.onnx").write_bytes(b"x" * 100)
    (tmp_path / "tokenizer.json").write_bytes(b"x" * 7)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))
    assert dense._model_disk_bytes("org/model") == 107


def test_embeddinggemma_loads_with_its_two_dense_layers():
    """The repo's modules after pooling (two Dense, Normalize) must be in the encoder the bench uses, in float32."""
    pytest.importorskip("sentence_transformers")
    from huggingface_hub import snapshot_download

    try:
        snapshot_download("google/embeddinggemma-300m", local_files_only=True)
    except Exception:
        pytest.skip("EmbeddingGemma's weights are not in the cache (gated: the reader downloads them)")
    from measurerag.retrievers.dense import DenseRetriever

    r = DenseRetriever(model="embeddinggemma")
    enc = r._load_encoder()
    kinds = [type(m).__name__ for m in enc]
    assert kinds == ["Transformer", "Pooling", "Dense", "Dense", "Normalize"], kinds
    assert r.pooling_source == "repo"
    assert getattr(enc, "get_embedding_dimension", enc.get_sentence_embedding_dimension)() == 768
    assert next(enc.parameters()).dtype.__str__() == "torch.float32"
    assert enc.max_seq_length == 512
