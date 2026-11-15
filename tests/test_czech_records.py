"""The Czech records, the re-score and the comparison with the archive.

Offline. The comparison runs on a tiny made-up archive, never on SQAD text. The re-score test runs
the shipped script on the shipped records and compares its output with the tree, byte for byte.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from measurerag.czech_labels import answer_tokens, answer_type, length_band, spaced_punct  # noqa: E402
from measurerag.czech_store import answers_path, read_jsonl_gz, write_jsonl_gz  # noqa: E402

RECORDS = REPO / "results/czech/records"
ANSWERS = REPO / "answers"
FIELDS = {"set", "q", "config", "kind", "p", "max_tokens", "seed", "context", "answer", "em", "f1", "f1_folded", "prefix5", "em_yes_no_mapped"}


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gold,kind", [
    (["ano"], "yes/no"), (["Ne"], "yes/no"), (["1889"], "date"), (["12. ledna 1900"], "date"), (["1. 5. 1900"], "date"),
    (["42"], "number"), (["3,5"], "number"), (["Praha"], "single word"), (["Karel Čapek"], "multi-word name"), (["Univerzita Karlova v Praze"], "multi-word name"),
    (["velmi dlouhá odpověď"], "other"), ([], "other"),
])
def test_answer_types_by_simple_rules(gold, kind):
    assert answer_type(gold) == kind


def test_lengths_and_spacing():
    assert answer_tokens(["Karel Čapek"]) == 2 and answer_tokens([]) == 0
    assert [length_band(n) for n in (1, 2, 3, 4, 5, 6, 40)] == ["1", "2", "3", "4-5", "4-5", "6+", "6+"]
    assert spaced_punct("U . S . A .") and spaced_punct("Jean - Paul") and not spaced_punct("Jean-Paul") and not spaced_punct("1. ledna")


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
def test_gzip_records_are_byte_reproducible(tmp_path):
    rows = [{"q": "1", "answer": "Příliš žluťoučký"}, {"q": "2", "answer": "ano"}]
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    write_jsonl_gz(a, rows)
    write_jsonl_gz(b, rows)
    assert a.read_bytes() == b.read_bytes() and read_jsonl_gz(a) == rows


def test_scores_are_recomputed_from_an_answer_and_a_gold():
    from rescore_czech import verify_rows
    from czech_records import scores

    gold = {"1": ["ano"], "2": ["Masarykova univerzita"], "3": ["Praha"]}
    rows = [{"q": "1", "answer": "Ano.", **scores("Ano.", gold["1"])}, {"q": "2", "answer": "Masarykovy univerzity", **scores("Masarykovy univerzity", gold["2"])},
            {"q": "3", "answer": "Yes", **scores("Yes", gold["3"])}, {"q": "not in the gold", "answer": "x", "em": 1, "f1": 1.0, "f1_folded": 1.0, "prefix5": 1, "em_yes_no_mapped": 1}]
    assert rows[0]["em"] == 1 and rows[1]["em"] == 0 and rows[1]["prefix5"] == 1
    assert verify_rows(rows, gold) == {"rows": 3, "mismatches": 0, "rows_with_no_gold": 1}   # counted, and fails --verify-scores
    rows[2]["em"] = 1                  # a score that does not follow from the answer
    assert verify_rows(rows, gold) == {"rows": 3, "mismatches": 1, "rows_with_no_gold": 1}


def test_a_verdict_follows_its_stated_rule():
    from rescore_czech import verdict_sign

    def e(point, established=True):
        return {"point": point, "established": established}

    assert verdict_sign(e(0.09), e(0.135)) == "HOLDS"
    assert verdict_sign(e(0.09), e(0.04)) == "WEAKENS"                 # same sign, under half the size
    assert verdict_sign(e(0.09), e(0.09, False)) == "WEAKENS"          # no longer established
    assert verdict_sign(e(0.09, False), e(0.09, False)) == "HOLDS"     # it never was
    assert verdict_sign(e(0.09), e(-0.09)) == "REVERSES"
    assert verdict_sign(e(0.09), e(-0.09, False)) == "WEAKENS"         # the other sign, not established


def test_the_bench_and_the_dial_must_give_the_same_string():
    """Only an answer the dial's cap could have cut may differ, and then only by being cut."""
    from sqad32_backfill import agreement

    rows = [("ostrov", "ostrov", 3), ("dlouhá odpověď", "dlouhá", 63), ("ostrov", "unanswerable", 3), ("dlouhá odpověď", "jiná", 63)]
    assert agreement(rows) == {"questions": 4, "identical": 1, "cut_by_the_cap": 1, "different": 2}


def test_a_question_is_paired_only_if_it_sits_in_both_splits_and_in_a_set():
    from czech_same_questions import pairs_in_a_set

    questions = {"copy": {"a1": "Kdo?", "a2": "Kde? ", "a3": "Kdy?", "a4": "Proč?"},
                 "sqad32": {"b1": "Kdo?", "b2": "Kde?", "b3": "Jak?", "b4": "Proč?"}}
    assert pairs_in_a_set(questions, {"copy": {"a1", "a3"}, "sqad32": {"b2", "b3"}}) == [("a1", "b1"), ("a2", "b2")]


def test_the_gap_on_part_of_two_sets_counts_only_that_part():
    from types import SimpleNamespace

    import numpy as np
    import rescore_czech as rc

    def sd(em, labels):
        qids = [f"q{i}" for i in range(len(em))]
        rows = [{"em": e} for e in em]
        return SimpleNamespace(config_rows=lambda c: rows, qids=qids, labels=dict(zip(qids, labels)),
                               counts=np.random.default_rng(1).multinomial(len(em), [1 / len(em)] * len(em), size=500).astype(float))

    copy = sd([0, 0, 1, 1], [{"k": 1}, {"k": 1}, {"k": 0}, {"k": 0}])
    target = sd([1, 1, 0, 0], [{"k": 1}, {"k": 1}, {"k": 0}, {"k": 0}])
    out = rc.subset_gap(copy, target, "c", lambda l: l["k"] == 1, lambda l: l["k"] == 1)
    assert (out["n_copy"], out["n_target"], out["em_copy"], out["em_target"], out["gap"]["point"]) == (2, 2, 0.0, 1.0, 1.0)
    assert rc.subset_gap(copy, target, "c", lambda l: False, lambda l: True) is None
    assert [rc.passage_band(n) for n in (499, 500, 799, 800)] == ["under 500", "500 to 799", "500 to 799", "800 or more"]


def test_yes_mapped_to_ano_is_a_labelled_secondary_metric_only():
    from czech_records import scores

    s = scores("yes", ["ano"])
    assert s["em"] == 0 and s["em_yes_no_mapped"] == 1


# --------------------------------------------------------------------------
# the comparison with the archive, on a made-up archive
# --------------------------------------------------------------------------
def _vert(words: list[str]) -> str:
    return "<s>\n" + "".join(f"{w}\t{w.lower()}\tk1\n" for w in words) + "</s>\n"


def _make_archive(path: Path) -> None:
    """Five records: two open their article, two come later, one is not located anywhere."""
    records = {
        "000001": ("Kdo je první", "První věta článku", "První", "První věta článku"),
        "000002": ("Kdo je druhý", "První věta článku", "Druhá", "Druhá věta článku"),
        "000003": ("Kdo je třetí", "První věta článku", "První", "První věta článku"),
        "000004": ("Kdo je čtvrtý", "První věta článku", "Třetí", "Třetí věta článku"),
        "000005": ("Kdo je pátý", "První věta článku", "Nikde", "Tato věta nikde není"),
    }
    article = "První věta článku .\nDruhá věta článku .\nTřetí věta článku .\n"
    text = "".join("<s>\n" + "".join(f"{w}\t{w.lower()}\tk1\n" for w in line.split()) + "</s>\n" for line in article.strip().split("\n"))
    with tarfile.open(path, "w:xz") as tf:
        def add(name: str, body: str) -> None:
            data = body.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        for rid, (question, _, answer, selection) in records.items():
            add(f"data/{rid}/01question.vert", _vert(question.split()))
            add(f"data/{rid}/03text.vert", text)
            add(f"data/{rid}/05metadata.txt", "<q_type>ENTITY</q_type>\n<a_type>ENTITY</a_type>\n")
            add(f"data/{rid}/06answer.selection.vert", _vert(selection.split()))
            add(f"data/{rid}/09answer_extraction.vert", _vert([answer]))


def _make_copy(folder: Path) -> None:
    """The copy keeps one of the two opening records and both later ones; one row has another question."""
    import pandas as pd

    folder.mkdir(parents=True, exist_ok=True)

    def row(i, q, ctx):
        return {"id": i, "question": q, "context": ctx, "answers": {"text": ["x"], "answer_start": [0]}}

    pd.DataFrame([row("000001", "Kdo je první", "Kdo je první člověk"), row("000002", "Kdo je druhý", "Slova která se neshodují vůbec")]).to_parquet(folder / "train-00000-of-00001.parquet")
    pd.DataFrame([row("000004", "Kdo je čtvrtý", "Další slova která se neshodují")]).to_parquet(folder / "validation-00000-of-00001.parquet")
    pd.DataFrame([row("000003", "Jiná otázka", "Ještě další slova")]).to_parquet(folder / "test-00000-of-00001.parquet")


def test_the_copy_is_compared_with_a_made_up_archive(tmp_path, capsys):
    import sqad_copy_vs_archive as cmp

    archive = tmp_path / "fake.tar.xz"
    _make_archive(archive)
    _make_copy(tmp_path / "copy")
    checks = tmp_path / "checks.json"
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    common = ["--archive", str(archive), "--copy-dir", str(tmp_path / "copy"), "--expect-md5", digest, "--checks", str(checks)]
    assert cmp.main([*common, "--write"]) == 0
    out = json.loads(checks.read_text(encoding="utf-8"))
    assert out["archive"]["records"] == 5 and out["archive"]["md5"] == digest
    assert out["copy"]["rows"] == 4 and out["copy"]["rows_with_same_id_and_question_in_the_archive"] == 3
    fl = out["filter_read_from_the_comparison"]
    assert fl["answer_sentence_opens_the_article"] == {"kept": 1, "dropped": 1}     # the row with another question is not kept
    assert fl["answer_sentence_comes_later"] == {"kept": 2, "dropped": 0}
    assert fl["answer_sentence_not_located"] == {"kept": 0, "dropped": 1}
    assert 0 < out["thresholds"]["archive_p90"] <= 1 and 0 < out["thresholds"]["copy_p90"] <= 1
    assert cmp.main(common) == 0                                                      # a second run agrees
    checks.write_text(checks.read_text(encoding="utf-8").replace('"records": 5', '"records": 6'), encoding="utf-8")
    assert cmp.main(common) == 1                                                      # a changed number is caught
    assert cmp.main([*common[:4], "--expect-md5", "0" * 32, "--checks", str(checks)]) == 1   # so is another file
    assert "not the archive" in capsys.readouterr().out


def test_the_text_scan_finds_what_it_looks_for_on_made_up_text(tmp_path):
    from sqad_text_scan import WINDOW, grams, pool_hits, read_file, words

    question = tuple(words("Kdo napsal knihu o válce s mloky"))
    sentence = words("Tady stojí Kdo napsal knihu o válce s mloky a nic dalšího")
    assert list(pool_hits(sentence, " " + " ".join(sentence) + " ", {question})) == [question]
    other = words("Tady stojí něco úplně jiného a nic dalšího")
    assert not list(pool_hits(other, " " + " ".join(other) + " ", {question}))
    passage = grams(words("jedna dvě tři čtyři pět šest sedm osm devět deset"), WINDOW)
    assert grams(words("x jedna dvě tři čtyři pět šest sedm osm y"), WINDOW) & passage
    assert not grams(words("jedna dvě tři čtyři pět šest sedm x"), WINDOW) & passage
    # an answer is counted apart from the rest of a record
    rec = tmp_path / "answers_x.jsonl.gz"
    write_jsonl_gz(rec, [{"q": "1", "answer": "Odpověď", "note": "poznámka"}])
    other_text, answers = read_file(rec)
    assert sorted(other_text) == sorted(["q", "1", "note", "poznámka"]) and answers == ["Odpověď"]
    # only the shipped answer files keep an answer apart; anywhere else it must hold no SQAD text either
    from sqad_text_scan import parts_to_check
    assert parts_to_check("answers/sqad32/answers_x.jsonl.gz", other_text, answers) == (other_text, ["Odpověď"])
    assert parts_to_check("results/czech/x.jsonl.gz", other_text, answers) == ([*other_text, "Odpověď"], [])


# --------------------------------------------------------------------------
# what ships
# --------------------------------------------------------------------------
shipped = pytest.mark.skipif(not (RECORDS / "sets.json").exists(), reason="no records in this tree yet")

# Addresses an answer may hold, by SHA-256: the two example addresses of a SQAD passage on XMPP (Juliet's and Romeo's, as the
# protocol's own documents name their examples), which two models quoted from that passage in the cross-version check.
QUOTED_EXAMPLE_ADDRESSES = {"216eab1e2b20f0fea7e3e56e8b6207cf40b32d51ce2f34b265e5c601f24f879e", "8026e96db376ba70985f9ddb78adea7c3e4e9706e56f9c30ce28a2a1bee4c77f"}


@shipped
def test_every_record_file_is_ids_and_scores():
    """No question, context, document or gold answer. Only these fields, and a generated answer."""
    files = sorted(ANSWERS.rglob("*.jsonl.gz"))
    assert files
    for path in files:
        for r in read_jsonl_gz(path):
            extra = set(r) - FIELDS - {"leak", "leak_span_only", "em_other_key", "corpus", "other_q", "from", "type", "gold_in_top3", "variant", "model", "device", "seconds", "completion_tokens", "load_seconds"}
            assert not extra and FIELDS <= set(r), (path.name, extra)
            assert all(re.fullmatch(r"ctx-[0-9a-f]{16}", d) for d in r["context"])
            addresses = {hashlib.sha256(x.encode()).hexdigest() for x in re.findall(r"[\w.]+@[\w.]+\.\w+", r["answer"])}
            assert addresses <= QUOTED_EXAMPLE_ADDRESSES, (path.name, r["q"], r["config"])
            assert not re.search(r"/(Users|home)/\w", r["answer"]), path.name
    for path in sorted(RECORDS.glob("rankings_*.jsonl.gz")):
        for r in read_jsonl_gz(path)[:50]:
            for k, v in r.items():
                assert k == "q" or all(re.fullmatch(r"ctx-[0-9a-f]{16}", d) for d in v), (path.name, k)


@shipped
def test_no_record_file_is_over_the_release_limit():
    assert max(p.stat().st_size for p in [*RECORDS.iterdir(), *ANSWERS.rglob("*.gz")]) < 5_000_000


@shipped
def test_each_answers_folder_carries_its_licence_and_attribution():
    from measurerag.czech_store import SOURCES

    assert (ANSWERS / "README.md").exists()
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in (ANSWERS / "sqad30_copy/LICENSE").read_text(encoding="utf-8")
    assert "Attribution-ShareAlike 3.0" in (ANSWERS / "sqad32/LICENSE").read_text(encoding="utf-8")
    assert "11234/1-3069" in (ANSWERS / "sqad30_copy/NOTICE.md").read_text(encoding="utf-8")
    assert "11234/1-5019" in (ANSWERS / "sqad32/NOTICE.md").read_text(encoding="utf-8")
    assert {p.name for p in ANSWERS.iterdir() if p.is_dir()} == set(SOURCES)


def test_an_answers_file_is_found_by_its_root(tmp_path, monkeypatch):
    """A file beside sets.json comes first; a first-cut root reads the first cut's folder before the current one."""
    import measurerag.czech_store as store

    monkeypatch.setattr(store, "REPO", tmp_path)
    for rel in ("answers/sqad32/answers_x.jsonl.gz", "answers/sqad32/first_cut/answers_x.jsonl.gz", "answers/sqad30_copy/answers_y.jsonl.gz"):
        write_jsonl_gz(tmp_path / rel, [{"q": rel}])
    now, first = tmp_path / "now", tmp_path / "first"
    now.mkdir()
    first.mkdir()
    (now / "sets.json").write_text('{"sets": {}}', encoding="utf-8")
    (first / "sets.json").write_text('{"sets": {}, "first_cut": true}', encoding="utf-8")
    assert store.load_answers(now, "x")[0]["q"] == "answers/sqad32/answers_x.jsonl.gz"
    assert store.load_answers(first, "x")[0]["q"] == "answers/sqad32/first_cut/answers_x.jsonl.gz"
    assert store.load_answers(first, "y")[0]["q"] == "answers/sqad30_copy/answers_y.jsonl.gz"
    write_jsonl_gz(now / "answers_x.jsonl.gz", [{"q": "beside"}])
    assert store.load_answers(now, "x")[0]["q"] == "beside"
    with pytest.raises(FileNotFoundError):
        store.load_answers(now, "z")


@shipped
def test_every_set_named_in_sets_json_is_complete():
    sets = json.loads((RECORDS / "sets.json").read_text(encoding="utf-8"))["sets"]
    for key, meta in sets.items():
        assert answers_path(RECORDS, f"answers_{key}.jsonl.gz").exists()
        assert meta["complete"] and meta["missing"] == 0, key


@shipped
def test_rescore_recomputes_every_czech_number_byte_for_byte():
    import subprocess

    p = subprocess.run([sys.executable, "scripts/rescore_czech.py", "--check"], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


@shipped
@pytest.mark.parametrize("script", ["scripts/czech_pilot_check.py", "scripts/czech_model_bench.py", "scripts/czech_model_bench_mistral.py", "scripts/czech_dev_prompts.py"])
def test_the_dev_set_results_recompute_from_the_shipped_records(script):
    """The pilot on the GX10, the model benchmark and the prompt's selection: no model, no cache, no dataset."""
    import subprocess

    p = subprocess.run([sys.executable, script, "--check"], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


@shipped
def test_the_czech_block_of_crossover_json_comes_from_the_records():
    import yaml

    cfg = yaml.safe_load((REPO / "configs/crossover.yaml").read_text(encoding="utf-8"))
    czech = cfg["datasets"]["czech"]
    assert czech["records"] == "results/czech/records" and czech["set"] == "released" and czech["harness"] == []
