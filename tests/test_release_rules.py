"""The documentation rules added for the public release: token names with "->"
and "+", the 3B licence line after every table with a 3B row, and the dash and
leftover-placeholder checks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fill  # noqa: E402
from check_docs import check  # noqa: E402
from test_check_docs import make_repo  # noqa: E402


def test_token_names_may_name_a_step_or_a_configuration():
    tokens = {"hotpotqa.step.tfidf-svd->bm25.k5.ret": "+0.214", "runs.cz_rag.e5-small+qwen2.5-1.5b.em": "0.290"}
    text, missing = fill.fill("a {{hotpotqa.step.tfidf-svd->bm25.k5.ret}} b {{ runs.cz_rag.e5-small+qwen2.5-1.5b.em }}", tokens)
    assert missing == [] and text == "a +0.214 b 0.290"


def test_a_table_with_a_3b_row_gets_the_licence_line_once():
    table = "| configuration | EM |\n|---|---|\n| bm25+qwen2.5-3b | 0.302 |\n"
    once = fill.add_3b_labels("Intro.\n\n" + table + "\nAfter.\n", "docs/X.md")
    assert once.count(fill.LABEL_3B_MARK) == 1
    assert "(../THIRD_PARTY.md)" in once
    assert fill.add_3b_labels(once, "docs/X.md") == once            # idempotent
    assert "\n\n*Qwen2.5 3B rows:" in once and ".*\n\nAfter." in once  # a paragraph of its own
    pair = "| configuration | EM 1.5B | EM 3B |\n|---|---|---|\n| bm25 | 0.24 | 0.30 |\n"
    assert fill.LABEL_3B_MARK in fill.add_3b_labels(pair, "README.md")   # a bare 3B beside Qwen2.5's 1.5B


def test_a_table_without_3b_and_a_table_in_code_are_left_alone():
    plain = "| configuration | EM |\n|---|---|\n| bm25+1.5b | 0.242 |\n"
    fenced = "```\n| x | qwen2.5-3b |\n|---|---|\n```\n"
    ministral = "| configuration | EM 3B | EM 8B |\n|---|---|---|\n| bm25+ministral-3-3b | 0.36 | 0.43 |\n"
    assert fill.add_3b_labels(plain, "README.md") == plain
    assert fill.add_3b_labels(fenced, "README.md") == fenced
    assert fill.add_3b_labels(ministral, "README.md") == ministral    # Ministral 3 3B is Apache 2.0


def test_check_docs_reports_an_unlabelled_3b_table_and_a_dash(tmp_path):
    guide = "# Guide\n\n| configuration | EM |\n|---|---|\n| bm25+qwen2.5-3b | 0.302 |\n\nA range 1\u20132.\n"
    root = make_repo(tmp_path, docs={"docs/GUIDE.md": guide})
    problems = check(root)
    assert any("has 3B rows but no Qwen Research License line" in p for p in problems)
    assert any("en or em dash" in p for p in problems)


def test_check_docs_reports_a_placeholder_the_token_pattern_cannot_read(tmp_path):
    root = make_repo(tmp_path, docs={"docs/GUIDE.md": "# Guide\n\nEM was {{ a b }} here.\n"})
    assert any("unfilled placeholder" in p for p in check(root))


def test_sqad_licence_line_is_the_one_lindat_states():
    """SQAD 3.0 is LGPL-3.0 on LINDAT. The live documents say so, with the handle,
    and the string written before the check is gone from every results file."""
    from measurerag.datasets.sqad import LICENCE, LICENCE_NOTE

    handle = "hdl.handle.net/11234/1-3069"
    assert "LGPL-3.0" in LICENCE and handle in LICENCE and handle in LICENCE_NOTE
    for rel in ("README.md", "THIRD_PARTY.md", "docs/LIMITATIONS.md", "docs/CROSSOVER.md",
                "docs/DATASETS_AND_LICENCES.md", "results/cz_rag/results.json", "results/cz_rag/manifest.json",
                "results/stats.json"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "LGPL-3.0" in text, rel
        assert "not been verified" not in text, rel
    for rel in ("README.md", "docs/LIMITATIONS.md", "docs/CROSSOVER.md", "docs/DATASETS_AND_LICENCES.md",
                "results/cz_rag/results.json", "results/cz_rag/manifest.json", "results/stats.json"):
        assert "BY-NC-SA" not in (REPO / rel).read_text(encoding="utf-8"), rel


def test_public_text_names_the_filtered_copy_and_scopes_the_czech_claims():
    """The copy by repository id and revision, its count and filter; LINDAT's licence, not the
    copy's label; the diacritics finding scoped; Czech exact match mostly an under-count, not strictly a lower bound, with why."""
    import json

    c = json.loads((REPO / "results/czech_checks.json").read_text(encoding="utf-8"))
    fl = c["filter_read_from_the_comparison"]
    assert sum(v["kept"] for v in fl.values()) == c["copy"]["rows"] == 7663
    assert sum(v["kept"] + v["dropped"] for v in fl.values()) == c["archive"]["records"] == 13476
    for rel in ("README.md", "docs/DATASETS_AND_LICENCES.md", "THIRD_PARTY.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert c["copy"]["repo_id"] in text and c["copy"]["revision"] in text, rel
        assert "7,663" in text and "13,476" in text and "5,798" in text, rel
        assert "LGPL-3.0" in text, rel
    limits = " ".join((REPO / "docs/LIMITATIONS.md").read_text(encoding="utf-8").split())
    assert c["copy"]["repo_id"] in limits and c["copy"]["revision"] in limits and "LGPL-3.0" in limits
    assert "Czech exact match mostly under-counts, but is not strictly a lower bound" in limits and "ano or ne" in limits and "declension" in limits
    for rel in ("README.md", "docs/LIMITATIONS.md"):
        text = " ".join((REPO / rel).read_text(encoding="utf-8").split())
        assert "e5-small leads" in text and "at large samples" in text, rel
    for rel in ("README.md", "docs/LIMITATIONS.md", "docs/CROSSOVER.md"):
        text = " ".join((REPO / rel).read_text(encoding="utf-8").split())
        assert "under declension" not in text and "ano or ne" in text, rel


def test_the_czech_bench_config_says_the_size_the_results_have():
    """v0.2.0's config capped the slice at 500 while its results came from 200."""
    import json

    import yaml

    cfg = yaml.safe_load((REPO / "configs/cz_rag.yaml").read_text(encoding="utf-8"))
    results = json.loads((REPO / "results/cz_rag/results.json").read_text(encoding="utf-8"))
    n = {int(x["answers"]["n_answered"]) for x in results["results"] if x.get("status") == "ok"}
    assert n == {cfg["dataset"]["max_queries"]} == {200}


def test_every_bench_config_says_the_caps_its_results_were_measured_with():
    """configs/cz_small.yaml once said 100,000 pages and 2,000 claims while results/cz_small came from 20,000 and 500."""
    import json

    import yaml

    for name in ("cz_small", "cz_full", "cz_rag", "cz_rag_32", "cz_v040_timed", "en_small", "en_nfcorpus", "en_fiqa", "en_rag"):
        cfg = yaml.safe_load((REPO / f"configs/{name}.yaml").read_text(encoding="utf-8"))["dataset"]
        ran = json.loads((REPO / f"results/{name}/manifest.json").read_text(encoding="utf-8"))["config"]["dataset"]
        for key in ("name", "max_docs", "max_queries"):
            assert cfg.get(key) == ran.get(key), (name, key, cfg.get(key), ran.get(key))


def test_limitations_quote_nothing_from_the_comparison_with_the_archive():
    """Limitations keeps only numbers the tree can recompute: of the check's values it may
    name the copy and its revision, which are identifiers, and nothing else."""
    import re

    template = (REPO / "templates/docs/LIMITATIONS.md").read_text(encoding="utf-8")
    tokens = set(re.findall(r"\{\{(czech_checks\.[a-z_]+)\}\}", template))
    assert tokens == {"czech_checks.copy_repo", "czech_checks.copy_revision"}


def test_the_sqad_loader_pins_the_revision_the_documents_name(monkeypatch):
    import json

    import huggingface_hub
    import pytest
    from measurerag.datasets import sqad
    from measurerag.datasets.base import DatasetUnavailable

    c = json.loads((REPO / "results/czech_checks.json").read_text(encoding="utf-8"))
    assert (sqad.MIRROR, sqad.REVISION) == (c["copy"]["repo_id"], c["copy"]["revision"])
    seen = {}

    def no_network(repo_id, filename, **kw):
        seen.update(kw, repo_id=repo_id)
        raise OSError("no network in this test")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    with pytest.raises(DatasetUnavailable):
        sqad.load()
    assert seen["repo_id"] == c["copy"]["repo_id"] and seen["revision"] == c["copy"]["revision"]


def test_no_ministral_3_configuration_in_the_english_text_carries_chinas_flag():
    """The flagger reads a bare "+ 3B" as Qwen2.5's, which is right in the Czech tables only: the English sections name
    Ministral 3's configurations in full."""
    import re

    for rel, end in (("README.md", "\n## Czech\n"), ("docs/BUDGETS.md", None), ("docs/CROSSOVER.md", "\n## Appendix: Czech")):
        text = (REPO / rel).read_text(encoding="utf-8")
        english = text if end is None else text[: text.index(end)]
        assert not re.search(r"\+ (?:3|8)B \U0001F1E8\U0001F1F3", english), rel
