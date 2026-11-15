"""The SQAD 3.2 reader, its windows, the debiasing flags and the sensitivity checks.
Offline, and on made-up text: no SQAD file is read here."""

from __future__ import annotations

from measurerag.czech_sensitivity import folded_f1, lemma_match, prefix_match, strip_diacritics, yes_no_in_czech, yes_no_variant
from measurerag.datasets.sqad32 import (
    answer_position,
    cut_window,
    debias_flags,
    find_ignoring_space,
    gold_answers,
    jaccard,
    locate,
    opens_article,
    parse_vert,
    sentences,
)
from measurerag.metrics import exact_match

VERT = "<s>\nKdo\tkdo\tk3\nvyhlásil\tvyhlásit\tk5\nDPMB\tDPMB\tkA\n<g/>\n,\t,\tkIx\nteď\tteď\tk6\n<g/>\n?\t?\tkIx.\n</s>\n<s desamb=\"1\">\nAno\tano\tk0\n</s>\n"


def test_parse_vert_glues_and_keeps_lemmas():
    p = parse_vert(VERT)
    assert p["text"] == "Kdo vyhlásil DPMB, teď?\nAno"
    assert p["lemmas"][1] == "vyhlásit" and p["words"][1] == "vyhlásil"


def test_a_window_is_whole_sentences_and_marks_a_cut():
    article = "\n".join(f"Věta číslo {i} je tady a má svou délku." for i in range(200))
    w = cut_window(article, "Věta číslo 100 je tady a má svou délku.", "t1")
    assert "Věta číslo 100 je" in w and w.startswith("... ") and w.endswith("...")
    assert len(w) <= 1010
    assert "Věta číslo 99 je" in w                        # a sentence before it, if one fits
    assert cut_window(article, "Věta číslo 100 je tady a má svou délku.", "t1") == w     # seeded
    assert not cut_window(article, "Věta číslo 0 je tady a má svou délku.", "t2").startswith("...")


def test_a_window_can_be_the_whole_of_a_short_article():
    assert cut_window("Jedna věta.\nDruhá věta.", "Druhá věta.", "t3") == "Jedna věta. Druhá věta."


def test_locate_and_opens_article_ignore_spacing():
    s = sentences("První, věta.\nDruhá   věta.\n")
    assert locate(s, "Druhá věta.") == (1, 1)
    assert locate(s, "Třetí") is None
    assert opens_article("První, věta.\nDruhá věta.", "První , věta.") is True
    assert opens_article("První, věta.\nDruhá věta.", "Druhá věta.") is False
    assert opens_article("První, věta.", "Jiná.") is None


def test_answer_position_finds_the_answer_after_its_sentence():
    ctx = "Metr je délka. Rok 1889 byl jiný. Metr byl zaveden roku 1889."
    assert answer_position(ctx, ["1889"]) == ctx.index("1889")
    assert answer_position(ctx, ["1889"], after=ctx.index("Metr byl")) == ctx.rindex("1889")
    assert answer_position(ctx, ["ano"]) is None             # a yes/no answer has no position
    assert find_ignoring_space("a b  c", "bc") == 2


def test_the_two_debiasing_rules():
    q = "kdy byl metr zaveden"
    ctx_long = " ".join(f"slovo{i}" for i in range(100))
    assert debias_flags(q, ctx_long, 49, 0.1)["early"] and debias_flags(q, ctx_long, 49, 0.1)["flagged"]
    assert not debias_flags(q, ctx_long, 50, 0.1)["flagged"]
    assert not debias_flags(q, ctx_long, None, 0.1)["flagged"]
    assert debias_flags(q, "metr byl zaveden", None, 0.1)["heavy"]
    assert jaccard("Kdy je tříska?", "kdy je triska") == 1.0
    assert jaccard("tříska", "triska", fold=False) == 0.0


def test_yes_no_golds_are_mapped_to_czech():
    assert gold_answers({"yes_no": "yes"}) == ["ano"] and gold_answers({"yes_no": "no"}) == ["ne"]
    assert gold_answers({"answer": "1774"}) == ["1774"]


def test_sensitivity_checks_are_kinder_than_exact_match():
    assert exact_match("v Praze", ["Praha"]) == 0.0
    assert prefix_match("Masarykovy univerzity", ["Masarykova univerzita"], n=5) == 1.0
    assert prefix_match("Praha", ["Plzeň"]) == 0.0
    assert prefix_match("Karel", ["Karlovy Vary"]) == 0.0          # a different token count is no match
    assert strip_diacritics("Příliš žluťoučký") == "Prilis zlutoucky"
    assert folded_f1("Prilis", ["příliš"]) == 1.0
    forms = {"praze": {"praha"}, "prahy": {"praha"}}
    assert exact_match("Praze", ["Prahy"]) == 0.0 and lemma_match("Praze", ["Prahy"], forms) == 1.0
    assert lemma_match("Brně", ["Prahy"], forms) == 0.0
    assert yes_no_variant("Yes.", ["ano"]) == 1.0 and yes_no_variant("no", ["ano"]) == 0.0


def test_the_yes_no_mapping_touches_a_whole_answer_only():
    assert yes_no_in_czech("Yes.") == "ano" and yes_no_in_czech("no") == "ne"
    assert yes_no_in_czech("No, it is not") == "No, it is not" and yes_no_in_czech("Praha") == "Praha"
    assert exact_match("Yes.", ["ano"]) == 0.0 and exact_match(yes_no_in_czech("Yes."), ["ano"]) == 1.0
    assert exact_match(yes_no_in_czech("no"), ["ano"]) == 0.0


def test_a_placeholder_lemma_does_not_make_two_tokens_agree():
    # the vertical files give every number and every URL the same lemma
    forms = {"1998": {"[number]", "#num#"}, "2001": {"[number]", "#num#"}, "metrů": {"metr"}, "metru": {"metr"}, "http://a.cz": {"[url]"}, "http://b.cz": {"[url]"}}
    assert lemma_match("1998", ["2001"], forms) == 0.0
    assert lemma_match("http://a.cz", ["http://b.cz"], forms) == 0.0
    assert lemma_match("7 metrů", ["15 metru"], forms) == 0.0
    assert lemma_match("15 metrů", ["15 metru"], forms) == 1.0
    assert lemma_match("1998", ["1998"], forms) == 1.0
