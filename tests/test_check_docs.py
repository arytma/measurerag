"""scripts/check_docs.py on a tiny fake repository, one rule per test.

Each test builds configs/docs.yaml, results/stats.json, a template and the
document scripts/fill.py generates from it in tmp_path, then calls
check_docs.check() directly. Nothing here touches the real documentation set.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_docs  # noqa: E402
import fill  # noqa: E402

TOKENS = {
    "em_base": "0.412",
    "n": "200",
    "gstar": "g* = 0.46 [0.30, 0.61]",
    "gens": "30,000",
}

README_TEMPLATE = """# Demo

EM was {{em_base}} on {{n}} questions, and {{gstar}}, under a cap of {{gens}} generations.

Released 14 Sep 2026 as version 0.2.0. In 2026 it was tested on Python 3.12; the data
are under CC BY-SA 4.0. There are 3 retrievers - e5-small among them - two generators,
Qwen2.5-1.5B and Qwen2.5-3B, and the p95 latency is reported.
The flag `--n-boot 10000` sets the number of resamples.

```bash
measurerag run --max-queries 12345
```

See [the guide](docs/GUIDE.md) and [its method](docs/GUIDE.md#the-method).
"""

GUIDE = """# Guide

## The method

Run the kit; EM was 0.412 here too, on 200 questions.
"""

LOG = """# Log

N1 (13 Sep): EM was 0.731 when this was decided.
"""

REPORT = """# Run report

## Run 1

EM was 0.731 then.

# Run 2

EM was 0.412 now.
"""

DOCS_YAML = {
    "stats": "results/stats.json",
    "templates": "templates",
    "documents": ["README.md", "docs/GUIDE.md", "LOG.md", "REPORT.md"],
    "number_rule_exempt": {"LOG.md": "a dated log"},
    "number_rule_from_heading": {"REPORT.md": "# Run 2"},
    "small_integer_max": 10,
}


def make_repo(
    root: Path,
    *,
    template: str = README_TEMPLATE,
    docs: dict[str, str] | None = None,
    tokens: dict[str, str] | None = None,
    cfg: dict | None = None,
    generate: bool = True,
) -> Path:
    """Write the fake repository; generate README.md from its template unless told not to."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "docs.yaml").write_text(yaml.safe_dump(cfg or DOCS_YAML), encoding="utf-8")
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "stats.json").write_text(
        json.dumps({"source": "results/crossover.json", "tokens": TOKENS if tokens is None else tokens}),
        encoding="utf-8",
    )
    (root / "templates").mkdir(exist_ok=True)
    (root / "templates" / "README.md").write_text(template, encoding="utf-8")
    hand = {"docs/GUIDE.md": GUIDE, "LOG.md": LOG, "REPORT.md": REPORT}
    hand.update(docs or {})
    for rel, text in hand.items():
        if text is None:
            continue
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    if generate:
        rendered, missing = fill.render_all(root)
        assert not missing, missing
        for path, text in rendered.items():
            path.write_text(text, encoding="utf-8")
    return root


def check(root: Path, skip_exec: bool = True) -> list[str]:
    return check_docs.check(root, skip_exec=skip_exec, timeout=30)


# -- the good fixture ----------------------------------------------------------
def test_good_fixture_passes(tmp_path):
    root = make_repo(tmp_path)
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("<!-- Generated from templates/README.md")
    assert "EM was 0.412 on 200 questions" in readme
    assert check(root) == []
    # the untagged ```bash block is not executed (it would fail: no such command)
    assert check(root, skip_exec=False) == []


@pytest.mark.parametrize(
    "phrase",
    [
        "EM was 0.412.",                          # rendered from a token
        "The cap is 30,000 generations.",         # token value with a thousands separator
        "Measured in 2026.",                      # a year
        "Run on 14 Sep.",                         # a date
        "Run on Sep 14, 2026.",                   # a date, month first
        "Run at 03:15 UTC.",                      # a time
        "Version 0.2.0 of the kit.",              # a version string
        "Tested on Python 3.12.",                 # a software version
        "Licensed CC BY-SA 4.0.",                 # a licence version
        "There are 3 retrievers.",                # a small counting integer
        "Pass `--n-boot 10000` to change it.",    # inline code
        "```text\nn_boot = 10000 and 0.999\n```", # a fenced block
        "The e5-small retriever.",                # identifiers with digits
        "The Qwen2.5-1.5B generator.",
        "The p95 latency.",
        "<!-- 0.999 in a comment -->",            # an HTML comment
    ],
)
def test_exempt_and_traced_numbers_pass(tmp_path, phrase):
    root = make_repo(tmp_path, docs={"docs/GUIDE.md": f"# Guide\n\n## The method\n\n{phrase}\n"})
    assert check(root) == []


# -- tokens and generated documents --------------------------------------------
def test_unfilled_token_fails(tmp_path):
    root = make_repo(tmp_path, docs={"docs/GUIDE.md": "# Guide\n\n## The method\n\nEM was {{em_base}}.\n"})
    problems = check(root)
    assert problems == ["docs/GUIDE.md: unfilled token {{em_base}}"]


def test_template_token_missing_from_stats_fails(tmp_path):
    root = make_repo(tmp_path)
    (root / "templates" / "README.md").write_text(README_TEMPLATE + "\nAnd {{nowhere}}.\n", encoding="utf-8")
    problems = check(root)
    assert any(p.startswith("templates/README.md: tokens missing") and "nowhere" in p for p in problems), problems


def test_generated_document_edited_by_hand_fails(tmp_path):
    root = make_repo(tmp_path)
    readme = root / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace("0.412", "0.413"), encoding="utf-8")
    problems = check(root)
    assert any(p.startswith("README.md: differs from its template") for p in problems), problems


def test_generated_document_never_generated_fails(tmp_path):
    root = make_repo(tmp_path, generate=False)
    problems = check(root)
    assert "README.md: missing" in problems
    assert any(p.startswith("README.md: differs from its template") for p in problems), problems


# -- the number rule -------------------------------------------------------------
def test_untraced_number_fails(tmp_path):
    root = make_repo(tmp_path, docs={"docs/GUIDE.md": "# Guide\n\n## The method\n\nEM was 0.731.\n"})
    problems = check(root)
    assert len(problems) == 1
    assert problems[0].startswith("docs/GUIDE.md: number not in results/stats.json: '0.731'")


def test_untraced_number_in_an_exempt_document_passes(tmp_path):
    # LOG.md quotes 0.731 in the default fixture; make it unambiguous
    root = make_repo(tmp_path, docs={"LOG.md": "# Log\n\nEM was 0.731, and 17.5 on 12 runs.\n"})
    assert check(root) == []


def test_untraced_number_in_a_generated_document_fails(tmp_path):
    root = make_repo(tmp_path, template=README_TEMPLATE + "\nEM was 0.731 by hand.\n")
    problems = check(root)
    assert any(p.startswith("README.md: number not in") and "'0.731'" in p for p in problems), problems


def test_large_integer_not_in_stats_fails(tmp_path):
    root = make_repo(tmp_path, docs={"docs/GUIDE.md": "# Guide\n\n## The method\n\nWe ran 11 configurations.\n"})
    problems = check(root)
    assert len(problems) == 1 and "'11'" in problems[0]


def test_number_before_the_from_heading_passes(tmp_path):
    root = make_repo(tmp_path, docs={"REPORT.md": REPORT})
    assert check(root) == []


def test_number_after_the_from_heading_fails(tmp_path):
    root = make_repo(tmp_path, docs={"REPORT.md": REPORT + "\nAnd EM was 0.731 by hand.\n"})
    problems = check(root)
    assert len(problems) == 1
    assert problems[0].startswith("REPORT.md: number not in results/stats.json: '0.731'")


def test_from_heading_on_the_first_line_still_checks_the_document(tmp_path):
    root = make_repo(tmp_path, docs={"REPORT.md": "# Run 2\n\nEM was 0.731 by hand.\n"})
    assert check(root) != []


@pytest.mark.parametrize("phrase", ["Peak RSS was 1.81GB.", "EM ranged 0.30-0.99.", "EM was 0.731/0.512."])
def test_numbers_glued_to_units_or_ranges_are_checked(tmp_path, phrase):
    root = make_repo(tmp_path, docs=_guide_with(phrase))
    assert check(root) != []


# -- links -------------------------------------------------------------------------
def _guide_with(line: str) -> dict[str, str]:
    return {"docs/GUIDE.md": f"# Guide\n\n## The method\n\n{line}\n"}


def test_broken_relative_link_fails(tmp_path):
    root = make_repo(tmp_path, docs=_guide_with("See [nothing](NOPE.md)."))
    assert check(root) == ["docs/GUIDE.md: broken link NOPE.md (no such file)"]


def test_link_to_an_existing_file_passes(tmp_path):
    root = make_repo(tmp_path, docs=_guide_with("See [the readme](../README.md) and [the config](../configs/docs.yaml)."))
    assert check(root) == []


def test_link_to_a_missing_anchor_fails(tmp_path):
    root = make_repo(tmp_path, docs=_guide_with("See [run three](../REPORT.md#run-three)."))
    assert check(root) == ["docs/GUIDE.md: broken link ../REPORT.md#run-three (no heading #run-three)"]


def test_link_to_an_existing_anchor_passes(tmp_path):
    root = make_repo(tmp_path, docs=_guide_with(
        "See [run 2](../REPORT.md#run-2), [one](../REPORT.md#run-1) "
        "and [here](#the-method)."))
    assert check(root) == []


def test_link_to_a_missing_anchor_in_the_same_document_fails(tmp_path):
    root = make_repo(tmp_path, docs=_guide_with("See [below](#the-results)."))
    assert check(root) == ["docs/GUIDE.md: broken link #the-results (no heading #the-results)"]


def test_html_image_with_a_missing_source_fails(tmp_path):
    (tmp_path / "fig").mkdir()
    (tmp_path / "fig" / "a_light.svg").write_text("<svg/>", encoding="utf-8")
    root = make_repo(tmp_path, docs=_guide_with(
        '<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="../fig/a_dark.svg">\n'
        '  <img alt="a figure" src="../fig/a_light.svg">\n</picture>'))
    assert check(root) == ["docs/GUIDE.md: broken link ../fig/a_dark.svg (no such file)"]


def test_html_image_whose_sources_exist_passes(tmp_path):
    (tmp_path / "fig").mkdir()
    for name in ("a_light.svg", "a_dark.svg"):
        (tmp_path / "fig" / name).write_text("<svg/>", encoding="utf-8")
    root = make_repo(tmp_path, docs=_guide_with(
        '<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="../fig/a_dark.svg">\n'
        '  <img alt="a figure" src="../fig/a_light.svg">\n</picture>'))
    assert check(root) == []


def test_link_outside_the_repository_fails(tmp_path):
    (tmp_path / "outside.md").write_text("# Out\n", encoding="utf-8")
    root = make_repo(tmp_path / "repo", docs=_guide_with("See [out](../../outside.md)."))
    assert check(root) == ["docs/GUIDE.md: broken link ../../outside.md (outside the repository)"]


# -- `# test` shell blocks -----------------------------------------------------------
def _block(body: str) -> dict[str, str]:
    return _guide_with(f"```bash\n# test\n{body}\n```")


def test_test_block_that_exits_zero_passes(tmp_path):
    root = make_repo(tmp_path, docs=_block("true"))
    assert check(root, skip_exec=False) == []


def test_test_block_that_exits_nonzero_fails(tmp_path):
    root = make_repo(tmp_path, docs=_block("exit 3"))
    problems = check(root, skip_exec=False)
    assert len(problems) == 1 and problems[0].startswith("docs/GUIDE.md: `# test` block 1 failed offline")
    # and --skip-exec does not run it
    assert check(root, skip_exec=True) == []


def test_test_block_runs_with_errexit(tmp_path):
    # bash -e: a failing command in the middle fails the block even if the last succeeds
    root = make_repo(tmp_path, docs=_block("false\ntrue"))
    assert len(check(root, skip_exec=False)) == 1


def test_test_block_that_reaches_for_the_network_fails_fast(tmp_path):
    # 192.0.2.1 is TEST-NET-1 (RFC 5737): even if the proxy were ignored, no real host is
    # contacted and no DNS lookup happens. With the proxy honoured the request goes to
    # 127.0.0.1:9 (check_docs.py:149), which refuses at once.
    body = (f'"{sys.executable}" -c "import urllib.request; '
            f"urllib.request.urlopen('http://192.0.2.1/', timeout=3)\"")
    root = make_repo(tmp_path, docs=_block(body))
    t0 = time.monotonic()
    problems = check(root, skip_exec=False)
    elapsed = time.monotonic() - t0
    assert len(problems) == 1 and "failed offline" in problems[0], problems
    assert "Connection refused" in problems[0], problems[0]
    assert elapsed < 2.5, f"took {elapsed:.1f} s: the dead proxy should refuse, not time out"


def test_untagged_shell_block_is_not_run(tmp_path):
    root = make_repo(tmp_path, docs=_guide_with("```bash\nexit 3\n```"))
    assert check(root, skip_exec=False) == []


# -- the documentation set ------------------------------------------------------------
def test_missing_document_fails(tmp_path):
    cfg = dict(DOCS_YAML, documents=DOCS_YAML["documents"] + ["docs/ABSENT.md"])
    root = make_repo(tmp_path, cfg=cfg)
    assert check(root) == ["docs/ABSENT.md: missing"]


def test_main_exit_codes(tmp_path, capsys, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_docs.py", "--root", str(root), "--skip-exec"])
    assert check_docs.main() == 0
    assert capsys.readouterr().out.strip().endswith("check_docs: OK")
    (root / "LOG.md").unlink()
    assert check_docs.main() == 1
    out = capsys.readouterr().out
    assert "check_docs: LOG.md: missing" in out and "FAILED, 1 problem(s)" in out


def test_a_missing_from_heading_is_reported_and_the_whole_document_checked(tmp_path):
    root = make_repo(tmp_path, docs={"REPORT.md": "# Run two\n\nEM was 0.731 by hand.\n"})
    problems = check(root)
    assert any("heading '# Run 2' not found" in p for p in problems)
    assert any("'0.731'" in p for p in problems)


def test_a_second_docs_config_adds_its_documents_and_exemptions(tmp_path):
    cfg = {k: v for k, v in DOCS_YAML.items() if k not in ("number_rule_exempt", "number_rule_from_heading")}
    cfg["documents"] = ["README.md", "docs/GUIDE.md"]
    root = make_repo(tmp_path, cfg=cfg)
    assert check(root) == []
    (root / "LOG.md").write_text("# Log\n\nEM was 0.731.\n", encoding="utf-8")
    (root / "configs" / "docs_more.yaml").write_text(yaml.safe_dump(
        {"documents": ["LOG.md", "NEW.md"], "number_rule_exempt": {"LOG.md": "a dated log"}}), encoding="utf-8")
    assert check(root) == ["NEW.md: missing"]


def test_token_syntax_quoted_as_code_is_not_an_unfilled_token(tmp_path):
    root = make_repo(tmp_path, docs={"LOG.md": LOG + "\nPlaceholders look like `{{token}}`.\n"})
    assert check(root) == []


# ---- what is not a result, and where a number is -------------------------------------------------------
def test_names_and_dates_are_not_results_even_across_a_line_break():
    text = ("Every slice's SHA-256 is kept. The licence is CC BY-SA\n   3.0 here, and CC BY\n  4.0 there. Measured 13 to 15 Sep. Apache License 2.0.\n"
            "Granite 4.0 and Gemma 4 answered; see sections 5 to 8 and section 11.")
    prose = check_docs.prose_of(text)
    assert not check_docs.NUMBER.findall(prose)
    assert prose.count("\n") == text.count("\n")                    # line numbers still point into the template


def test_prose_keeps_its_line_count_when_code_and_comments_go():
    text = "a\n```bash\nx = 256\ny = 3\n```\nb 0.412\n<!-- 7\n8 -->\nc"
    prose = check_docs.prose_of(text)
    assert prose.count("\n") == text.count("\n") and prose.split("\n")[5].strip() == "b 0.412"
